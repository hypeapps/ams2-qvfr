#include "mvr_ps.h"

#include <intrin.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "dxbc.h"
#include "hooks.h"
#include "mvr_ps_patch.h"
#include "mvr_vs.h"
#include "sun_fx.h"

namespace qvfr::mvrps {

namespace {

// ID3D11Device vtable (d3d11.h order).
constexpr int kSlotCreateGeometryShader = 13, kSlotCreateGeometryShaderSo = 14, kSlotCreatePixelShader = 15;

using CreateGSFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                               ID3D11GeometryShader**);
using CreateGSSoFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, const D3D11_SO_DECLARATION_ENTRY*,
                                                 UINT, const UINT*, UINT, UINT, ID3D11ClassLinkage*,
                                                 ID3D11GeometryShader**);
CreateGSFn o_createGs = nullptr;
CreateGSSoFn o_createGsSo = nullptr;
void* g_target = nullptr;
bool g_installed = false;
std::atomic<bool> g_on{false};
std::atomic<bool> g_dead{false};
bool g_testing = false;
uintptr_t g_selfBase = 0, g_selfEnd = 0;
ID3D11DeviceContext* g_checkedCtx = nullptr;
std::atomic<bool> g_submitted{false};

std::atomic<uint64_t> g_made{0};        // calls with bytecode and an output pointer
std::atomic<uint64_t> g_patched{0};
std::atomic<uint64_t> g_notNeeded{0};
std::atomic<uint64_t> g_refused{0};
std::atomic<uint64_t> g_failed{0};      // CreatePixelShader of the patched shader failed: the original
std::atomic<uint64_t> g_noMvr{0};
std::atomic<uint64_t> g_own{0};
std::atomic<uint64_t> g_other{0};       // class linkage or no output pointer
std::atomic<uint64_t> g_beforeSubmit{0};
std::atomic<uint64_t> g_patchUs{0};
std::atomic<uint64_t> g_inputs{0}, g_wrapped{0}, g_scaled{0};
std::atomic<uint64_t> g_tex[kTextures] = {};
std::atomic<uint64_t> g_gsMade{0}, g_gsLayered{0};  // geometry shaders, of them writing the array or viewport index
uint64_t g_loggedMade = 0;

std::mutex g_m;
std::vector<std::pair<std::string, uint64_t>> g_reasons;
std::shared_mutex g_sm;
std::unordered_map<void*, Info> g_shaders;
// O1: InfoOfRt's direct-mapped memo (render thread only); every new shader starts a new generation.
std::atomic<uint32_t> g_gen{1};
struct Memo {
    void* ps = nullptr;
    uint32_t gen = 0;
    Info info;
};
constexpr size_t kMemo = 2048;
Memo g_memo[kMemo];

void Fail(const char* where, unsigned long code) {
    g_on = false;
    if (g_dead.exchange(true)) return;
    Log("MVR PS patch: ERROR: exception 0x%08lX in %s; off for this session (the game gets its own pixel shaders)",
        code, where);
}

template <class F>
void Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

double Us(const LARGE_INTEGER& a, const LARGE_INTEGER& b) {
    static const double freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return (double)f.QuadPart;
    }();
    return (double)(b.QuadPart - a.QuadPart) * 1e6 / freq;
}

void SumText(const void* code, size_t len, char out[40]) {
    uint32_t s[4] = {};
    if (len >= 20) memcpy(s, (const uint8_t*)code + 4, 16);
    snprintf(out, 40, "%08x-%08x-%08x-%08x", s[0], s[1], s[2], s[3]);
}

void Refuse(std::atomic<uint64_t>& counter, std::string reason, const void* code, size_t len) {
    counter.fetch_add(1, std::memory_order_relaxed);
    if (const size_t at = reason.find(" at token"); at != std::string::npos) reason.resize(at);
    bool first = false;
    {
        std::lock_guard lock(g_m);
        auto it = g_reasons.begin();
        for (; it != g_reasons.end(); ++it)
            if (it->first == reason) break;
        if (it == g_reasons.end()) {
            first = true;
            g_reasons.emplace_back(reason, 1);
        } else {
            ++it->second;
        }
    }
    if (first) {
        char sum[40];
        SumText(code, len, sum);
        Log("MVR PS patch: original shader for %s (%zu bytes): %s (counted from now on)", sum, len, reason.c_str());
    }
}

// The patched shader (kind Patched), or null and the kind of the original the caller makes.
ID3D11PixelShader* MakePatched(ID3D11Device* dev, const void* code, SIZE_T len, ID3D11ClassLinkage* link,
                               ID3D11PixelShader** out, sunfx::CreatePsFn create, uintptr_t caller, Info& made) {
    made = Info{};
    Kind& kind = made.kind;
    if (!code || !len || !out || link) {
        g_other.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    {
        const char* b = (const char*)code;
        made.liveTrack = std::search(b, b + len, kLiveTrackName, kLiveTrackName + strlen(kLiveTrackName)) != b + len;
    }
    g_made.fetch_add(1, std::memory_order_relaxed);
    if (!g_testing && caller >= g_selfBase && caller < g_selfEnd) {
        g_own.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (!mvrvs::DeviceHasMvr(dev)) {
        g_noMvr.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    dxbc::MvrPsPatchOptions opt;
    opt.cbSlot = kCbSlot;
    opt.texBase = kTexBase;
    opt.screenTextures.push_back(kLiveTrackName);  // texture 5 (B9d run 5)
    std::vector<uint8_t> patched;
    dxbc::MvrPsPatchInfo info;
    std::string err;
    const dxbc::MvrPsPatchResult r = dxbc::PatchPsForMvr(code, len, opt, patched, &info, &err);
    if (r == dxbc::MvrPsPatchResult::NotNeeded) {
        g_notNeeded.fetch_add(1, std::memory_order_relaxed);
        kind = Kind::NotNeeded;
        return nullptr;
    }
    kind = Kind::Original;
    if (r == dxbc::MvrPsPatchResult::Refused) {
        Refuse(g_refused, err, code, len);
        return nullptr;
    }
    ID3D11PixelShader* ps = nullptr;
    const HRESULT hr = create(dev, patched.data(), patched.size(), nullptr, &ps);
    if (FAILED(hr) || !ps) {
        if (ps) ps->Release();
        char why[64];
        snprintf(why, sizeof(why), "CreatePixelShader of the patched shader 0x%08lX", (unsigned long)hr);
        Refuse(g_failed, why, code, len);
        return nullptr;
    }
    QueryPerformanceCounter(&t1);
    kind = Kind::Patched;
    for (unsigned k = 0; k < kTextures && k < info.textureRegs.size(); ++k)
        if (info.textureRegs[k] < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) made.slot[k] = (uint8_t)info.textureRegs[k];
    g_patchUs.fetch_add((uint64_t)Us(t0, t1), std::memory_order_relaxed);
    g_inputs.fetch_add(info.screenInputs, std::memory_order_relaxed);
    g_wrapped.fetch_add(info.wrapped, std::memory_order_relaxed);
    g_scaled.fetch_add(info.scaled, std::memory_order_relaxed);
    for (unsigned k = 0; k < kTextures; ++k)
        if (info.textureMask & (1u << k)) g_tex[k].fetch_add(1, std::memory_order_relaxed);
    if (!g_submitted.load(std::memory_order_relaxed)) g_beforeSubmit.fetch_add(1, std::memory_order_relaxed);
    if (g_patched.fetch_add(1, std::memory_order_relaxed) == 0) {
        char sum[40];
        SumText(code, len, sum);
        Log("MVR PS patch: first patched pixel shader %s (%zu -> %zu bytes, %u screen inputs, textures 0x%x, %u reads "
            "with a branch, view number v%u)",
            sum, len, patched.size(), info.screenInputs, info.textureMask, info.wrapped, info.viewReg);
    }
    return ps;
}

// In the sun effects module's CreatePixelShader hook.
HRESULT CreatePixelShader(ID3D11Device* dev, const void* code, SIZE_T len, ID3D11ClassLinkage* link,
                          ID3D11PixelShader** out, sunfx::CreatePsFn create, uintptr_t caller) {
    if (!g_on.load(std::memory_order_relaxed)) return create(dev, code, len, link, out);
    ID3D11PixelShader* ps = nullptr;
    Info made;
    Protect("CreatePixelShader", [&] { ps = MakePatched(dev, code, len, link, out, create, caller, made); });
    HRESULT hr = S_OK;
    if (ps)
        *out = ps;
    else
        hr = create(dev, code, len, link, out);
    if (SUCCEEDED(hr) && out && *out && made.kind != Kind::Unknown) {
        std::unique_lock lock(g_sm);
        g_shaders[*out] = made;
        g_gen.fetch_add(1, std::memory_order_acq_rel);  // the render thread's memo (InfoOfRt) looks again
    }
    return hr;
}

void NoteGeometryShader(const void* code, SIZE_T len) {
    if (!code || !len || !g_on.load(std::memory_order_relaxed)) return;
    Protect("CreateGeometryShader", [&] {
        g_gsMade.fetch_add(1, std::memory_order_relaxed);
        if (!dxbc::HasOutputSystemValue(code, len, 4) && !dxbc::HasOutputSystemValue(code, len, 5)) return;
        if (g_gsLayered.fetch_add(1, std::memory_order_relaxed) == 0) {
            char sum[40];
            SumText(code, len, sum);
            Log("MVR PS patch: geometry shader %s writes the render target array or viewport index (counted from now "
                "on; a patched pixel shader behind it would read view 1's textures)",
                sum);
        }
    });
}

HRESULT STDMETHODCALLTYPE D_CreateGeometryShader(ID3D11Device* dev, const void* code, SIZE_T len,
                                                 ID3D11ClassLinkage* link, ID3D11GeometryShader** out) {
    NoteGeometryShader(code, len);
    return o_createGs(dev, code, len, link, out);
}

HRESULT STDMETHODCALLTYPE D_CreateGeometryShaderSo(ID3D11Device* dev, const void* code, SIZE_T len,
                                                   const D3D11_SO_DECLARATION_ENTRY* so, UINT entries,
                                                   const UINT* strides, UINT numStrides, UINT stream,
                                                   ID3D11ClassLinkage* link, ID3D11GeometryShader** out) {
    NoteGeometryShader(code, len);
    return o_createGsSo(dev, code, len, so, entries, strides, numStrides, stream, link, out);
}

bool Hook(ID3D11Device* dev) {
    void** vt = *(void***)dev;
    if (!sunfx::SetCreatePsFilter(&CreatePixelShader)) {
        Log("MVR PS patch: the sun effects module's CreatePixelShader hook is not in place; off");
        return false;
    }
    g_target = vt[kSlotCreatePixelShader];
    // Diagnostics only: the pixel shader patch works without them.
    if (!hooks::CreateAt(vt[kSlotCreateGeometryShader], (void*)D_CreateGeometryShader, (void**)&o_createGs,
                         "CreateGeometryShader") ||
        !hooks::CreateAt(vt[kSlotCreateGeometryShaderSo], (void*)D_CreateGeometryShaderSo, (void**)&o_createGsSo,
                         "CreateGeometryShaderWithStreamOutput"))
        Log("MVR PS patch: no geometry shader hooks (layered geometry shaders are not counted)");
    return true;
}

void LogLine(const char* what) {
    const uint64_t patched = g_patched.load();
    Log("MVR PS patch stats%s: %s; pixel shaders made %llu: patched %llu (%llu before the first Submit, %.0f us each; "
        "screen inputs %llu, reads with a branch %llu, of them in UV %llu; globalTexture0..4 and the live track map in "
        "%llu/%llu/%llu/%llu/%llu/%llu), "
        "not needed %llu, refused %llu, creation failed %llu, device without MVR %llu, proxy's own %llu; other calls "
        "%llu; geometry shaders %llu, writing the array or viewport index %llu",
        what, g_dead ? "STOPPED" : g_on ? "on" : "off", (unsigned long long)g_made.load(), (unsigned long long)patched,
        (unsigned long long)g_beforeSubmit.load(), patched ? (double)g_patchUs.load() / (double)patched : 0.0,
        (unsigned long long)g_inputs.load(), (unsigned long long)g_wrapped.load(), (unsigned long long)g_scaled.load(),
        (unsigned long long)g_tex[0].load(), (unsigned long long)g_tex[1].load(), (unsigned long long)g_tex[2].load(),
        (unsigned long long)g_tex[3].load(), (unsigned long long)g_tex[4].load(), (unsigned long long)g_tex[5].load(),
        (unsigned long long)g_notNeeded.load(),
        (unsigned long long)g_refused.load(), (unsigned long long)g_failed.load(), (unsigned long long)g_noMvr.load(),
        (unsigned long long)g_own.load(), (unsigned long long)g_other.load(), (unsigned long long)g_gsMade.load(),
        (unsigned long long)g_gsLayered.load());
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    bool ok = false;
    Protect("Install", [&] {
        if (!hooks::Init()) return;
        HMODULE self = nullptr;
        MODULEINFO mi{};
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)&Install, &self) &&
            GetModuleInformation(GetCurrentProcess(), self, &mi, sizeof(mi))) {
            g_selfBase = (uintptr_t)mi.lpBaseOfDll;
            g_selfEnd = g_selfBase + mi.SizeOfImage;
        }
        using CreateFn = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL*, UINT,
                                          UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);
        HMODULE d3d = LoadLibraryW(L"d3d11.dll");
        auto create = d3d ? (CreateFn)GetProcAddress(d3d, "D3D11CreateDevice") : nullptr;
        ID3D11Device* dev = nullptr;
        HRESULT hr = create ? create(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                     nullptr, nullptr)
                            : E_NOINTERFACE;
        if (SUCCEEDED(hr) && dev) ok = Hook(dev);
        if (dev) dev->Release();
        Log("MVR PS patch: %s (CreatePixelShader through the sun effects hook; extra view: cb slot %u, "
            "globalTexture0..4 in t%u..t%u; MVR stays off)%s",
            ok ? "on" : "FAILED", kCbSlot, kTexBase, kTexBase + kTextures - 1,
            SUCCEEDED(hr) ? "" : "; no WARP device");
    });
    g_installed = ok;
    g_on = ok;
    return ok;
}

void OnFrameStart(ID3D11DeviceContext* ctx) {
    if (!g_installed || !ctx || ctx == g_checkedCtx || g_testing) return;
    Protect("OnFrameStart", [&] {
        g_checkedCtx = ctx;
        if (!g_submitted.exchange(true))
            Log("MVR PS patch: first Submit; %llu pixel shaders patched so far", (unsigned long long)g_patched.load());
        ID3D11Device* dev = nullptr;
        ctx->GetDevice(&dev);
        if (!dev) return;
        void* fn = (*(void***)dev)[kSlotCreatePixelShader];
        if (fn != g_target)
            Log("MVR PS patch: the game's D3D11 device has another CreatePixelShader (%p, hooked %p): a layer in "
                "between; the stats show whether its shaders get patched",
                fn, g_target);
        dev->Release();
    });
}

Kind KindOf(ID3D11PixelShader* ps) { return InfoOf(ps).kind; }

Info InfoOf(ID3D11PixelShader* ps) {
    if (!ps) return Info{};
    std::shared_lock lock(g_sm);
    auto it = g_shaders.find(ps);
    return it == g_shaders.end() ? Info{} : it->second;
}

Info InfoOfRt(ID3D11PixelShader* ps) {
    if (!ps) return Info{};
    Memo& m = g_memo[((uintptr_t)ps >> 4) & (kMemo - 1)];
    const uint32_t gen = g_gen.load(std::memory_order_acquire);  // before the lookup: a shader made meanwhile misses next
    if (m.ps == ps && m.gen == gen) return m.info;
    const Info info = InfoOf(ps);
    m.ps = ps;
    m.gen = gen;
    m.info = info;
    return info;
}

bool On() { return g_installed && g_on.load(std::memory_order_relaxed); }

std::string Describe() {
    if (!g_installed) return "MVR PS patch off";
    char buf[160];
    snprintf(buf, sizeof(buf), "MVR PS patch %s (patched %llu of %llu, not needed %llu, refused %llu)",
             g_dead ? "STOPPED" : g_on ? "on" : "off", (unsigned long long)g_patched.load(),
             (unsigned long long)g_made.load(), (unsigned long long)g_notNeeded.load(),
             (unsigned long long)(g_refused.load() + g_failed.load()));
    return buf;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    const uint64_t made = g_made.load();
    if (made != g_loggedMade || totals) {
        g_loggedMade = made;
        LogLine(totals ? " (session)" : "");
    }
    if (!totals) return;
    std::lock_guard lock(g_m);
    for (const auto& [reason, n] : g_reasons)
        Log("MVR PS patch: original shader %llu times: %s", (unsigned long long)n, reason.c_str());
}

bool TestInstall() {
    g_testing = true;
    return Install();
}

void TestSetEnabled(bool on) { g_on = on && g_installed && !g_dead; }

TestState TestGet() {
    TestState t;
    t.made = g_made.load();
    t.patched = g_patched.load();
    t.notNeeded = g_notNeeded.load();
    t.refused = g_refused.load();
    t.failed = g_failed.load();
    t.gsMade = g_gsMade.load();
    t.gsLayered = g_gsLayered.load();
    t.faults = g_dead ? 1 : 0;
    return t;
}

}  // namespace qvfr::mvrps

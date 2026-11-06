#include "mvr_vs.h"

#include <intrin.h>
#include <psapi.h>

#include <nvapi.h>

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
#include "hooks.h"
#include "mvr_patch.h"

namespace qvfr::mvrvs {

namespace {

constexpr int kSlotCreateVertexShader = 12;  // ID3D11Device vtable (d3d11.h order)

using CreateVSFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                               ID3D11VertexShader**);
CreateVSFn o_createVs = nullptr;
void* g_target = nullptr;
bool g_installed = false;
std::atomic<bool> g_on{false};    // hook does its work (installed, not faulted; smoke test: TestSetEnabled)
std::atomic<bool> g_dead{false};  // an exception in this module: off for the session
bool g_testing = false;
uintptr_t g_selfBase = 0, g_selfEnd = 0;  // this DLL: its own shaders (composition, sun effects) stay unpatched
ID3D11DeviceContext* g_checkedCtx = nullptr;
std::atomic<bool> g_submitted{false};  // the first Submit was seen

// Our call into NVAPI may come back through the hooked CreateVertexShader.
thread_local bool t_inside = false;

std::atomic<uint64_t> g_made{0};         // calls of the hook with bytecode and an output pointer
std::atomic<uint64_t> g_patched{0};      // patched shaders the game got
std::atomic<uint64_t> g_refused{0};      // PatchVsForMvr refused: the original shader
std::atomic<uint64_t> g_nvFailed{0};     // NvAPI_D3D11_CreateVertexShaderEx failed: the original shader
std::atomic<uint64_t> g_noMvr{0};        // a device without Multi-View support
std::atomic<uint64_t> g_own{0};          // this DLL's shaders
std::atomic<uint64_t> g_other{0};        // class linkage or no output pointer (validation call)
std::atomic<uint64_t> g_beforeSubmit{0}; // patched before the first Submit
std::atomic<uint64_t> g_patchUs{0};      // time in the patch and NVAPI creation of the patched shaders
std::atomic<unsigned> g_maxReg{0};       // largest output register after the patch
uint64_t g_loggedMade = 0;
double g_firstMs = -1, g_lastMs = -1;

std::mutex g_m;  // reasons, devices, first/last times
std::vector<std::pair<std::string, uint64_t>> g_reasons;
struct DeviceInfo {
    ID3D11Device* dev;  // only compared
    bool mvr;
};
std::vector<DeviceInfo> g_devices;
// Every shader made through the hook: patched or not, and its DXBC checksum (gate B7 counts the draws by it). A new
// shader at the address of a released one replaces its entry.
struct ShaderInfo {
    bool patched;
    char sum[40];
};
std::shared_mutex g_sm;  // shared for the lookups (gate B8: every VSSetShader of an MVR pass)
std::unordered_map<void*, ShaderInfo> g_shaders;

std::string NvErr(NvAPI_Status s) {
    NvAPI_ShortString msg = {};
    NvAPI_GetErrorMessage(s, msg);
    char buf[96];
    snprintf(buf, sizeof(buf), "%s (%d)", msg, (int)s);
    return buf;
}

void Fail(const char* where, unsigned long code) {
    g_on = false;
    if (g_dead.exchange(true)) return;
    Log("MVR VS patch: ERROR: exception 0x%08lX in %s; off for this session (the game gets its own vertex shaders)",
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

// The DXBC checksum (bytes 4..19), the "Shader hash" RenderDoc shows.
void SumText(const void* code, size_t len, char out[40]) {
    uint32_t s[4] = {};
    if (len >= 20) memcpy(s, (const uint8_t*)code + 4, 16);
    snprintf(out, 40, "%08x-%08x-%08x-%08x", s[0], s[1], s[2], s[3]);
}

// One counter per reason; the first shader with a reason goes to the log at once.
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
        Log("MVR VS patch: original shader for %s (%zu bytes): %s (counted from now on)", sum, len, reason.c_str());
    }
}

bool DeviceHasMvr(ID3D11Device* dev) {
    std::lock_guard lock(g_m);
    for (const DeviceInfo& d : g_devices)
        if (d.dev == dev) return d.mvr;
    NV_QUERY_MULTIVIEW_SUPPORT_PARAMS q = {};
    q.version = NV_QUERY_MULTIVIEW_SUPPORT_PARAMS_VER;
    const NvAPI_Status st = NvAPI_D3D_QueryMultiViewSupport(dev, &q);
    const bool mvr = st == NVAPI_OK && q.bMultiViewSupported;
    g_devices.push_back({dev, mvr});
    Log("MVR VS patch: device %p: NvAPI_D3D_QueryMultiViewSupport %s, MultiView %u%s", (void*)dev, NvErr(st).c_str(),
        (unsigned)q.bMultiViewSupported, mvr ? "; its vertex shaders get patched" : "; its vertex shaders stay as they are");
    return mvr;
}

// The patched shader, or null (the caller makes the original).
ID3D11VertexShader* MakePatched(ID3D11Device* dev, const void* code, SIZE_T len, ID3D11ClassLinkage* link,
                                ID3D11VertexShader** out, uintptr_t caller) {
    if (!code || !len || !out || link) {
        g_other.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    g_made.fetch_add(1, std::memory_order_relaxed);
    if (!g_testing && caller >= g_selfBase && caller < g_selfEnd) {
        g_own.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (!DeviceHasMvr(dev)) {
        g_noMvr.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    dxbc::MvrVsPatchOptions opt;
    opt.cbSlot = kCbSlot;
    opt.extraViews = 1;
    opt.clip = true;
    std::vector<uint8_t> patched;
    dxbc::MvrVsPatchInfo info;
    std::string err;
    if (!dxbc::PatchVsForMvr(code, len, opt, patched, &info, &err)) {
        Refuse(g_refused, err, code, len);
        return nullptr;
    }
    NV_CUSTOM_SEMANTIC sem[2] = {};
    sem[0].version = NV_CUSTOM_SEMANTIC_VERSION;
    sem[0].NVCustomSemanticType = NV_POSITION_SEMANTIC;
    strcpy_s(sem[0].NVCustomSemanticNameString, "NV_POSITION");
    sem[1].version = NV_CUSTOM_SEMANTIC_VERSION;
    sem[1].NVCustomSemanticType = NV_CLIP_DISTANCE_0_SEMANTIC;
    strcpy_s(sem[1].NVCustomSemanticNameString, "NV_CLIP_DISTANCE_0");
    NvAPI_D3D11_CREATE_VERTEX_SHADER_EX args = {};
    args.version = NVAPI_D3D11_CREATEVERTEXSHADEREX_VERSION;
    args.NumCustomSemantics = 2;
    args.pCustomSemantics = sem;
    ID3D11VertexShader* vs = nullptr;
    const NvAPI_Status st = NvAPI_D3D11_CreateVertexShaderEx(dev, patched.data(), patched.size(), nullptr, &args, &vs);
    if (st != NVAPI_OK || !vs) {
        if (vs) vs->Release();
        Refuse(g_nvFailed, "NvAPI_D3D11_CreateVertexShaderEx " + NvErr(st), code, len);
        return nullptr;
    }
    QueryPerformanceCounter(&t1);
    g_patchUs.fetch_add((uint64_t)Us(t0, t1), std::memory_order_relaxed);
    const uint64_t n = g_patched.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!g_submitted.load(std::memory_order_relaxed)) g_beforeSubmit.fetch_add(1, std::memory_order_relaxed);
    unsigned prev = g_maxReg.load(std::memory_order_relaxed);
    while (info.viewClipReg > prev && !g_maxReg.compare_exchange_weak(prev, info.viewClipReg)) {
    }
    const double now = NowMs();
    {
        std::lock_guard lock(g_m);
        if (g_firstMs < 0) g_firstMs = now;
        g_lastMs = now;
    }
    if (n == 1) {
        char sum[40];
        SumText(code, len, sum);
        Log("MVR VS patch: first patched vertex shader %s (%zu -> %zu bytes, position o%u, view 1 o%u, clip o%u/o%u)",
            sum, len, patched.size(), info.positionReg, info.viewPositionReg, info.clipReg, info.viewClipReg);
    }
    return vs;
}

HRESULT STDMETHODCALLTYPE D_CreateVertexShader(ID3D11Device* dev, const void* code, SIZE_T len,
                                               ID3D11ClassLinkage* link, ID3D11VertexShader** out) {
    const uintptr_t caller = (uintptr_t)_ReturnAddress();
    if (t_inside || !g_on.load(std::memory_order_relaxed)) return o_createVs(dev, code, len, link, out);
    ID3D11VertexShader* vs = nullptr;
    t_inside = true;
    Protect("CreateVertexShader", [&] { vs = MakePatched(dev, code, len, link, out, caller); });
    t_inside = false;
    const bool patched = vs != nullptr;
    HRESULT hr = S_OK;
    if (patched)
        *out = vs;
    else
        hr = o_createVs(dev, code, len, link, out);
    if (SUCCEEDED(hr) && out && *out && code && len) {
        ShaderInfo info{patched, {}};
        SumText(code, len, info.sum);
        std::unique_lock lock(g_sm);
        g_shaders[*out] = info;
    }
    return hr;
}

bool Hook(ID3D11Device* dev) {
    void* t = (*(void***)dev)[kSlotCreateVertexShader];
    const bool ok = hooks::CreateAt(t, (void*)D_CreateVertexShader, (void**)&o_createVs, "CreateVertexShader");
    if (ok) g_target = t;
    return ok;
}

bool InitNvapi() {
    const NvAPI_Status st = NvAPI_Initialize();
    if (st != NVAPI_OK) {
        Log("MVR VS patch: NvAPI_Initialize %s; off (NVIDIA only)", NvErr(st).c_str());
        return false;
    }
    NvU32 driver = 0;
    NvAPI_ShortString branch = {};
    if (NvAPI_SYS_GetDriverAndBranchVersion(&driver, branch) == NVAPI_OK)
        Log("MVR VS patch: NVAPI ready, NVIDIA driver %u.%02u (%s)", driver / 100, driver % 100, branch);
    return true;
}

void LogLine(const char* what) {
    const uint64_t patched = g_patched.load();
    double firstS, lastS;
    {
        std::lock_guard lock(g_m);
        firstS = g_firstMs / 1000.0;
        lastS = g_lastMs / 1000.0;
    }
    Log("MVR VS patch stats%s: %s; vertex shaders made %llu: patched %llu (%llu before the first Submit, %.0f us each, "
        "%.1f..%.1f s), refused %llu, NVAPI failures %llu, device without MVR %llu, proxy's own %llu; other calls %llu; "
        "largest output register o%u",
        what, g_dead ? "STOPPED" : g_on ? "on" : "off", (unsigned long long)g_made.load(), (unsigned long long)patched,
        (unsigned long long)g_beforeSubmit.load(), patched ? (double)g_patchUs.load() / (double)patched : 0.0, firstS,
        lastS, (unsigned long long)g_refused.load(), (unsigned long long)g_nvFailed.load(),
        (unsigned long long)g_noMvr.load(), (unsigned long long)g_own.load(), (unsigned long long)g_other.load(),
        g_maxReg.load());
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    bool ok = false;
    Protect("Install", [&] {
        if (!hooks::Init() || !InitNvapi()) return;
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
        Log("MVR VS patch: D3D11 hook %s (CreateVertexShader; cb slot %u, 1 extra view with clip distances; MVR stays "
            "off)%s",
            ok ? "installed" : "FAILED", kCbSlot, SUCCEEDED(hr) ? "" : "; no WARP device");
    });
    g_installed = ok;
    g_on = ok;
    return ok;
}

void OnFrameStart(ID3D11DeviceContext* ctx) {
    if (!g_installed || !ctx || ctx == g_checkedCtx || g_testing) return;
    Protect("OnFrameStart", [&] {
        g_checkedCtx = ctx;
        if (!g_submitted.exchange(true)) {
            Log("MVR VS patch: first Submit; %llu vertex shaders patched so far", (unsigned long long)g_patched.load());
        }
        ID3D11Device* dev = nullptr;
        ctx->GetDevice(&dev);
        if (!dev) return;
        void* fn = (*(void***)dev)[kSlotCreateVertexShader];
        if (fn != g_target)
            Log("MVR VS patch: the game's D3D11 device has another CreateVertexShader (%p, hooked %p): a layer in "
                "between (a debug layer still calls the hooked one); the stats show whether its shaders get patched",
                fn, g_target);
        dev->Release();
    });
}

bool IsPatched(ID3D11VertexShader* vs, char* sum) {
    if (sum) sum[0] = 0;
    if (!vs) return false;
    std::shared_lock lock(g_sm);
    auto it = g_shaders.find(vs);
    if (it == g_shaders.end()) return false;
    if (sum) memcpy(sum, it->second.sum, sizeof(it->second.sum));
    return it->second.patched;
}

std::string Describe() {
    if (!g_installed) return "MVR VS patch off";
    char buf[160];
    snprintf(buf, sizeof(buf), "MVR VS patch %s (patched %llu of %llu, refused %llu, NVAPI failures %llu)",
             g_dead ? "STOPPED" : g_on ? "on" : "off", (unsigned long long)g_patched.load(),
             (unsigned long long)g_made.load(), (unsigned long long)g_refused.load(),
             (unsigned long long)g_nvFailed.load());
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
        Log("MVR VS patch: original shader %llu times: %s", (unsigned long long)n, reason.c_str());
}

bool TestInstall(ID3D11Device* dev) {
    // The game's path (hook through a WARP device); the test exe's own shaders (this module is compiled into it) get
    // patched. A debug layer device has its own function in the vtable, which calls the hooked one of the core device.
    if (!Install()) return false;
    g_testing = true;
    void* fn = (*(void***)dev)[kSlotCreateVertexShader];
    Log("MVR VS patch: the test device's CreateVertexShader %s the hooked one (%p, hooked %p)",
        fn == g_target ? "is" : "is NOT", fn, g_target);
    return true;
}

void TestSetEnabled(bool on) { g_on = on && g_installed && !g_dead; }

TestState TestGet() {
    TestState t;
    t.made = g_made.load();
    t.patched = g_patched.load();
    t.refused = g_refused.load();
    t.nvapiFailed = g_nvFailed.load();
    t.skipped = g_noMvr.load() + g_own.load() + g_other.load();
    t.faults = g_dead ? 1 : 0;
    t.maxOutputReg = g_maxReg.load();
    return t;
}

}  // namespace qvfr::mvrvs

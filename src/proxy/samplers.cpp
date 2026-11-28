#include "samplers.h"

#include <windows.h>

#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "hooks.h"

namespace qvfr::samplers {

namespace {

constexpr int kSlotCreateSamplerState = 23;  // ID3D11Device vtable
constexpr int kSlotPSSetSamplers = 10;       // ID3D11DeviceContext vtable
constexpr size_t kMaxEntries = 256;

using CreateFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_SAMPLER_DESC*, ID3D11SamplerState**);
CreateFn o_create = nullptr;
using PsSetFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11SamplerState* const*);
PsSetFn o_psSet = nullptr;
void* g_psTarget = nullptr;

constexpr size_t kMaxSites = 8;

struct Entry {
    D3D11_SAMPLER_DESC d{};
    std::string where;  // module+offset of the first caller
    bool game = false;  // a caller in the game's exe
    bool scene = false;  // the mip bias applies to it
    std::vector<uintptr_t> sites;  // distinct callers, at most kMaxSites
    uint64_t count = 0;
};

std::mutex g_m;
std::vector<Entry> g_list;
size_t g_logged = 0;  // g_list[0, g_logged) already in the log
uint64_t g_creations = 0, g_failed = 0, g_overflow = 0;
std::atomic<bool> g_on{false};
std::atomic<bool> g_dead{false};
bool g_installed = false;
uintptr_t g_exe = 0;

// Mip bias: a game scene sampler and its copies, one per bias step in use (step k = bias -k/8 over the game's).
constexpr int kSteps = 33;  // 0 .. -4
constexpr float kStep = 0.125f;
constexpr size_t kMaxSets = 128;
constexpr size_t kTableSize = 2048;  // power of two, above kMaxSets * a few steps in use
constexpr int kBuckets = 5;          // eye groups 0..3, outside the groups
const char* const kBucketNames[kBuckets] = {"outer L", "outer R", "focus L", "focus R", "outside the eye groups"};

struct SceneSampler {
    D3D11_SAMPLER_DESC d{};
    std::atomic<ID3D11SamplerState*> s[kSteps] = {};  // [0] the game's
};
SceneSampler g_sets[kMaxSets];
std::atomic<size_t> g_setCount{0};
uint64_t g_copyFailed = 0, g_setOverflow = 0;

// Open addressing on the object's address; written under g_m, read without a lock (the set index before the key).
struct Slot {
    std::atomic<ID3D11SamplerState*> key{nullptr};
    std::atomic<int> set{-1};
    std::atomic<bool> original{false};
};
Slot g_table[kTableSize];

std::atomic<int> g_switch{-1};
std::atomic<int> g_want[2] = {0, 0};  // the step of outer's and focus's views; 0 = the game's samplers
bool g_dlss = false, g_ini = false;
std::atomic<int> g_group{-1};
std::atomic<bool> g_post{false};        // the group is past its TAA resolve
std::atomic<bool> g_groupsSeen{false};  // quad eye groups run (else outer's step holds everywhere)
std::atomic<uint64_t> g_replaced[kBuckets];
std::atomic<uint64_t> g_rebound[kBuckets];
uint64_t g_groupStarts[kBuckets] = {}, g_sceneEnds[kBuckets] = {};
uint64_t g_replacedTotal = 0;  // TestReplacedBinds: kept over the setting changes
ULONGLONG g_periodStart = 0;
ID3D11DeviceContext* g_checkedCtx = nullptr;

void Fail(const char* where, unsigned long code) {
    g_on = false;
    if (g_dead.exchange(true)) return;
    Log("Samplers: ERROR: exception 0x%08lX in %s; the sampler log stops", code, where);
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

std::string Where(uintptr_t a) {
    char buf[MAX_PATH + 32];
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)a, &m) ||
        !m) {
        snprintf(buf, sizeof(buf), "0x%llX", (unsigned long long)a);
        return buf;
    }
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(m, path, MAX_PATH);
    const wchar_t* name = wcsrchr(path, L'\\');
    snprintf(buf, sizeof(buf), "%s+0x%llX", Narrow(name ? name + 1 : path).c_str(),
             (unsigned long long)(a - (uintptr_t)m));
    return buf;
}

bool FromExe(uintptr_t caller) {
    HMODULE m = nullptr;
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR)caller, &m) &&
           (uintptr_t)m == g_exe;
}

// A scene texture sampler: no comparison or reduction, linear minification with linear mips (or anisotropic), mips
// allowed. Shadow maps, point sampling and LOD 0..0 (post-process passes, render targets) stay as they are.
bool Eligible(const D3D11_SAMPLER_DESC& d) {
    const unsigned f = (unsigned)d.Filter;
    if (f & 0x180) return false;
    const bool trilinear = (f & 0x11) == 0x11, aniso = (f & 0x40) != 0;
    return (trilinear || aniso) && d.MaxLOD > 0 && d.MaxLOD > d.MinLOD;
}

size_t Hash(const void* p) { return (size_t)((((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull) >> 54) & (kTableSize - 1); }

Slot* Lookup(const ID3D11SamplerState* s) {
    for (size_t i = Hash(s), n = 0; n < kTableSize; i = (i + 1) & (kTableSize - 1), ++n) {
        ID3D11SamplerState* k = g_table[i].key.load(std::memory_order_acquire);
        if (k == s) return &g_table[i];
        if (!k) return nullptr;
    }
    return nullptr;
}

// Under g_m. An original replaces a variant entry of the same object (the runtime gave an equal description the
// object it made for a variant); a variant never replaces an entry.
void Insert(ID3D11SamplerState* s, int set, bool original) {
    if (!s) return;
    if (Slot* e = Lookup(s)) {
        if (original && !e->original.load(std::memory_order_relaxed)) {
            e->set.store(set, std::memory_order_relaxed);
            e->original.store(true, std::memory_order_relaxed);
        }
        return;
    }
    for (size_t i = Hash(s), n = 0; n < kTableSize; i = (i + 1) & (kTableSize - 1), ++n) {
        if (g_table[i].key.load(std::memory_order_relaxed)) continue;
        g_table[i].set.store(set, std::memory_order_relaxed);
        g_table[i].original.store(original, std::memory_order_relaxed);
        g_table[i].key.store(s, std::memory_order_release);
        return;
    }
}

// Under g_m: set `i`'s copy for `step` (made once, on the device of the game's sampler).
void EnsureLocked(size_t i, int step) {
    SceneSampler& v = g_sets[i];
    if (step <= 0 || step >= kSteps || v.s[step].load(std::memory_order_relaxed)) return;
    ID3D11SamplerState* orig = v.s[0].load(std::memory_order_relaxed);
    ID3D11Device* dev = nullptr;
    orig->GetDevice(&dev);
    if (!dev) return;
    D3D11_SAMPLER_DESC d = v.d;
    d.MipLODBias = std::clamp(v.d.MipLODBias - step * kStep, -16.0f, 15.99f);
    ID3D11SamplerState* copy = nullptr;
    if (SUCCEEDED(o_create(dev, &d, &copy)) && copy) {
        Insert(copy, (int)i, false);
        v.s[step].store(copy, std::memory_order_release);
    } else {
        ++g_copyFailed;
    }
    dev->Release();
}

// Under g_m: a game scene sampler, once per object, with copies for the steps in use.
bool AddSceneLocked(const D3D11_SAMPLER_DESC& d, ID3D11SamplerState* orig) {
    if (Slot* e = Lookup(orig); e && e->original.load(std::memory_order_relaxed)) return true;
    const size_t i = g_setCount.load(std::memory_order_relaxed);
    if (i >= kMaxSets) {
        ++g_setOverflow;
        return false;
    }
    SceneSampler& v = g_sets[i];
    v.d = d;
    orig->AddRef();
    v.s[0].store(orig, std::memory_order_relaxed);
    Insert(orig, (int)i, true);
    EnsureLocked(i, g_want[0].load());
    EnsureLocked(i, g_want[1].load());
    g_setCount.store(i + 1, std::memory_order_release);
    return true;
}

// The step the current group wants now.
int WantNow() {
    if (g_post.load(std::memory_order_relaxed)) return 0;
    const int g = g_group.load(std::memory_order_relaxed);
    if (g < 0) return g_groupsSeen.load(std::memory_order_relaxed) ? 0 : g_want[0].load(std::memory_order_relaxed);
    return g_want[g >= 2 ? 1 : 0].load(std::memory_order_relaxed);
}

int Bucket() {
    const int g = g_group.load(std::memory_order_relaxed);
    return g >= 0 && g < 4 ? g : 4;
}

// The object to bind for `s` (a game scene sampler or one of its copies) at `step`; others as they are.
ID3D11SamplerState* Map(ID3D11SamplerState* s, int step) {
    if (!s) return s;
    const Slot* e = Lookup(s);
    if (!e) return s;
    const int set = e->set.load(std::memory_order_relaxed);
    if (set < 0) return s;
    ID3D11SamplerState* to = g_sets[set].s[step].load(std::memory_order_acquire);
    return to ? to : g_sets[set].s[0].load(std::memory_order_relaxed);
}

void Note(const D3D11_SAMPLER_DESC& d, uintptr_t caller, bool ok, bool game, bool variants) {
    std::lock_guard lock(g_m);
    ++g_creations;
    if (!ok) ++g_failed;
    for (Entry& e : g_list) {
        if (memcmp(&e.d, &d, sizeof(d)) != 0) continue;
        ++e.count;
        e.game |= game;
        e.scene |= variants;
        if (e.sites.size() < kMaxSites && std::find(e.sites.begin(), e.sites.end(), caller) == e.sites.end())
            e.sites.push_back(caller);
        return;
    }
    if (g_list.size() >= kMaxEntries) {
        ++g_overflow;
        return;
    }
    Entry e;
    e.d = d;
    e.where = Where(caller);
    e.game = game;
    e.scene = variants;
    e.sites.push_back(caller);
    e.count = 1;
    g_list.push_back(std::move(e));
}

HRESULT STDMETHODCALLTYPE D_CreateSamplerState(ID3D11Device* dev, const D3D11_SAMPLER_DESC* d,
                                               ID3D11SamplerState** out) {
    const HRESULT hr = o_create(dev, d, out);
    if (d && g_on.load(std::memory_order_relaxed)) {
        const uintptr_t caller = (uintptr_t)_ReturnAddress();
        Protect("CreateSamplerState", [&] {
            const bool game = FromExe(caller);
            bool scene = false;
            if (SUCCEEDED(hr) && game && out && *out && Eligible(*d)) {
                std::lock_guard lock(g_m);
                scene = AddSceneLocked(*d, *out);
            }
            Note(*d, caller, SUCCEEDED(hr), game, scene);
        });
    }
    return hr;
}

void STDMETHODCALLTYPE D_PSSetSamplers(ID3D11DeviceContext* c, UINT start, UINT n, ID3D11SamplerState* const* s) {
    ID3D11SamplerState* mapped[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
    ID3D11SamplerState* const* use = s;
    if (s && n && n <= D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT && g_setCount.load(std::memory_order_acquire) &&
        g_on.load(std::memory_order_relaxed)) {
        const int step = WantNow();
        if (step)
            Protect("PSSetSamplers", [&] {
                unsigned replaced = 0;
                for (UINT i = 0; i < n; ++i) {
                    mapped[i] = Map(s[i], step);
                    replaced += mapped[i] != s[i];
                }
                if (!replaced) return;
                use = mapped;
                g_replaced[Bucket()].fetch_add(replaced, std::memory_order_relaxed);
            });
    }
    o_psSet(c, start, n, use);
}

// The pixel shader slots as bound now, with the step of the current group and phase.
void RebindLocked(ID3D11DeviceContext* c) {
    ID3D11SamplerState* bound[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT] = {};
    c->PSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, bound);
    ID3D11SamplerState* mapped[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
    const int step = WantNow();
    UINT lo = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, hi = 0;
    for (UINT i = 0; i < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT; ++i) {
        mapped[i] = Map(bound[i], step);
        if (mapped[i] != bound[i]) {
            lo = std::min(lo, i);
            hi = i + 1;
        }
    }
    if (hi > lo) {
        o_psSet(c, lo, hi - lo, mapped + lo);
        g_rebound[Bucket()].fetch_add(hi - lo, std::memory_order_relaxed);
    }
    for (ID3D11SamplerState* b : bound)
        if (b) b->Release();
}

std::string BiasText() {
    const int o = g_want[0].load(), f = g_want[1].load();
    if (!o && !f) return "off";
    char buf[96];
    snprintf(buf, sizeof(buf), "outer %+.3f, focus %+.3f", -o * kStep, -f * kStep);
    return buf;
}

// Under g_m: the samplers replaced at bind under the current bias; `end`: its period ends (a change, the session's
// end) and the counts start again.
void LogPeriodLocked(const char* why, bool end) {
    const ULONGLONG now = GetTickCount64();
    uint64_t replaced[kBuckets], rebound[kBuckets], all = 0;
    for (int b = 0; b < kBuckets; ++b) {
        replaced[b] = end ? g_replaced[b].exchange(0) : g_replaced[b].load();
        rebound[b] = end ? g_rebound[b].exchange(0) : g_rebound[b].load();
        all += replaced[b] + rebound[b];
    }
    if (end) g_replacedTotal += all;
    std::string groups;
    for (int b = 0; b < kBuckets; ++b) {
        char buf[160];
        if (b < 4)
            snprintf(buf, sizeof(buf), "%s %llu (group starts %llu, TAA resolves %llu, slots rebound %llu), ",
                     kBucketNames[b], (unsigned long long)replaced[b], (unsigned long long)g_groupStarts[b],
                     (unsigned long long)g_sceneEnds[b], (unsigned long long)rebound[b]);
        else
            snprintf(buf, sizeof(buf), "%s %llu (rebound %llu)", kBucketNames[b], (unsigned long long)replaced[b],
                     (unsigned long long)rebound[b]);
        groups += buf;
        if (end) g_groupStarts[b] = g_sceneEnds[b] = 0;
    }
    Log("Samplers mip bias (%s): %s; %zu game scene samplers (%llu copies failed, %llu past the %zu kept); over %.1f s "
        "with this bias samplers replaced at bind %llu: %s",
        why, BiasText().c_str(), g_setCount.load(), (unsigned long long)g_copyFailed,
        (unsigned long long)g_setOverflow, kMaxSets, g_periodStart ? (now - g_periodStart) / 1000.0 : 0.0,
        (unsigned long long)all, groups.c_str());
    if (end) g_periodStart = now;
}

std::string FilterText(unsigned f) {
    const unsigned reduction = f & 0x180;
    std::string s = reduction == 0x80 ? "comparison " : reduction == 0x100 ? "minimum " : reduction == 0x180 ? "maximum " : "";
    if (f & 0x40) return s + "anisotropic";
    s += "min ";
    s += (f & 0x10) ? "linear" : "point";
    s += (f & 0x04) ? ", mag linear" : ", mag point";
    s += (f & 0x01) ? ", mip linear" : ", mip point";
    return s;
}

const char* AddressText(D3D11_TEXTURE_ADDRESS_MODE m) {
    switch (m) {
        case D3D11_TEXTURE_ADDRESS_WRAP: return "wrap";
        case D3D11_TEXTURE_ADDRESS_MIRROR: return "mirror";
        case D3D11_TEXTURE_ADDRESS_CLAMP: return "clamp";
        case D3D11_TEXTURE_ADDRESS_BORDER: return "border";
        case D3D11_TEXTURE_ADDRESS_MIRROR_ONCE: return "mirror once";
        default: return "?";
    }
}

const char* ComparisonText(D3D11_COMPARISON_FUNC c) {
    static const char* const kNames[] = {"?", "never", "less", "equal", "less equal", "greater", "not equal",
                                         "greater equal", "always"};
    return (unsigned)c < sizeof(kNames) / sizeof(kNames[0]) ? kNames[c] : "?";
}

void LogEntry(size_t i, const Entry& e) {
    const D3D11_SAMPLER_DESC& d = e.d;
    char minLod[32], maxLod[32];
    if (d.MinLOD <= -FLT_MAX)
        snprintf(minLod, sizeof(minLod), "min");
    else
        snprintf(minLod, sizeof(minLod), "%.2f", d.MinLOD);
    if (d.MaxLOD >= FLT_MAX)
        snprintf(maxLod, sizeof(maxLod), "max");
    else
        snprintf(maxLod, sizeof(maxLod), "%.2f", d.MaxLOD);
    Log("Samplers: #%zu %s: filter 0x%X (%s), address %s/%s/%s, MipLODBias %.3f, MaxAnisotropy %u, comparison %s, "
        "LOD %s..%s, border (%g, %g, %g, %g), created %llu from %zu%s call sites%s",
        i + 1, e.where.c_str(), (unsigned)d.Filter, FilterText((unsigned)d.Filter).c_str(), AddressText(d.AddressU),
        AddressText(d.AddressV), AddressText(d.AddressW), d.MipLODBias, d.MaxAnisotropy, ComparisonText(d.ComparisonFunc),
        minLod, maxLod, d.BorderColor[0], d.BorderColor[1], d.BorderColor[2], d.BorderColor[3],
        (unsigned long long)e.count, e.sites.size(), e.sites.size() >= kMaxSites ? "+" : "",
        e.scene ? "; scene sampler (mip bias)" : "");
}

}  // namespace

void Install() {
    if (g_installed || (!Cfg().samplerLog && !Cfg().dlaa)) return;
    Protect("Install", [&] {
        if (!hooks::Init()) return;
        g_exe = (uintptr_t)GetModuleHandleW(nullptr);
        using DevFn = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL*, UINT,
                                       UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);
        HMODULE d3d = LoadLibraryW(L"d3d11.dll");
        auto create = d3d ? (DevFn)GetProcAddress(d3d, "D3D11CreateDevice") : nullptr;
        ID3D11Device* dev = nullptr;
        const HRESULT hr = create ? create(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                           &dev, nullptr, nullptr)
                                  : E_NOINTERFACE;
        const bool ok = SUCCEEDED(hr) && dev && TestInstall(dev);
        if (dev) dev->Release();
        Log("Samplers: D3D11 hooks %s (CreateSamplerState, PSSetSamplers; sampler log %s, texture mip bias under DLSS "
            "%s)%s",
            ok ? "installed" : "FAILED", Cfg().samplerLog ? "on" : "off", Cfg().dlaa ? "available" : "off",
            SUCCEEDED(hr) ? "" : "; no WARP device");
    });
}

bool TestInstall(ID3D11Device* dev) {
    if (g_installed) return true;
    if (!dev || !hooks::Init()) return false;
    if (!g_exe) g_exe = (uintptr_t)GetModuleHandleW(nullptr);
    ID3D11DeviceContext* ctx = nullptr;
    dev->GetImmediateContext(&ctx);
    if (!ctx) return false;
    void* t = (*(void***)dev)[kSlotCreateSamplerState];
    void* p = (*(void***)ctx)[kSlotPSSetSamplers];
    ctx->Release();
    g_installed = hooks::CreateAt(t, (void*)D_CreateSamplerState, (void**)&o_create, "CreateSamplerState") &&
                  hooks::CreateAt(p, (void*)D_PSSetSamplers, (void**)&o_psSet, "PSSetSamplers");
    g_psTarget = p;
    g_periodStart = GetTickCount64();
    g_on = g_installed;
    return g_installed;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    Protect("LogStats", [&] {
        std::lock_guard lock(g_m);
        if (totals || g_want[0].load() || g_want[1].load())
            LogPeriodLocked(totals ? "session end" : "stats", totals);
        if (!Cfg().samplerLog) return;
        const size_t from = totals ? 0 : g_logged;
        if (from == g_list.size() && !totals) return;
        size_t negative = 0, negativeGame = 0, aniso = 0, game = 0;
        for (const Entry& e : g_list) {
            game += e.game;
            aniso += (e.d.Filter & 0x40) != 0;
            if (e.d.MipLODBias < 0) {
                ++negative;
                negativeGame += e.game;
            }
        }
        Log("Samplers%s: %zu distinct descriptions (%zu from the game's exe, %zu new) in %llu creations (%llu failed, "
            "%llu past the %zu kept); negative MipLODBias in %zu (from the game's exe %zu), anisotropic %zu",
            totals ? " (session total)" : "", g_list.size(), game, g_list.size() - g_logged,
            (unsigned long long)g_creations, (unsigned long long)g_failed, (unsigned long long)g_overflow, kMaxEntries,
            negative, negativeGame, aniso);
        for (size_t i = from; i < g_list.size(); ++i) LogEntry(i, g_list[i]);
        g_logged = g_list.size();
    });
}

void SetViewBias(bool dlss, bool ini, float outer, float focus) {
    if (!g_installed || !g_on.load(std::memory_order_relaxed)) return;
    const int sw = g_switch.load();
    const bool on = dlss && (sw >= 0 ? sw == 1 : ini);
    auto step = [&](float b) { return on ? std::clamp((int)std::lround(-b / kStep), 0, kSteps - 1) : 0; };
    const int so = step(outer), sf = step(focus);
    if (so == g_want[0].load() && sf == g_want[1].load() && dlss == g_dlss && ini == g_ini) return;
    Protect("SetViewBias", [&] {
        std::lock_guard lock(g_m);
        LogPeriodLocked("before the change", true);
        const size_t n = g_setCount.load(std::memory_order_acquire);
        for (size_t i = 0; i < n; ++i) {
            EnsureLocked(i, so);
            EnsureLocked(i, sf);
        }
        g_want[0] = so;
        g_want[1] = sf;
        g_dlss = dlss;
        g_ini = ini;
        const char* why = !dlss ? "DLSS is not running" : sw == 0 ? "switch dlss_mip_bias=0" : !on ? "[dlaa] mip_bias=0" : "";
        if (on)
            Log("Samplers: texture mip bias outer %+.3f, focus %+.3f over the game's (asked for %+.3f, %+.3f; %s)",
                -so * kStep, -sf * kStep, outer, focus, sw == 1 ? "switch dlss_mip_bias=1" : "[dlaa] mip_bias=1");
        else
            Log("Samplers: texture mip bias off (%s)", why);
    });
}

int GetSwitch() { return g_switch.load(); }

// The next frame start's SetViewBias applies it.
void SetSwitch(int v) { g_switch = std::clamp(v, -1, 1); }

void BeginGroup(int group, ID3D11DeviceContext* ctx) {
    g_group.store(group, std::memory_order_relaxed);
    g_post.store(false, std::memory_order_relaxed);
    if (!g_installed || !g_on.load(std::memory_order_relaxed) || !ctx) return;
    g_groupsSeen.store(true, std::memory_order_relaxed);
    Protect("BeginGroup", [&] {
        std::lock_guard lock(g_m);
        if (group >= 0 && group < 4) ++g_groupStarts[group];
        if (ctx != g_checkedCtx) {
            g_checkedCtx = ctx;
            void* now = (*(void***)ctx)[kSlotPSSetSamplers];
            if (now != g_psTarget)
                Log("Samplers: ERROR: the game's context calls PSSetSamplers at %p, the hook is at %p; the texture "
                    "mip bias cannot replace samplers",
                    now, g_psTarget);
        }
        if (g_setCount.load(std::memory_order_acquire)) RebindLocked(ctx);
    });
}

void EndGroup() {
    g_group.store(-1, std::memory_order_relaxed);
    g_post.store(false, std::memory_order_relaxed);
}

void SceneEnd(ID3D11DeviceContext* ctx) {
    if (!g_installed || !g_on.load(std::memory_order_relaxed) || !ctx || !g_groupsSeen.load() ||
        g_group.load(std::memory_order_relaxed) < 0 || g_post.exchange(true))
        return;
    Protect("SceneEnd", [&] {
        std::lock_guard lock(g_m);
        ++g_sceneEnds[Bucket()];
        if (g_setCount.load(std::memory_order_acquire)) RebindLocked(ctx);
    });
}

size_t TestDistinct() {
    std::lock_guard lock(g_m);
    return g_list.size();
}

size_t TestNegativeBias() {
    std::lock_guard lock(g_m);
    size_t n = 0;
    for (const Entry& e : g_list) n += e.d.MipLODBias < 0;
    return n;
}

uint64_t TestCreations() {
    std::lock_guard lock(g_m);
    return g_creations;
}

size_t TestSceneSamplers() { return g_setCount.load(); }

uint64_t TestReplacedBinds() {
    std::lock_guard lock(g_m);
    uint64_t n = g_replacedTotal;
    for (int b = 0; b < kBuckets; ++b) n += g_replaced[b].load() + g_rebound[b].load();
    return n;
}

}  // namespace qvfr::samplers

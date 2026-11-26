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
constexpr size_t kMaxEntries = 256;

using CreateFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_SAMPLER_DESC*, ID3D11SamplerState**);
CreateFn o_create = nullptr;

constexpr size_t kMaxSites = 8;

struct Entry {
    D3D11_SAMPLER_DESC d{};
    std::string where;  // module+offset of the first caller
    bool game = false;  // a caller in the game's exe
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

void Note(const D3D11_SAMPLER_DESC& d, uintptr_t caller, bool ok) {
    std::lock_guard lock(g_m);
    ++g_creations;
    if (!ok) ++g_failed;
    HMODULE m = nullptr;
    const bool game =
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)caller, &m) &&
        (uintptr_t)m == g_exe;
    for (Entry& e : g_list) {
        if (memcmp(&e.d, &d, sizeof(d)) != 0) continue;
        ++e.count;
        e.game |= game;
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
    e.sites.push_back(caller);
    e.count = 1;
    g_list.push_back(std::move(e));
}

HRESULT STDMETHODCALLTYPE D_CreateSamplerState(ID3D11Device* dev, const D3D11_SAMPLER_DESC* d,
                                               ID3D11SamplerState** out) {
    const HRESULT hr = o_create(dev, d, out);
    if (d && g_on.load(std::memory_order_relaxed)) {
        const uintptr_t caller = (uintptr_t)_ReturnAddress();
        Protect("CreateSamplerState", [&] { Note(*d, caller, SUCCEEDED(hr)); });
    }
    return hr;
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
    char maxLod[32];
    if (d.MaxLOD >= FLT_MAX)
        snprintf(maxLod, sizeof(maxLod), "max");
    else
        snprintf(maxLod, sizeof(maxLod), "%.2f", d.MaxLOD);
    Log("Samplers: #%zu %s: filter 0x%X (%s), address %s/%s/%s, MipLODBias %.3f, MaxAnisotropy %u, comparison %s, "
        "LOD %.2f..%s, border (%g, %g, %g, %g), created %llu from %zu%s call sites",
        i + 1, e.where.c_str(), (unsigned)d.Filter, FilterText((unsigned)d.Filter).c_str(), AddressText(d.AddressU),
        AddressText(d.AddressV), AddressText(d.AddressW), d.MipLODBias, d.MaxAnisotropy, ComparisonText(d.ComparisonFunc),
        d.MinLOD, maxLod, d.BorderColor[0], d.BorderColor[1], d.BorderColor[2], d.BorderColor[3],
        (unsigned long long)e.count, e.sites.size(), e.sites.size() >= kMaxSites ? "+" : "");
}

}  // namespace

void Install() {
    if (g_installed || !Cfg().samplerLog) return;
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
        Log("Samplers: D3D11 hook %s (CreateSamplerState; the stats interval logs new sampler descriptions)%s",
            ok ? "installed" : "FAILED", SUCCEEDED(hr) ? "" : "; no WARP device");
    });
}

bool TestInstall(ID3D11Device* dev) {
    if (g_installed) return true;
    if (!dev || !hooks::Init()) return false;
    if (!g_exe) g_exe = (uintptr_t)GetModuleHandleW(nullptr);
    void* t = (*(void***)dev)[kSlotCreateSamplerState];
    g_installed = hooks::CreateAt(t, (void*)D_CreateSamplerState, (void**)&o_create, "CreateSamplerState");
    g_on = g_installed;
    return g_installed;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    Protect("LogStats", [&] {
        std::lock_guard lock(g_m);
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

}  // namespace qvfr::samplers

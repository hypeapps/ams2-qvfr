#include "clones.h"

#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "hooks.h"

namespace qvfr::clones {

namespace {

using hooks::Anchor;

// Engine object layout, AMS2AVX.exe build aa43bd68 (M7.1 and M7.3, docs/ams2-renderer.md).
// Texture object (0x108 B, vtable 0x21192f8, dtor 0x169c280): the D3D11 texture at +0x58, its views after it (the dtor
// releases +0x58, +0x60, +0x70, ...), the name (engine string) at +0xf0.
constexpr size_t kTexRes = 0x58;
constexpr size_t kTexEnd = 0x108;
constexpr size_t kTexName = 0xf0;
// Render target object (0xd0 B, vtable 0x21148f8, dtor 0x164e760): u16 size at +0x24/+0x26, u16 counts at +0x64
// (depth views) and +0x66 (color views); view arrays +0x78 (RTV), +0x80 (resolve RTV, may be null), +0x88 (second RTV),
// all +0x66 long, and +0x90 (DSV), +0x64 long; the name at +0xb8. Its textures (+0x98, +0xa0, +0xb0) are texture objects.
constexpr size_t kRtW = 0x24, kRtH = 0x26;
constexpr size_t kRtDepthCount = 0x64, kRtViewCount = 0x66;
struct RtArray {
    size_t off, countOff;
};
constexpr RtArray kRtArrays[] = {{0x78, kRtViewCount}, {0x80, kRtViewCount}, {0x88, kRtViewCount}, {0x90, kRtDepthCount}};
constexpr size_t kRtName = 0xb8;
constexpr unsigned kMaxRtViews = 16;

constexpr int kLearnGroups = 16;   // focus groups watched before the clones are made (8 frames)
constexpr int kProofGroups = 240;  // focus groups watched after the clones went in (120 frames)
constexpr int kMaxLeakLogs = 12;
constexpr int kMaxOtherLogs = 24;
constexpr int kMaxResetLogs = 20;
constexpr size_t kMaxPending = 1 << 18;
constexpr UINT kSrvSlots = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
constexpr UINT kUavSlots = D3D11_PS_CS_UAV_REGISTER_COUNT;

enum Phase { Waiting, Learning, Ready, Failed };
const char* const kPhaseNames[] = {"waiting", "learning", "ready", "failed"};
enum Use { Read, Write };

struct Tex {
    uint8_t* obj = nullptr;
    ID3D11Resource* res = nullptr;  // the engine's (owned by the object, not referenced here)
    D3D11_TEXTURE2D_DESC desc{};
    int fraction = 1;               // 1, 2, 4 or 8: the eye size divided by this
    char name[64] = "";
    bool read = false, written = false;  // seen in the focus groups while learning
    ID3D11Texture2D* clone = nullptr;    // ours
    int views = 0;                       // its view fields swapped
};
struct Rt {
    uint8_t* obj = nullptr;
    uint16_t w = 0, h = 0;
    char name[64] = "";
    int views = 0;  // array entries swapped
};
// One pointer field of an engine object: the original while no focus group runs, the clone inside one.
struct Entry {
    uint8_t* obj;
    uint64_t* addr;
    uint64_t orig, clone;
};

struct Counters {
    uint64_t swaps = 0, translated = 0, lateRestores = 0, changed = 0, sets = 0;
};

std::mutex g_cm;
bool g_installed = false;  // all four engine hooks in place
bool g_failed = false;     // a fault or a missing hook: no clones for the rest of the process
Phase g_phase = Waiting;
uint32_t g_eyeW = 0, g_eyeH = 0;
ID3D11DeviceContext* g_imm = nullptr;  // not referenced (gpu.cpp holds it)
uintptr_t g_d3dLo = 0, g_d3dHi = 0;    // module of the D3D11 runtime's object vtables
bool g_d3dHooked = false;
bool g_enabled = false;
bool g_dirty = false;      // the set of listed objects changed
bool g_swapped = false;    // clones in the engine objects now
bool g_inFocus = false;
int g_learned = 0;         // focus groups watched in this set
int g_proofLeft = 0;
int g_resetLogs = 0;
bool g_changedLogged = false;

std::unordered_map<uint8_t*, bool> g_pending;  // created since the last frame start (true: render target)
std::unordered_map<uint8_t*, Tex> g_tex;
std::unordered_map<uint8_t*, Rt> g_rt;
std::unordered_map<ID3D11Resource*, uint8_t*> g_byRes, g_byClone;
std::vector<Entry> g_entries;
std::unordered_map<void*, void*> g_fwd, g_back;  // D3D11 view: original -> clone, clone -> original
std::vector<IUnknown*> g_owned;                  // clone views
std::vector<IUnknown*> g_grave;                  // released at the next frame start (on the render thread)

// Learning and proof: what the focus groups touched.
std::atomic<bool> g_watch{false};
thread_local bool t_internal = false;  // our own D3D11 calls (translation) are not observed
uint64_t g_cloneUses = 0, g_leakReads = 0, g_leakWrites = 0;
int g_leakLogs = 0;
std::unordered_set<ID3D11Resource*> g_otherWritten;  // written in learning groups, not listed
std::vector<std::string> g_otherLines;
Counters g_c, g_ctot;

// ---------------------------------------------------------------- helpers

void Fail(const char* where, unsigned long code);

// Runs f under SEH; a fault fails clones for the session and puts the originals back.
template <class F>
void Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

bool IsD3D(const void* p) {
    uintptr_t vt = 0;
    return p && g_d3dLo && hooks::TryRead(p, &vt, 8) && vt >= g_d3dLo && vt < g_d3dHi;
}

// Engine string (char*) at obj+off, printable, empty when unreadable.
void ReadName(const uint8_t* obj, size_t off, char (&out)[64]) {
    out[0] = 0;
    const char* s = nullptr;
    if (!hooks::TryRead(obj + off, &s, 8) || !s) return;
    char buf[64] = {};
    size_t n = 0;
    for (; n + 1 < sizeof(buf); ++n) {
        char ch = 0;
        if (!hooks::TryRead(s + n, &ch, 1) || !ch) break;
        buf[n] = (ch >= 0x20 && ch < 0x7f) ? ch : '?';
    }
    buf[n] = 0;
    memcpy(out, buf, sizeof(out));
}

int Fraction(uint32_t w, uint32_t h) {
    if (!g_eyeW || !g_eyeH) return 0;
    for (int d = 1; d <= 8; d *= 2) {
        const uint32_t fw = g_eyeW / d, cw = (g_eyeW + d - 1) / d, fh = g_eyeH / d, ch = (g_eyeH + d - 1) / d;
        if ((w == fw || w == cw) && (h == fh || h == ch)) return d;
    }
    return 0;
}

uint32_t BitsPerPixel(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT: return 128;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return 64;
        case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_D16_UNORM:
        case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R8G8_UNORM: return 16;
        case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_A8_UNORM: return 8;
        default: return 32;
    }
}

double Megabytes(const D3D11_TEXTURE2D_DESC& d) {
    double px = 0;
    for (UINT m = 0; m < std::max(1u, d.MipLevels); ++m)
        px += (double)std::max(1u, d.Width >> m) * std::max(1u, d.Height >> m);
    return px * d.ArraySize * std::max(1u, d.SampleDesc.Count) * BitsPerPixel(d.Format) / 8.0 / (1024.0 * 1024.0);
}

// The resource of a D3D11 view (referenced), or null.
ID3D11Resource* ViewResource(void* v) {
    if (!IsD3D(v)) return nullptr;
    ID3D11View* view = nullptr;
    if (FAILED(static_cast<IUnknown*>(v)->QueryInterface(__uuidof(ID3D11View), (void**)&view)) || !view) return nullptr;
    ID3D11Resource* r = nullptr;
    view->GetResource(&r);
    view->Release();
    return r;
}

const char* TexName(const Tex& t) { return t.name[0] ? t.name : "(no name)"; }

// ---------------------------------------------------------------- object lists

void RegisterTexLocked(uint8_t* obj) {
    if (g_tex.count(obj)) return;
    void* res = nullptr;
    if (!hooks::TryRead(obj + kTexRes, &res, 8) || !IsD3D(res)) return;
    ID3D11Texture2D* t = nullptr;
    if (FAILED(static_cast<IUnknown*>(res)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) || !t) return;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    t->Release();
    const int fr = Fraction(d.Width, d.Height);
    if (!fr || d.Usage != D3D11_USAGE_DEFAULT) return;
    Tex x;
    x.obj = obj;
    x.res = static_cast<ID3D11Resource*>(res);
    x.desc = d;
    x.fraction = fr;
    ReadName(obj, kTexName, x.name);
    g_byRes[x.res] = obj;
    g_tex[obj] = x;
    g_dirty = true;
}

void RegisterRtLocked(uint8_t* obj) {
    if (g_rt.count(obj)) return;
    uint16_t wh[2] = {};
    if (!hooks::TryRead(obj + kRtW, &wh[0], 2) || !hooks::TryRead(obj + kRtH, &wh[1], 2) || !Fraction(wh[0], wh[1])) return;
    Rt r;
    r.obj = obj;
    r.w = wh[0];
    r.h = wh[1];
    ReadName(obj, kRtName, r.name);
    g_rt[obj] = r;
    g_dirty = true;
}

void ProcessPendingLocked() {
    if (!g_eyeW || !g_d3dLo || g_pending.empty()) return;
    // Textures first: a render target's textures are created inside its constructor.
    for (auto& [obj, rt] : g_pending)
        if (!rt) RegisterTexLocked(obj);
    for (auto& [obj, rt] : g_pending)
        if (rt) RegisterRtLocked(obj);
    g_pending.clear();
}

// Puts the originals back into one object's fields (it is being destroyed or the set is dropped).
void RestoreObjectLocked(uint8_t* obj) {
    for (const Entry& e : g_entries) {
        if (e.obj != obj) continue;
        uint64_t cur = 0;
        if (hooks::TryRead(e.addr, &cur, 8) && cur == e.clone) *e.addr = e.orig;
    }
}

void DestroyLocked(uint8_t* obj, bool rt) {
    g_pending.erase(obj);
    if (g_swapped) RestoreObjectLocked(obj);
    const size_t before = g_entries.size();
    g_entries.erase(std::remove_if(g_entries.begin(), g_entries.end(), [&](const Entry& e) { return e.obj == obj; }),
                    g_entries.end());
    bool listed = before != g_entries.size();
    if (rt) {
        listed |= g_rt.erase(obj) > 0;
    } else if (auto it = g_tex.find(obj); it != g_tex.end()) {
        g_byRes.erase(it->second.res);
        if (it->second.clone) {
            g_byClone.erase(it->second.clone);
            g_grave.push_back(it->second.clone);
        }
        g_tex.erase(it);
        listed = true;
    }
    if (listed) g_dirty = true;
}

// ---------------------------------------------------------------- clones

void DropClonesLocked() {
    for (IUnknown* u : g_owned) g_grave.push_back(u);
    g_owned.clear();
    for (auto& [obj, t] : g_tex) {
        if (t.clone) g_grave.push_back(t.clone);
        t.clone = nullptr;
        t.views = 0;
    }
    for (auto& [obj, r] : g_rt) r.views = 0;
    g_byClone.clear();
    g_entries.clear();
    g_fwd.clear();
    g_back.clear();
}

// A changed set: no clones until the focus groups were watched again.
void ResetLocked(const char* why) {
    DropClonesLocked();
    for (auto& [obj, t] : g_tex) t.read = t.written = false;
    g_otherWritten.clear();
    g_otherLines.clear();
    g_learned = 0;
    g_proofLeft = 0;
    g_changedLogged = false;
    if (g_failed) return;
    g_phase = Learning;
    ++g_c.sets;
    if (g_resetLogs < kMaxResetLogs) {
        ++g_resetLogs;
        Log("Clones: %s: %zu textures and %zu render targets in the eye size %u x %u (or 1/2, 1/4, 1/8); watching which "
            "ones the focus views write (%d focus groups)%s",
            why, g_tex.size(), g_rt.size(), g_eyeW, g_eyeH, kLearnGroups,
            g_resetLogs == kMaxResetLogs ? " (further set changes only counted)" : "");
    }
}

// A clone of view v (same description) on the clone texture; reused when v appears twice.
void* CloneViewLocked(ID3D11Device* dev, void* v, ID3D11Resource* clone) {
    if (auto it = g_fwd.find(v); it != g_fwd.end()) return it->second;
    IUnknown* u = static_cast<IUnknown*>(v);
    IUnknown* out = nullptr;
    HRESULT hr = E_NOINTERFACE;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11ShaderResourceView), (void**)&srv)) && srv) {
        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        srv->GetDesc(&d);
        srv->Release();
        hr = dev->CreateShaderResourceView(clone, &d, (ID3D11ShaderResourceView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11RenderTargetView), (void**)&rtv)) && rtv) {
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        rtv->GetDesc(&d);
        rtv->Release();
        hr = dev->CreateRenderTargetView(clone, &d, (ID3D11RenderTargetView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11DepthStencilView), (void**)&dsv)) && dsv) {
        D3D11_DEPTH_STENCIL_VIEW_DESC d{};
        dsv->GetDesc(&d);
        dsv->Release();
        hr = dev->CreateDepthStencilView(clone, &d, (ID3D11DepthStencilView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11UnorderedAccessView), (void**)&uav)) && uav) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
        uav->GetDesc(&d);
        uav->Release();
        hr = dev->CreateUnorderedAccessView(clone, &d, (ID3D11UnorderedAccessView**)&out);
    }
    if (FAILED(hr) || !out) {
        Log("Clones: cannot clone view %p (0x%08lX)", v, (unsigned long)hr);
        return nullptr;
    }
    g_owned.push_back(out);
    g_fwd[v] = out;
    g_back[out] = v;
    return out;
}

void AddEntry(uint8_t* obj, void* addr, void* orig, void* clone) {
    g_entries.push_back({obj, (uint64_t*)addr, (uint64_t)orig, (uint64_t)clone});
}

// Clones for the textures the focus groups wrote, entries for their view fields and the render targets' view arrays.
bool BuildLocked(ID3D11DeviceContext* imm) {
    ID3D11Device* dev = nullptr;
    imm->GetDevice(&dev);
    if (!dev) return false;
    bool ok = true;
    double mb = 0;
    int clones = 0;
    for (auto& [obj, t] : g_tex) {
        if (!t.written) continue;
        D3D11_TEXTURE2D_DESC d = t.desc;  // M7.3: the same size
        d.MiscFlags &= ~(UINT)(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                               D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
        HRESULT hr = dev->CreateTexture2D(&d, nullptr, &t.clone);
        if (FAILED(hr) || !t.clone) {
            t.clone = nullptr;
            Log("Clones: cannot create the clone of %s (%u x %u fmt %d x%u, 0x%08lX)", TexName(t), d.Width, d.Height,
                (int)d.Format, d.SampleDesc.Count, (unsigned long)hr);
            ok = false;
            break;
        }
        g_byClone[t.clone] = obj;
        mb += Megabytes(d);
        ++clones;
        AddEntry(obj, obj + kTexRes, t.res, t.clone);
        for (size_t off = kTexRes + 8; off + 8 <= kTexEnd; off += 8) {
            void* v = nullptr;
            if (!hooks::TryRead(obj + off, &v, 8)) continue;
            ID3D11Resource* r = ViewResource(v);
            if (!r) continue;
            const bool mine = r == t.res;
            r->Release();
            if (!mine) continue;
            void* cv = CloneViewLocked(dev, v, t.clone);
            if (!cv) {
                ok = false;
                break;
            }
            AddEntry(obj, obj + off, v, cv);
            ++t.views;
        }
        if (!ok) break;
    }
    for (auto& [obj, rt] : g_rt) {
        if (!ok) break;
        for (const RtArray& a : kRtArrays) {
            uint16_t count = 0;
            void** arr = nullptr;
            if (!hooks::TryRead(obj + a.countOff, &count, 2) || !hooks::TryRead(obj + a.off, &arr, 8) || !arr) continue;
            for (unsigned i = 0; i < std::min<unsigned>(count, kMaxRtViews) && ok; ++i) {
                void* v = nullptr;
                if (!hooks::TryRead(arr + i, &v, 8)) continue;
                ID3D11Resource* r = ViewResource(v);
                if (!r) continue;
                auto it = g_byRes.find(r);
                r->Release();
                if (it == g_byRes.end()) continue;
                const Tex& t = g_tex[it->second];
                if (!t.clone) continue;
                void* cv = CloneViewLocked(dev, v, t.clone);
                if (!cv) {
                    ok = false;
                    break;
                }
                AddEntry(obj, arr + i, v, cv);
                ++rt.views;
            }
        }
    }
    dev->Release();
    // The list: every listed texture with its role in the focus groups.
    Log("Clones: focus views watched in %d groups; %zu listed textures:", g_learned, g_tex.size());
    std::vector<const Tex*> list;
    for (auto& [obj, t] : g_tex) list.push_back(&t);
    std::sort(list.begin(), list.end(), [](const Tex* a, const Tex* b) {
        if (a->written != b->written) return a->written;
        return strcmp(a->name, b->name) < 0;
    });
    for (const Tex* t : list) {
        char line[320];
        snprintf(line, sizeof(line), "  %-40s %5u x %-5u (1/%d) fmt %3d x%u mips %u arr %u  %-12s %s", TexName(*t),
                 t->desc.Width, t->desc.Height, t->fraction, (int)t->desc.Format, t->desc.SampleDesc.Count,
                 t->desc.MipLevels, t->desc.ArraySize,
                 t->written ? (t->read ? "read+written" : "written") : (t->read ? "read only" : "not used"),
                 t->clone ? "-> clone" : "");
        LogRaw(line);
    }
    int rtViews = 0;
    for (auto& [obj, rt] : g_rt) rtViews += rt.views;
    if (!g_otherLines.empty()) {
        Log("Clones: %zu other targets written by the focus views (not in the eye size, or not from the engine factory):",
            g_otherWritten.size());
        for (const std::string& s : g_otherLines) LogRaw(s.c_str());
    }
    if (!ok) {
        DropClonesLocked();
        g_phase = Failed;
        Log("Clones: OFF for this set (see the line above); the focus views share the targets with outer");
        return false;
    }
    g_phase = Ready;
    g_proofLeft = kProofGroups;
    g_cloneUses = g_leakReads = g_leakWrites = 0;
    g_leakLogs = 0;
    Log("Clones: %d clones (%.0f MB), %zu pointer fields swapped per focus group (%zu D3D11 views; %d in %zu render "
        "targets); %s",
        clones, mb, g_entries.size(), g_fwd.size(), rtViews, g_rt.size(),
        g_enabled ? "on from the next focus group" : "made, but off (setting or hotkey)");
    return true;
}

// The D3D11 state bound now, with every view found in `map` replaced (RTVs, DSV, SRVs of all stages, CS UAVs).
void TranslateLocked(ID3D11DeviceContext* imm, const std::unordered_map<void*, void*>& map) {
    if (!imm || map.empty()) return;
    t_internal = true;
    auto tr = [&](void* p) -> void* {
        if (!p) return p;
        auto it = map.find(p);
        return it == map.end() ? p : it->second;
    };
    bool any = false;
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    imm->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
    ID3D11RenderTargetView* nrtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
    bool ch = false;
    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
        nrtv[i] = (ID3D11RenderTargetView*)tr(rtv[i]);
        ch |= nrtv[i] != rtv[i];
    }
    ID3D11DepthStencilView* ndsv = (ID3D11DepthStencilView*)tr(dsv);
    ch |= ndsv != dsv;
    if (ch) {
        imm->OMSetRenderTargetsAndUnorderedAccessViews(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, nrtv, ndsv, 0,
                                                       D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr);
        any = true;
    }
    for (ID3D11RenderTargetView* v : rtv)
        if (v) v->Release();
    if (dsv) dsv->Release();
    using GetFn = void (STDMETHODCALLTYPE ID3D11DeviceContext::*)(UINT, UINT, ID3D11ShaderResourceView**);
    using SetFn = void (STDMETHODCALLTYPE ID3D11DeviceContext::*)(UINT, UINT, ID3D11ShaderResourceView* const*);
    const struct {
        GetFn get;
        SetFn set;
    } stages[] = {
        {&ID3D11DeviceContext::VSGetShaderResources, &ID3D11DeviceContext::VSSetShaderResources},
        {&ID3D11DeviceContext::PSGetShaderResources, &ID3D11DeviceContext::PSSetShaderResources},
        {&ID3D11DeviceContext::GSGetShaderResources, &ID3D11DeviceContext::GSSetShaderResources},
        {&ID3D11DeviceContext::HSGetShaderResources, &ID3D11DeviceContext::HSSetShaderResources},
        {&ID3D11DeviceContext::DSGetShaderResources, &ID3D11DeviceContext::DSSetShaderResources},
        {&ID3D11DeviceContext::CSGetShaderResources, &ID3D11DeviceContext::CSSetShaderResources},
    };
    for (const auto& s : stages) {
        ID3D11ShaderResourceView* srv[kSrvSlots] = {};
        (imm->*s.get)(0, kSrvSlots, srv);
        ID3D11ShaderResourceView* nsrv[kSrvSlots];
        UINT lo = kSrvSlots, hi = 0;
        for (UINT i = 0; i < kSrvSlots; ++i) {
            nsrv[i] = (ID3D11ShaderResourceView*)tr(srv[i]);
            if (nsrv[i] != srv[i]) {
                lo = std::min(lo, i);
                hi = i + 1;
            }
        }
        if (lo < hi) {
            (imm->*s.set)(lo, hi - lo, nsrv + lo);
            any = true;
        }
        for (ID3D11ShaderResourceView* v : srv)
            if (v) v->Release();
    }
    ID3D11UnorderedAccessView* uav[kUavSlots] = {};
    imm->CSGetUnorderedAccessViews(0, kUavSlots, uav);
    ID3D11UnorderedAccessView* nuav[kUavSlots];
    ch = false;
    for (UINT i = 0; i < kUavSlots; ++i) {
        nuav[i] = (ID3D11UnorderedAccessView*)tr(uav[i]);
        ch |= nuav[i] != uav[i];
    }
    if (ch) {
        UINT keep[kUavSlots];
        for (UINT& k : keep) k = (UINT)-1;
        imm->CSSetUnorderedAccessViews(0, kUavSlots, nuav, keep);
        any = true;
    }
    for (ID3D11UnorderedAccessView* v : uav)
        if (v) v->Release();
    if (any) ++g_c.translated;
    t_internal = false;
}

void RestoreLocked(ID3D11DeviceContext* imm) {
    if (!g_swapped) return;
    g_swapped = false;
    int changed = 0;
    for (const Entry& e : g_entries) {
        uint64_t cur = 0;
        if (!hooks::TryRead(e.addr, &cur, 8)) continue;
        if (cur == e.clone)
            *e.addr = e.orig;
        else
            ++changed;
    }
    TranslateLocked(imm, g_back);
    if (changed) {
        // The engine replaced a view inside a focus group: its value stays, the set is watched again.
        ++g_c.changed;
        g_dirty = true;
        Log("Clones: %d fields changed by the engine during a focus group; set dropped", changed);
    }
}

bool PutBackRaw() {
    __try {
        if (g_swapped) {
            g_swapped = false;
            for (const Entry& e : g_entries) {
                uint64_t cur = 0;
                if (hooks::TryRead(e.addr, &cur, 8) && cur == e.clone) *e.addr = e.orig;
            }
        }
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Protect's handler runs after the unwinding (/EHa), so the lambda's lock is released; no Protect runs while this
// thread holds g_cm (our own D3D11 calls are not observed), so the lock below cannot be recursive.
void Fail(const char* where, unsigned long code) {
    std::lock_guard lock(g_cm);
    t_internal = false;
    if (g_failed) return;
    g_failed = true;
    g_phase = Failed;
    g_watch = false;
    Log("ERROR: exception 0x%08lX in clones (%s); clones OFF for this session%s", code, where,
        PutBackRaw() ? "" : "; the originals could NOT be put back");
}

// ---------------------------------------------------------------- observation (learning and proof groups)

void NoteLocked(ID3D11Resource* r, Use use, const char* method) {
    if (!r) return;
    if (g_phase == Learning) {
        if (auto it = g_byRes.find(r); it != g_byRes.end()) {
            Tex& t = g_tex[it->second];
            (use == Write ? t.written : t.read) = true;
        } else if (use == Write && g_otherWritten.insert(r).second && (int)g_otherLines.size() < kMaxOtherLogs) {
            char line[200];
            ID3D11Texture2D* t = nullptr;
            if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
                D3D11_TEXTURE2D_DESC d{};
                t->GetDesc(&d);
                t->Release();
                snprintf(line, sizeof(line), "  %p %u x %u fmt %d x%u (%s)", (void*)r, d.Width, d.Height, (int)d.Format,
                         d.SampleDesc.Count, method);
            } else {
                snprintf(line, sizeof(line), "  %p not a 2D texture (%s)", (void*)r, method);
            }
            g_otherLines.push_back(line);
        }
        return;
    }
    if (g_byClone.count(r)) {
        ++g_cloneUses;
        return;
    }
    auto it = g_byRes.find(r);
    if (it == g_byRes.end() || !g_tex[it->second].clone) return;
    ++(use == Write ? g_leakWrites : g_leakReads);
    if (g_leakLogs < kMaxLeakLogs) {
        ++g_leakLogs;
        Log("Clones: LEAK: a focus view %s the original %s through %s%s", use == Write ? "writes" : "reads",
            TexName(g_tex[it->second]), method, g_leakLogs == kMaxLeakLogs ? " (further leaks only counted)" : "");
    }
}

template <class V>
void ObserveViews(ID3D11DeviceContext* c, Use use, const char* method, V* const* views, UINT n) {
    if (t_internal || c != g_imm || !views) return;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (!g_inFocus) return;
        for (UINT i = 0; i < n && i < 128; ++i) {
            if (!views[i]) continue;
            ID3D11Resource* r = nullptr;
            views[i]->GetResource(&r);
            NoteLocked(r, use, method);
            if (r) r->Release();
        }
    });
}

void ObserveRes(ID3D11DeviceContext* c, Use use, const char* method, ID3D11Resource* r) {
    if (t_internal || c != g_imm || !r) return;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (g_inFocus) NoteLocked(r, use, method);
    });
}

bool Watching() { return g_watch.load(std::memory_order_relaxed); }

// ---------------------------------------------------------------- D3D11 detours

using SetSrvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11ShaderResourceView* const*);
SetSrvFn o_psSrv = nullptr, o_vsSrv = nullptr, o_csSrv = nullptr, o_gsSrv = nullptr;
using OmUavFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                         ID3D11DepthStencilView*, UINT, UINT, ID3D11UnorderedAccessView* const*,
                                         const UINT*);
OmUavFn o_omUav = nullptr;
using CsUavFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11UnorderedAccessView* const*, const UINT*);
CsUavFn o_csUav = nullptr;
using CopyRegionFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, UINT, UINT, UINT,
                                              ID3D11Resource*, UINT, const D3D11_BOX*);
CopyRegionFn o_copyRegion = nullptr;
using CopyFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, ID3D11Resource*);
CopyFn o_copy = nullptr;
using ResolveFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, ID3D11Resource*, UINT,
                                           DXGI_FORMAT);
ResolveFn o_resolve = nullptr;
using ClearRtvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, const FLOAT[4]);
ClearRtvFn o_clearRtv = nullptr;
using ClearDsvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
ClearDsvFn o_clearDsv = nullptr;
using ClearUavUFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11UnorderedAccessView*, const UINT[4]);
ClearUavUFn o_clearUavU = nullptr;
using ClearUavFFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11UnorderedAccessView*, const FLOAT[4]);
ClearUavFFn o_clearUavF = nullptr;
using MipsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11ShaderResourceView*);
MipsFn o_mips = nullptr;

void STDMETHODCALLTYPE D_PSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    if (Watching()) ObserveViews(c, Read, "PSSetShaderResources", v, n);
    o_psSrv(c, s, n, v);
}
void STDMETHODCALLTYPE D_VSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    if (Watching()) ObserveViews(c, Read, "VSSetShaderResources", v, n);
    o_vsSrv(c, s, n, v);
}
void STDMETHODCALLTYPE D_CSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    if (Watching()) ObserveViews(c, Read, "CSSetShaderResources", v, n);
    o_csSrv(c, s, n, v);
}
void STDMETHODCALLTYPE D_GSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    if (Watching()) ObserveViews(c, Read, "GSSetShaderResources", v, n);
    o_gsSrv(c, s, n, v);
}
void STDMETHODCALLTYPE D_OMSetRtUav(ID3D11DeviceContext* c, UINT nr, ID3D11RenderTargetView* const* rtv,
                                    ID3D11DepthStencilView* dsv, UINT us, UINT nu, ID3D11UnorderedAccessView* const* uav,
                                    const UINT* counts) {
    if (Watching()) {
        const char* m = "OMSetRenderTargetsAndUnorderedAccessViews";
        if (nr != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL) {
            ObserveViews(c, Write, m, rtv, nr);
            ObserveViews(c, Write, m, &dsv, 1);
        }
        if (nu != D3D11_KEEP_UNORDERED_ACCESS_VIEWS) ObserveViews(c, Write, m, uav, nu);
    }
    o_omUav(c, nr, rtv, dsv, us, nu, uav, counts);
}
void STDMETHODCALLTYPE D_CSSetUav(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11UnorderedAccessView* const* uav,
                                  const UINT* counts) {
    if (Watching()) ObserveViews(c, Write, "CSSetUnorderedAccessViews", uav, n);
    o_csUav(c, s, n, uav, counts);
}
void STDMETHODCALLTYPE D_CopyRegion(ID3D11DeviceContext* c, ID3D11Resource* dst, UINT ds, UINT x, UINT y, UINT z,
                                    ID3D11Resource* src, UINT ss, const D3D11_BOX* box) {
    if (Watching()) {
        ObserveRes(c, Write, "CopySubresourceRegion", dst);
        ObserveRes(c, Read, "CopySubresourceRegion", src);
    }
    o_copyRegion(c, dst, ds, x, y, z, src, ss, box);
}
void STDMETHODCALLTYPE D_Copy(ID3D11DeviceContext* c, ID3D11Resource* dst, ID3D11Resource* src) {
    if (Watching()) {
        ObserveRes(c, Write, "CopyResource", dst);
        ObserveRes(c, Read, "CopyResource", src);
    }
    o_copy(c, dst, src);
}
void STDMETHODCALLTYPE D_Resolve(ID3D11DeviceContext* c, ID3D11Resource* dst, UINT ds, ID3D11Resource* src, UINT ss,
                                 DXGI_FORMAT f) {
    if (Watching()) {
        ObserveRes(c, Write, "ResolveSubresource", dst);
        ObserveRes(c, Read, "ResolveSubresource", src);
    }
    o_resolve(c, dst, ds, src, ss, f);
}
void STDMETHODCALLTYPE D_ClearRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]) {
    if (Watching()) ObserveViews(c, Write, "ClearRenderTargetView", &v, 1);
    o_clearRtv(c, v, col);
}
void STDMETHODCALLTYPE D_ClearDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT f, FLOAT d, UINT8 s) {
    if (Watching()) ObserveViews(c, Write, "ClearDepthStencilView", &v, 1);
    o_clearDsv(c, v, f, d, s);
}
void STDMETHODCALLTYPE D_ClearUavU(ID3D11DeviceContext* c, ID3D11UnorderedAccessView* v, const UINT val[4]) {
    if (Watching()) ObserveViews(c, Write, "ClearUnorderedAccessViewUint", &v, 1);
    o_clearUavU(c, v, val);
}
void STDMETHODCALLTYPE D_ClearUavF(ID3D11DeviceContext* c, ID3D11UnorderedAccessView* v, const FLOAT val[4]) {
    if (Watching()) ObserveViews(c, Write, "ClearUnorderedAccessViewFloat", &v, 1);
    o_clearUavF(c, v, val);
}
void STDMETHODCALLTYPE D_Mips(ID3D11DeviceContext* c, ID3D11ShaderResourceView* v) {
    if (Watching()) ObserveViews(c, Write, "GenerateMips", &v, 1);
    o_mips(c, v);
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_createTex = nullptr, o_createRt = nullptr, o_texDtor = nullptr, o_rtDtor = nullptr;

void NoteCreated(void* obj, bool rt) {
    if (!obj || g_failed) return;
    Protect("factory", [&] {
        std::lock_guard lock(g_cm);
        if (g_pending.size() < kMaxPending) g_pending[(uint8_t*)obj] = rt;
    });
}

void NoteDestroyed(void* obj, bool rt) {
    if (!obj) return;
    Protect("destructor", [&] {
        std::lock_guard lock(g_cm);
        DestroyLocked((uint8_t*)obj, rt);
    });
}

void* __fastcall D_CreateTexture(void* dev, void* spec, void* ctx, void* d) {
    void* r = o_createTex(dev, spec, ctx, d);
    NoteCreated(r, false);
    return r;
}

void* __fastcall D_CreateRenderTarget(void* dev, void* spec, void* ctx, void* d) {
    void* r = o_createRt(dev, spec, ctx, d);
    NoteCreated(r, true);
    return r;
}

// The originals go back before the engine releases the object's views.
void* __fastcall D_TextureDtor(void* obj, void* b, void* c, void* d) {
    NoteDestroyed(obj, false);
    return o_texDtor(obj, b, c, d);
}

void* __fastcall D_RenderTargetDtor(void* obj, void* b, void* c, void* d) {
    NoteDestroyed(obj, true);
    return o_rtDtor(obj, b, c, d);
}

void ReleaseGraveLocked() {
    for (IUnknown* u : g_grave) u->Release();
    g_grave.clear();
}

// The immediate context's vtable lives on the heap (a copy per context), but the device's lies in the runtime's module,
// like the views' and textures' (d3d11.dll, or a wrapper such as RenderDoc that wraps all of them).
void SetD3DModule(ID3D11DeviceContext* imm) {
    HMODULE mod = nullptr;
    MODULEINFO mi{};
    ID3D11Device* dev = nullptr;
    imm->GetDevice(&dev);
    void* vt = dev ? *(void**)dev : nullptr;
    if (dev) dev->Release();
    if (vt &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)vt, &mod) &&
        GetModuleInformation(GetCurrentProcess(), mod, &mi, sizeof(mi))) {
        g_d3dLo = (uintptr_t)mi.lpBaseOfDll;
        g_d3dHi = g_d3dLo + mi.SizeOfImage;
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(mod, path, MAX_PATH);
        Log("Clones: D3D11 objects recognized by their vtables in %s", Narrow(fs::path(path).filename().wstring()).c_str());
    } else {
        Log("Clones: module of the D3D11 device vtable not found (error %lu); no clones", GetLastError());
    }
}

void LogProofLocked() {
    const uint64_t leaks = g_leakReads + g_leakWrites;
    Log("Clones: proof over %d focus groups: %llu binds and copies of clones, %llu reads and %llu writes of the originals "
        "of cloned targets (%s)",
        kProofGroups, (unsigned long long)g_cloneUses, (unsigned long long)g_leakReads,
        (unsigned long long)g_leakWrites,
        leaks ? "LEAKS: the engine reaches those originals other than through the swapped fields"
              : "0 expected, ok: the focus views render into the clones");
}

}  // namespace

// ---------------------------------------------------------------- public

bool InstallEngineHooks() {
    bool ok = true;
    ok &= hooks::Create(Anchor::TextureDtor, (void*)D_TextureDtor, (void**)&o_texDtor);
    ok &= hooks::Create(Anchor::RenderTargetDtor, (void*)D_RenderTargetDtor, (void**)&o_rtDtor);
    // Without both destructors an object could be freed with the clones in it: no factory hooks then.
    if (ok) {
        ok &= hooks::Create(Anchor::CreateTexture, (void*)D_CreateTexture, (void**)&o_createTex);
        ok &= hooks::Create(Anchor::CreateRenderTarget, (void*)D_CreateRenderTarget, (void**)&o_createRt);
    }
    std::lock_guard lock(g_cm);
    g_installed = ok;
    if (!ok) {
        g_failed = true;
        g_phase = Failed;
    }
    Log("Clones: engine factory and destructor hooks %s", ok ? "installed" : "INCOMPLETE; no clones");
    return ok;
}

void InstallD3DHooks(ID3D11DeviceContext* imm) {
    if (!imm || g_d3dHooked) return;
    Protect("InstallD3DHooks", [&] {
        std::lock_guard lock(g_cm);
        g_imm = imm;
        SetD3DModule(imm);
        void** vt = *(void***)imm;
        // ID3D11DeviceContext vtable slots (d3d11.h order).
        struct H {
            int slot;
            void* detour;
            void** orig;
            const char* name;
        } const hs[] = {
            {8, (void*)D_PSSetSrv, (void**)&o_psSrv, "PSSetShaderResources"},
            {25, (void*)D_VSSetSrv, (void**)&o_vsSrv, "VSSetShaderResources"},
            {31, (void*)D_GSSetSrv, (void**)&o_gsSrv, "GSSetShaderResources"},
            {34, (void*)D_OMSetRtUav, (void**)&o_omUav, "OMSetRenderTargetsAndUnorderedAccessViews"},
            {46, (void*)D_CopyRegion, (void**)&o_copyRegion, "CopySubresourceRegion"},
            {47, (void*)D_Copy, (void**)&o_copy, "CopyResource"},
            {50, (void*)D_ClearRtv, (void**)&o_clearRtv, "ClearRenderTargetView"},
            {51, (void*)D_ClearUavU, (void**)&o_clearUavU, "ClearUnorderedAccessViewUint"},
            {52, (void*)D_ClearUavF, (void**)&o_clearUavF, "ClearUnorderedAccessViewFloat"},
            {53, (void*)D_ClearDsv, (void**)&o_clearDsv, "ClearDepthStencilView"},
            {54, (void*)D_Mips, (void**)&o_mips, "GenerateMips"},
            {57, (void*)D_Resolve, (void**)&o_resolve, "ResolveSubresource"},
            {67, (void*)D_CSSetSrv, (void**)&o_csSrv, "CSSetShaderResources"},
            {68, (void*)D_CSSetUav, (void**)&o_csUav, "CSSetUnorderedAccessViews"},
        };
        bool ok = g_d3dLo != 0;
        for (const H& h : hs)
            if (ok) ok &= hooks::CreateAt(vt[h.slot], h.detour, h.orig, h.name);
        g_d3dHooked = true;
        if (!ok) {
            // A hook may be in place already; it only observes, so it stays, but nothing is learned or cloned.
            g_failed = true;
            g_phase = Failed;
        }
        Log("Clones: D3D11 observation hooks %s (14 context methods, active only in learning and proof groups)",
            ok ? "installed" : "INCOMPLETE; no clones");
    });
}

void SetEyeSize(uint32_t w, uint32_t h) {
    Protect("SetEyeSize", [&] {
        std::lock_guard lock(g_cm);
        if (w == g_eyeW && h == g_eyeH) return;
        g_eyeW = w;
        g_eyeH = h;
        // Listed under the old size: dropped (a new size comes with a new render target pool).
        DropClonesLocked();
        g_tex.clear();
        g_rt.clear();
        g_byRes.clear();
        g_dirty = true;
    });
}

void OnFrameStart(ID3D11DeviceContext* imm, bool enabled) {
    Protect("frame start", [&] {
        std::lock_guard lock(g_cm);
        g_watch = false;
        g_inFocus = false;
        if (g_swapped) {
            ++g_c.lateRestores;
            RestoreLocked(g_imm);
        }
        ReleaseGraveLocked();
        if (!g_installed || g_failed) {
            g_pending.clear();
            return;
        }
        if (imm && g_imm && imm != g_imm) {
            Log("Clones: D3D11 context changed; clones dropped");
            g_imm = imm;
            g_dirty = true;
        }
        if (enabled != g_enabled) {
            g_enabled = enabled;
            Log("Clones: %s", enabled ? "ON (focus views into their own targets)" : "OFF (focus views share outer's targets)");
            if (enabled && g_phase == Ready) {
                g_proofLeft = kProofGroups;
                g_cloneUses = g_leakReads = g_leakWrites = 0;
                g_leakLogs = 0;
            }
        }
        ProcessPendingLocked();
        if (g_eyeW && g_d3dLo && (g_dirty || g_phase == Waiting || g_phase == Failed)) {
            const bool wasFailed = g_phase == Failed;
            if (g_dirty || !wasFailed) ResetLocked(g_phase == Waiting ? "first set" : "set changed");
            g_dirty = false;
        }
        if (g_phase == Learning && g_learned >= kLearnGroups && g_enabled && imm) BuildLocked(imm);
    });
}

void BeginFocus(ID3D11DeviceContext* imm) {
    Protect("BeginFocus", [&] {
        std::lock_guard lock(g_cm);
        if (g_failed || g_swapped) return;
        g_inFocus = true;
        if (g_phase == Learning) {
            g_watch = g_d3dHooked && g_learned < kLearnGroups;
            return;
        }
        // A set that changed in this frame waits for the frame start (its entries may point to dropped clones).
        if (g_phase != Ready || !g_enabled || g_dirty || g_entries.empty()) return;
        for (const Entry& e : g_entries) {
            uint64_t cur = 0;
            if (!hooks::TryRead(e.addr, &cur, 8) || cur != e.orig) {
                ++g_c.changed;
                g_dirty = true;
                if (!g_changedLogged) {
                    g_changedLogged = true;
                    Log("Clones: a swapped field of object %p (+0x%llX) changed outside the focus views; no clones until "
                        "the set is watched again",
                        (void*)e.obj, (unsigned long long)((uint8_t*)e.addr - e.obj));
                }
                return;
            }
        }
        for (const Entry& e : g_entries) *e.addr = e.clone;
        g_swapped = true;
        ++g_c.swaps;
        TranslateLocked(imm, g_fwd);
        g_watch = g_proofLeft > 0 && g_d3dHooked;
    });
}

void EndFocus(ID3D11DeviceContext* imm) {
    Protect("EndFocus", [&] {
        std::lock_guard lock(g_cm);
        g_watch = false;
        if (!g_inFocus) return;
        g_inFocus = false;
        if (g_phase == Learning) ++g_learned;
        if (!g_swapped) return;
        RestoreLocked(imm);
        if (g_proofLeft > 0 && --g_proofLeft == 0) LogProofLocked();
    });
}

void OnOMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                          ID3D11DepthStencilView* dsv) {
    if (!Watching()) return;
    ObserveViews(c, Write, "OMSetRenderTargets", rtv, n);
    ObserveViews(c, Write, "OMSetRenderTargets", &dsv, 1);
}

void* Original(void* res) {
    void* out = res;
    if (!res) return out;
    Protect("Original", [&] {
        std::lock_guard lock(g_cm);
        if (auto it = g_byClone.find((ID3D11Resource*)res); it != g_byClone.end()) out = g_tex[it->second].res;
    });
    return out;
}

std::string Describe() {
    char buf[256] = "clones: (error)";
    Protect("Describe", [&] {
        std::lock_guard lock(g_cm);
        int clones = 0;
        double mb = 0;
        for (auto& [obj, t] : g_tex)
            if (t.clone) {
                ++clones;
                mb += Megabytes(t.desc);
            }
        snprintf(buf, sizeof(buf), "clones %s (%s; %zu eye-size textures, %zu render targets, %d clones %.0f MB)",
                 g_enabled ? "on" : "off", kPhaseNames[g_phase], g_tex.size(), g_rt.size(), clones, mb);
    });
    return buf;
}

void LogStats(bool totals) {
    Protect("LogStats", [&] {
        std::lock_guard lock(g_cm);
        g_ctot.swaps += g_c.swaps;
        g_ctot.translated += g_c.translated;
        g_ctot.lateRestores += g_c.lateRestores;
        g_ctot.changed += g_c.changed;
        g_ctot.sets += g_c.sets;
        const Counters& c = totals ? g_ctot : g_c;
        if (c.swaps || c.sets || c.changed)
            Log("Clones stats%s: %s, %llu focus groups with clones, %llu with bound views translated, %llu late "
                "restores, %llu sets watched, %llu changed fields",
                totals ? " (session total)" : "", kPhaseNames[g_phase], (unsigned long long)c.swaps,
                (unsigned long long)c.translated, (unsigned long long)c.lateRestores, (unsigned long long)c.sets,
                (unsigned long long)c.changed);
        g_c = Counters{};
    });
}

void Shutdown() {
    Protect("Shutdown", [&] {
        std::lock_guard lock(g_cm);
        g_watch = false;
        g_inFocus = false;
        if (g_swapped) RestoreLocked(nullptr);
        DropClonesLocked();
        ReleaseGraveLocked();
        g_tex.clear();
        g_rt.clear();
        g_byRes.clear();
        g_pending.clear();
        g_eyeW = g_eyeH = 0;
        g_enabled = false;
        g_dirty = false;
        if (!g_failed) g_phase = Waiting;
        g_c = g_ctot = Counters{};
        g_resetLogs = 0;
    });
}

void TestRegister(bool renderTarget, void* obj) {
    std::lock_guard lock(g_cm);
    g_installed = true;
    g_pending[(uint8_t*)obj] = renderTarget;
}

void TestDestroy(bool renderTarget, void* obj) { NoteDestroyed(obj, renderTarget); }

TestState TestGet() {
    std::lock_guard lock(g_cm);
    TestState s;
    s.phase = (int)g_phase;
    s.textures = (int)g_tex.size();
    s.renderTargets = (int)g_rt.size();
    for (auto& [obj, t] : g_tex) s.clones += t.clone ? 1 : 0;
    s.entries = (int)g_entries.size();
    s.swapped = g_swapped;
    s.cloneUses = g_cloneUses;
    s.leaks = g_leakReads + g_leakWrites;
    s.translated = g_c.translated + g_ctot.translated;
    return s;
}

}  // namespace qvfr::clones

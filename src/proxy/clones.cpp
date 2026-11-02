#include "clones.h"

#include <intrin.h>
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

#pragma intrinsic(_ReturnAddress)

namespace qvfr::clones {

namespace {

using hooks::Anchor;

// Engine object layout, AMS2AVX.exe build aa43bd68 (M7.1, M7.3 and M7.4, docs/ams2-renderer.md).
// Texture object (0x108 B, vtable 0x21192f8, ctor 0x1699170, dtor 0x169c280): the D3D11 texture at +0x58, its views
// after it (the dtor releases +0x58, +0x60, +0x70, ...), u16 width and height at +0x68/+0x6a, the name (engine string)
// at +0xf0.
constexpr size_t kTexRes = 0x58;
constexpr size_t kTexSize = 0x68;
constexpr size_t kTexEnd = 0x108;
constexpr size_t kTexName = 0xf0;
// Render target object (0xd0 B, vtable 0x21148f8, dtor 0x164e760): u16 size at +0x24/+0x26, u16 counts at +0x64
// (depth views) and +0x66 (color views); view arrays +0x78 (RTV), +0x80 (resolve RTV, may be null), +0x88 (second RTV),
// all +0x66 long, and +0x90 (DSV), +0x64 long; the name at +0xb8. Its textures (+0x98, +0xa0, +0xb0) are texture objects.
constexpr size_t kRtSize = 0x24;
constexpr size_t kRtDepthCount = 0x64, kRtViewCount = 0x66;
struct RtArray {
    size_t off, countOff;
};
constexpr RtArray kRtArrays[] = {{0x78, kRtViewCount}, {0x80, kRtViewCount}, {0x88, kRtViewCount}, {0x90, kRtDepthCount}};
constexpr size_t kRtName = 0xb8;
constexpr unsigned kMaxRtViews = 16;
// Engine device (the first argument of CreateTexture): per command context k (0..3, stride 0xe60) the bound render
// target's size as two floats at +0x9b18/+0x9b1c, and a byte at +0xe448+k that marks the context's constants dirty.
constexpr size_t kDevRtSize = 0x9b18, kDevStride = 0xe60, kDevDirty = 0xe448;
constexpr int kDevContexts = 4;
// The device's postprocess object (device+0x17b0, getter device+0x440) keeps the scene size as two ints at +0x100/+0x104;
// its tonemap (0x16319c0, called through vtable +0x190 by the task 0x9d4900) runs only when the scene texture has that
// size, so a group with clones needs the clone's size there (M7.4 run 7).
constexpr size_t kDevPost = 0x17b0, kPostSize = 0x100;

constexpr int kLearnGroups = 16;   // groups of each set watched before the clones are made (8 frames)
constexpr int kProofGroups = 240;  // groups of a set watched after its clones went in (120 frames)
constexpr int kProofDetail = 8;    // leaks counted per group for the first proof groups
constexpr int kMaxLeakLogs = 12;
constexpr int kMaxOtherLogs = 24;
constexpr int kMaxResetLogs = 20;
constexpr int kMaxCallerLogs = 16;
constexpr size_t kMaxPending = 1 << 18;
constexpr UINT kSrvSlots = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
constexpr UINT kUavSlots = D3D11_PS_CS_UAV_REGISTER_COUNT;
constexpr UINT kMaxViewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;

enum Phase { Waiting, Learning, Ready, Failed, NotNeeded };
const char* const kPhaseNames[] = {"waiting", "learning", "ready", "failed", "not needed"};
const char* const kSetNames[] = {"outer", "focus"};
enum Use { Read, Write };

// A size that is the game's render size divided by d, each axis rounded down or up.
struct Frac {
    int d = 0;
    bool upW = false, upH = false;
};

struct Tex {
    uint8_t* obj = nullptr;
    ID3D11Resource* res = nullptr;  // the engine's (owned by the object, not referenced here)
    D3D11_TEXTURE2D_DESC desc{};
    Frac fr;
    char name[64] = "";
    bool sizeField = false;              // u16 at +0x68/+0x6a match the D3D11 description
    bool read[2] = {}, written[2] = {};  // seen in the groups of each set while learning
    ID3D11Texture2D* clone[2] = {};      // ours, per set
    int views[2] = {};                   // view fields swapped
};
struct Rt {
    uint8_t* obj = nullptr;
    uint16_t w = 0, h = 0;
    Frac fr;
    char name[64] = "";
    int views[2] = {};  // array entries swapped
};
// One field of an engine object (a pointer, 8 bytes, or a u16 size pair, 4 bytes): the original outside the set's
// groups, the clone's inside one.
struct Entry {
    uint8_t* obj;
    uint8_t* addr;
    int bytes;
    uint64_t orig, clone;
};
// A render target or depth view of a clone: its size and the original's (at the view's mip).
struct Target {
    uint32_t cw = 0, ch = 0, ow = 0, oh = 0;
    char name[64] = "";  // the texture's
};

struct SetState {
    bool wanted = false;  // its size differs from the game's
    uint32_t w = 0, h = 0;
    bool failed = false;  // its clones could not be made for this list
    int learned = 0;      // groups watched in this list
    std::vector<Entry> entries;
    std::unordered_map<void*, void*> fwd, back;  // D3D11 view: original -> clone, clone -> original
    std::unordered_map<void*, Target> targets;   // clone RTVs and DSVs
    // Proof
    int proofLeft = 0, proofGroup = 0;
    uint64_t cloneUses = 0, leakReads = 0, leakWrites = 0, fixed = 0;
    uint64_t detail[kProofDetail] = {};
    int leakLogs = 0;
    bool changedLogged = false;
};

struct Counters {
    uint64_t swaps[2] = {}, translated = 0, lateRestores = 0, changed = 0, lists = 0;
    uint64_t vpMismatch = 0, vpFixed = 0, vpOutside = 0, devSizes = 0, postSizes = 0;
    uint64_t fixFwd = 0, fixBack = 0, lazy = 0, sizeMismatch = 0;
};

std::mutex g_cm;
bool g_installed = false;  // all four engine hooks in place
bool g_failed = false;     // a fault or a missing hook: no clones for the rest of the process
Phase g_phase = Waiting;
uint32_t g_gameW = 0, g_gameH = 0;
SetState g_set[2];
uint8_t* g_dev = nullptr;              // engine device
ID3D11DeviceContext* g_imm = nullptr;  // not referenced (gpu.cpp holds it)
uintptr_t g_d3dLo = 0, g_d3dHi = 0;    // module of the D3D11 runtime's object vtables
bool g_d3dHooked = false;
bool g_enabled = false;
bool g_dirty = false;    // the list of objects or the sizes changed
bool g_swapped = false;  // clones in the engine objects now (set g_cur)
bool g_inGroup = false;
Set g_cur = Focus;
int g_resetLogs = 0;
uint64_t g_frame = 0;
bool g_texSizeLogged = false, g_devLogged = false;

std::unordered_map<uint8_t*, bool> g_pending;  // created since the last frame start (true: render target)
std::unordered_map<uint8_t*, Tex> g_tex;
std::unordered_map<uint8_t*, Rt> g_rt;
std::unordered_map<ID3D11Resource*, uint8_t*> g_byRes, g_byClone;
std::vector<IUnknown*> g_owned;  // clone views
std::vector<IUnknown*> g_grave;  // released at the next frame start (on the render thread)

// Learning and proof: what the groups touched.
std::atomic<bool> g_watch{false};
thread_local bool t_internal = false;  // our own D3D11 calls (translation, viewports) are not observed
std::unordered_set<ID3D11Resource*> g_otherWritten;  // written in learning groups, not listed
std::vector<std::string> g_otherLines;
int g_otherCount[2] = {};
std::vector<void*> g_outsideCallers, g_mismatchCallers;
Counters g_c, g_ctot;

// Viewport while clones are in: what the engine asked for, what is set, the bound target.
std::atomic<bool> g_vpActive{false};
UINT g_vpN = 0;
D3D11_VIEWPORT g_vpWant[kMaxViewports] = {}, g_vpSet[kMaxViewports] = {};
UINT g_vpSetN = 0;
void* g_bound = nullptr;  // first RTV, else the DSV, as bound now
std::vector<void*> g_callers;

// Bind fix (M7.4): engine passes that keep their own D3D11 views of a target bypass the swapped fields. While an eye
// group runs with clones listed, every hooked bind, clear and copy goes through the maps: LiveSwap (a group with clones
// in) original -> clone, LiveBack (an eye group without them) clone -> original.
enum Live { LiveOff = 0, LiveSwap = 1, LiveBack = 2 };
std::atomic<int> g_live{LiveOff};
constexpr UINT kMapMax = 128;
std::vector<IUnknown*> g_lazyRefs;          // originals of views cloned on the fly (referenced: their address stays theirs)
std::unordered_set<void*> g_lazyFailed;     // views that could not be cloned (logged once)
std::vector<std::pair<void*, bool>> g_fixCallers;  // logged callers of the bind fix (caller, to the clone)

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

// module+0xRVA of a code address.
std::string Where(const void* p) {
    HMODULE mod = nullptr;
    char buf[160];
    if (p && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)p, &mod)) {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(mod, path, MAX_PATH);
        snprintf(buf, sizeof(buf), "%s+0x%llX", Narrow(fs::path(path).filename().wstring()).c_str(),
                 (unsigned long long)((uintptr_t)p - (uintptr_t)mod));
    } else {
        snprintf(buf, sizeof(buf), "%p", p);
    }
    return buf;
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

uint32_t Scaled(uint32_t size, int d, bool up) { return std::max(1u, up ? (size + d - 1) / d : size / d); }

// The game's render size divided by 1, 2, 4 or 8 (rounded down or up per axis), or d = 0.
Frac FractionOf(uint32_t w, uint32_t h) {
    Frac f;
    if (!g_gameW || !g_gameH) return f;
    for (int d = 1; d <= 8; d *= 2) {
        const uint32_t fw = g_gameW / d, cw = Scaled(g_gameW, d, true), fh = g_gameH / d, ch = Scaled(g_gameH, d, true);
        if ((w == fw || w == cw) && (h == fh || h == ch)) {
            f.d = d;
            f.upW = w != fw;
            f.upH = h != fh;
            return f;
        }
    }
    return f;
}

// (w, h) is (fromW, fromH) / d for some d (rounded either way): the same fraction of (toW, toH).
bool MapSize(uint32_t fromW, uint32_t fromH, uint32_t toW, uint32_t toH, float w, float h, float& ow, float& oh) {
    for (int d = 1; d <= 8; d *= 2) {
        bool okW = false, okH = false;
        for (bool up : {false, true}) {
            if (!okW && w == (float)Scaled(fromW, d, up)) {
                okW = true;
                ow = (float)Scaled(toW, d, up);
            }
            if (!okH && h == (float)Scaled(fromH, d, up)) {
                okH = true;
                oh = (float)Scaled(toH, d, up);
            }
        }
        if (okW && okH) return true;
    }
    return false;
}

UINT MaxMips(uint32_t w, uint32_t h) {
    UINT n = 1;
    for (uint32_t s = std::max(w, h); s > 1; s >>= 1) ++n;
    return n;
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

D3D11_TEXTURE2D_DESC DescOf(ID3D11Texture2D* t) {
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    return d;
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

bool ReadField(const Entry& e, uint64_t& v) {
    v = 0;
    return hooks::TryRead(e.addr, &v, e.bytes);
}
void WriteField(const Entry& e, uint64_t v) { memcpy(e.addr, &v, e.bytes); }

// ---------------------------------------------------------------- object lists

void RegisterTexLocked(uint8_t* obj) {
    if (g_tex.count(obj)) return;
    void* res = nullptr;
    if (!hooks::TryRead(obj + kTexRes, &res, 8) || !IsD3D(res)) return;
    ID3D11Texture2D* t = nullptr;
    if (FAILED(static_cast<IUnknown*>(res)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) || !t) return;
    const D3D11_TEXTURE2D_DESC d = DescOf(t);
    t->Release();
    const Frac fr = FractionOf(d.Width, d.Height);
    if (!fr.d || d.Usage != D3D11_USAGE_DEFAULT) return;
    Tex x;
    x.obj = obj;
    x.res = static_cast<ID3D11Resource*>(res);
    x.desc = d;
    x.fr = fr;
    ReadName(obj, kTexName, x.name);
    uint16_t wh[2] = {};
    x.sizeField = hooks::TryRead(obj + kTexSize, wh, 4) && wh[0] == d.Width && wh[1] == d.Height;
    if (!x.sizeField && !g_texSizeLogged) {
        g_texSizeLogged = true;
        Log("Clones: texture object %s has %u x %u at +0x68, not its D3D11 size %u x %u; such size fields stay as they "
            "are (logged once)",
            TexName(x), wh[0], wh[1], d.Width, d.Height);
    }
    g_byRes[x.res] = obj;
    g_tex[obj] = x;
    g_dirty = true;
}

void RegisterRtLocked(uint8_t* obj) {
    if (g_rt.count(obj)) return;
    uint16_t wh[2] = {};
    if (!hooks::TryRead(obj + kRtSize, wh, 4)) return;
    const Frac fr = FractionOf(wh[0], wh[1]);
    if (!fr.d) return;
    Rt r;
    r.obj = obj;
    r.w = wh[0];
    r.h = wh[1];
    r.fr = fr;
    ReadName(obj, kRtName, r.name);
    g_rt[obj] = r;
    g_dirty = true;
}

void ProcessPendingLocked() {
    if (!g_gameW || !g_d3dLo || g_pending.empty()) return;
    // Textures first: a render target's textures are created inside its constructor.
    for (auto& [obj, rt] : g_pending)
        if (!rt) RegisterTexLocked(obj);
    for (auto& [obj, rt] : g_pending)
        if (rt) RegisterRtLocked(obj);
    g_pending.clear();
}

// Puts the originals back into one object's fields (it is being destroyed while its clones are in).
void RestoreObjectLocked(uint8_t* obj) {
    for (const Entry& e : g_set[g_cur].entries) {
        if (e.obj != obj) continue;
        uint64_t cur = 0;
        if (ReadField(e, cur) && cur == e.clone) WriteField(e, e.orig);
    }
}

void DestroyLocked(uint8_t* obj, bool rt) {
    g_pending.erase(obj);
    if (g_swapped) RestoreObjectLocked(obj);
    bool listed = false;
    for (SetState& s : g_set) {
        const size_t before = s.entries.size();
        s.entries.erase(std::remove_if(s.entries.begin(), s.entries.end(), [&](const Entry& e) { return e.obj == obj; }),
                        s.entries.end());
        listed |= before != s.entries.size();
    }
    if (rt) {
        listed |= g_rt.erase(obj) > 0;
    } else if (auto it = g_tex.find(obj); it != g_tex.end()) {
        g_byRes.erase(it->second.res);
        for (ID3D11Texture2D*& c : it->second.clone) {
            if (!c) continue;
            g_byClone.erase(c);
            g_grave.push_back(c);
            c = nullptr;
        }
        g_tex.erase(it);
        listed = true;
    }
    if (listed) g_dirty = true;
}

// ---------------------------------------------------------------- clones

void DropSetLocked(SetState& s) {
    s.entries.clear();
    s.fwd.clear();
    s.back.clear();
    s.targets.clear();
    s.proofLeft = 0;
}

void DropClonesLocked() {
    for (IUnknown* u : g_owned) g_grave.push_back(u);
    g_owned.clear();
    for (auto& [obj, t] : g_tex)
        for (int s = 0; s < 2; ++s) {
            if (t.clone[s]) g_grave.push_back(t.clone[s]);
            t.clone[s] = nullptr;
            t.views[s] = 0;
        }
    for (auto& [obj, r] : g_rt) r.views[0] = r.views[1] = 0;
    g_byClone.clear();
    for (SetState& s : g_set) DropSetLocked(s);
    for (IUnknown* u : g_lazyRefs) g_grave.push_back(u);
    g_lazyRefs.clear();
    g_lazyFailed.clear();
    g_live = LiveOff;
}

bool AnyWanted() { return g_set[Outer].wanted || g_set[Focus].wanted; }

// A changed list: no clones until the groups were watched again.
void ResetLocked(const char* why) {
    DropClonesLocked();
    for (auto& [obj, t] : g_tex)
        for (int s = 0; s < 2; ++s) t.read[s] = t.written[s] = false;
    for (SetState& s : g_set) {
        s.failed = false;
        s.learned = 0;
        s.changedLogged = false;
    }
    g_otherWritten.clear();
    g_otherLines.clear();
    g_otherCount[0] = g_otherCount[1] = 0;
    if (g_failed) return;
    if (!AnyWanted()) {
        if (g_phase != NotNeeded)
            Log("Clones: not needed: outer and focus both want the game's render size %u x %u", g_gameW, g_gameH);
        g_phase = NotNeeded;
        return;
    }
    g_phase = Learning;
    ++g_c.lists;
    if (g_resetLogs < kMaxResetLogs) {
        ++g_resetLogs;
        Log("Clones: %s: %zu textures and %zu render targets in the game's render size %u x %u (or 1/2, 1/4, 1/8); "
            "watching which ones outer and focus write (%d groups each); clones for %s%s",
            why, g_tex.size(), g_rt.size(), g_gameW, g_gameH, kLearnGroups,
            g_set[Outer].wanted && g_set[Focus].wanted ? "outer and focus"
            : g_set[Outer].wanted                      ? "outer"
                                                       : "focus",
            g_resetLogs == kMaxResetLogs ? " (further list changes only counted)" : "");
    }
}

uint32_t MipSize(uint32_t s, UINT mip) { return std::max(1u, s >> std::min(mip, 31u)); }

// A clone of view v (the same description) on set s's clone texture; reused when v appears twice. Render target and
// depth views are noted with their sizes for the viewport check.
void* CloneViewLocked(SetState& s, ID3D11Device* dev, void* v, ID3D11Texture2D* clone, const Tex& t) {
    if (auto it = s.fwd.find(v); it != s.fwd.end()) return it->second;
    IUnknown* u = static_cast<IUnknown*>(v);
    IUnknown* out = nullptr;
    HRESULT hr = E_NOINTERFACE;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    int mip = -1;  // a target view's mip
    if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11ShaderResourceView), (void**)&srv)) && srv) {
        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        srv->GetDesc(&d);
        srv->Release();
        hr = dev->CreateShaderResourceView(clone, &d, (ID3D11ShaderResourceView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11RenderTargetView), (void**)&rtv)) && rtv) {
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        rtv->GetDesc(&d);
        rtv->Release();
        mip = d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D        ? (int)d.Texture2D.MipSlice
              : d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY ? (int)d.Texture2DArray.MipSlice
                                                                      : 0;
        hr = dev->CreateRenderTargetView(clone, &d, (ID3D11RenderTargetView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11DepthStencilView), (void**)&dsv)) && dsv) {
        D3D11_DEPTH_STENCIL_VIEW_DESC d{};
        dsv->GetDesc(&d);
        dsv->Release();
        mip = d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D        ? (int)d.Texture2D.MipSlice
              : d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DARRAY ? (int)d.Texture2DArray.MipSlice
                                                                      : 0;
        hr = dev->CreateDepthStencilView(clone, &d, (ID3D11DepthStencilView**)&out);
    } else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11UnorderedAccessView), (void**)&uav)) && uav) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
        uav->GetDesc(&d);
        uav->Release();
        hr = dev->CreateUnorderedAccessView(clone, &d, (ID3D11UnorderedAccessView**)&out);
    }
    if (FAILED(hr) || !out) {
        const D3D11_TEXTURE2D_DESC cd = DescOf(clone);
        Log("Clones: cannot clone view %p of %s on its %u x %u clone with %u mips (0x%08lX)", v, TexName(t), cd.Width,
            cd.Height, cd.MipLevels, (unsigned long)hr);
        return nullptr;
    }
    g_owned.push_back(out);
    s.fwd[v] = out;
    s.back[out] = v;
    if (mip >= 0) {
        const D3D11_TEXTURE2D_DESC cd = DescOf(clone);
        Target& tg = s.targets[out];
        tg = {MipSize(cd.Width, mip), MipSize(cd.Height, mip), MipSize(t.desc.Width, mip), MipSize(t.desc.Height, mip)};
        memcpy(tg.name, t.name, sizeof(tg.name));
    }
    return out;
}

void AddEntry(SetState& s, uint8_t* obj, void* addr, int bytes, uint64_t orig, uint64_t clone) {
    s.entries.push_back({obj, (uint8_t*)addr, bytes, orig, clone});
}

uint64_t PackSize(uint32_t w, uint32_t h) { return (uint64_t)(w & 0xffff) | ((uint64_t)(h & 0xffff) << 16); }

// Outer keeps a texture focus reads but does not write: focus needs outer's data in it.
bool KeptForFocus(const Tex& t) { return t.read[Focus] && !t.written[Focus]; }

bool CloneWanted(const Tex& t, Set s) { return t.written[s] && !(s == Outer && KeptForFocus(t)); }

// Clones for the textures set s writes, entries for their fields and the render targets' view arrays and sizes.
// A new texture's memory may hold anything (NaN included). Passes that accumulate over frames (history) read their
// target before writing it and never recover from a NaN, so every clone starts at zero (run 2 of M7.4: the focus went
// black). Multisampled textures take no initial data; the engine clears them every frame.
struct Zeros {
    std::vector<uint8_t> mem;
    std::vector<D3D11_SUBRESOURCE_DATA> subs;
    // Initial data for d, or null when it cannot have any.
    const D3D11_SUBRESOURCE_DATA* For(const D3D11_TEXTURE2D_DESC& d) {
        if (d.SampleDesc.Count > 1) return nullptr;
        const size_t pitch = ((size_t)d.Width * BitsPerPixel(d.Format) + 7) / 8;
        const size_t bytes = pitch * d.Height;
        if (mem.size() < bytes) mem.assign(bytes, 0);
        subs.clear();
        for (UINT a = 0; a < d.ArraySize; ++a)
            for (UINT m = 0; m < d.MipLevels; ++m) {
                const size_t p = ((size_t)std::max(1u, d.Width >> m) * BitsPerPixel(d.Format) + 7) / 8;
                subs.push_back({mem.data(), (UINT)p, (UINT)(p * std::max(1u, d.Height >> m))});
            }
        return subs.data();
    }
};

bool BuildSetLocked(ID3D11Device* dev, Set si, int& clones, double& mb) {
    SetState& s = g_set[si];
    bool ok = true;
    Zeros zeros;
    int zeroed = 0, notZeroed = 0;
    for (auto& [obj, t] : g_tex) {
        if (!CloneWanted(t, si)) continue;
        D3D11_TEXTURE2D_DESC d = t.desc;
        d.Width = Scaled(s.w, t.fr.d, t.fr.upW);
        d.Height = Scaled(s.h, t.fr.d, t.fr.upH);
        d.MipLevels = std::min(std::max(1u, d.MipLevels), MaxMips(d.Width, d.Height));
        d.MiscFlags &= ~(UINT)(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                               D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
        ID3D11Texture2D* c = nullptr;
        const D3D11_SUBRESOURCE_DATA* init = zeros.For(d);
        HRESULT hr = init ? dev->CreateTexture2D(&d, init, &c) : E_INVALIDARG;
        if (SUCCEEDED(hr) && c) {
            ++zeroed;
        } else {
            c = nullptr;
            hr = dev->CreateTexture2D(&d, nullptr, &c);
            ++notZeroed;
        }
        if (FAILED(hr) || !c) {
            Log("Clones: cannot create the %s clone of %s (%u x %u fmt %d x%u, 0x%08lX)", kSetNames[si], TexName(t),
                d.Width, d.Height, (int)d.Format, d.SampleDesc.Count, (unsigned long)hr);
            ok = false;
            break;
        }
        t.clone[si] = c;
        g_byClone[c] = obj;
        mb += Megabytes(d);
        ++clones;
        AddEntry(s, obj, obj + kTexRes, 8, (uint64_t)t.res, (uint64_t)c);
        if (t.sizeField)
            AddEntry(s, obj, obj + kTexSize, 4, PackSize(t.desc.Width, t.desc.Height), PackSize(d.Width, d.Height));
        for (size_t off = kTexRes + 8; off + 8 <= kTexEnd; off += 8) {
            if (off == kTexSize) continue;
            void* v = nullptr;
            if (!hooks::TryRead(obj + off, &v, 8)) continue;
            ID3D11Resource* r = ViewResource(v);
            if (!r) continue;
            const bool mine = r == t.res;
            r->Release();
            if (!mine) continue;
            void* cv = CloneViewLocked(s, dev, v, c, t);
            if (!cv) {
                ok = false;
                break;
            }
            AddEntry(s, obj, obj + off, 8, (uint64_t)v, (uint64_t)cv);
            ++t.views[si];
        }
        if (!ok) break;
    }
    int mixed = 0;
    for (auto& [obj, rt] : g_rt) {
        if (!ok) break;
        int other = 0;  // views of listed textures without a clone for this set
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
                if (!t.clone[si]) {
                    ++other;
                    continue;
                }
                void* cv = CloneViewLocked(s, dev, v, t.clone[si], t);
                if (!cv) {
                    ok = false;
                    break;
                }
                AddEntry(s, obj, arr + i, 8, (uint64_t)v, (uint64_t)cv);
                ++rt.views[si];
            }
        }
        if (rt.views[si]) {
            AddEntry(s, obj, obj + kRtSize, 4, PackSize(rt.w, rt.h),
                     PackSize(Scaled(s.w, rt.fr.d, rt.fr.upW), Scaled(s.h, rt.fr.d, rt.fr.upH)));
            if (other) ++mixed;
        }
    }
    if (mixed)
        Log("Clones: %d render targets have views of both cloned and shared textures for %s (sizes may not match)",
            mixed, kSetNames[si]);
    Log("Clones: %s: %d clones start at zero, %d without initial data (multisampled, or refused with it)",
        kSetNames[si], zeroed, notZeroed);
    return ok;
}

void LogListLocked() {
    Log("Clones: outer watched in %d groups, focus in %d; %zu listed textures (O outer, F focus; r read, w written):",
        g_set[Outer].learned, g_set[Focus].learned, g_tex.size());
    std::vector<const Tex*> list;
    for (auto& [obj, t] : g_tex) list.push_back(&t);
    std::sort(list.begin(), list.end(), [](const Tex* a, const Tex* b) {
        const int wa = a->written[0] + a->written[1], wb = b->written[0] + b->written[1];
        if (wa != wb) return wa > wb;
        return strcmp(a->name, b->name) < 0;
    });
    auto role = [](bool r, bool w) { return w ? (r ? "rw" : "w ") : (r ? "r " : "- "); };
    for (const Tex* t : list) {
        char clones[96] = "";
        size_t n = 0;
        for (int s = 0; s < 2; ++s)
            if (t->clone[s]) {
                const D3D11_TEXTURE2D_DESC d = DescOf(t->clone[s]);
                n += snprintf(clones + n, sizeof(clones) - n, " -> %s %u x %u", kSetNames[s], d.Width, d.Height);
            }
        const bool kept = g_set[Outer].wanted && t->written[Outer] && KeptForFocus(*t);
        char line[360];
        snprintf(line, sizeof(line), "  %-40s %5u x %-5u (1/%d) fmt %3d x%u mips %u arr %u  O %s F %s%s%s", TexName(*t),
                 t->desc.Width, t->desc.Height, t->fr.d, (int)t->desc.Format, t->desc.SampleDesc.Count,
                 t->desc.MipLevels, t->desc.ArraySize, role(t->read[Outer], t->written[Outer]),
                 role(t->read[Focus], t->written[Focus]), clones, kept ? "  (kept for outer: focus reads it)" : "");
        LogRaw(line);
    }
    if (!g_otherLines.empty()) {
        Log("Clones: %zu other targets written by outer or focus (not in the game's size, or not from the engine "
            "factory; up to %d per set listed):",
            g_otherWritten.size(), kMaxOtherLogs);
        for (const std::string& s : g_otherLines) LogRaw(s.c_str());
    }
}

void StartProofLocked(SetState& s) {
    s.proofLeft = s.entries.empty() ? 0 : kProofGroups;
    s.proofGroup = 0;
    s.cloneUses = s.leakReads = s.leakWrites = s.fixed = 0;
    memset(s.detail, 0, sizeof(s.detail));
    s.leakLogs = 0;
}

bool BuildLocked(ID3D11DeviceContext* imm) {
    ID3D11Device* dev = nullptr;
    imm->GetDevice(&dev);
    if (!dev) return false;
    int clones[2] = {};
    double mb[2] = {};
    for (int si = 0; si < 2; ++si) {
        SetState& s = g_set[si];
        if (!s.wanted) continue;
        if (!BuildSetLocked(dev, (Set)si, clones[si], mb[si])) {
            s.failed = true;
            // Its clones stay listed until the next list change (released then); its entries are dropped now.
            DropSetLocked(s);
            Log("Clones: OFF for %s in this list (see the line above); %s renders at the game's size", kSetNames[si],
                kSetNames[si]);
        }
    }
    dev->Release();
    LogListLocked();
    const bool any = !g_set[Outer].entries.empty() || !g_set[Focus].entries.empty();
    g_phase = any ? Ready : Failed;
    for (int si = 0; si < 2; ++si) {
        SetState& s = g_set[si];
        if (!s.wanted || s.failed) continue;
        StartProofLocked(s);
        size_t targets = 0;
        for (auto& [obj, rt] : g_rt) targets += rt.views[si] ? 1 : 0;
        Log("Clones: %s: %d clones in %u x %u (%.0f MB), %zu fields swapped per group (%zu D3D11 views, %zu render "
            "targets); %s",
            kSetNames[si], clones[si], s.w, s.h, mb[si], s.entries.size(), s.fwd.size(), targets,
            g_enabled ? "on from its next group" : "made, but off (setting or hotkey)");
    }
    return any;
}

// The D3D11 state bound now, with every view found in `map` replaced (RTVs, DSV, SRVs of all stages, CS UAVs).
// Returns the bound target after it (the first RTV, else the DSV).
void* TranslateLocked(ID3D11DeviceContext* imm, const std::unordered_map<void*, void*>& map) {
    if (!imm) return nullptr;
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
    void* bound = nullptr;
    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
        nrtv[i] = (ID3D11RenderTargetView*)tr(rtv[i]);
        ch |= nrtv[i] != rtv[i];
        if (!bound) bound = nrtv[i];
    }
    ID3D11DepthStencilView* ndsv = (ID3D11DepthStencilView*)tr(dsv);
    ch |= ndsv != dsv;
    if (!bound) bound = ndsv;
    if (ch && !map.empty()) {
        imm->OMSetRenderTargetsAndUnorderedAccessViews(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, nrtv, ndsv, 0,
                                                       D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr);
        any = true;
    }
    for (ID3D11RenderTargetView* v : rtv)
        if (v) v->Release();
    if (dsv) dsv->Release();
    if (map.empty()) {
        t_internal = false;
        return bound;
    }
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
    return bound;
}

// ---------------------------------------------------------------- engine device: render target size per context

// Every context whose size is a fraction of (fromW, fromH) gets the same fraction of (toW, toH) and its dirty flag.
// Raw (no lock, no logging): also used after a fault.
int MapDeviceSizes(uint32_t fromW, uint32_t fromH, uint32_t toW, uint32_t toH) {
    if (!g_dev) return 0;
    int n = 0;
    for (int k = 0; k < kDevContexts; ++k) {
        float* wh = (float*)(g_dev + kDevRtSize + k * kDevStride);
        float cur[2] = {}, to[2] = {};
        if (!hooks::TryRead(wh, cur, 8) || !MapSize(fromW, fromH, toW, toH, cur[0], cur[1], to[0], to[1])) continue;
        if (cur[0] == to[0] && cur[1] == to[1]) continue;
        wh[0] = to[0];
        wh[1] = to[1];
        g_dev[kDevDirty + k] = 1;
        ++n;
    }
    return n;
}

// The postprocess object's scene size, when it is (fromW, fromH), becomes (toW, toH). Raw, like MapDeviceSizes.
int MapPostSize(uint32_t fromW, uint32_t fromH, uint32_t toW, uint32_t toH) {
    uint8_t* post = nullptr;
    int32_t wh[2] = {};
    if (!g_dev || !hooks::TryRead(g_dev + kDevPost, &post, 8) || !post || !hooks::TryRead(post + kPostSize, wh, 8))
        return 0;
    if (wh[0] != (int32_t)fromW || wh[1] != (int32_t)fromH) return 0;
    wh[0] = (int32_t)toW;
    wh[1] = (int32_t)toH;
    memcpy(post + kPostSize, wh, 8);
    return 1;
}

// ---------------------------------------------------------------- viewports while clones are in

using RsVpFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_VIEWPORT*);
RsVpFn o_rsVp = nullptr;

void NoteCallerLocked(void* caller, const D3D11_VIEWPORT& v, const Target& t, bool fixed) {
    if ((int)g_callers.size() >= kMaxCallerLogs) return;
    if (std::find(g_callers.begin(), g_callers.end(), caller) != g_callers.end()) return;
    g_callers.push_back(caller);
    Log("Clones: viewport %.0f x %.0f at (%.0f, %.0f) for %s (clone %u x %u, original %u x %u) in a %s group, from %s; "
        "%s%s",
        v.Width, v.Height, v.TopLeftX, v.TopLeftY, t.name[0] ? t.name : "(no name)", t.cw, t.ch, t.ow, t.oh,
        kSetNames[g_cur], caller ? Where(caller).c_str() : "the state bound before the group",
        fixed ? "scaled to the clone" : "NOT scaled (clones_viewport_fix=0)",
        (int)g_callers.size() == kMaxCallerLogs ? " (further callers only counted)" : "");
}

// The viewports for the engine's request and the bound target: a viewport larger than the clone that fits in the
// original was computed for the original and is scaled to the clone. Set when it differs from what is set (always
// when `force`).
void ApplyViewportsLocked(void* caller, bool force) {
    if (!o_rsVp || !g_imm) return;
    D3D11_VIEWPORT eff[kMaxViewports];
    memcpy(eff, g_vpWant, sizeof(D3D11_VIEWPORT) * g_vpN);
    const SetState& s = g_set[g_cur];
    auto it = g_bound ? s.targets.find(g_bound) : s.targets.end();
    if (it != s.targets.end()) {
        const Target& t = it->second;
        const bool fix = Cfg().clonesViewportFix;
        for (UINT i = 0; i < g_vpN; ++i) {
            D3D11_VIEWPORT& v = eff[i];
            if (v.Width <= t.cw + 0.5f && v.Height <= t.ch + 0.5f) continue;
            if (v.TopLeftX + v.Width > t.ow + 0.5f || v.TopLeftY + v.Height > t.oh + 0.5f) {
                ++g_c.vpOutside;
                if (g_outsideCallers.size() < 8 &&
                    std::find(g_outsideCallers.begin(), g_outsideCallers.end(), caller) == g_outsideCallers.end()) {
                    g_outsideCallers.push_back(caller);
                    Log("Clones: viewport %.0f x %.0f at (%.0f, %.0f) larger than the original of %s (clone %u x %u, "
                        "original %u x %u) in a %s group, from %s; left as it is",
                        v.Width, v.Height, v.TopLeftX, v.TopLeftY, t.name[0] ? t.name : "(no name)", t.cw, t.ch, t.ow,
                        t.oh, kSetNames[g_cur], caller ? Where(caller).c_str() : "the state bound before the group");
                }
                continue;
            }
            ++g_c.vpMismatch;
            NoteCallerLocked(caller, v, t, fix);
            if (!fix) continue;
            const float sx = (float)t.cw / t.ow, sy = (float)t.ch / t.oh;
            v.TopLeftX *= sx;
            v.Width *= sx;
            v.TopLeftY *= sy;
            v.Height *= sy;
            ++g_c.vpFixed;
        }
    }
    if (!force && g_vpSetN == g_vpN && !memcmp(eff, g_vpSet, sizeof(D3D11_VIEWPORT) * g_vpN)) return;
    t_internal = true;
    o_rsVp(g_imm, g_vpN, eff);
    t_internal = false;
    memcpy(g_vpSet, eff, sizeof(D3D11_VIEWPORT) * g_vpN);
    g_vpSetN = g_vpN;
}

void OnBindLocked(void* bound, void* caller) {
    if (!g_swapped || bound == g_bound) return;
    g_bound = bound;
    ApplyViewportsLocked(caller, false);
}

// ---------------------------------------------------------------- swap in and out

void RestoreLocked(ID3D11DeviceContext* imm) {
    g_live = LiveOff;
    if (!g_swapped) return;
    g_swapped = false;
    g_vpActive = false;
    SetState& s = g_set[g_cur];
    int changed = 0;
    for (const Entry& e : s.entries) {
        uint64_t cur = 0;
        if (!ReadField(e, cur)) continue;
        if (cur == e.clone)
            WriteField(e, e.orig);
        else
            ++changed;
    }
    g_c.devSizes += MapDeviceSizes(s.w, s.h, g_gameW, g_gameH);
    g_c.postSizes += MapPostSize(s.w, s.h, g_gameW, g_gameH);
    TranslateLocked(imm, s.back);
    // The engine's last viewport request, as asked.
    if (imm && o_rsVp && g_vpN &&
        (g_vpSetN != g_vpN || memcmp(g_vpSet, g_vpWant, sizeof(D3D11_VIEWPORT) * g_vpN))) {
        t_internal = true;
        o_rsVp(imm, g_vpN, g_vpWant);
        t_internal = false;
    }
    g_bound = nullptr;
    if (changed) {
        // The engine replaced a field inside a group: its value stays, the list is watched again.
        ++g_c.changed;
        g_dirty = true;
        Log("Clones: %d fields changed by the engine during a %s group; list dropped", changed, kSetNames[g_cur]);
    }
}

bool PutBackRaw() {
    __try {
        g_vpActive = false;
        g_live = LiveOff;
        if (g_swapped) {
            g_swapped = false;
            const SetState& s = g_set[g_cur];
            for (const Entry& e : s.entries) {
                uint64_t cur = 0;
                if (ReadField(e, cur) && cur == e.clone) WriteField(e, e.orig);
            }
            MapDeviceSizes(s.w, s.h, g_gameW, g_gameH);
            MapPostSize(s.w, s.h, g_gameW, g_gameH);
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
    g_live = LiveOff;
    Log("ERROR: exception 0x%08lX in clones (%s); clones OFF for this session%s", code, where,
        PutBackRaw() ? "" : "; the originals could NOT be put back");
}

// ---------------------------------------------------------------- bind fix

bool Live(ID3D11DeviceContext* c) {
    return g_live.load(std::memory_order_relaxed) != LiveOff && !t_internal && c == g_imm;
}

const Tex* TexOfResLocked(ID3D11Resource* r) {
    if (auto it = g_byRes.find(r); it != g_byRes.end()) return &g_tex[it->second];
    if (auto it = g_byClone.find(r); it != g_byClone.end()) return &g_tex[it->second];
    return nullptr;
}

const Tex* TexOfViewLocked(void* v) {
    ID3D11Resource* r = nullptr;
    static_cast<ID3D11View*>(v)->GetResource(&r);
    const Tex* t = r ? TexOfResLocked(r) : nullptr;
    if (r) r->Release();
    return t;
}

// Counts a translation and logs its caller once (up to 16). `t` null: found from `item` (a view or, with `res`, a
// resource) only for the log line.
void NoteFixLocked(bool toClone, const char* method, void* caller, void* item, bool res, bool lazy) {
    ++(toClone ? g_c.fixFwd : g_c.fixBack);
    if (lazy) ++g_c.lazy;
    if (toClone && g_set[g_cur].proofLeft > 0) ++g_set[g_cur].fixed;
    if ((int)g_fixCallers.size() >= kMaxCallerLogs) return;
    const std::pair<void*, bool> key{caller, toClone};
    if (std::find(g_fixCallers.begin(), g_fixCallers.end(), key) != g_fixCallers.end()) return;
    g_fixCallers.push_back(key);
    const Tex* t = res ? TexOfResLocked((ID3D11Resource*)item) : TexOfViewLocked(item);
    char what[96];
    if (t)
        snprintf(what, sizeof(what), "%s %u x %u fmt %d", TexName(*t), t->desc.Width, t->desc.Height,
                 (int)t->desc.Format);
    else
        snprintf(what, sizeof(what), "?");
    if (toClone)
        Log("Clones: bind fix: a %s group reaches the original %s through %s from %s, not through the swapped fields; "
            "its clone used instead%s%s",
            kSetNames[g_cur], what, method, Where(caller).c_str(), lazy ? " (a view made on the clone)" : "",
            (int)g_fixCallers.size() == kMaxCallerLogs ? " (further callers only counted)" : "");
    else
        Log("Clones: bind fix: %s %s group without clones reaches a clone of %s through %s from %s; the original "
            "used instead%s",
            g_cur == Outer ? "an" : "a", kSetNames[g_cur], what, method, Where(caller).c_str(),
            (int)g_fixCallers.size() == kMaxCallerLogs ? " (further callers only counted)" : "");
}

// The view to bind instead of v (v itself when nothing to translate).
void* MapViewLocked(void* v, const char* method, void* caller) {
    if (!v) return v;
    const int live = g_live.load(std::memory_order_relaxed);
    if (live == LiveBack) {
        for (const SetState& s : g_set)
            if (auto it = s.back.find(v); it != s.back.end()) {
                NoteFixLocked(false, method, caller, v, false, false);
                return it->second;
            }
        return v;
    }
    if (live != LiveSwap) return v;
    SetState& s = g_set[g_cur];
    if (auto it = s.fwd.find(v); it != s.fwd.end()) {
        NoteFixLocked(true, method, caller, v, false, false);
        return it->second;
    }
    if (s.back.count(v) || g_lazyFailed.count(v)) return v;
    // A view the engine keeps outside the swapped fields: cloned on the fly when its texture has a clone.
    ID3D11Resource* r = nullptr;
    static_cast<ID3D11View*>(v)->GetResource(&r);
    if (!r) return v;
    auto bt = g_byRes.find(r);
    r->Release();
    if (bt == g_byRes.end()) return v;
    const Tex& t = g_tex[bt->second];
    if (!t.clone[g_cur]) return v;
    ID3D11Device* dev = nullptr;
    g_imm->GetDevice(&dev);
    void* cv = dev ? CloneViewLocked(s, dev, v, t.clone[g_cur], t) : nullptr;
    if (dev) dev->Release();
    if (!cv) {
        g_lazyFailed.insert(v);
        return v;
    }
    static_cast<IUnknown*>(v)->AddRef();
    g_lazyRefs.push_back(static_cast<IUnknown*>(v));
    NoteFixLocked(true, method, caller, v, false, true);
    return cv;
}

ID3D11Resource* MapResLocked(ID3D11Resource* r, const char* method, void* caller) {
    if (!r) return r;
    const int live = g_live.load(std::memory_order_relaxed);
    if (live == LiveSwap) {
        auto it = g_byRes.find(r);
        if (it == g_byRes.end() || !g_tex[it->second].clone[g_cur]) return r;
        NoteFixLocked(true, method, caller, r, true, false);
        return g_tex[it->second].clone[g_cur];
    }
    if (live == LiveBack) {
        auto it = g_byClone.find(r);
        if (it == g_byClone.end()) return r;
        NoteFixLocked(false, method, caller, r, true, false);
        return g_tex[it->second].res;
    }
    return r;
}

// The views to bind: `views` itself, or `tmp` (n entries) with translations.
template <class V>
V* const* MapViews(ID3D11DeviceContext* c, V* const* views, UINT n, V** tmp, const char* method, void* caller) {
    if (!views || !n || n > kMapMax || !Live(c)) return views;
    bool changed = false;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (g_live.load(std::memory_order_relaxed) == LiveOff) return;
        for (UINT i = 0; i < n; ++i) {
            tmp[i] = (V*)MapViewLocked(views[i], method, caller);
            changed |= tmp[i] != views[i];
        }
    });
    return changed ? tmp : views;
}

template <class V>
V* MapView(ID3D11DeviceContext* c, V* v, const char* method, void* caller) {
    V* tmp[1] = {};
    return *MapViews(c, &v, 1, tmp, method, caller);
}

ID3D11Resource* MapRes(ID3D11DeviceContext* c, ID3D11Resource* r, const char* method, void* caller) {
    if (!r || !Live(c)) return r;
    ID3D11Resource* out = r;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        out = MapResLocked(r, method, caller);
    });
    return out;
}

// ---------------------------------------------------------------- observation (learning and proof groups)

void NoteLocked(ID3D11Resource* r, Use use, const char* method, void* caller) {
    if (!r) return;
    if (g_phase == Learning) {
        if (auto it = g_byRes.find(r); it != g_byRes.end()) {
            Tex& t = g_tex[it->second];
            (use == Write ? t.written : t.read)[g_cur] = true;
        } else if (use == Write && g_otherCount[g_cur] < kMaxOtherLogs && g_otherWritten.insert(r).second) {
            ++g_otherCount[g_cur];
            char line[280];
            ID3D11Texture2D* t = nullptr;
            if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
                const D3D11_TEXTURE2D_DESC d = DescOf(t);
                t->Release();
                snprintf(line, sizeof(line), "  %p %u x %u fmt %d x%u (%s %s, from %s)%s", (void*)r, d.Width, d.Height,
                         (int)d.Format, d.SampleDesc.Count, kSetNames[g_cur], method, Where(caller).c_str(),
                         FractionOf(d.Width, d.Height).d ? "  GAME SIZE (fraction), not from the engine factory" : "");
            } else {
                snprintf(line, sizeof(line), "  %p not a 2D texture (%s %s, from %s)", (void*)r, kSetNames[g_cur],
                         method, Where(caller).c_str());
            }
            g_otherLines.push_back(line);
        }
        return;
    }
    SetState& s = g_set[g_cur];
    if (g_byClone.count(r)) {
        ++s.cloneUses;
        return;
    }
    auto it = g_byRes.find(r);
    if (it == g_byRes.end() || !g_tex[it->second].clone[g_cur]) return;
    ++(use == Write ? s.leakWrites : s.leakReads);
    if (s.proofGroup < kProofDetail) ++s.detail[s.proofGroup];
    if (s.leakLogs < kMaxLeakLogs) {
        ++s.leakLogs;
        Log("Clones: LEAK: %s group %d of the proof (frame %llu) %s the original %s through %s from %s%s",
            kSetNames[g_cur], s.proofGroup + 1, (unsigned long long)g_frame, use == Write ? "writes" : "reads",
            TexName(g_tex[it->second]), method, Where(caller).c_str(),
            s.leakLogs == kMaxLeakLogs ? " (further leaks only counted)" : "");
    }
}

bool Watching() { return g_watch.load(std::memory_order_relaxed); }

template <class V>
void ObserveViews(ID3D11DeviceContext* c, Use use, const char* method, V* const* views, UINT n, void* caller) {
    if (t_internal || c != g_imm || !views) return;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (!g_inGroup) return;
        for (UINT i = 0; i < n && i < 128; ++i) {
            if (!views[i]) continue;
            ID3D11Resource* r = nullptr;
            views[i]->GetResource(&r);
            NoteLocked(r, use, method, caller);
            if (r) r->Release();
        }
    });
}

void ObserveRes(ID3D11DeviceContext* c, Use use, const char* method, ID3D11Resource* r, void* caller) {
    if (t_internal || c != g_imm || !r) return;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (g_inGroup) NoteLocked(r, use, method, caller);
    });
}

// A target bind while clones are in: the bound target for the viewport check.
void TrackBind(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
               void* caller) {
    if (!g_vpActive.load(std::memory_order_relaxed) || t_internal || c != g_imm) return;
    void* bound = nullptr;
    for (UINT i = 0; rtv && i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && !bound; ++i) bound = rtv[i];
    if (!bound) bound = dsv;
    Protect("bind", [&] {
        std::lock_guard lock(g_cm);
        OnBindLocked(bound, caller);
    });
}

// The size of a render target or depth view at its mip, and what its texture is to the clones.
bool TargetSize(void* v, uint32_t& w, uint32_t& h, char* what, size_t n) {
    ID3D11Resource* r = nullptr;
    static_cast<ID3D11View*>(v)->GetResource(&r);
    if (!r) return false;
    ID3D11Texture2D* t = nullptr;
    bool ok = SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t;
    if (ok) {
        const D3D11_TEXTURE2D_DESC d = DescOf(t);
        t->Release();
        UINT mip = 0;
        ID3D11RenderTargetView* rtv = nullptr;
        ID3D11DepthStencilView* dsv = nullptr;
        if (SUCCEEDED(static_cast<IUnknown*>(v)->QueryInterface(__uuidof(ID3D11RenderTargetView), (void**)&rtv)) && rtv) {
            D3D11_RENDER_TARGET_VIEW_DESC vd{};
            rtv->GetDesc(&vd);
            rtv->Release();
            mip = vd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D ? vd.Texture2D.MipSlice
                  : vd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY ? vd.Texture2DArray.MipSlice : 0;
        } else if (SUCCEEDED(static_cast<IUnknown*>(v)->QueryInterface(__uuidof(ID3D11DepthStencilView), (void**)&dsv)) &&
                   dsv) {
            D3D11_DEPTH_STENCIL_VIEW_DESC vd{};
            dsv->GetDesc(&vd);
            dsv->Release();
            mip = vd.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D ? vd.Texture2D.MipSlice
                  : vd.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DARRAY ? vd.Texture2DArray.MipSlice : 0;
        }
        w = MipSize(d.Width, mip);
        h = MipSize(d.Height, mip);
        const char* role = "unlisted";
        const Tex* x = nullptr;
        if (auto it = g_byClone.find(r); it != g_byClone.end()) {
            role = "clone of";
            x = &g_tex[it->second];
        } else if (auto it2 = g_byRes.find(r); it2 != g_byRes.end()) {
            x = &g_tex[it2->second];
            role = x->clone[g_cur] ? "ORIGINAL of cloned" : "shared";
        }
        snprintf(what, n, "%u x %u fmt %d x%u %s%s%s", w, h, (int)d.Format, d.SampleDesc.Count, role, x ? " " : "",
                 x ? TexName(*x) : "");
    }
    r->Release();
    return ok;
}

// In proof groups: render targets and depth bound together must have one size; a mix of clone and original sizes is
// refused by D3D11 (nothing is drawn). Counted, the first callers logged with every bound target.
void CheckTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                  const char* method, void* caller) {
    if (t_internal || c != g_imm) return;
    Protect(method, [&] {
        std::lock_guard lock(g_cm);
        if (!g_swapped) return;
        uint32_t w0 = 0, h0 = 0;
        bool first = true, mismatch = false;
        char desc[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT + 1][160] = {};
        UINT nd = 0;
        auto add = [&](void* v, const char* slot) {
            uint32_t w = 0, h = 0;
            char what[128];
            if (!v || !TargetSize(v, w, h, what, sizeof(what))) return;
            snprintf(desc[nd++], sizeof(desc[0]), "%s %s", slot, what);
            if (first) {
                w0 = w;
                h0 = h;
                first = false;
            } else if (w != w0 || h != h0) {
                mismatch = true;
            }
        };
        static const char* const slots[] = {"RTV0", "RTV1", "RTV2", "RTV3", "RTV4", "RTV5", "RTV6", "RTV7"};
        for (UINT i = 0; rtv && i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) add(rtv[i], slots[i]);
        add(dsv, "DSV");
        if (!mismatch) return;
        ++g_c.sizeMismatch;
        if (g_mismatchCallers.size() >= kMaxCallerLogs ||
            std::find(g_mismatchCallers.begin(), g_mismatchCallers.end(), caller) != g_mismatchCallers.end())
            return;
        g_mismatchCallers.push_back(caller);
        std::string line;
        for (UINT i = 0; i < nd; ++i) line += std::string(i ? "; " : "") + desc[i];
        Log("Clones: SIZE MISMATCH: a %s group binds targets of different sizes through %s from %s: %s", kSetNames[g_cur],
            method, Where(caller).c_str(), line.c_str());
    });
}

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

// Each detour first translates (bind fix), then observes what is really bound (learning and proof).
void SetSrv(SetSrvFn orig, const char* m, ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v,
            void* caller) {
    ID3D11ShaderResourceView* tmp[kMapMax];
    v = MapViews(c, v, n, tmp, m, caller);
    if (Watching()) ObserveViews(c, Read, m, v, n, caller);
    orig(c, s, n, v);
}
void STDMETHODCALLTYPE D_PSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    SetSrv(o_psSrv, "PSSetShaderResources", c, s, n, v, _ReturnAddress());
}
void STDMETHODCALLTYPE D_VSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    SetSrv(o_vsSrv, "VSSetShaderResources", c, s, n, v, _ReturnAddress());
}
void STDMETHODCALLTYPE D_CSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    SetSrv(o_csSrv, "CSSetShaderResources", c, s, n, v, _ReturnAddress());
}
void STDMETHODCALLTYPE D_GSSetSrv(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11ShaderResourceView* const* v) {
    SetSrv(o_gsSrv, "GSSetShaderResources", c, s, n, v, _ReturnAddress());
}
void STDMETHODCALLTYPE D_OMSetRtUav(ID3D11DeviceContext* c, UINT nr, ID3D11RenderTargetView* const* rtv,
                                    ID3D11DepthStencilView* dsv, UINT us, UINT nu, ID3D11UnorderedAccessView* const* uav,
                                    const UINT* counts) {
    void* caller = _ReturnAddress();
    const bool targets = nr != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL;
    const char* m = "OMSetRenderTargetsAndUnorderedAccessViews";
    ID3D11RenderTargetView* trtv[kMapMax];
    ID3D11UnorderedAccessView* tuav[kMapMax];
    if (targets) {
        rtv = MapViews(c, rtv, nr, trtv, m, caller);
        dsv = MapView(c, dsv, m, caller);
    }
    if (nu != D3D11_KEEP_UNORDERED_ACCESS_VIEWS) uav = MapViews(c, uav, nu, tuav, m, caller);
    if (Watching()) {
        if (targets) {
            ObserveViews(c, Write, m, rtv, nr, caller);
            ObserveViews(c, Write, m, &dsv, 1, caller);
        }
        if (nu != D3D11_KEEP_UNORDERED_ACCESS_VIEWS) ObserveViews(c, Write, m, uav, nu, caller);
    }
    o_omUav(c, nr, rtv, dsv, us, nu, uav, counts);
    if (targets) {
        TrackBind(c, nr, rtv, dsv, caller);
        if (Watching()) CheckTargets(c, nr, rtv, dsv, m, caller);
    }
}
void STDMETHODCALLTYPE D_CSSetUav(ID3D11DeviceContext* c, UINT s, UINT n, ID3D11UnorderedAccessView* const* uav,
                                  const UINT* counts) {
    void* caller = _ReturnAddress();
    ID3D11UnorderedAccessView* tmp[kMapMax];
    uav = MapViews(c, uav, n, tmp, "CSSetUnorderedAccessViews", caller);
    if (Watching()) ObserveViews(c, Write, "CSSetUnorderedAccessViews", uav, n, caller);
    o_csUav(c, s, n, uav, counts);
}
void STDMETHODCALLTYPE D_CopyRegion(ID3D11DeviceContext* c, ID3D11Resource* dst, UINT ds, UINT x, UINT y, UINT z,
                                    ID3D11Resource* src, UINT ss, const D3D11_BOX* box) {
    void* caller = _ReturnAddress();
    dst = MapRes(c, dst, "CopySubresourceRegion", caller);
    src = MapRes(c, src, "CopySubresourceRegion", caller);
    if (Watching()) {
        ObserveRes(c, Write, "CopySubresourceRegion", dst, caller);
        ObserveRes(c, Read, "CopySubresourceRegion", src, caller);
    }
    o_copyRegion(c, dst, ds, x, y, z, src, ss, box);
}
void STDMETHODCALLTYPE D_Copy(ID3D11DeviceContext* c, ID3D11Resource* dst, ID3D11Resource* src) {
    void* caller = _ReturnAddress();
    dst = MapRes(c, dst, "CopyResource", caller);
    src = MapRes(c, src, "CopyResource", caller);
    if (Watching()) {
        ObserveRes(c, Write, "CopyResource", dst, caller);
        ObserveRes(c, Read, "CopyResource", src, caller);
    }
    o_copy(c, dst, src);
}
void STDMETHODCALLTYPE D_Resolve(ID3D11DeviceContext* c, ID3D11Resource* dst, UINT ds, ID3D11Resource* src, UINT ss,
                                 DXGI_FORMAT f) {
    void* caller = _ReturnAddress();
    dst = MapRes(c, dst, "ResolveSubresource", caller);
    src = MapRes(c, src, "ResolveSubresource", caller);
    if (Watching()) {
        ObserveRes(c, Write, "ResolveSubresource", dst, caller);
        ObserveRes(c, Read, "ResolveSubresource", src, caller);
    }
    o_resolve(c, dst, ds, src, ss, f);
}
void STDMETHODCALLTYPE D_ClearRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]) {
    v = MapView(c, v, "ClearRenderTargetView", _ReturnAddress());
    if (Watching()) ObserveViews(c, Write, "ClearRenderTargetView", &v, 1, _ReturnAddress());
    o_clearRtv(c, v, col);
}
void STDMETHODCALLTYPE D_ClearDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT f, FLOAT d, UINT8 s) {
    v = MapView(c, v, "ClearDepthStencilView", _ReturnAddress());
    if (Watching()) ObserveViews(c, Write, "ClearDepthStencilView", &v, 1, _ReturnAddress());
    o_clearDsv(c, v, f, d, s);
}
void STDMETHODCALLTYPE D_ClearUavU(ID3D11DeviceContext* c, ID3D11UnorderedAccessView* v, const UINT val[4]) {
    v = MapView(c, v, "ClearUnorderedAccessViewUint", _ReturnAddress());
    if (Watching()) ObserveViews(c, Write, "ClearUnorderedAccessViewUint", &v, 1, _ReturnAddress());
    o_clearUavU(c, v, val);
}
void STDMETHODCALLTYPE D_ClearUavF(ID3D11DeviceContext* c, ID3D11UnorderedAccessView* v, const FLOAT val[4]) {
    v = MapView(c, v, "ClearUnorderedAccessViewFloat", _ReturnAddress());
    if (Watching()) ObserveViews(c, Write, "ClearUnorderedAccessViewFloat", &v, 1, _ReturnAddress());
    o_clearUavF(c, v, val);
}
void STDMETHODCALLTYPE D_Mips(ID3D11DeviceContext* c, ID3D11ShaderResourceView* v) {
    v = MapView(c, v, "GenerateMips", _ReturnAddress());
    if (Watching()) ObserveViews(c, Write, "GenerateMips", &v, 1, _ReturnAddress());
    o_mips(c, v);
}
// Fast path: one atomic load unless clones are in.
void STDMETHODCALLTYPE D_RSSetViewports(ID3D11DeviceContext* c, UINT n, const D3D11_VIEWPORT* vps) {
    if (!g_vpActive.load(std::memory_order_relaxed) || t_internal || c != g_imm) {
        o_rsVp(c, n, vps);
        return;
    }
    void* caller = _ReturnAddress();
    bool done = false;
    Protect("RSSetViewports", [&] {
        std::lock_guard lock(g_cm);
        if (!g_swapped) return;
        g_vpN = std::min(n, kMaxViewports);
        if (g_vpN && vps) memcpy(g_vpWant, vps, sizeof(D3D11_VIEWPORT) * g_vpN);
        ApplyViewportsLocked(caller, true);
        done = true;
    });
    if (!done) o_rsVp(c, n, vps);
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

void NoteDevice(void* dev) {
    if (!dev || dev == g_dev || g_failed) return;
    Protect("device", [&] {
        std::lock_guard lock(g_cm);
        if (g_dev && !g_devLogged) {
            g_devLogged = true;
            Log("Clones: engine device changed %p -> %p (logged once)", (void*)g_dev, dev);
        }
        g_dev = (uint8_t*)dev;
    });
}

void* __fastcall D_CreateTexture(void* dev, void* spec, void* ctx, void* d) {
    NoteDevice(dev);
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

void LogProofLocked(Set si) {
    const SetState& s = g_set[si];
    const uint64_t leaks = s.leakReads + s.leakWrites;
    char detail[160];
    size_t n = 0;
    for (int i = 0; i < kProofDetail; ++i)
        n += snprintf(detail + n, sizeof(detail) - n, "%s%llu", i ? " " : "", (unsigned long long)s.detail[i]);
    Log("Clones: proof for %s over %d groups: %llu binds and copies of clones (%llu of them by the bind fix), %llu "
        "reads and %llu writes of the originals of its clones (first %d groups: %s) (%s)",
        kSetNames[si], kProofGroups, (unsigned long long)s.cloneUses, (unsigned long long)s.fixed,
        (unsigned long long)s.leakReads, (unsigned long long)s.leakWrites, kProofDetail, detail,
        leaks ? "LEAKS: the engine reaches those originals other than through the swapped fields"
              : "0 expected, ok: its groups render into the clones");
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
            {44, (void*)D_RSSetViewports, (void**)&o_rsVp, "RSSetViewports"},
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
            // A hook may be in place already; it only observes while no clones are in, so it stays, but nothing is
            // learned or cloned.
            g_failed = true;
            g_phase = Failed;
        }
        Log("Clones: D3D11 hooks %s (15 context methods: observation in learning and proof groups, bind fix in eye "
            "groups and the bound target and viewport while clones are in)",
            ok ? "installed" : "INCOMPLETE; no clones");
    });
}

void SetSizes(uint32_t gameW, uint32_t gameH, uint32_t outerW, uint32_t outerH, uint32_t focusW, uint32_t focusH) {
    Protect("SetSizes", [&] {
        std::lock_guard lock(g_cm);
        const bool gameChanged = gameW != g_gameW || gameH != g_gameH;
        const uint32_t sizes[2][2] = {{outerW, outerH}, {focusW, focusH}};
        bool changed = gameChanged;
        for (int s = 0; s < 2; ++s) changed |= g_set[s].w != sizes[s][0] || g_set[s].h != sizes[s][1];
        if (!changed) return;
        DropClonesLocked();
        g_gameW = gameW;
        g_gameH = gameH;
        for (int s = 0; s < 2; ++s) {
            g_set[s].w = sizes[s][0];
            g_set[s].h = sizes[s][1];
            g_set[s].wanted = gameW && gameH && g_set[s].w && g_set[s].h && (g_set[s].w != gameW || g_set[s].h != gameH);
        }
        if (gameChanged) {
            // Listed under the old size: dropped (a new size comes with a new render target pool).
            g_tex.clear();
            g_rt.clear();
            g_byRes.clear();
        }
        g_dirty = true;
    });
}

void OnFrameStart(ID3D11DeviceContext* imm, bool enabled) {
    Protect("frame start", [&] {
        std::lock_guard lock(g_cm);
        ++g_frame;
        g_watch = false;
        g_live = LiveOff;
        g_inGroup = false;
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
            Log("Clones: %s", enabled ? "ON (views in their own size render into their own targets)"
                                      : "OFF (outer and focus share the game-size targets)");
            if (enabled && g_phase == Ready)
                for (SetState& s : g_set) StartProofLocked(s);
        }
        ProcessPendingLocked();
        if (g_gameW && g_d3dLo && (g_dirty || g_phase == Waiting || g_phase == Failed)) {
            const bool wasFailed = g_phase == Failed;
            if (g_dirty || !wasFailed) ResetLocked(g_phase == Waiting ? "first list" : "list changed");
            g_dirty = false;
        }
        if (g_phase == Learning && g_set[Outer].learned >= kLearnGroups && g_set[Focus].learned >= kLearnGroups &&
            g_enabled && imm)
            BuildLocked(imm);
    });
}

void BeginGroup(ID3D11DeviceContext* imm, Set set) {
    Protect("BeginGroup", [&] {
        std::lock_guard lock(g_cm);
        if (g_failed || g_swapped) return;
        g_inGroup = true;
        g_cur = set;
        SetState& s = g_set[set];
        if (g_phase == Learning) {
            g_watch = g_d3dHooked && s.learned < kLearnGroups;
            return;
        }
        // Clones listed but not in this group: the bind fix sends clones back to their originals.
        const bool fix = Cfg().clonesBindFix;
        if (g_phase == Ready && fix && !g_byClone.empty()) g_live = LiveBack;
        // A list that changed in this frame waits for the frame start (its entries may point to dropped clones).
        if (g_phase != Ready || !g_enabled || g_dirty || s.entries.empty()) return;
        for (const Entry& e : s.entries) {
            uint64_t cur = 0;
            if (!ReadField(e, cur) || cur != e.orig) {
                ++g_c.changed;
                g_dirty = true;
                if (!s.changedLogged) {
                    s.changedLogged = true;
                    Log("Clones: a swapped field of object %p (+0x%llX) changed outside the %s groups; no clones until "
                        "the list is watched again",
                        (void*)e.obj, (unsigned long long)(e.addr - e.obj), kSetNames[set]);
                }
                return;
            }
        }
        for (const Entry& e : s.entries) WriteField(e, e.clone);
        g_swapped = true;
        ++g_c.swaps[set];
        g_c.devSizes += MapDeviceSizes(g_gameW, g_gameH, s.w, s.h);
        g_c.postSizes += MapPostSize(g_gameW, g_gameH, s.w, s.h);
        g_bound = TranslateLocked(imm, s.fwd);
        // The viewport the engine set last, checked against the bound target.
        g_vpN = 0;
        if (imm) {
            imm->RSGetViewports(&g_vpN, nullptr);  // the number bound
            g_vpN = std::min(g_vpN, kMaxViewports);
            if (g_vpN) imm->RSGetViewports(&g_vpN, g_vpWant);
        }
        memcpy(g_vpSet, g_vpWant, sizeof(D3D11_VIEWPORT) * g_vpN);
        g_vpSetN = g_vpN;
        if (g_vpN) ApplyViewportsLocked(nullptr, false);
        g_vpActive = true;
        g_live = fix ? LiveSwap : LiveOff;
        g_watch = s.proofLeft > 0 && g_d3dHooked;
    });
}

void EndGroup(ID3D11DeviceContext* imm) {
    Protect("EndGroup", [&] {
        std::lock_guard lock(g_cm);
        g_watch = false;
        g_live = LiveOff;
        if (!g_inGroup) return;
        g_inGroup = false;
        SetState& s = g_set[g_cur];
        if (g_phase == Learning) ++s.learned;
        if (!g_swapped) return;
        RestoreLocked(imm);
        if (s.proofLeft > 0) {
            ++s.proofGroup;
            if (--s.proofLeft == 0) LogProofLocked(g_cur);
        }
    });
}

bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut, void* caller) {
    if (!Live(c) || n > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT) return false;
    const char* m = "OMSetRenderTargets";
    ID3D11RenderTargetView* const* r = MapViews(c, rtv, n, rtvOut, m, caller);
    if (r != rtvOut)
        for (UINT i = 0; i < n; ++i) rtvOut[i] = rtv ? rtv[i] : nullptr;
    *dsvOut = MapView(c, dsv, m, caller);
    return r == rtvOut || *dsvOut != dsv;
}

void OnOMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                          ID3D11DepthStencilView* dsv, void* caller) {
    if (Watching()) {
        ObserveViews(c, Write, "OMSetRenderTargets", rtv, n, caller);
        ObserveViews(c, Write, "OMSetRenderTargets", &dsv, 1, caller);
        CheckTargets(c, n, rtv, dsv, "OMSetRenderTargets", caller);
    }
    TrackBind(c, n, rtv, dsv, caller);
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

int ListForShot(Set set, ID3D11Resource** out, char (*labels)[64], int max) {
    int n = 0;
    Protect("ListForShot", [&] {
        std::lock_guard lock(g_cm);
        std::vector<const Tex*> list;
        for (auto& [obj, t] : g_tex)
            if (t.clone[set]) list.push_back(&t);
        std::sort(list.begin(), list.end(), [](const Tex* a, const Tex* b) {
            if (a->desc.Width != b->desc.Width) return a->desc.Width > b->desc.Width;
            if (a->desc.Format != b->desc.Format) return a->desc.Format < b->desc.Format;
            return a->obj < b->obj;
        });
        for (const Tex* t : list) {
            if (n >= max) break;
            const D3D11_TEXTURE2D_DESC d = DescOf(t->clone[set]);
            const bool r = t->read[set], w = t->written[set];
            out[n] = t->clone[set];
            snprintf(labels[n], 64, "%s_%02d_%ux%u_f%d%s_%s", kSetNames[set], n, d.Width, d.Height, (int)d.Format,
                     d.SampleDesc.Count > 1 ? "ms" : "", r && w ? "rw" : w ? "w" : "r");
            Log("Clones: shot %s = clone of %s (object %p, original %p %u x %u)", labels[n], TexName(*t), (void*)t->obj,
                (void*)t->res, t->desc.Width, t->desc.Height);
            ++n;
        }
    });
    return n;
}

std::string Describe() {
    char buf[320] = "clones: (error)";
    Protect("Describe", [&] {
        std::lock_guard lock(g_cm);
        int clones[2] = {};
        double mb[2] = {};
        for (auto& [obj, t] : g_tex)
            for (int s = 0; s < 2; ++s)
                if (t.clone[s]) {
                    ++clones[s];
                    mb[s] += Megabytes(DescOf(t.clone[s]));
                }
        char sets[2][96];
        for (int s = 0; s < 2; ++s) {
            const SetState& st = g_set[s];
            if (!st.wanted)
                snprintf(sets[s], sizeof(sets[s]), "%s at the game size", kSetNames[s]);
            else
                snprintf(sets[s], sizeof(sets[s]), "%s %u x %u: %d clones %.0f MB%s", kSetNames[s], st.w, st.h,
                         clones[s], mb[s], st.failed ? " (FAILED)" : "");
        }
        snprintf(buf, sizeof(buf), "clones %s (%s; %zu textures, %zu render targets listed; %s; %s)",
                 g_enabled ? "on" : "off", kPhaseNames[g_phase], g_tex.size(), g_rt.size(), sets[0], sets[1]);
    });
    return buf;
}

void LogStats(bool totals) {
    Protect("LogStats", [&] {
        std::lock_guard lock(g_cm);
        for (int s = 0; s < 2; ++s) g_ctot.swaps[s] += g_c.swaps[s];
        g_ctot.translated += g_c.translated;
        g_ctot.lateRestores += g_c.lateRestores;
        g_ctot.changed += g_c.changed;
        g_ctot.lists += g_c.lists;
        g_ctot.vpMismatch += g_c.vpMismatch;
        g_ctot.vpFixed += g_c.vpFixed;
        g_ctot.vpOutside += g_c.vpOutside;
        g_ctot.devSizes += g_c.devSizes;
        g_ctot.postSizes += g_c.postSizes;
        g_ctot.fixFwd += g_c.fixFwd;
        g_ctot.fixBack += g_c.fixBack;
        g_ctot.lazy += g_c.lazy;
        g_ctot.sizeMismatch += g_c.sizeMismatch;
        const Counters& c = totals ? g_ctot : g_c;
        if (c.swaps[0] || c.swaps[1] || c.lists || c.changed)
            Log("Clones stats%s: %s, groups with clones outer %llu focus %llu, %llu with bound views translated, %llu "
                "late restores, %llu lists watched, %llu changed fields, %llu device sizes and %llu postprocess sizes "
                "swapped; viewports for an original on its clone %llu (scaled %llu), larger than the original %llu; "
                "bind fix: originals to clones %llu (views made on clones %llu), clones to originals %llu; target size "
                "mismatches in proof groups %llu",
                totals ? " (session total)" : "", kPhaseNames[g_phase], (unsigned long long)c.swaps[0],
                (unsigned long long)c.swaps[1], (unsigned long long)c.translated, (unsigned long long)c.lateRestores,
                (unsigned long long)c.lists, (unsigned long long)c.changed, (unsigned long long)c.devSizes,
                (unsigned long long)c.postSizes,
                (unsigned long long)c.vpMismatch, (unsigned long long)c.vpFixed, (unsigned long long)c.vpOutside,
                (unsigned long long)c.fixFwd, (unsigned long long)c.lazy, (unsigned long long)c.fixBack,
                (unsigned long long)c.sizeMismatch);
        g_c = Counters{};
    });
}

void Shutdown() {
    Protect("Shutdown", [&] {
        std::lock_guard lock(g_cm);
        g_watch = false;
        g_inGroup = false;
        if (g_swapped) RestoreLocked(nullptr);
        DropClonesLocked();
        ReleaseGraveLocked();
        g_tex.clear();
        g_rt.clear();
        g_byRes.clear();
        g_pending.clear();
        g_gameW = g_gameH = 0;
        for (SetState& s : g_set) s = SetState{};
        g_enabled = false;
        g_dirty = false;
        if (!g_failed) g_phase = Waiting;
        g_c = g_ctot = Counters{};
        g_resetLogs = 0;
        g_callers.clear();
        g_fixCallers.clear();
        g_outsideCallers.clear();
        g_mismatchCallers.clear();
        g_vpN = g_vpSetN = 0;
        g_bound = nullptr;
    });
}

void TestRegister(bool renderTarget, void* obj) {
    std::lock_guard lock(g_cm);
    g_installed = true;
    g_pending[(uint8_t*)obj] = renderTarget;
}

void TestDestroy(bool renderTarget, void* obj) { NoteDestroyed(obj, renderTarget); }

void TestSetDevice(void* device) {
    std::lock_guard lock(g_cm);
    g_dev = (uint8_t*)device;
}

TestState TestGet() {
    std::lock_guard lock(g_cm);
    TestState s;
    s.phase = (int)g_phase;
    s.textures = (int)g_tex.size();
    s.renderTargets = (int)g_rt.size();
    for (auto& [obj, t] : g_tex)
        for (int i = 0; i < 2; ++i) s.clones[i] += t.clone[i] ? 1 : 0;
    for (const SetState& st : g_set) {
        s.entries += (int)st.entries.size();
        s.cloneUses += st.cloneUses;
        s.leaks += st.leakReads + st.leakWrites;
    }
    s.swapped = g_swapped;
    s.translated = g_c.translated + g_ctot.translated;
    s.viewportsFixed = g_c.vpFixed + g_ctot.vpFixed;
    s.viewportMismatches = g_c.vpMismatch + g_ctot.vpMismatch;
    s.bindsToClone = g_c.fixFwd + g_ctot.fixFwd;
    s.bindsToOriginal = g_c.fixBack + g_ctot.fixBack;
    s.lazyViews = g_c.lazy + g_ctot.lazy;
    return s;
}

}  // namespace qvfr::clones

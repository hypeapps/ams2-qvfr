#include "mvr_pass.h"

#include <d3d11_1.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "gpu.h"
#include "hooks.h"
#include "mvr_vs.h"

namespace qvfr::mvrpass {

namespace {

using Microsoft::WRL::ComPtr;

// Engine layout (docs/mvr.md B5), AMS2AVX.exe build aa43bd68.
constexpr uintptr_t kPhase1Vtable = 0x204adc8;   // CPhase1Render vtable (RVA)
constexpr uintptr_t kPhase1Execute = 0x9c96c0;   // its Execute at +0x48 (RVA)
constexpr int kExecuteSlot = 0x48 / 8;
constexpr int kVtSlots = 16;                     // vtable bytes checked at Install (slots)
constexpr int kOuterL = 0, kFocusL = 2;          // eye groups
constexpr int kMaxTargets = 4;                   // G-buffer: 3 colors and the depth
constexpr int kMaxCopies = 8;                    // two-slice copies per test
constexpr int kMaxSeen = 8;                      // targets of an observed pass
enum { ImgOuter, ImgView0, ImgView1, ImgFocus, kImgPerTarget };
const char* const kImgNames[kImgPerTarget] = {"outer", "view0", "view1", "focus"};

bool g_installed = false;
std::atomic<bool> g_on{false};
std::atomic<bool> g_dead{false};
uintptr_t g_phase1Vt = 0;  // absolute address of the vtable the test runs for
ID3D11DeviceContext* g_testCtx = nullptr;
uint32_t g_gen = 0;

ID3D11DeviceContext* Imm() { return g_testCtx ? g_testCtx : gpu::Context(); }

void Fail(const char* where, unsigned long code);

template <class F>
void Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

std::string NvErr(NvAPI_Status s) {
    NvAPI_ShortString msg = {};
    NvAPI_GetErrorMessage(s, msg);
    char buf[96];
    snprintf(buf, sizeof(buf), "%s (%d)", msg, (int)s);
    return buf;
}

// ---------------------------------------------------------------- test state (render thread)

struct Copy {
    ID3D11Resource* orig = nullptr;  // the game's texture (only compared)
    ComPtr<ID3D11Texture2D> arr;
    D3D11_TEXTURE2D_DESC desc{};
};
struct ViewMap {
    void* orig = nullptr;  // the game's view (only compared)
    ComPtr<ID3D11View> arr;
};
struct PlainVs {
    char sum[40];
    unsigned draws;
};

struct Test {
    bool armed = false;
    bool reported = false;
    bool ran = false;
    bool faulted = false;
    bool outerGrabbed = false, focusGrabbed = false;
    bool window = false, focusProjected = false, focusSeen = false;
    bool active = false;  // the engine's run is going through the copies now
    int base = -1;  // first shot image (-1: no images, smoke test)
    int targets = kMaxTargets;
    uint64_t frame = 0;
    uint32_t w = 0, h = 0, fw = 0, fh = 0;
    double sx = 1, ox = 0, sy = 1, oy = 0;
    float rows[8][4] = {};
    std::vector<Copy> copies;
    std::vector<ViewMap> views;
    ComPtr<ID3D11Buffer> cb;
    ComPtr<ID3D11DeviceContext1> c1;
    unsigned binds = 0, clears = 0, mapFailures = 0, sizeMismatches = 0, patchedDraws = 0, plainDraws = 0,
             cbSwaps = 0, viewsFailures = 0, copiedBack = 0, focusMatched = 0;
    std::vector<PlainVs> plain;
    double ms = 0;
    std::string why;  // reason the run did not happen
};
Test g_t;
uint64_t g_tests = 0, g_testsRan = 0;
uint64_t g_frame = 0;

// The latest observed size of the first target of outer L's and focus L's CPhase1Render.
uint32_t g_obsW[2] = {}, g_obsH[2] = {};

// Observed pass on the render thread: CPhase1Render of outer L (0) or focus L (1).
thread_local int t_obs = -1;
thread_local uint8_t* t_obsTask = nullptr;
thread_local bool t_extra = false;  // inside the test run (the engine's CPhase1Render of outer L with MVR)
std::vector<ComPtr<ID3D11Resource>> g_seen;

// One draw's saved state (BeforeDraw -> AfterDraw).
struct DrawSave {
    bool cb = false, views = false;
    ID3D11Buffer* buf = nullptr;  // referenced
    UINT first = 0, num = 0;
};
DrawSave g_draw;

char g_names[kMaxTargets][kImgPerTarget][32];
char g_labels[kMaxTargets][2][96];

bool SetViews(ID3D11DeviceContext* c, NvU32 views);

void Fail(const char* where, unsigned long code) {
    g_on = false;
    if (t_extra && Imm()) SetViews(Imm(), 1);
    t_extra = false;
    g_t.faulted = true;
    if (g_dead.exchange(true)) return;
    Log("MVR G-buffer test: ERROR: exception 0x%08lX in %s; off for this session", code, where);
}

bool SetViews(ID3D11DeviceContext* c, NvU32 views) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = views;
    for (NvU32 i = 0; i < 4; ++i) p.renderTargetIndexOffset[i] = i;
    const NvAPI_Status st = NvAPI_D3D_SetMultiViewMode(c, &p);
    if (st == NVAPI_OK) return true;
    if (!g_t.viewsFailures++)
        Log("MVR G-buffer test: NvAPI_D3D_SetMultiViewMode(%u views) %s", (unsigned)views, NvErr(st).c_str());
    return false;
}

void ReleaseTest() {
    if (g_draw.buf) g_draw.buf->Release();
    g_draw = DrawSave{};
    g_t = Test{};
}

// ---------------------------------------------------------------- copies of the targets

Copy* CopyFor(ID3D11DeviceContext* c, ID3D11Resource* res) {
    for (Copy& k : g_t.copies)
        if (k.orig == res) return &k;
    if (g_t.copies.size() >= kMaxCopies) return nullptr;
    ComPtr<ID3D11Texture2D> tex;
    if (!res || FAILED(res->QueryInterface(IID_PPV_ARGS(&tex)))) return nullptr;
    D3D11_TEXTURE2D_DESC d{};
    tex->GetDesc(&d);
    if (d.ArraySize != 1) return nullptr;
    if (d.Width != g_t.w || d.Height != g_t.h) ++g_t.sizeMismatches;
    Copy k;
    k.orig = res;
    k.desc = d;
    D3D11_TEXTURE2D_DESC a = d;
    a.ArraySize = 2;
    a.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE |
                     D3D11_RESOURCE_MISC_TEXTURECUBE);
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    if (!dev || FAILED(dev->CreateTexture2D(&a, nullptr, &k.arr))) return nullptr;
    g_t.copies.push_back(std::move(k));
    return &g_t.copies.back();
}

// The copy's view for a view of the game; a view of a copy itself (the pass read the bound views back) stays.
ID3D11View* FindView(void* v) {
    for (const ViewMap& m : g_t.views)
        if (m.orig == v || m.arr.Get() == v) return m.arr.Get();
    return nullptr;
}

// The copy's view on both slices for a view of the game, or null (the draw writes nothing; counted).
ID3D11RenderTargetView* ArrayRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v) {
    if (ID3D11View* m = FindView(v)) return static_cast<ID3D11RenderTargetView*>(m);
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    Copy* k = CopyFor(c, res.Get());
    D3D11_RENDER_TARGET_VIEW_DESC d{};
    v->GetDesc(&d);
    bool ok = k != nullptr;
    if (d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D) {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, 0, 2};
    } else if (d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, 2};
    } else {
        ok = false;
    }
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    ComPtr<ID3D11RenderTargetView> out;
    if (!ok || !dev || FAILED(dev->CreateRenderTargetView(k->arr.Get(), &d, &out))) {
        ++g_t.mapFailures;
        return nullptr;
    }
    g_t.views.push_back({v, out});
    return out.Get();
}

ID3D11DepthStencilView* ArrayDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v) {
    if (ID3D11View* m = FindView(v)) return static_cast<ID3D11DepthStencilView*>(m);
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    Copy* k = CopyFor(c, res.Get());
    D3D11_DEPTH_STENCIL_VIEW_DESC d{};
    v->GetDesc(&d);
    bool ok = k != nullptr;
    if (d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D) {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, 0, 2};
    } else if (d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, 2};
    } else {
        ok = false;
    }
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    ComPtr<ID3D11DepthStencilView> out;
    if (!ok || !dev || FAILED(dev->CreateDepthStencilView(k->arr.Get(), &d, &out))) {
        ++g_t.mapFailures;
        return nullptr;
    }
    g_t.views.push_back({v, out});
    return out.Get();
}

// ---------------------------------------------------------------- the test run

// View 1 = focus L in the top-left corner of slice 1 (a = fw / w, b = fh / h of the outer size). From outer L's clip
// space: focus x_f = sx x + ox w, y_f = sy y + oy w (NDC y up); corner x_c = a x_f + (a - 1) w, y_c = b y_f + (1 - b) w,
// so focus pixel (i, j) lands exactly on pixel (i, j) of the slice. The clip planes keep -w <= x_f, y_f <= w (the
// window; given in view 0's clip space, as the patch wants them).
void BuildRows() {
    Test& t = g_t;
    const double a = (double)t.fw / t.w, b = (double)t.fh / t.h;
    const float rows[8][4] = {
        {(float)(a * t.sx), 0, 0, (float)(a * t.ox + a - 1)},
        {0, (float)(b * t.sy), 0, (float)(b * t.oy + 1 - b)},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {(float)t.sx, 0, 0, (float)(t.ox + 1)},
        {(float)-t.sx, 0, 0, (float)(1 - t.ox)},
        {0, (float)t.sy, 0, (float)(t.oy + 1)},
        {0, (float)-t.sy, 0, (float)(1 - t.oy)},
    };
    memcpy(t.rows, rows, sizeof(rows));
}

bool MakeCb(ID3D11DeviceContext* c) {
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(g_t.rows);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA init{g_t.rows, 0, 0};
    return dev && SUCCEEDED(dev->CreateBuffer(&bd, &init, &g_t.cb));
}

// Before the engine's own Execute of outer L's CPhase1Render: its run goes through the copies with MVR. The task runs
// once: it replays the engine's software command buffers, and each replay releases one reference of its buffer
// (0x1620ad0), so a second run in the same frame uses a released buffer (the first B7 build crashed there).
void TestBegin() {
    Test& t = g_t;
    t.ran = true;
    ID3D11DeviceContext* c = Imm();
    if (!c) {
        t.why = "no D3D11 context";
        return;
    }
    if (!t.window) {
        t.why = "no focus window in this frame";
        return;
    }
    BuildRows();
    if (!MakeCb(c)) {
        t.why = "the constant buffer could not be made";
        return;
    }
    c->QueryInterface(IID_PPV_ARGS(&t.c1));
    if (!SetViews(c, 2)) {
        t.why = "Multi-View Rendering could not be turned on";
        return;
    }
    // The targets the previous tasks left (CBeginEye): bound again through the detour, so the pass starts on copies.
    ID3D11RenderTargetView* rt[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* ds = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, &ds);
    t.ms = NowMs();
    t_extra = true;
    t.active = true;
    bool bound = ds != nullptr;
    for (ID3D11RenderTargetView* v : rt) bound |= v != nullptr;
    if (bound) c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, ds);
    for (ID3D11RenderTargetView* v : rt)
        if (v) v->Release();
    if (ds) ds->Release();
}

// The game's view behind a bound view of a copy (null for null; a view that is not a copy's stays).
void* OrigView(ID3D11View* v) {
    if (!v) return nullptr;
    for (const ViewMap& m : g_t.views)
        if (m.arr.Get() == v) return m.orig;
    return v;
}

// After the engine's Execute: MVR off, the game's views bound instead of the copies' (as the pass left them), slice 0
// copied into the game's textures, so the frame goes on with what the pass would have drawn without MVR.
void TestFinish() {
    Test& t = g_t;
    ID3D11DeviceContext* c = Imm();
    t_extra = false;
    t.active = false;
    t.ms = NowMs() - t.ms;
    if (g_draw.buf) g_draw.buf->Release();
    g_draw = DrawSave{};
    if (!c) return;
    SetViews(c, 1);
    ID3D11RenderTargetView* rt[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* ds = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, &ds);
    ID3D11RenderTargetView* back[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    for (int i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
        back[i] = static_cast<ID3D11RenderTargetView*>(OrigView(rt[i]));
    ID3D11DepthStencilView* backDs = static_cast<ID3D11DepthStencilView*>(OrigView(ds));
    c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, back, backDs);
    for (ID3D11RenderTargetView* v : rt)
        if (v) v->Release();
    if (ds) ds->Release();
    for (const Copy& k : t.copies) {
        for (UINT m = 0; m < k.desc.MipLevels; ++m)
            c->CopySubresourceRegion(k.orig, D3D11CalcSubresource(m, 0, k.desc.MipLevels), 0, 0, 0, k.arr.Get(),
                                     D3D11CalcSubresource(m, 0, k.desc.MipLevels), nullptr);
        ++t.copiedBack;
    }
    ++g_testsRan;
    if (t.base < 0) return;
    for (int k = 0; k < t.targets && k < (int)t.copies.size(); ++k) {
        const int i = t.base + k * kImgPerTarget;
        shots::Grab(c, t.copies[k].orig, i + ImgOuter);
        shots::Grab(c, t.copies[k].arr.Get(), i + ImgView0, 0);
        shots::Grab(c, t.copies[k].arr.Get(), i + ImgView1, 1);
    }
    t.outerGrabbed = true;
}

void Report() {
    Test& t = g_t;
    if (!t.armed || t.reported) return;
    t.reported = true;
    if (!t.ran || !t.why.empty()) {
        Log("MVR G-buffer test (frame %llu): not run (%s)", (unsigned long long)t.frame,
            t.ran ? t.why.c_str() : "outer L's CPhase1Render did not run in the shot frame");
        return;
    }
    std::string tg;
    for (size_t k = 0; k < t.copies.size(); ++k) {
        const D3D11_TEXTURE2D_DESC& d = t.copies[k].desc;
        char b[96];
        snprintf(b, sizeof(b), "%s%zu: %ux%u fmt %d bind 0x%X", k ? ", " : "", k, d.Width, d.Height, (int)d.Format,
                 d.BindFlags);
        tg += b;
    }
    std::string pl;
    for (const PlainVs& p : t.plain) {
        char b[80];
        snprintf(b, sizeof(b), "%s%s x%u", pl.empty() ? "" : ", ", p.sum[0] ? p.sum : "(not seen by the patch)", p.draws);
        pl += b;
    }
    Log("MVR G-buffer test (frame %llu): outer L's CPhase1Render ran with MVR (2 views) into two-slice copies in "
        "%.2f ms, slice 0 copied back into %u game target(s); targets %zu (%s); binds %u, clears %u, map failures %u, "
        "size mismatches %u; draws with a patched VS %u (slot 3 held a game buffer in %u), with an unpatched VS %u "
        "(one view%s%s), SetMultiViewMode failures %u",
        (unsigned long long)t.frame, t.ms, t.copiedBack, t.copies.size(), tg.c_str(), t.binds, t.clears,
        t.mapFailures, t.sizeMismatches, t.patchedDraws, t.cbSwaps, t.plainDraws, pl.empty() ? "" : ": ", pl.c_str(),
        t.viewsFailures);
    Log("MVR G-buffer test: view 1 = focus L in the corner %u x %u of %u x %u; window sx %.6f ox %.6f sy %.6f oy %.6f; "
        "M rows [%.6f 0 0 %.6f] [0 %.6f 0 %.6f]; focus L %s, its targets %s (%u matched to a copy by format)",
        t.fw, t.fh, t.w, t.h, t.sx, t.ox, t.sy, t.oy, t.rows[0][0], t.rows[0][3], t.rows[1][1], t.rows[1][3],
        t.focusProjected ? "with the focus projection" : "WITHOUT the focus projection (no comparison)",
        t.focusSeen ? "grabbed" : "NOT seen", t.focusMatched);
}

// ---------------------------------------------------------------- instanced and indirect draws

using DrawIndexedInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DrawInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using DrawAutoFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*);
using DrawIndirectFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Buffer*, UINT);
DrawIndexedInstancedFn o_dii = nullptr;
DrawInstancedFn o_di = nullptr;
DrawAutoFn o_da = nullptr;
DrawIndirectFn o_diii = nullptr, o_dii2 = nullptr;
bool g_d3dHooked = false;

void STDMETHODCALLTYPE D_DrawIndexedInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, INT e, UINT f) {
    const bool m = BeforeDraw(c);
    o_dii(c, a, b, d, e, f);
    if (m) AfterDraw(c);
}
void STDMETHODCALLTYPE D_DrawInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, UINT e) {
    const bool m = BeforeDraw(c);
    o_di(c, a, b, d, e);
    if (m) AfterDraw(c);
}
void STDMETHODCALLTYPE D_DrawAuto(ID3D11DeviceContext* c) {
    const bool m = BeforeDraw(c);
    o_da(c);
    if (m) AfterDraw(c);
}
void STDMETHODCALLTYPE D_DrawIndexedInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    const bool m = BeforeDraw(c);
    o_diii(c, b, o);
    if (m) AfterDraw(c);
}
void STDMETHODCALLTYPE D_DrawInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    const bool m = BeforeDraw(c);
    o_dii2(c, b, o);
    if (m) AfterDraw(c);
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    Protect("Install", [&] {
        const uintptr_t base = hooks::ExeBase();
        uintptr_t exec = 0;
        if (!base || !hooks::Readable((void*)(base + kPhase1Vtable), kVtSlots * 8) ||
            !hooks::TryRead((void*)(base + kPhase1Vtable + kExecuteSlot * 8), &exec, 8) ||
            exec != base + kPhase1Execute) {
            Log("MVR G-buffer test: CPhase1Render vtable does not match (Execute at AMS2AVX.exe+0x%llX, expected "
                "+0x%llX); off",
                (unsigned long long)(exec ? exec - base : 0), (unsigned long long)kPhase1Execute);
            return;
        }
        g_phase1Vt = base + kPhase1Vtable;
        g_installed = true;
        Log("MVR G-buffer test: on; each shot runs outer L's CPhase1Render once more with MVR into two-slice copies of "
            "its targets (view 1 = focus L in the corner of slice 1) and compares them with outer L and focus L");
    });
    g_on = g_installed;
    return g_installed;
}

void InstallD3DHooks(ID3D11DeviceContext* imm) {
    if (!g_installed || g_d3dHooked || !imm) return;
    g_d3dHooked = true;
    Protect("InstallD3DHooks", [&] {
        void** vt = *(void***)imm;
        // ID3D11DeviceContext vtable slots (d3d11.h order).
        const bool a = hooks::CreateAt(vt[20], (void*)D_DrawIndexedInstanced, (void**)&o_dii, "DrawIndexedInstanced");
        const bool b = hooks::CreateAt(vt[21], (void*)D_DrawInstanced, (void**)&o_di, "DrawInstanced");
        const bool d = hooks::CreateAt(vt[38], (void*)D_DrawAuto, (void**)&o_da, "DrawAuto");
        const bool e = hooks::CreateAt(vt[39], (void*)D_DrawIndexedInstancedIndirect, (void**)&o_diii,
                                       "DrawIndexedInstancedIndirect");
        const bool f = hooks::CreateAt(vt[40], (void*)D_DrawInstancedIndirect, (void**)&o_dii2, "DrawInstancedIndirect");
        Log("MVR G-buffer test: D3D11 draw hooks: DrawIndexedInstanced %s, DrawInstanced %s, DrawAuto %s, indirect %s/%s "
            "(Draw and DrawIndexed through the sun effects hooks)",
            a ? "ok" : "FAILED", b ? "ok" : "FAILED", d ? "ok" : "FAILED", e ? "ok" : "FAILED", f ? "ok" : "FAILED");
    });
}

bool On() { return g_on.load(std::memory_order_relaxed); }

void OnFrameStart() {
    if (!g_installed) return;
    ++g_frame;
    Protect("OnFrameStart", [&] {
        const uint32_t gen = gpu::Generation();
        if (gen != g_gen) {
            g_gen = gen;
            g_seen.clear();
            ReleaseTest();
        }
        if (g_t.armed) {
            Report();
            ReleaseTest();
        }
    });
}

int PrepareShot(const char** names, int base, int max, shots::Pair* pairs, int maxPairs, int* pairCount) {
    *pairCount = 0;
    if (!On()) return 0;
    int n = 0;
    Protect("PrepareShot", [&] {
        if (!g_obsW[0] || !g_obsW[1]) {
            Log("MVR G-buffer test: not in this shot (outer L's or focus L's CPhase1Render not seen yet)");
            return;
        }
        const int targets = std::min({kMaxTargets, max / kImgPerTarget, maxPairs / 2});
        if (targets < 1) {
            Log("MVR G-buffer test: not in this shot (no room for its images)");
            return;
        }
        ReleaseTest();
        Test& t = g_t;
        t.armed = true;
        t.base = base;
        t.targets = targets;
        t.frame = g_frame;
        t.w = g_obsW[0];
        t.h = g_obsH[0];
        t.fw = g_obsW[1];
        t.fh = g_obsH[1];
        ++g_tests;
        for (int k = 0; k < targets; ++k) {
            for (int j = 0; j < kImgPerTarget; ++j) {
                snprintf(g_names[k][j], sizeof(g_names[k][j]), "mvr_t%d_%s", k, kImgNames[j]);
                names[base + k * kImgPerTarget + j] = g_names[k][j];
            }
            snprintf(g_labels[k][0], sizeof(g_labels[k][0]), "MVR target %d: view 0 (slice 0) vs outer L", k);
            snprintf(g_labels[k][1], sizeof(g_labels[k][1]), "MVR target %d: focus L vs view 1 (corner of slice 1)", k);
            const int i = base + k * kImgPerTarget;
            pairs[2 * k] = {i + ImgView0, i + ImgOuter, g_labels[k][0]};
            pairs[2 * k + 1] = {i + ImgFocus, i + ImgView1, g_labels[k][1], 0, 0, (float)t.fw, (float)t.fh, 0};
        }
        *pairCount = 2 * targets;
        n = targets * kImgPerTarget;
    });
    return n;
}

void CancelShot() {
    if (g_t.armed) Protect("CancelShot", [&] { ReleaseTest(); });
}

void SetWindow(bool valid, double sx, double ox, double sy, double oy) {
    if (!g_t.armed || g_t.ran) return;
    g_t.window = valid;
    g_t.sx = sx;
    g_t.ox = ox;
    g_t.sy = sy;
    g_t.oy = oy;
}

void FocusProjected(bool projected) {
    if (g_t.armed) g_t.focusProjected = projected;
}

void BeforeTask(uint8_t* task, int group) {
    if (!On() || !task || (group != kOuterL && group != kFocusL)) return;
    uintptr_t vt = 0;
    if (!hooks::TryRead(task, &vt, 8) || vt != g_phase1Vt) return;
    Protect("BeforeTask", [&] {
        t_obs = group == kOuterL ? 0 : 1;
        t_obsTask = task;
        g_seen.clear();
        if (group == kOuterL && g_t.armed && !g_t.ran) TestBegin();
    });
}

void AfterTask(uint8_t* task) {
    if (t_obs < 0 || task != t_obsTask) return;
    const int o = t_obs;
    t_obs = -1;
    t_obsTask = nullptr;
    Protect("AfterTask", [&] {
        if (o == 0 && g_t.active) TestFinish();
        if (!g_seen.empty()) {
            ComPtr<ID3D11Texture2D> tex;
            if (SUCCEEDED(g_seen[0]->QueryInterface(IID_PPV_ARGS(&tex)))) {
                D3D11_TEXTURE2D_DESC d{};
                tex->GetDesc(&d);
                g_obsW[o] = d.Width;
                g_obsH[o] = d.Height;
            }
        }
        Test& t = g_t;
        ID3D11DeviceContext* c = Imm();
        if (o == 1 && t.armed && t.ran && t.why.empty() && c && !t.focusGrabbed) {
            t.focusGrabbed = true;
            t.focusSeen = !g_seen.empty();
            // Focus L binds its targets in another order than outer L's copies were made (outer's start with the depth
            // CBeginEye left bound): each goes to the copy with its format and role (depth or color).
            bool used[kMaxCopies] = {};
            for (const auto& r : g_seen) {
                ComPtr<ID3D11Texture2D> tex;
                if (t.base < 0 || FAILED(r->QueryInterface(IID_PPV_ARGS(&tex)))) continue;
                D3D11_TEXTURE2D_DESC d{};
                tex->GetDesc(&d);
                for (int k = 0; k < t.targets && k < (int)t.copies.size(); ++k) {
                    const D3D11_TEXTURE2D_DESC& cd = t.copies[k].desc;
                    if (used[k] || cd.Format != d.Format ||
                        ((cd.BindFlags ^ d.BindFlags) & D3D11_BIND_DEPTH_STENCIL))
                        continue;
                    used[k] = true;
                    ++t.focusMatched;
                    shots::Grab(c, r.Get(), t.base + k * kImgPerTarget + ImgFocus);
                    break;
                }
            }
            Report();
        }
        g_seen.clear();
    });
}

bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut) {
    if (!t_extra || c != Imm()) return false;
    bool ok = false;
    Protect("MapTargets", [&] {
        ++g_t.binds;
        for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            rtvOut[i] = rtv && rtv[i] ? ArrayRtv(c, rtv[i]) : nullptr;
        *dsvOut = dsv ? ArrayDsv(c, dsv) : nullptr;
        ok = true;
    });
    return ok;
}

ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v) {
    if (!t_extra || !v || c != Imm()) return v;
    ID3D11RenderTargetView* out = v;
    Protect("ClearRenderTargetView", [&] {
        ++g_t.clears;
        if (ID3D11RenderTargetView* m = ArrayRtv(c, v)) out = m;  // else the game's (a clear, the engine's run follows)
    });
    return out;
}

ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v) {
    if (!t_extra || !v || c != Imm()) return v;
    ID3D11DepthStencilView* out = v;
    Protect("ClearDepthStencilView", [&] {
        ++g_t.clears;
        if (ID3D11DepthStencilView* m = ArrayDsv(c, v)) out = m;
    });
    return out;
}

void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    if (t_obs < 0 || t_extra || c != Imm()) return;
    Protect("OnSetTargets", [&] {
        auto add = [](ID3D11View* v) {
            if (!v || g_seen.size() >= kMaxSeen) return;
            ComPtr<ID3D11Resource> res;
            v->GetResource(&res);
            for (const auto& s : g_seen)
                if (s.Get() == res.Get()) return;
            g_seen.push_back(res);
        };
        for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) add(rtv ? rtv[i] : nullptr);
        add(dsv);
    });
}

bool BeforeDraw(ID3D11DeviceContext* c) {
    if (!t_extra || c != Imm()) return false;
    bool after = false;
    Protect("BeforeDraw", [&] {
        Test& t = g_t;
        ID3D11VertexShader* vs = nullptr;
        c->VSGetShader(&vs, nullptr, nullptr);
        char sum[40];
        const bool patched = mvrvs::IsPatched(vs, sum);
        if (vs) vs->Release();
        DrawSave& s = g_draw;
        if (patched) {
            ++t.patchedDraws;
            if (t.c1)
                t.c1->VSGetConstantBuffers1(mvrvs::kCbSlot, 1, &s.buf, &s.first, &s.num);
            else
                c->VSGetConstantBuffers(mvrvs::kCbSlot, 1, &s.buf);
            if (s.buf == t.cb.Get()) {
                s.buf->Release();
                s.buf = nullptr;
                return;
            }
            if (s.buf) ++t.cbSwaps;
            ID3D11Buffer* cb = t.cb.Get();
            c->VSSetConstantBuffers(mvrvs::kCbSlot, 1, &cb);
            s.cb = true;
        } else {
            ++t.plainDraws;
            auto it = std::find_if(t.plain.begin(), t.plain.end(),
                                   [&](const PlainVs& p) { return !strcmp(p.sum, sum); });
            if (it != t.plain.end())
                ++it->draws;
            else if (t.plain.size() < 8) {
                PlainVs p{};
                memcpy(p.sum, sum, sizeof(p.sum));
                p.draws = 1;
                t.plain.push_back(p);
            }
            s.views = SetViews(c, 1);
        }
        after = s.cb || s.views;
    });
    return after;
}

void AfterDraw(ID3D11DeviceContext* c) {
    Protect("AfterDraw", [&] {
        DrawSave& s = g_draw;
        if (s.cb) {
            if (g_t.c1 && s.buf)
                g_t.c1->VSSetConstantBuffers1(mvrvs::kCbSlot, 1, &s.buf, &s.first, &s.num);
            else
                c->VSSetConstantBuffers(mvrvs::kCbSlot, 1, &s.buf);
        }
        if (s.views) SetViews(c, 2);
        if (s.buf) s.buf->Release();
        s = DrawSave{};
    });
}

std::string Describe() {
    if (!g_installed) return "MVR G-buffer test off";
    char buf[96];
    snprintf(buf, sizeof(buf), "MVR G-buffer test %s (%llu of %llu run)", g_dead ? "STOPPED" : "on",
             (unsigned long long)g_testsRan, (unsigned long long)g_tests);
    return buf;
}

void LogStats(bool totals) {
    if (!g_installed || !totals) return;
    Log("MVR G-buffer test (session): %llu shots with the test, %llu runs", (unsigned long long)g_tests,
        (unsigned long long)g_testsRan);
}

void Shutdown() {
    if (!g_installed) return;
    Protect("Shutdown", [&] {
        g_seen.clear();
        ReleaseTest();
    });
}

void TestSetup(ID3D11DeviceContext* ctx, const void* taskVtable) {
    g_testCtx = ctx;
    g_phase1Vt = (uintptr_t)taskVtable;
    g_installed = true;
    g_on = true;
    g_obsW[0] = g_obsW[1] = 0;
}

void TestArm(uint32_t w, uint32_t h, uint32_t fw, uint32_t fh, double sx, double ox, double sy, double oy) {
    ReleaseTest();
    g_t.armed = true;
    g_t.base = -1;
    g_t.w = w;
    g_t.h = h;
    g_t.fw = fw;
    g_t.fh = fh;
    g_t.focusProjected = true;
    ++g_tests;
    SetWindow(true, sx, ox, sy, oy);
}

ID3D11Texture2D* TestArray(int k) {
    return k >= 0 && k < (int)g_t.copies.size() ? g_t.copies[k].arr.Get() : nullptr;
}

TestState TestGet() {
    TestState s;
    s.targets = (unsigned)g_t.copies.size();
    s.binds = g_t.binds;
    s.clears = g_t.clears;
    s.mapFailures = g_t.mapFailures;
    s.patchedDraws = g_t.patchedDraws;
    s.plainDraws = g_t.plainDraws;
    s.cbSwaps = g_t.cbSwaps;
    s.copiedBack = g_t.copiedBack;
    s.viewsFailures = g_t.viewsFailures;
    s.ran = g_t.ran && g_t.why.empty();
    s.faulted = g_t.faulted;
    return s;
}

void TestEnd() {
    Report();
    ReleaseTest();
}

}  // namespace qvfr::mvrpass

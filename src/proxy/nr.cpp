#include "nr.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "common.h"
#include "gpu.h"
#include "nr_bridge.h"

namespace qvfr::nr {

namespace {

// The images, one bridge slot each: 0/1 focus L/R, 2/3 outer L/R (outer only in mode=final).
constexpr int kImages = 4;
constexpr const char* kName[kImages] = {"focus L", "focus R", "outer L", "outer R"};
constexpr int kMotionCache = 16;  // motion vectors and depth of up to four views, twice

// dlaa's view index (1 outer L, 2 outer R, 3 focus L, 4 focus R) to the image, -1 for the others.
int ImageOf(int view) {
    switch (view) {
        case 3: return 0;
        case 4: return 1;
        case 1: return 2;
        case 2: return 3;
        default: return -1;
    }
}

nrbridge::Bridge* g_bridge = nullptr;
nrbridge::Settings g_set;  // what the bridge was configured with
uint32_t g_gen = 0;
bool g_failed = false;  // the rest of the session: DLAA only
std::string g_why;
std::atomic<bool> g_active{true};
std::atomic<bool> g_resetPending[kImages] = {};  // NR turned on again: the model's history starts over
bool g_keyDown = false;
bool g_faultLogged = false;
uint64_t g_calls[kImages] = {}, g_lastCalls[kImages] = {};
bool g_firstLogged[kImages] = {};
nrbridge::Stats g_last{};
Window g_window;  // the last statistics window (LastWindow)

// SRVs of the motion vector and depth textures DLAA hands over (kept with a reference on the texture).
struct InputSrv {
    ID3D11Resource* res = nullptr;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    ID3D11ShaderResourceView* srv = nullptr;
};
InputSrv g_motion[kMotionCache];
int g_motionNext = 0;
bool g_motionLogged[2] = {};

// The built-in test sequence (dlaa-plan D7, test 13: the outer images with the focus or in their own evaluation):
// [nr] modifier + test_key goes to the next step, on top of the ini. Step 0 is the ini as it is. `mode` 0 keeps the
// ini's mode, 1 after DLAA (focus), 2 the final image; `outer`, `batch` and `separate` -1 keep the ini's [nr] outer /
// batch / outer_atlas, 0 off, 1 on; `outerStep` -1 keeps the ini's [nr] outer_resolution, else the outer images go
// into the atlas at 1/outerStep of their size per axis; `smoothen` >= 0 replaces [quad] smoothen_focus_view_edges at
// the composition. Earlier sequences (tests 5 to 12) are in dlaa-plan D7.
struct TestStep {
    const char* name;
    int mode;
    int outer;
    int batch;
    int outerStep;
    int separate;
    float smoothen;
    void (*apply)(nrbridge::Settings&);
};
const TestStep kSteps[] = {
    {"the ini as it is", 0, -1, -1, -1, -1, -1.0f, [](nrbridge::Settings&) {}},
    {"the outer images in their own evaluation (outer_atlas=separate)", 0, 1, 1, -1, 1, -1.0f,
     [](nrbridge::Settings&) {}},
    {"the outer images with the focus (outer_atlas=shared)", 0, 1, 1, -1, 0, -1.0f, [](nrbridge::Settings&) {}},
    {"focus only (outer=0)", 0, 0, 1, -1, -1, -1.0f, [](nrbridge::Settings&) {}},
};
constexpr int kStepCount = (int)(sizeof(kSteps) / sizeof(kSteps[0]));
std::atomic<int> g_step{0};
bool g_testKeyDown = false;

// The mode now: the ini's (0 off, 1 after DLAA, 2 the final image), or the test step's while the ini has NR on.
int ModeNow() {
    const int ini = Cfg().nrMode;
    const int step = kSteps[g_step.load()].mode;
    return ini && step ? step : ini;
}

// Benchmark sequence switches (quad.cpp, Knob): -1 keeps the ini's value; a test step still wins over them.
std::atomic<int> g_knob[(int)Knob::Count] = {-1, -1, -1, -1, -1, -1, -1, -1};

int KnobOr(Knob k, int ini) {
    const int v = g_knob[(int)k].load();
    return v >= 0 ? v : ini;
}

// mode=final: the outer images through the model too (the ini's [nr] outer, a benchmark switch or the test step's).
bool OuterNow() {
    const int step = kSteps[g_step.load()].outer;
    return ModeNow() == 2 && (step >= 0 ? step != 0 : KnobOr(Knob::Outer, Cfg().nrOuter) != 0);
}

// mode=final: all images of a frame in one atlas and one evaluation of the model (the ini's [nr] batch, a benchmark
// switch or the test step's). The model has a fixed cost per evaluation of about 1 ms besides its cost per pixel (test
// 7).
bool BatchNow() {
    const int step = kSteps[g_step.load()].batch;
    return ModeNow() == 2 && (step >= 0 ? step != 0 : KnobOr(Knob::Batch, Cfg().nrBatch) != 0);
}

// The atlas's source pixels per outer pixel in each axis (1 full, 2 half, ...; [nr] outer_resolution, a benchmark
// switch or the test step's). The model's cost follows its pixels (about 0.5 ms per MP, test 9); in the periphery its
// tone and color matter, not its finest detail.
UINT OuterStepNow() {
    const int step = kSteps[g_step.load()].outerStep;
    return step > 0 ? (UINT)step : (UINT)KnobOr(Knob::OuterRes, Cfg().nrOuterStep);
}

// The centre part of the outer images that goes through the model, percent of each axis ([nr] outer_area or a
// benchmark switch); outside it the model's edit fades out (the far periphery keeps the game's image).
UINT OuterAreaNow() { return (UINT)KnobOr(Knob::OuterArea, Cfg().nrOuterAreaPct); }

// The outer images in an atlas of their own and a second evaluation ([nr] outer_atlas=separate, a benchmark switch or
// the test step's): the model judges the light of the whole image it gets, so the outer images in the focus images'
// atlas change the focus's result (test 12: a lighter, flatter cockpit); separate, the focus looks as without the
// outer, for about 1 ms more.
bool SeparateNow() {
    const int step = kSteps[g_step.load()].separate;
    return step >= 0 ? step != 0 : KnobOr(Knob::OuterSeparate, Cfg().nrOuterSeparate) != 0;
}

// The atlases (batch=1). Shared: focus L and R in the top row, outer L and R (with outer on: their centre part at
// 1/step of its size) below or right of them. Separate: the focus images in atlas 0, the outer images side by side in
// atlas 1. Images kAtlasGap pixels apart. The motion vectors and depth go into the bridge's slot kAtlasSlot + atlas
// region by region at DLAA time; at the left eye's composition each image goes straight into its region (downsampled
// with a step), each atlas through the model once, and each image straight back into its copy (or the model's edit
// enlarged onto it). The atlases are the bridge's shared textures: no copy of them in D3D11.
constexpr int kAtlases = 2;
constexpr int kAtlasSlot = 4;
constexpr UINT kAtlasGap = 16;
struct Region {
    UINT w = 0, h = 0;    // the image's size (DLAA's)
    UINT step = 1;        // source pixels per atlas pixel in each axis
    UINT cx = 0, cy = 0;  // the part of the image in the atlas: origin and size in source pixels
    UINT cw = 0, ch = 0;
    UINT fade = 0;        // with a centre part: the edit's fade-out band inside its edge, source pixels
    int atlas = 0;        // which atlas it is in
    UINT aw = 0, ah = 0;  // its size in the atlas
    UINT x = 0, y = 0;    // its place in the atlas
};
Region g_region[kImages];
UINT g_atlasW[kAtlases] = {}, g_atlasH[kAtlases] = {};  // 0: not used, or not all sizes known yet
int g_atlasCount = 0;             // atlases in the layout (0: not all sizes known yet)
bool g_layoutOuter = false;
bool g_layoutSeparate = false;    // the outer images in atlas 1
UINT g_layoutStep = 1;            // the outer images' step in the layout
UINT g_layoutArea = 100;          // and their centre part, percent
uint32_t g_layoutGen = 1;
uint32_t g_guided[kImages] = {};  // the layout generation of the region's guides written since the last atlas run
uint32_t g_atlasRunGen[kAtlases] = {};  // the layout of the atlas's last run (another one: the history starts over)
bool g_atlasReset[kAtlases] = {};       // DLAA reset a history of its images since its last run
bool g_atlasOut = false;          // this frame's atlases ran: the eyes compose from g_final
bool g_atlasLogged = false, g_atlasSkipLogged = false;
double g_atlasMp = 0;             // the atlases' size together, for the benchmark summary

void Relayout() {
    ++g_layoutGen;
    g_atlasCount = 0;
    for (int a = 0; a < kAtlases; ++a) g_atlasW[a] = g_atlasH[a] = 0;
    const int n = g_layoutOuter ? 4 : 2;
    for (int i = 0; i < n; ++i)
        if (!g_region[i].w) return;
    Region* r = g_region;
    const bool separate = g_layoutOuter && g_layoutSeparate;
    for (int i = 0; i < n; ++i) {
        const bool outer = i >= 2;
        r[i].step = outer ? g_layoutStep : 1;
        const UINT area = outer ? g_layoutArea : 100;
        r[i].cw = (r[i].w * area + 50) / 100;
        r[i].ch = (r[i].h * area + 50) / 100;
        r[i].cx = (r[i].w - r[i].cw) / 2;
        r[i].cy = (r[i].h - r[i].ch) / 2;
        r[i].fade = area < 100 ? (std::max)(16u, (std::min)(r[i].cw, r[i].ch) / 10) : 0;
        r[i].atlas = outer && separate ? 1 : 0;
        r[i].aw = (r[i].cw + r[i].step - 1) / r[i].step;
        r[i].ah = (r[i].ch + r[i].step - 1) / r[i].step;
    }
    r[0].x = 0;
    r[0].y = 0;
    r[1].x = r[0].aw + kAtlasGap;
    r[1].y = 0;
    UINT w = r[1].x + r[1].aw, h = (std::max)(r[0].ah, r[1].ah);
    bool column = false;
    if (separate) {  // the outer images side by side in atlas 1
        r[2].x = 0;
        r[2].y = r[3].y = 0;
        r[3].x = r[2].aw + kAtlasGap;
        g_atlasW[1] = r[3].x + r[3].aw;
        g_atlasH[1] = (std::max)(r[2].ah, r[3].ah);
    } else if (g_layoutOuter) {
        // The outer images in a row below the focus images, or (smaller ones) one above the other right of them:
        // whichever atlas is smaller, since the model pays for the empty pixels too.
        const UINT rowW = (std::max)(w, r[2].aw + kAtlasGap + r[3].aw);
        const UINT rowH = h + kAtlasGap + (std::max)(r[2].ah, r[3].ah);
        const UINT colW = w + kAtlasGap + (std::max)(r[2].aw, r[3].aw);
        const UINT colH = (std::max)(h, r[2].ah + kAtlasGap + r[3].ah);
        column = (uint64_t)colW * colH < (uint64_t)rowW * rowH;
        if (column) {
            r[2].x = r[3].x = w + kAtlasGap;
            r[2].y = 0;
            r[3].y = r[2].ah + kAtlasGap;
            w = colW;
            h = colH;
        } else {
            r[2].x = 0;
            r[2].y = r[3].y = h + kAtlasGap;
            r[3].x = r[2].aw + kAtlasGap;
            w = rowW;
            h = rowH;
        }
    }
    g_atlasW[0] = w;
    g_atlasH[0] = h;
    g_atlasCount = separate ? 2 : 1;
    g_atlasMp = ((double)g_atlasW[0] * g_atlasH[0] + (double)g_atlasW[1] * g_atlasH[1]) / 1e6;
    Log("NR: atlas %ux%u (%.1f MP) for one evaluation: focus %ux%u + %ux%u%s", w, h, (double)w * h / 1e6, r[0].aw,
        r[0].ah, r[1].aw, r[1].ah,
        !g_layoutOuter ? "" : separate ? "; the outer images in a second atlas" : column ? ", outer right of them"
                                                                                  : ", outer below");
    if (separate)
        Log("NR: atlas 2: %ux%u (%.1f MP) for a second evaluation: the outer images side by side", g_atlasW[1],
            g_atlasH[1], (double)g_atlasW[1] * g_atlasH[1] / 1e6);
    if (g_layoutOuter)
        Log("NR: atlas: outer %ux%u + %ux%u: the centre %u%% (%ux%u) of %ux%u at 1/%u, the edit fading out over %u px",
            r[2].aw, r[2].ah, r[3].aw, r[3].ah, g_layoutArea, r[2].cw, r[2].ch, r[2].w, r[2].h, g_layoutStep,
            r[2].fade);
}

// mode=final: per image, a copy of the finished image the model writes into (compose reads it instead).
struct FinalImage {
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    D3D11_TEXTURE2D_DESC desc{};  // of the game's image it copies
};
FinalImage g_final[kImages];
// batch=1: the copy of an image the bridge cannot read in place (no SRV binding).
FinalImage g_orig[kImages];
// batch=1: the bridge's view of an image, while it is the same image.
struct View {
    ID3D11Texture2D* tex = nullptr;  // not a reference of its own (the view holds one)
    ID3D11ShaderResourceView* srv = nullptr;
};
View g_view[kImages];
bool g_viewLogged = false;
bool g_finalReset[kImages] = {};  // DLAA reset its history since the last final image
bool g_finalLogged[kImages] = {};
bool g_finalSkipLogged[kImages] = {};

void DropImage(FinalImage& f) {
    if (f.uav) f.uav->Release();
    if (f.srv) f.srv->Release();
    if (f.tex) f.tex->Release();
    f = {};
}

void DropView(View& v) {
    if (v.srv) v.srv->Release();
    v = {};
}

void DropFinal() {
    for (FinalImage& f : g_final) DropImage(f);
    for (FinalImage& f : g_orig) DropImage(f);
    for (View& v : g_view) DropView(v);
    for (auto& g : g_guided) g = 0;
    g_atlasOut = false;
}

// The copy's texture format (the image's own: compose wants outer and focus in one format) and the view format the
// bridge reads and writes (the stored, display-encoded values, as the model wants them). False: not usable (an _SRGB
// image cannot have an UNORM unordered access view).
bool FinalFormats(DXGI_FORMAT f, DXGI_FORMAT& tex, DXGI_FORMAT& view) {
    tex = f;
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            view = DXGI_FORMAT_R8G8B8A8_UNORM;
            return true;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            view = DXGI_FORMAT_R10G10B10A2_UNORM;
            return true;
        default:
            return false;
    }
}

// The step by ear: one long low tone per 5, one short high tone per 1 more; step 0 one very low tone.
void StepBeeps(int step) {
    std::thread([step] {
        if (step == 0) {
            Beep(330, 500);
            return;
        }
        for (int i = 0; i < step / 5; ++i) {
            Beep(523, 350);
            Sleep(120);
        }
        for (int i = 0; i < step % 5; ++i) {
            Beep(1047, 110);
            Sleep(120);
        }
    }).detach();
}

bool Pressed(int mod, int key, bool& wasDown) {
    const bool down = key && (GetAsyncKeyState(key) & 0x8000) && (!mod || (GetAsyncKeyState(mod) & 0x8000));
    const bool pressed = down && !wasDown;
    wasDown = down;
    return pressed;
}

void DropMotion() {
    for (InputSrv& m : g_motion) {
        if (m.srv) m.srv->Release();
        if (m.res) m.res->Release();
        m = {};
    }
    g_motionNext = 0;
}

void Drop() {
    delete g_bridge;
    g_bridge = nullptr;
    g_last = {};  // the next bridge's statistics start at zero
    DropMotion();
    DropFinal();
}

void TurnOff(const std::string& why) {
    g_failed = true;
    g_why = why;
    Log("NR: off for the rest of the session (%s); the focus views keep the DLAA result", why.c_str());
}

// An SRV in `fmt` of a texture DLAA used (motion vectors R16G16_FLOAT, depth R32_FLOAT).
ID3D11ShaderResourceView* InputFor(ID3D11Device* dev, ID3D11Resource* res, DXGI_FORMAT fmt) {
    if (!res) return nullptr;
    for (const InputSrv& m : g_motion)
        if (m.res == res && m.fmt == fmt) return m.srv;
    const bool isMotion = fmt == DXGI_FORMAT_R16G16_FLOAT;
    ID3D11Texture2D* tex = nullptr;
    if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex)) || !tex) return nullptr;
    D3D11_TEXTURE2D_DESC d;
    tex->GetDesc(&d);
    tex->Release();
    ID3D11ShaderResourceView* srv = nullptr;
    if (d.ArraySize == 1 && d.SampleDesc.Count == 1 && (d.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = fmt;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        if (FAILED(dev->CreateShaderResourceView(res, &sd, &srv))) srv = nullptr;
    }
    if (!srv && !g_motionLogged[isMotion]) {
        g_motionLogged[isMotion] = true;
        Log("NR: the %s (%ux%u fmt %d, %u slices) cannot be read; %s for them", isMotion ? "motion vectors" : "depth",
            d.Width, d.Height, (int)d.Format, d.ArraySize, isMotion ? "zero vectors" : "the constant depth");
    }
    InputSrv& m = g_motion[g_motionNext];
    g_motionNext = (g_motionNext + 1) % kMotionCache;
    if (m.srv) m.srv->Release();
    if (m.res) m.res->Release();
    m.res = res;
    res->AddRef();
    m.fmt = fmt;
    m.srv = srv;
    return srv;
}

nrbridge::Settings SettingsNow() {
    const Config& c = Cfg();
    nrbridge::Settings s = g_set;
    if (s.modelPath.empty()) {  // fixed for the session
        const fs::path dir = ModuleDir() / L"qvfr";
        s.modelPath = (dir / L"nvngx_dlssnr.dll").wstring();
        s.forwarderPath = (dir / L"nvngx.dll_qvfr_nr.dll").wstring();
        s.searchDir = dir.wstring();
        s.dataDir = LogDir().wstring();
        s.appId = c.nrAppId;
        s.ngx = c.nrNgx;
    }
    s.model = c.nrModel;
    s.intensity = c.nrIntensity;
    s.style = (unsigned)KnobOr(Knob::Style, c.nrStyle);
    s.preset = (unsigned)KnobOr(Knob::Preset, c.nrPreset);
    s.autoMask = c.nrAutoMask;
    s.uiCorrection = c.nrUiCorrection;
    s.debug = c.nrDebug;
    s.inputScale = c.nrInputScale;
    s.tonemap = c.nrTonemap;
    s.transfer = c.nrTransfer;
    s.motion = c.nrMotion;
    s.depthMode = c.nrDepth;
    s.depthK = c.nrDepthK;
    s.strength = (float)KnobOr(Knob::Strength, (int)(c.nrStrength * 100.0f + 0.5f)) / 100.0f;
    kSteps[g_step.load()].apply(s);
    if (ModeNow() == 2) {  // the final image is display-encoded already: into the model and back as it is
        s.inputScale = 1.0f;
        s.tonemap = 0;
        s.transfer = 0;
    }
    return s;
}

bool Same(const nrbridge::Settings& a, const nrbridge::Settings& b) {
    return a.model == b.model && a.intensity == b.intensity && a.style == b.style && a.preset == b.preset &&
           a.autoMask == b.autoMask && a.uiCorrection == b.uiCorrection && a.debug == b.debug &&
           a.strength == b.strength &&
           a.inputScale == b.inputScale && a.tonemap == b.tonemap && a.transfer == b.transfer && a.motion == b.motion &&
           a.depthMode == b.depthMode && a.depthK == b.depthK && a.modelPath == b.modelPath && a.appId == b.appId &&
           a.ngx == b.ngx;
}

void ProcessImpl(ID3D11DeviceContext* c, int view, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav,
                 ID3D11Resource* motion, ID3D11Resource* depth, UINT w, UINT h, bool reset) {
    if (g_bridge && g_gen != gpu::Generation()) {
        Log("NR: the device changed; the bridge starts again");
        Drop();
    }
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return;
    if (!g_bridge) {
        g_bridge = new nrbridge::Bridge;
        g_gen = gpu::Generation();
        const bool ok = g_bridge->Init(dev);
        if (!ok) {
            dev->Release();
            TurnOff("bridge: " + g_bridge->why());
            Drop();
            return;
        }
        g_set = SettingsNow();
        g_bridge->Configure(g_set);
        Log("NR: prototype NR-P1: the focus views' DLAA result goes through the D3D12 device%s",
            g_set.model ? " and NVIDIA's Neural Rendering model (x64\\qvfr\\nvngx_dlssnr.dll)" : " and back (passthrough)");
    } else {
        const nrbridge::Settings s = SettingsNow();
        if (!Same(s, g_set)) {
            g_bridge->Configure(s);
            g_set = s;
            Log("NR: settings: mode=%s outer=%d model=%d intensity=%.2f style=%u preset=%u auto_mask=%d "
                "ui_correction=%d debug=%d strength=%.2f input_scale=%g tonemap=%d transfer=%s motion_vectors=%d "
                "depth=%d depth_k=%g",
                ModeNow() == 2 ? "final" : "focus", OuterNow(), s.model, s.intensity, s.style, s.preset, s.autoMask,
                s.uiCorrection, s.debug, s.strength, s.inputScale, s.tonemap, s.transfer ? "ratio" : "inverse",
                s.motion, s.depthMode, s.depthK);
        }
    }
    ID3D11ShaderResourceView* mv = InputFor(dev, motion, DXGI_FORMAT_R16G16_FLOAT);
    ID3D11ShaderResourceView* dp = g_set.depthMode ? InputFor(dev, depth, DXGI_FORMAT_R32_FLOAT) : nullptr;
    dev->Release();
    const int i = ImageOf(view);
    if (i < 0) return;
    // A change between after-DLAA and final images: the model's history is of the other kind of image.
    static int lastMode[kImages] = {};
    const int mode = ModeNow();
    if (lastMode[i] && lastMode[i] != mode) g_resetPending[i] = true;
    lastMode[i] = mode;
    if (mode == 2 && BatchNow()) {  // the image's motion vectors and depth into its region of the atlas
        const bool outer = OuterNow(), separate = SeparateNow();
        const UINT step = OuterStepNow(), area = OuterAreaNow();
        if (g_region[i].w != w || g_region[i].h != h || g_layoutOuter != outer ||
            (outer && (g_layoutStep != step || g_layoutArea != area || g_layoutSeparate != separate))) {
            g_region[i].w = w;
            g_region[i].h = h;
            g_layoutOuter = outer;
            g_layoutStep = step;
            g_layoutArea = area;
            g_layoutSeparate = separate;
            Relayout();
        }
        if (!g_atlasCount) return;  // not all images' sizes known yet
        const Region& r = g_region[i];
        g_bridge->TimeMark(c, nrbridge::Mark::Guides);
        const bool guided = g_bridge->Guides(c, kAtlasSlot + r.atlas, mv, dp, r.aw, r.ah, r.x, r.y, g_atlasW[r.atlas],
                                             g_atlasH[r.atlas], r.step, r.cx, r.cy);
        g_bridge->TimeMark(c, nrbridge::Mark::Other);
        if (guided) {
            g_guided[i] = g_layoutGen;
            if (reset) g_atlasReset[r.atlas] = true;
        } else if (g_bridge->failed()) {
            TurnOff("bridge: " + g_bridge->why());
        }
        return;
    }
    if (mode == 2) {  // the final image: only the motion vectors and depth now, the color at the composition
        if (g_bridge->Guides(c, i, mv, dp, w, h)) {
            if (reset) g_finalReset[i] = true;
        } else if (g_bridge->failed()) {
            TurnOff("bridge: " + g_bridge->why());
        }
        return;
    }
    const bool again = g_resetPending[i].exchange(false);
    if (!g_bridge->Process(c, i, srv, uav, mv, dp, w, h, reset || again)) {
        if (g_bridge->failed()) TurnOff("bridge: " + g_bridge->why());
        return;
    }
    ++g_calls[i];
    if (!g_firstLogged[i]) {
        g_firstLogged[i] = true;
        Log("NR: first image of %s (%ux%u) through %s, motion vectors %s, depth %s", kName[i], w, h,
            g_bridge->modelPath(), mv ? "from the game" : "zero", dp ? "from the game" : "constant");
    }
}

// The compute shader state ProcessFinal changes (the bridge's shaders), saved before and restored after.
struct CsState {
    ID3D11ComputeShader* cs = nullptr;
    ID3D11ClassInstance* ci[256] = {};
    UINT nci = 256;
    ID3D11ShaderResourceView* srv[3] = {};
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11Buffer* cb = nullptr;
    void Save(ID3D11DeviceContext* c) {
        c->CSGetShader(&cs, ci, &nci);
        c->CSGetShaderResources(0, 3, srv);
        c->CSGetUnorderedAccessViews(0, 1, &uav);
        c->CSGetConstantBuffers(0, 1, &cb);
    }
    void Restore(ID3D11DeviceContext* c) {
        c->CSSetShader(cs, ci, nci);
        c->CSSetShaderResources(0, 3, srv);
        c->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
        c->CSSetConstantBuffers(0, 1, &cb);
        if (cs) cs->Release();
        for (UINT i = 0; i < nci && i < 256; ++i)
            if (ci[i]) ci[i]->Release();
        for (auto* s : srv)
            if (s) s->Release();
        if (uav) uav->Release();
        if (cb) cb->Release();
    }
};

// `f` as a d.Width x d.Height texture in d.Format's family with an SRV and a UAV in the stored (UNORM) format; made or
// remade when needed. False (logged, NR off for the session): it cannot be made.
bool EnsureImage(ID3D11DeviceContext* c, FinalImage& f, const D3D11_TEXTURE2D_DESC& d, const char* name) {
    if (f.tex && f.desc.Width == d.Width && f.desc.Height == d.Height && f.desc.Format == d.Format) return true;
    DropImage(f);
    DXGI_FORMAT texFmt = DXGI_FORMAT_UNKNOWN, viewFmt = DXGI_FORMAT_UNKNOWN;
    if (!FinalFormats(d.Format, texFmt, viewFmt)) return false;
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return false;
    D3D11_TEXTURE2D_DESC n = d;
    n.Format = texFmt;
    n.MipLevels = 1;
    n.ArraySize = 1;
    n.SampleDesc = {1, 0};
    n.Usage = D3D11_USAGE_DEFAULT;
    n.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    n.CPUAccessFlags = 0;
    n.MiscFlags = 0;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = viewFmt;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = viewFmt;
    ud.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    const HRESULT hr = dev->CreateTexture2D(&n, nullptr, &f.tex);
    if (SUCCEEDED(hr)) dev->CreateShaderResourceView(f.tex, &sd, &f.srv);
    if (f.srv) dev->CreateUnorderedAccessView(f.tex, &ud, &f.uav);
    dev->Release();
    if (!f.uav) {
        Log("NR: cannot make the copy of the final %s image (%ux%u fmt %d, 0x%08lX); it stays", name, d.Width, d.Height,
            (int)d.Format, (unsigned long)hr);
        DropImage(f);
        g_failed = true;
        g_why = "no copy of the final image";
        return false;
    }
    f.desc = d;
    return true;
}

ID3D11Texture2D* ProcessFinalImpl(ID3D11DeviceContext* c, int i, ID3D11Texture2D* img) {
    if (!g_bridge || !g_bridge->ready()) return img;  // the bridge starts at the first DLAA it is wanted in
    D3D11_TEXTURE2D_DESC d{};
    img->GetDesc(&d);
    DXGI_FORMAT texFmt = DXGI_FORMAT_UNKNOWN, viewFmt = DXGI_FORMAT_UNKNOWN;
    if (!FinalFormats(d.Format, texFmt, viewFmt) || d.ArraySize != 1 || d.SampleDesc.Count != 1) {
        if (!g_finalSkipLogged[i]) {
            g_finalSkipLogged[i] = true;
            Log("NR: the final %s image (%ux%u fmt %d, %u slices, %u samples) cannot go through the model; it stays",
                kName[i], d.Width, d.Height, (int)d.Format, d.ArraySize, d.SampleDesc.Count);
        }
        return img;
    }
    FinalImage& f = g_final[i];
    if (!EnsureImage(c, f, d, kName[i])) return img;
    c->CopySubresourceRegion(f.tex, 0, 0, 0, 0, img, 0, nullptr);
    const bool reset = g_finalReset[i] || g_resetPending[i].exchange(false);
    g_finalReset[i] = false;
    CsState st;
    st.Save(c);
    const bool ok = g_bridge->Process(c, i, f.srv, f.uav, nullptr, nullptr, d.Width, d.Height, reset, true);
    st.Restore(c);
    if (!ok) {
        if (g_bridge->failed()) {
            TurnOff("bridge: " + g_bridge->why());
        } else if (!g_finalSkipLogged[i]) {
            g_finalSkipLogged[i] = true;
            Log("NR: the final %s image (%ux%u) has no motion vectors and depth of its size from DLAA; it stays",
                kName[i], d.Width, d.Height);
        }
        return img;
    }
    ++g_calls[i];
    if (!g_finalLogged[i]) {
        g_finalLogged[i] = true;
        Log("NR: first final %s image (%ux%u fmt %d, the game's bind flags 0x%X) through %s; the composition reads the "
            "copy",
            kName[i], d.Width, d.Height, (int)d.Format, d.BindFlags, g_bridge->modelPath());
    }
    return f.tex;
}

// The output merger's targets, saved and unbound while the bridge reads the images (the game's Submit texture may still
// be bound as a render target, and an SRV of it would be unbound), restored after.
struct OmState {
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    void Save(ID3D11DeviceContext* c) {
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
        c->OMSetRenderTargets(0, nullptr, nullptr);
    }
    void Restore(ID3D11DeviceContext* c) {
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
        for (auto* v : rtv)
            if (v) v->Release();
        if (dsv) dsv->Release();
    }
};

// batch=1: an UNORM view of the image for the bridge (cached while the image stays); an image without an SRV binding
// goes through its copy in g_orig. Null: neither can be made (logged once).
ID3D11ShaderResourceView* ImageView(ID3D11DeviceContext* c, int i, ID3D11Texture2D* img,
                                    const D3D11_TEXTURE2D_DESC& d) {
    if (!(d.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
        if (!EnsureImage(c, g_orig[i], d, kName[i])) return nullptr;
        c->CopySubresourceRegion(g_orig[i].tex, 0, 0, 0, 0, img, 0, nullptr);
        return g_orig[i].srv;
    }
    View& v = g_view[i];
    if (v.tex == img) return v.srv;
    DropView(v);
    DXGI_FORMAT texFmt = DXGI_FORMAT_UNKNOWN, viewFmt = DXGI_FORMAT_UNKNOWN;
    FinalFormats(d.Format, texFmt, viewFmt);
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = viewFmt;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    const HRESULT hr = dev->CreateShaderResourceView(img, &sd, &v.srv);
    dev->Release();
    if (FAILED(hr) || !v.srv) {
        v = {};
        if (!g_viewLogged) {
            g_viewLogged = true;
            Log("NR: no view of the final %s image (fmt %d, 0x%08lX); it goes through a copy", kName[i], (int)d.Format,
                (unsigned long)hr);
        }
        if (!EnsureImage(c, g_orig[i], d, kName[i])) return nullptr;
        c->CopySubresourceRegion(g_orig[i].tex, 0, 0, 0, 0, img, 0, nullptr);
        return g_orig[i].srv;
    }
    v.tex = img;  // the view holds a reference, so the address stays this image's
    return v.srv;
}

// batch=1, at the left eye's composition: img = focus L, focus R, outer L, outer R (the outer ones only with the
// layout's outer). Each image goes straight into its region of its atlas (the bridge's shared model input), each atlas
// through the model once, and each image straight back out into its copy in g_final. True: g_final holds every image
// through the model.
bool RunAtlas(ID3D11DeviceContext* c, ID3D11Texture2D* const img[kImages]) {
    if (!g_bridge || !g_bridge->ready()) return false;
    const int n = g_layoutOuter ? 4 : 2;
    const char* why = nullptr;
    D3D11_TEXTURE2D_DESC d[kImages] = {};
    DXGI_FORMAT texFmt = DXGI_FORMAT_UNKNOWN, viewFmt = DXGI_FORMAT_UNKNOWN;
    if (!g_atlasCount) why = "not all images' sizes are known yet";
    for (int i = 0; i < n && !why; ++i) {
        if (!img[i]) {
            why = "an image of this frame is missing";
            break;
        }
        img[i]->GetDesc(&d[i]);
        if (g_guided[i] != g_layoutGen)
            why = "the motion vectors and depth of an image are not from this frame's layout";
        else if (d[i].Width != g_region[i].w || d[i].Height != g_region[i].h)
            why = "an image's size is not DLAA's";
        else if (d[i].Format != d[0].Format || !FinalFormats(d[i].Format, texFmt, viewFmt) || d[i].ArraySize != 1 ||
                 d[i].SampleDesc.Count != 1)
            why = "an image's format cannot go through the model";
    }
    if (why) {
        if (!g_atlasSkipLogged) {
            g_atlasSkipLogged = true;
            Log("NR: atlas skipped (%s; logged once); the images stay", why);
        }
        return false;
    }
    for (int i = 0; i < n; ++i)
        if (!EnsureImage(c, g_final[i], d[i], kName[i])) return false;
    CsState st;
    st.Save(c);
    OmState om;
    om.Save(c);
    ID3D11ShaderResourceView* view[kImages] = {};
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = (view[i] = ImageView(c, i, img[i], d[i])) != nullptr;
    bool reset[kAtlases] = {};
    for (int a = 0; a < g_atlasCount; ++a) {
        reset[a] = g_atlasReset[a] || g_atlasRunGen[a] != g_layoutGen;
        if (g_atlasRunGen[a] != g_layoutGen) g_bridge->ClearInput(kAtlasSlot + a);  // the new layout's gaps black
    }
    for (int i = 0; i < n; ++i) reset[g_region[i].atlas] |= g_resetPending[i].exchange(false);
    g_bridge->TimeMark(c, nrbridge::Mark::Pack);
    for (int i = 0; i < n && ok; ++i) {
        const Region& r = g_region[i];
        ok = g_bridge->PackRegion(c, kAtlasSlot + r.atlas, view[i], r.x, r.y, r.aw, r.ah, r.step, r.cx, r.cy);
    }
    for (int a = 0; a < g_atlasCount && ok; ++a) {
        g_atlasReset[a] = false;
        g_atlasRunGen[a] = g_layoutGen;
        ok = g_bridge->Evaluate(c, kAtlasSlot + a, reset[a]);
    }
    for (auto& g : g_guided) g = 0;  // the next run needs the next frame's guides
    for (int i = 0; i < n && ok; ++i) {
        const Region& r = g_region[i];
        ok = g_bridge->UnpackRegion(c, kAtlasSlot + r.atlas, view[i], g_final[i].uav, r.x, r.y, r.aw, r.ah, r.step,
                                    r.w, r.h, r.cx, r.cy, r.fade);
    }
    om.Restore(c);
    st.Restore(c);
    if (!ok) {
        if (g_bridge->failed()) {
            TurnOff("bridge: " + g_bridge->why());
        } else if (!g_atlasSkipLogged) {
            g_atlasSkipLogged = true;
            Log("NR: atlas skipped (no view of an image, or no motion vectors and depth of the atlas size; logged "
                "once); the images stay");
        }
        return false;
    }
    for (int i = 0; i < n; ++i) ++g_calls[i];
    if (!g_atlasLogged) {
        g_atlasLogged = true;
        Log("NR: first atlas run (%ux%u%s fmt %d, %d images, outer at 1/%u, its centre %u%%) through %s in %d "
            "evaluation(s), the images read in place%s; the composition reads the copies",
            g_atlasW[0], g_atlasH[0], g_atlasCount > 1 ? " and the outer atlas" : "", (int)d[0].Format, n,
            g_region[2].step, g_layoutArea, g_bridge->modelPath(), g_atlasCount,
            (d[0].BindFlags & D3D11_BIND_SHADER_RESOURCE) ? "" : " (through a copy: no SRV binding)");
    }
    return true;
}

void FinalEyeImpl(ID3D11DeviceContext* c, int eye, ID3D11Texture2D*& focus, ID3D11Texture2D*& outer,
                  ID3D11Texture2D* rightFocus, ID3D11Texture2D* rightOuter) {
    if (!BatchNow()) {
        g_atlasOut = false;
        focus = ProcessFinalImpl(c, eye, focus);
        if (OuterNow()) outer = ProcessFinalImpl(c, eye + 2, outer);
        return;
    }
    if (eye == 0) {
        ID3D11Texture2D* const img[kImages] = {focus, rightFocus, outer, rightOuter};
        g_atlasOut = RunAtlas(c, img);
        if (g_bridge) g_bridge->TimeFrameEnd(c);  // the atlas frame's D3D11 timestamps (none counted without a run)
    }
    if (g_atlasOut) {
        focus = g_final[eye].tex;
        if (g_layoutOuter) outer = g_final[eye + 2].tex;
    }
    if (eye == 1) g_atlasOut = false;  // the next frame's atlas first
}

}  // namespace

bool Wanted(int view) {
    const int i = ImageOf(view);
    return ModeNow() != 0 && (i == 0 || i == 1 || (i >= 2 && OuterNow())) && !g_failed &&
           g_active.load(std::memory_order_relaxed);
}

bool WantedFinal() { return ModeNow() == 2 && !g_failed && g_active.load(std::memory_order_relaxed); }

float Smoothen(float ini) {
    const float step = kSteps[g_step.load()].smoothen;
    return step >= 0 ? step : ini;
}

void ProcessFinal(ID3D11DeviceContext* c, int eye, ID3D11Texture2D*& focus, ID3D11Texture2D*& outer,
                  ID3D11Texture2D* rightFocus, ID3D11Texture2D* rightOuter) {
    if (!c || !focus || !outer || eye < 0 || eye > 1) return;
    ID3D11Texture2D* f = focus;
    ID3D11Texture2D* o = outer;
    unsigned long code = 0;
    __try {
        FinalEyeImpl(c, eye, f, o, rightFocus, rightOuter);
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_faultLogged) {
            g_faultLogged = true;
            Log("ERROR: exception 0x%08lX in NR (final image); off for the rest of the session", code);
        }
        g_failed = true;
        g_why = "exception";
        f = focus;
        o = outer;
    }
    focus = f;
    outer = o;
}

void Process(ID3D11DeviceContext* c, int view, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav,
             ID3D11Resource* motion, ID3D11Resource* depth, UINT w, UINT h, bool reset) {
    if (!c || !srv || !uav) return;
    unsigned long code = 0;
    __try {
        ProcessImpl(c, view, srv, uav, motion, depth, w, h, reset);
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_faultLogged) {
            g_faultLogged = true;
            Log("ERROR: exception 0x%08lX in NR; off for the rest of the session", code);
        }
        g_failed = true;
        g_why = "exception";
    }
}

bool Active() { return g_active.load(); }

int GetKnob(Knob k) {
    const Config& c = Cfg();
    switch (k) {
        case Knob::Outer: return KnobOr(k, c.nrOuter);
        case Knob::Batch: return KnobOr(k, c.nrBatch);
        case Knob::OuterRes: return KnobOr(k, c.nrOuterStep);
        case Knob::OuterArea: return KnobOr(k, c.nrOuterAreaPct);
        case Knob::Preset: return KnobOr(k, c.nrPreset);
        case Knob::Style: return KnobOr(k, c.nrStyle);
        case Knob::Strength: return KnobOr(k, (int)(c.nrStrength * 100.0f + 0.5f));
        case Knob::OuterSeparate: return KnobOr(k, c.nrOuterSeparate);
        default: return 0;
    }
}

void SetKnob(Knob k, int value) {
    if (k < Knob::Count) g_knob[(int)k] = value;
}

Window LastWindow() { return g_window; }

void SetActive(bool on) {
    if (on && !g_active.load())
        for (auto& r : g_resetPending) r = true;
    g_active = on;
}

void OnFrameStart() {
    const Config& c = Cfg();
    if (c.nrMode == 0) return;
    if (c.nrTestKey && Pressed(c.nrKeyMod, c.nrTestKey, g_testKeyDown)) {
        const int step = (g_step.load() + 1) % kStepCount;
        g_step = step;
        StepBeeps(step);
        Log("NR: test step %d of %d: %s", step, kStepCount - 1, kSteps[step].name);
    }
    // The outer images on again: their model history is from before the pause.
    static bool outerWas = false;
    const bool outer = OuterNow();
    if (outer && !outerWas) g_resetPending[2] = g_resetPending[3] = true;
    outerWas = outer;
    if (!c.nrKey) return;
    if (Pressed(c.nrKeyMod, c.nrKey, g_keyDown)) {
        const bool on = !g_active.load();
        SetActive(on);
        Log("NR: hotkey: %s", on ? "on" : "off (DLAA only)");
    }
}

void LogStats(bool totals) {
    if (Cfg().nrMode == 0 && !g_bridge) return;
    const nrbridge::Stats s = g_bridge ? g_bridge->stats() : nrbridge::Stats{};
    nrbridge::Stats d = s;
    if (!totals) {
        d.calls -= g_last.calls;
        d.modelCalls -= g_last.modelCalls;
        d.modelFails -= g_last.modelFails;
        d.cpuMs -= g_last.cpuMs;
        d.gpuMs -= g_last.gpuMs;
        d.gpuSamples -= g_last.gpuSamples;
        for (int i = 0; i < nrbridge::kSlots; ++i) {
            d.slotGpuMs[i] -= g_last.slotGpuMs[i];
            d.slotGpuSamples[i] -= g_last.slotGpuSamples[i];
        }
        d.d11GuidesMs -= g_last.d11GuidesMs;
        d.d11PackMs -= g_last.d11PackMs;
        d.d11HandoffMs -= g_last.d11HandoffMs;
        d.d11UnpackMs -= g_last.d11UnpackMs;
        d.d11Frames -= g_last.d11Frames;
    }
    // Per image how many went through, per evaluation (one image, or the atlas of all) the model's D3D12 GPU time.
    static const char* const kSlotName[nrbridge::kSlots] = {"focus L", "focus R", "outer L", "outer R", "atlas",
                                                            "outer atlas"};
    std::string images, gpu;
    for (int i = 0; i < kImages; ++i) {
        const uint64_t n = totals ? g_calls[i] : g_calls[i] - g_lastCalls[i];
        if (!n && i >= 2) continue;
        char b[64];
        snprintf(b, sizeof(b), "%s%s %llu", images.empty() ? "" : ", ", kName[i], (unsigned long long)n);
        images += b;
    }
    for (int i = 0; i < nrbridge::kSlots; ++i) {
        if (!d.slotGpuSamples[i]) continue;
        char b[64];
        snprintf(b, sizeof(b), "%s%s %.3f", gpu.empty() ? "" : ", ", kSlotName[i],
                 d.slotGpuMs[i] / (double)d.slotGpuSamples[i]);
        gpu += b;
    }
    // The D3D12 GPU time per frame: all evaluations of a frame (focus L counts the frames).
    const uint64_t frames = totals ? g_calls[0] : g_calls[0] - g_lastCalls[0];
    const double perFrame =
        frames && d.gpuSamples ? d.gpuMs / (double)d.gpuSamples * (double)d.calls / (double)frames : 0.0;
    Log("NR%s: %s images; %llu evaluations (failed %llu), D3D12 GPU ms per evaluation: %s; %.2f ms per frame; CPU "
        "%.3f ms per evaluation (with the Flush); %s",
        totals ? " (session total)" : "", images.c_str(), (unsigned long long)d.modelCalls,
        (unsigned long long)d.modelFails, gpu.empty() ? "none" : gpu.c_str(), perFrame,
        d.calls ? d.cpuMs / (double)d.calls : 0.0, Describe().c_str());
    // The game's D3D11 queue around the atlas: what NR costs the frame there. The hand-off holds the model (D3D12
    // time above) plus the switches between the queues.
    const double f11 = (double)d.d11Frames;
    const double g11 = f11 ? d.d11GuidesMs / f11 : 0, p11 = f11 ? d.d11PackMs / f11 : 0,
                 h11 = f11 ? d.d11HandoffMs / f11 : 0, u11 = f11 ? d.d11UnpackMs / f11 : 0;
    if (d.d11Frames)
        Log("NR%s: D3D11 GPU ms per atlas frame (%llu frames): guides %.3f, pack %.3f, hand-off %.3f (D3D12 %.3f, so "
            "%.3f between the queues), unpack %.3f; total %.3f",
            totals ? " (session total)" : "", (unsigned long long)d.d11Frames, g11, p11, h11, perFrame,
            h11 - perFrame, u11, g11 + p11 + h11 + u11);
    if (!totals) {
        g_window.any = frames > 0;
        g_window.gpuMsPerFrame = perFrame;
        g_window.atlasMp = BatchNow() ? g_atlasMp : 0.0;
        g_window.d11 = d.d11Frames > 0;
        g_window.d11Guides = g11;
        g_window.d11Pack = p11;
        g_window.d11Handoff = h11;
        g_window.d11Unpack = u11;
    }
    g_last = s;
    for (int i = 0; i < kImages; ++i) g_lastCalls[i] = g_calls[i];
}

std::string Describe() {
    if (Cfg().nrMode == 0) return "NR: off";
    if (g_failed) return "NR: off (" + g_why + ")";
    if (!g_bridge) return "NR: waiting for the first focus DLAA";
    std::string s = std::string(ModeNow() != 2  ? "NR: focus views through "
                                : OuterNow()    ? "NR: final focus and outer images through "
                                                : "NR: final focus images through ") +
                    g_bridge->modelPath() + " on " + g_bridge->adapter();
    if (BatchNow())
        s += !OuterNow()     ? ", one evaluation per frame (atlas)"
             : SeparateNow() ? ", two evaluations per frame (focus atlas, outer atlas at 1/" +
                                   std::to_string(OuterStepNow()) + ", its centre " + std::to_string(OuterAreaNow()) + "%)"
                             : ", one evaluation per frame (atlas, outer at 1/" + std::to_string(OuterStepNow()) +
                                   ", its centre " + std::to_string(OuterAreaNow()) + "%)";
    if (!g_active.load()) s += ", hotkey: off";
    if (g_step.load()) s += std::string(", test step ") + std::to_string(g_step.load()) + ": " + kSteps[g_step.load()].name;
    return s;
}

void Shutdown() {
    Drop();
    g_set = {};
    for (int i = 0; i < kImages; ++i) g_firstLogged[i] = g_finalLogged[i] = false;
    g_atlasLogged = g_atlasSkipLogged = false;
}

}  // namespace qvfr::nr

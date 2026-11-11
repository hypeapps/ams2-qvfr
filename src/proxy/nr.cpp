#include "nr.h"

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "common.h"
#include "gpu.h"
#include "nr_bridge.h"

namespace qvfr::nr {

namespace {

constexpr int kFocusL = 3;  // dlaa's view index of focus L (focus R = 4)
constexpr const char* kName[2] = {"focus L", "focus R"};
constexpr int kMotionCache = 8;

nrbridge::Bridge* g_bridge = nullptr;
nrbridge::Settings g_set;  // what the bridge was configured with
uint32_t g_gen = 0;
bool g_failed = false;  // the rest of the session: DLAA only
std::string g_why;
std::atomic<bool> g_active{true};
std::atomic<bool> g_resetPending[2] = {false, false};  // NR turned on again: the model's history starts over
bool g_keyDown = false;
bool g_faultLogged = false;
uint64_t g_calls[2] = {}, g_lastCalls[2] = {};
bool g_firstLogged[2] = {};
nrbridge::Stats g_last{};

// SRVs of the motion vector and depth textures DLAA hands over (kept with a reference on the texture).
struct InputSrv {
    ID3D11Resource* res = nullptr;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    ID3D11ShaderResourceView* srv = nullptr;
};
InputSrv g_motion[kMotionCache];
int g_motionNext = 0;
bool g_motionLogged[2] = {};

// The built-in test sequence (dlaa-plan D7, test 6): [nr] modifier + test_key goes to the next step, on top of the ini.
// Step 0 is the ini as it is. `mode` 0 keeps the ini's mode, 1 after DLAA (focus), 2 the final image. Test 5's sequence
// (input_scale, transfer after DLAA) is in dlaa-plan D7.
struct TestStep {
    const char* name;
    int mode;
    void (*apply)(nrbridge::Settings&);
};
const TestStep kSteps[] = {
    {"the ini as it is", 0, [](nrbridge::Settings&) {}},
    {"edit map (debug=1)", 0, [](nrbridge::Settings& s) { s.debug = 1; }},
    {"intensity=0 (the output should equal the input)", 0, [](nrbridge::Settings& s) { s.intensity = 0.0f; }},
    {"strength=2 (the model's edit x2)", 0, [](nrbridge::Settings& s) { s.strength = 2.0f; }},
    {"strength=3 (the model's edit x3)", 0, [](nrbridge::Settings& s) { s.strength = 3.0f; }},
    {"style=1", 0, [](nrbridge::Settings& s) { s.style = 1; }},
    {"style=2 (Cinematic)", 0, [](nrbridge::Settings& s) { s.style = 2; }},
    {"style=3", 0, [](nrbridge::Settings& s) { s.style = 3; }},
    {"auto_mask=0", 0, [](nrbridge::Settings& s) { s.autoMask = false; }},
    {"ui_correction=0", 0, [](nrbridge::Settings& s) { s.uiCorrection = false; }},
    {"motion_vectors=0", 0, [](nrbridge::Settings& s) { s.motion = false; }},
    {"after DLAA (mode=focus), input_scale=0.0005", 1, [](nrbridge::Settings& s) { s.inputScale = 0.0005f; }},
    {"after DLAA (mode=focus), input_scale=0.0005, strength=3", 1,
     [](nrbridge::Settings& s) { s.inputScale = 0.0005f; s.strength = 3.0f; }},
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

// mode=final: per eye, a copy of the finished focus image the model writes into (compose reads it instead).
struct FinalImage {
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    D3D11_TEXTURE2D_DESC desc{};  // of the game's image it copies
};
FinalImage g_final[2];
bool g_finalReset[2] = {};       // DLAA reset its history since the last final image
bool g_finalLogged[2] = {};
bool g_finalSkipLogged[2] = {};

void DropFinal() {
    for (FinalImage& f : g_final) {
        if (f.uav) f.uav->Release();
        if (f.srv) f.srv->Release();
        if (f.tex) f.tex->Release();
        f = {};
    }
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
    s.style = (unsigned)c.nrStyle;
    s.preset = (unsigned)c.nrPreset;
    s.autoMask = c.nrAutoMask;
    s.uiCorrection = c.nrUiCorrection;
    s.debug = c.nrDebug;
    s.inputScale = c.nrInputScale;
    s.tonemap = c.nrTonemap;
    s.transfer = c.nrTransfer;
    s.motion = c.nrMotion;
    s.depthMode = c.nrDepth;
    s.depthK = c.nrDepthK;
    s.strength = c.nrStrength;
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
            Log("NR: settings: mode=%s model=%d intensity=%.2f style=%u preset=%u auto_mask=%d ui_correction=%d "
                "debug=%d strength=%.2f input_scale=%g tonemap=%d transfer=%s motion_vectors=%d depth=%d depth_k=%g",
                ModeNow() == 2 ? "final" : "focus", s.model, s.intensity, s.style, s.preset, s.autoMask,
                s.uiCorrection, s.debug, s.strength, s.inputScale, s.tonemap, s.transfer ? "ratio" : "inverse",
                s.motion, s.depthMode, s.depthK);
        }
    }
    ID3D11ShaderResourceView* mv = InputFor(dev, motion, DXGI_FORMAT_R16G16_FLOAT);
    ID3D11ShaderResourceView* dp = g_set.depthMode ? InputFor(dev, depth, DXGI_FORMAT_R32_FLOAT) : nullptr;
    dev->Release();
    const int i = view - kFocusL;
    // A change between after-DLAA and final images: the model's history is of the other kind of image.
    static int lastMode[2] = {};
    const int mode = ModeNow();
    if (lastMode[i] && lastMode[i] != mode) g_resetPending[i] = true;
    lastMode[i] = mode;
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

ID3D11Texture2D* ProcessFinalImpl(ID3D11DeviceContext* c, int e, ID3D11Texture2D* focus) {
    if (!g_bridge || !g_bridge->ready()) return focus;  // the bridge starts at the first focus DLAA
    D3D11_TEXTURE2D_DESC d{};
    focus->GetDesc(&d);
    DXGI_FORMAT texFmt = DXGI_FORMAT_UNKNOWN, viewFmt = DXGI_FORMAT_UNKNOWN;
    if (!FinalFormats(d.Format, texFmt, viewFmt) || d.ArraySize != 1 || d.SampleDesc.Count != 1) {
        if (!g_finalSkipLogged[e]) {
            g_finalSkipLogged[e] = true;
            Log("NR: the final %s image (%ux%u fmt %d, %u slices, %u samples) cannot go through the model; it stays",
                kName[e], d.Width, d.Height, (int)d.Format, d.ArraySize, d.SampleDesc.Count);
        }
        return focus;
    }
    FinalImage& f = g_final[e];
    if (f.tex && (f.desc.Width != d.Width || f.desc.Height != d.Height || f.desc.Format != d.Format)) {
        f.uav->Release();
        f.srv->Release();
        f.tex->Release();
        f = {};
    }
    if (!f.tex) {
        ID3D11Device* dev = nullptr;
        c->GetDevice(&dev);
        if (!dev) return focus;
        D3D11_TEXTURE2D_DESC n = d;
        n.Format = texFmt;
        n.MipLevels = 1;
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
            Log("NR: cannot make the copy of the final %s image (%ux%u fmt %d, 0x%08lX); it stays", kName[e], d.Width,
                d.Height, (int)d.Format, (unsigned long)hr);
            if (f.srv) f.srv->Release();
            if (f.tex) f.tex->Release();
            f = {};
            g_failed = true;
            g_why = "no copy of the final image";
            return focus;
        }
        f.desc = d;
    }
    c->CopySubresourceRegion(f.tex, 0, 0, 0, 0, focus, 0, nullptr);
    const bool reset = g_finalReset[e] || g_resetPending[e].exchange(false);
    g_finalReset[e] = false;
    CsState st;
    st.Save(c);
    const bool ok = g_bridge->Process(c, e, f.srv, f.uav, nullptr, nullptr, d.Width, d.Height, reset, true);
    st.Restore(c);
    if (!ok) {
        if (g_bridge->failed()) {
            TurnOff("bridge: " + g_bridge->why());
        } else if (!g_finalSkipLogged[e]) {
            g_finalSkipLogged[e] = true;
            Log("NR: the final %s image (%ux%u) has no motion vectors and depth of its size from DLAA; it stays",
                kName[e], d.Width, d.Height);
        }
        return focus;
    }
    ++g_calls[e];
    if (!g_finalLogged[e]) {
        g_finalLogged[e] = true;
        Log("NR: first final %s image (%ux%u fmt %d, the game's bind flags 0x%X) through %s; the composition reads the "
            "copy",
            kName[e], d.Width, d.Height, (int)d.Format, d.BindFlags, g_bridge->modelPath());
    }
    return f.tex;
}

}  // namespace

bool Wanted(int view) {
    return ModeNow() != 0 && (view == kFocusL || view == kFocusL + 1) && !g_failed &&
           g_active.load(std::memory_order_relaxed);
}

bool WantedFinal() { return ModeNow() == 2 && !g_failed && g_active.load(std::memory_order_relaxed); }

ID3D11Texture2D* ProcessFinal(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* focus) {
    if (!c || !focus || eye < 0 || eye > 1) return focus;
    unsigned long code = 0;
    ID3D11Texture2D* out = focus;
    __try {
        out = ProcessFinalImpl(c, eye, focus);
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_faultLogged) {
            g_faultLogged = true;
            Log("ERROR: exception 0x%08lX in NR (final image); off for the rest of the session", code);
        }
        g_failed = true;
        g_why = "exception";
        out = focus;
    }
    return out;
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

void SetActive(bool on) {
    if (on && !g_active.load()) g_resetPending[0] = g_resetPending[1] = true;
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
    }
    Log("NR%s: %s %llu, %s %llu images; model %llu (failed %llu); CPU %.3f ms per image (with the Flush), D3D12 GPU "
        "%.3f ms per image; %s",
        totals ? " (session total)" : "", kName[0], (unsigned long long)(totals ? g_calls[0] : g_calls[0] - g_lastCalls[0]),
        kName[1], (unsigned long long)(totals ? g_calls[1] : g_calls[1] - g_lastCalls[1]),
        (unsigned long long)d.modelCalls, (unsigned long long)d.modelFails, d.calls ? d.cpuMs / (double)d.calls : 0.0,
        d.gpuSamples ? d.gpuMs / (double)d.gpuSamples : 0.0, Describe().c_str());
    g_last = s;
    g_lastCalls[0] = g_calls[0];
    g_lastCalls[1] = g_calls[1];
}

std::string Describe() {
    if (Cfg().nrMode == 0) return "NR: off";
    if (g_failed) return "NR: off (" + g_why + ")";
    if (!g_bridge) return "NR: waiting for the first focus DLAA";
    std::string s = std::string(ModeNow() == 2 ? "NR: final focus images through " : "NR: focus views through ") +
                    g_bridge->modelPath() + " on " + g_bridge->adapter();
    if (!g_active.load()) s += ", hotkey: off";
    if (g_step.load()) s += std::string(", test step ") + std::to_string(g_step.load()) + ": " + kSteps[g_step.load()].name;
    return s;
}

void Shutdown() {
    Drop();
    g_set = {};
    g_firstLogged[0] = g_firstLogged[1] = false;
    g_finalLogged[0] = g_finalLogged[1] = false;
}

}  // namespace qvfr::nr

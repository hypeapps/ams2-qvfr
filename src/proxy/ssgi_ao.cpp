#include "ssgi_ao.h"

#include <d3d11_1.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "gpu.h"
#include "mvr_pass.h"
#include "ssgi_ao_cs.h"       // kSsgiAoCS (fxc from shaders/ssgi_ao.hlsl)
#include "ssgi_blur_cs.h"     // kSsgiBlurCS
#include "ssgi_full_vs.h"     // kSsgiFullVS
#include "ssgi_gi_ps.h"       // kSsgiGiPS
#include "ssgi_history_cs.h"  // kSsgiHistoryCS
#include "ssgi_preview_ps.h"  // kSsgiPreviewPS
#include "ssgi_probe_cs.h"    // kSsgiProbeCS
#include "sun_fx.h"

namespace qvfr::ssgiao {

namespace {

// Slots of the game's draws (docs/ssgi-plan.md "Ustalenia G0"): the overlay's SSAO, normals and depth, the ambient
// pass's SSAO and normals.
constexpr UINT kOverlaySsao = 1, kOverlayNormals = 2, kOverlayDepth = 3;
// The ambient pass's slots (SSAO, normals, the deferred-ambient weight): PS 15549 and the lightmap variant.
struct AmbientSlots {
    UINT ssao, normals, weight;
};
constexpr AmbientSlots kAmbient = {2, 4, 3}, kAmbientLightmap = {3, 5, 4};
// Constant slots of the compute shaders: the game's windows (the overlay PS's _GLOBAL_ and COMMON_BUFFER2) and ours.
constexpr UINT kCbGlobal = 0, kCbParams = 1, kCbHist = 3, kCbProbe = 4, kCbCommon = 12;
// Shader slots of shaders/ssgi_ao.hlsl.
constexpr UINT kTDepth = 0, kTNormals = 1, kTRaw = 2, kTAo = 3, kTWeight = 4, kTHist = 5, kTGi = 6, kTHistSrc = 7,
               kTProbe = 8;  // 8 the bounce, 9 the diffuse light, 10 the history, 11 the weight
constexpr UINT kUAo = 0, kUCams = 1, kUGi = 2, kURaw = 3, kUHist = 4, kUProbe = 5;
constexpr UINT kCsSrvs = 12, kCsUavs = 6;
// The probe (diagnostics): views 0 and 2 (outer L, focus L), 64 groups of 3 float4 each.
constexpr int kProbes = 2;
constexpr UINT kProbeFloats = 64 * 3 * 4;
constexpr UINT kCamSize = 5;  // float4 per camera entry (viewI rows, tangents)
constexpr int kViews = 4;
const char* const kViewNames[kViews] = {"outer L", "outer R", "focus L", "focus R"};

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// Why a draw went without own AO or a view without the bounce (counted, the first of each logged).
enum Why {
    WhyGroup,       // outside the eye groups
    WhyBindings,    // the overlay's normals or depth are not the expected 2D views
    WhyWindows,     // no constant window in a slot the compute shader needs (or no ID3D11DeviceContext1)
    WhyTexture,     // own textures could not be made
    WhyAmbient,     // the ambient draw's view has no AO of this frame for its normals
    WhyPreview,     // the resolve's view has no AO of its size in this frame
    WhyWeight,      // the overlay's render target is not the readable deferred-ambient weight
    WhyHistory,     // no history of the previous frame for the view
    WhyHistSource,  // the view's final image cannot be read
    kWhys
};
const char* const kWhyText[kWhys] = {
    "a draw outside the eye groups",
    "the overlay's normals or depth are not 2D views of the expected formats",
    "a constant window the compute shader needs is not bound (or no ID3D11DeviceContext1)",
    "own AO textures could not be made",
    "the ambient draw's view has no own AO of this frame for its normals (the game's SSAO stays there)",
    "the MSAA resolve's view has no own AO of its size in this frame (no preview or history there)",
    "the overlay's render target is not a readable R8G8B8A8_UNORM texture (the deferred-ambient weight): no bounce",
    "a view has no image and camera of the previous frame (its first frame, or no MSAA resolve or DLSS output to take "
    "it from): no bounce there in that frame",
    "a view's final image is not a readable single-sample texture: no history from it",
};

struct Stats {
    uint64_t frames = 0;             // frames with own AO on
    uint64_t passes[kViews] = {};    // compute passes per view
    uint64_t giPasses[kViews] = {};  // of them with the bounce
    uint64_t captures[kViews] = {};  // history copies per view
    uint64_t ambient = 0;            // ambient draws with own AO
    uint64_t composites = 0;         // the bounce added into the diffuse light target
    uint64_t previews = 0;           // preview draws
    uint64_t why[kWhys] = {};
};

// Own AO and bounce of one view: the raw result (rgb the bounce, a the AO) and the blurred ones at the G-buffer's size;
// the history (the view's final image of the previous frame, half the view's size).
struct View {
    ID3D11Texture2D* raw = nullptr;
    ID3D11Texture2D* fin = nullptr;
    ID3D11Texture2D* gi = nullptr;
    ID3D11UnorderedAccessView* rawUav = nullptr;
    ID3D11UnorderedAccessView* finUav = nullptr;
    ID3D11UnorderedAccessView* giUav = nullptr;
    ID3D11ShaderResourceView* rawSrv = nullptr;
    ID3D11ShaderResourceView* finSrv = nullptr;
    ID3D11ShaderResourceView* giSrv = nullptr;
    UINT w = 0, h = 0;
    bool failed = false, giFailed = false;
    uint64_t frame = 0;       // frame of the latest result (0 = none)
    uint64_t giFrame = 0;     // frame of the latest bounce
    void* normals = nullptr;  // the G-buffer normals texture it was made from (only compared)
    float vpW = 0, vpH = 0;   // the overlay draw's viewport size (the preview's AO region)
    UINT loggedW = 0, loggedH = 0;
    ID3D11Texture2D* hist = nullptr;
    ID3D11UnorderedAccessView* histUav = nullptr;
    ID3D11ShaderResourceView* histSrv = nullptr;
    UINT hw = 0, hh = 0;
    bool histFailed = false;
    uint64_t histFrame = 0;       // frame the history is of
    uint64_t camFrame[2] = {};    // frame whose camera each entry (by frame parity) holds
    UINT histLoggedW = 0, histLoggedH = 0;
};

struct Pipeline {
    ID3D11Device* dev = nullptr;  // referenced
    ID3D11ComputeShader* ao = nullptr;
    ID3D11ComputeShader* blur = nullptr;
    ID3D11ComputeShader* history = nullptr;
    ID3D11ComputeShader* probe = nullptr;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11PixelShader* giPs = nullptr;
    ID3D11Buffer* params = nullptr;
    ID3D11Buffer* histParams = nullptr;
    ID3D11Buffer* preview = nullptr;
    ID3D11Buffer* probeParams = nullptr;
    ID3D11Buffer* probeBuf[kProbes] = {};  // the probe's result per probed view, and its CPU copy
    ID3D11UnorderedAccessView* probeUav[kProbes] = {};
    ID3D11Buffer* probeStaging[kProbes] = {};
    ID3D11Buffer* cams = nullptr;  // the views' cameras, two entries each (by frame parity)
    ID3D11UnorderedAccessView* camsUav = nullptr;
    ID3D11ShaderResourceView* white = nullptr;  // 1 x 1 AO = 1 (ao=2: the overlay's sun term without AO)
    ID3D11SamplerState* point = nullptr;        // CS s0: the depth gather
    ID3D11SamplerState* linear = nullptr;       // CS s1: the history
    ID3D11BlendState* blend = nullptr;          // opaque (previews)
    ID3D11BlendState* add = nullptr;            // One, One (the bounce into the diffuse light)
    ID3D11DepthStencilState* dss = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    bool failed = false;
};

ID3D11DeviceContext* g_imm = nullptr;
ID3D11DeviceContext1* g_ctx1 = nullptr;  // referenced
uint32_t g_gen = 0;
int g_mode = 0;           // [ssgi] ao in effect this frame (0 also when stopped)
int g_gi = 0;             // [ssgi] gi in effect this frame (0 without own AO)
int g_modeLogged = -1, g_giLogged = -1;
bool g_dead = false;      // a fault: off for the session
bool g_testing = false;
int g_testMode = 0, g_testGi = 0;
uint64_t g_frame = 0;
Pipeline g_p;
View g_view[kViews];
Stats g_st, g_total;
bool g_whyLogged[kWhys] = {};

// AfterDraw's work: a PS slot back to the game's view, the bounce into the ambient draw's target, the history from the
// resolve, the preview over it.
struct Pending {
    bool active = false;
    bool restore = false;
    UINT slot = 0;
    ID3D11ShaderResourceView* srv = nullptr;  // referenced
    int gi = -1;                              // view whose bounce goes into the bound target 0
    int capture = -1;                         // view whose history the bound target gives
    int preview = -1;                         // view whose AO (or bounce) the preview shows
    bool previewGi = false;
    UINT weightSlot = 0;                      // the ambient draw's weight slot (the probe reads it)
};
Pending g_pending;

// The probe of each probed view while the bounce is on: wanted every kProbeFrames frames, made at the view's next
// bounce, copied to the CPU copy and read a few frames later without waiting, logged at once with the settings it ran
// with (the tester may switch scenarios every few seconds).
constexpr uint64_t kProbeFrames = 270;  // about 3 s at 90 Hz
struct Probe {
    bool wanted = false;
    bool copied = false;
    uint64_t frame = 0;
    uint64_t next = 0;  // the frame from which the next one is wanted
    int gi = 0;
    float radius = 0, maxPx = 0, strength = 0;
    int steps = 0;
};
Probe g_probe[kProbes];
std::string g_testProbe;  // the smoke test's copy of the latest line

// Draw dump: frames left to dump, draws left after the latest overlay, its view.
int g_dumpFrames = 0;
int g_dumpDraws = 0;
int g_dumpView = -1;
uint64_t g_dumpLines = 0;  // the smoke test's count

const char* FormatName(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "RGBA8";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "RGBA8_SRGB";
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "RGBA8_TYPELESS";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "BGRA8";
        case DXGI_FORMAT_R16G16B16A16_UNORM: return "RGBA16";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "RGBA16F";
        case DXGI_FORMAT_R11G11B10_FLOAT: return "R11G11B10F";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "RGB10A2";
        case DXGI_FORMAT_R32_FLOAT: return "R32F";
        case DXGI_FORMAT_R16_FLOAT: return "R16F";
        case DXGI_FORMAT_R16_UNORM: return "R16";
        case DXGI_FORMAT_R8_UNORM: return "R8";
        case DXGI_FORMAT_R16G16_FLOAT: return "RG16F";
        case DXGI_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT: return "D32F";
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return "D24S8";
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return "D32FS8";
        default: return nullptr;
    }
}

// "fmt WxH[xN samples] @ptr" of a view's resource (2D textures; other kinds by name).
void DescribeView(ID3D11View* v, DXGI_FORMAT fmt, char* out, size_t n) {
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    ID3D11Texture2D* t = nullptr;
    if (r) r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    const char* name = FormatName(fmt);
    char f[24];
    if (!name) {
        snprintf(f, sizeof(f), "fmt%d", (int)fmt);
        name = f;
    }
    if (t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        if (d.SampleDesc.Count > 1)
            snprintf(out, n, "%s %ux%u x%u @%p", name, d.Width, d.Height, d.SampleDesc.Count, (void*)t);
        else
            snprintf(out, n, "%s %ux%u @%p", name, d.Width, d.Height, (void*)t);
    } else {
        snprintf(out, n, "%s (not 2D) @%p", name, (void*)r);
    }
    SafeRelease(t);
    SafeRelease(r);
}

// One line: the pixel shader, the render targets, t0..t7 of the bound pixel shader.
void DumpLine(ID3D11DeviceContext* c, const char* what, UINT count) {
    char line[1800];
    size_t len = 0;
    auto put = [&](const char* fmt, auto... a) {
        if (len < sizeof(line)) len += snprintf(line + len, sizeof(line) - len, fmt, a...);
    };
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    uint32_t sum[4] = {};
    if (ps && sunfx::PsChecksum(ps, sum))
        put("PS %08x-%08x-%08x-%08x", sum[0], sum[1], sum[2], sum[3]);
    else
        put("PS %p (checksum unknown)", (void*)ps);
    SafeRelease(ps);
    put(", %u vertices", count);
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
    for (int i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
        if (!rtv[i]) continue;
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        rtv[i]->GetDesc(&d);
        char b[96];
        DescribeView(rtv[i], d.Format, b, sizeof(b));
        put("; RT%d %s", i, b);
        SafeRelease(rtv[i]);
    }
    if (dsv) {
        D3D11_DEPTH_STENCIL_VIEW_DESC d{};
        dsv->GetDesc(&d);
        char b[96];
        DescribeView(dsv, d.Format, b, sizeof(b));
        put("; DS %s", b);
        SafeRelease(dsv);
    }
    ID3D11BlendState* bs = nullptr;
    float factor[4];
    UINT mask = 0;
    c->OMGetBlendState(&bs, factor, &mask);
    if (bs) {
        D3D11_BLEND_DESC bd{};
        bs->GetDesc(&bd);
        const D3D11_RENDER_TARGET_BLEND_DESC& r = bd.RenderTarget[0];
        put("; blend0 %s %d,%d op %d mask 0x%X", r.BlendEnable ? "on" : "off", (int)r.SrcBlend, (int)r.DestBlend,
            (int)r.BlendOp, (unsigned)r.RenderTargetWriteMask);
        SafeRelease(bs);
    } else {
        put("; blend default");
    }
    ID3D11ShaderResourceView* srv[8] = {};
    c->PSGetShaderResources(0, 8, srv);
    for (int i = 0; i < 8; ++i) {
        if (!srv[i]) continue;
        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        srv[i]->GetDesc(&d);
        char b[96];
        if (d.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D || d.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2DMS)
            DescribeView(srv[i], d.Format, b, sizeof(b));
        else
            snprintf(b, sizeof(b), "dimension %d fmt%d", (int)d.ViewDimension, (int)d.Format);
        put("; t%d %s", i, b);
        SafeRelease(srv[i]);
    }
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    if (nvp) put("; viewport %.0f,%.0f %.0fx%.0f", vp.TopLeftX, vp.TopLeftY, vp.Width, vp.Height);
    Log("SSGI dump (frame %llu, %s, %s): %s", (unsigned long long)g_frame,
        g_dumpView >= 0 ? kViewNames[g_dumpView] : "?", what, line);
    ++g_dumpLines;
}

// Render thread, frame start: a copied probe a few frames old read without waiting (still drawing: next frame).
void ReadProbes() {
    for (int i = 0; i < kProbes; ++i) {
        Probe& pr = g_probe[i];
        if (!pr.copied || !g_imm || !g_p.probeStaging[i] || pr.frame + 3 > g_frame) continue;
        D3D11_MAPPED_SUBRESOURCE m{};
        const HRESULT hr = g_imm->Map(g_p.probeStaging[i], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
        if (hr == DXGI_ERROR_WAS_STILL_DRAWING) continue;
        pr.copied = false;
        if (FAILED(hr)) continue;
        const float* f = (const float*)m.pData;
        double gi = 0, light = 0, hist = 0, n = 0, lit = 0, vGi = 0, vHist = 0, vn = 0;
        float giMax = 0;
        for (int g = 0; g < 64; ++g) {
            const float* a = f + g * 12;
            gi += a[0];
            light += a[1];
            hist += a[2];
            giMax = std::max(giMax, a[3]);
            n += a[4];
            lit += a[5];
            vGi += a[8];
            vHist += a[9];
            vn += a[10];
        }
        g_imm->Unmap(g_p.probeStaging[i], 0);
        if (n + vn <= 0) continue;
        const double nw = std::max(n, 1.0), nv = std::max(vn, 1.0);
        char b[640];
        snprintf(b, sizeof(b),
                 "SSGI GI probe (%s, frame %llu, gi=%d, reach %.1f m, max %.0f px, %d steps; luminance in the game's HDR "
                 "units, the bounce before its strength x %.1f): world %.0f samples: bounce mean %.4g, the game's "
                 "diffuse light before it %.4g (bounce / light %.3f), image %.4g, samples with a bounce over 1 %% of "
                 "the light %.1f %%; vehicles %.0f samples: bounce mean %.4g, image %.4g (bounce / image %.3f); "
                 "bounce max %.4g",
                 kViewNames[i ? 2 : 0], (unsigned long long)pr.frame, pr.gi, pr.radius, pr.maxPx, pr.steps,
                 pr.strength, n, gi / nw, light / nw, light > 0 ? gi / light : 0.0, hist / nw, 100 * lit / nw, vn,
                 vGi / nv, vHist / nv, vHist > 0 ? vGi / vHist : 0.0, giMax);
        Log("%s", b);
        g_testProbe = b;
    }
}
int ProbeSlot(int view) { return view == 0 ? 0 : view == 2 ? 1 : -1; }

// Views of textures this module does not own (the game's weight target, DLSS outputs, resolve targets), made once per
// texture and format. A view keeps its texture, so an address in the table is never another texture's.
struct SrvEntry {
    ID3D11Texture2D* tex = nullptr;  // only compared
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    int slice = -1;
    ID3D11ShaderResourceView* srv = nullptr;
};
SrvEntry g_srvs[12];
int g_srvNext = 0;

// PreviewInto: a render target view of each group's DLSS output (made once per texture).
struct PreviewTarget {
    ID3D11Texture2D* tex = nullptr;  // only compared
    ID3D11RenderTargetView* rtv = nullptr;
};
PreviewTarget g_previewTarget[kViews];

void Fail(const char* where, unsigned long code) {
    if (g_dead) return;
    g_dead = true;
    Log("SSGI AO: ERROR: exception 0x%08lX in %s; off for this session (the game's SSAO stays)", code, where);
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

void Count(Why w) {
    ++g_st.why[w];
    if (!g_whyLogged[w]) {
        g_whyLogged[w] = true;
        Log("SSGI AO: %s (counted from now on)", kWhyText[w]);
    }
}

void ReleaseView(View& v) {
    SafeRelease(v.rawSrv);
    SafeRelease(v.finSrv);
    SafeRelease(v.giSrv);
    SafeRelease(v.rawUav);
    SafeRelease(v.finUav);
    SafeRelease(v.giUav);
    SafeRelease(v.raw);
    SafeRelease(v.fin);
    SafeRelease(v.gi);
    SafeRelease(v.histSrv);
    SafeRelease(v.histUav);
    SafeRelease(v.hist);
    const UINT lw = v.loggedW, lh = v.loggedH, hlw = v.histLoggedW, hlh = v.histLoggedH;
    v = View{};
    v.loggedW = lw;
    v.loggedH = lh;
    v.histLoggedW = hlw;
    v.histLoggedH = hlh;
}

void ReleaseObjects() {
    for (View& v : g_view) ReleaseView(v);
    for (PreviewTarget& t : g_previewTarget) {
        SafeRelease(t.rtv);
        t.tex = nullptr;
    }
    for (SrvEntry& e : g_srvs) {
        SafeRelease(e.srv);
        e = SrvEntry{};
    }
    g_srvNext = 0;
    SafeRelease(g_p.ao);
    SafeRelease(g_p.blur);
    SafeRelease(g_p.history);
    SafeRelease(g_p.probe);
    SafeRelease(g_p.probeParams);
    for (int i = 0; i < kProbes; ++i) {
        SafeRelease(g_p.probeUav[i]);
        SafeRelease(g_p.probeBuf[i]);
        SafeRelease(g_p.probeStaging[i]);
        g_probe[i] = Probe{};
    }
    SafeRelease(g_p.vs);
    SafeRelease(g_p.ps);
    SafeRelease(g_p.giPs);
    SafeRelease(g_p.params);
    SafeRelease(g_p.histParams);
    SafeRelease(g_p.preview);
    SafeRelease(g_p.camsUav);
    SafeRelease(g_p.cams);
    SafeRelease(g_p.white);
    SafeRelease(g_p.point);
    SafeRelease(g_p.linear);
    SafeRelease(g_p.blend);
    SafeRelease(g_p.add);
    SafeRelease(g_p.dss);
    SafeRelease(g_p.rs);
    SafeRelease(g_p.dev);
    g_p = Pipeline{};
    SafeRelease(g_ctx1);
}

bool EnsurePipeline(ID3D11Device* dev) {
    if (g_p.dev != dev) {
        ReleaseObjects();
        g_p.dev = dev;
        dev->AddRef();
    }
    if (g_p.failed) return false;
    if (g_p.ao) return true;
    HRESULT hr = dev->CreateComputeShader(kSsgiAoCS, sizeof(kSsgiAoCS), nullptr, &g_p.ao);
    if (SUCCEEDED(hr)) hr = dev->CreateComputeShader(kSsgiBlurCS, sizeof(kSsgiBlurCS), nullptr, &g_p.blur);
    if (SUCCEEDED(hr)) hr = dev->CreateComputeShader(kSsgiHistoryCS, sizeof(kSsgiHistoryCS), nullptr, &g_p.history);
    if (SUCCEEDED(hr)) hr = dev->CreateComputeShader(kSsgiProbeCS, sizeof(kSsgiProbeCS), nullptr, &g_p.probe);
    if (SUCCEEDED(hr)) hr = dev->CreateVertexShader(kSsgiFullVS, sizeof(kSsgiFullVS), nullptr, &g_p.vs);
    if (SUCCEEDED(hr)) hr = dev->CreatePixelShader(kSsgiPreviewPS, sizeof(kSsgiPreviewPS), nullptr, &g_p.ps);
    if (SUCCEEDED(hr)) hr = dev->CreatePixelShader(kSsgiGiPS, sizeof(kSsgiGiPS), nullptr, &g_p.giPs);
    const struct {
        ID3D11Buffer** b;
        UINT size;
    } cbs[] = {{&g_p.params, 64}, {&g_p.histParams, 32}, {&g_p.preview, 32}, {&g_p.probeParams, 16}};
    for (const auto& cb : cbs) {
        if (FAILED(hr)) break;
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = cb.size;
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = dev->CreateBuffer(&d, nullptr, cb.b);
    }
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = kViews * 2 * kCamSize * 16;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        d.StructureByteStride = 16;
        hr = dev->CreateBuffer(&d, nullptr, &g_p.cams);
        if (SUCCEEDED(hr)) hr = dev->CreateUnorderedAccessView(g_p.cams, nullptr, &g_p.camsUav);
        for (int i = 0; i < kProbes && SUCCEEDED(hr); ++i) {
            d.ByteWidth = kProbeFloats * 4;
            hr = dev->CreateBuffer(&d, nullptr, &g_p.probeBuf[i]);
            if (SUCCEEDED(hr)) hr = dev->CreateUnorderedAccessView(g_p.probeBuf[i], nullptr, &g_p.probeUav[i]);
            if (SUCCEEDED(hr)) {
                D3D11_BUFFER_DESC sd = d;
                sd.Usage = D3D11_USAGE_STAGING;
                sd.BindFlags = 0;
                sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                hr = dev->CreateBuffer(&sd, nullptr, &g_p.probeStaging[i]);
            }
        }
    }
    if (SUCCEEDED(hr)) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = d.Height = 1;
        d.MipLevels = d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R8_UNORM;
        d.SampleDesc = {1, 0};
        d.Usage = D3D11_USAGE_IMMUTABLE;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        const uint8_t one = 255;
        const D3D11_SUBRESOURCE_DATA init = {&one, 1, 1};
        ID3D11Texture2D* t = nullptr;
        hr = dev->CreateTexture2D(&d, &init, &t);
        if (SUCCEEDED(hr)) hr = dev->CreateShaderResourceView(t, nullptr, &g_p.white);
        SafeRelease(t);
    }
    for (ID3D11SamplerState** s : {&g_p.point, &g_p.linear}) {
        if (FAILED(hr)) break;
        D3D11_SAMPLER_DESC sd{};
        sd.Filter = s == &g_p.point ? D3D11_FILTER_MIN_MAG_MIP_POINT : D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        hr = dev->CreateSamplerState(&sd, s);
    }
    if (SUCCEEDED(hr)) {
        D3D11_BLEND_DESC b{};
        b.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        hr = dev->CreateBlendState(&b, &g_p.blend);
        if (SUCCEEDED(hr)) {
            D3D11_RENDER_TARGET_BLEND_DESC& r = b.RenderTarget[0];
            r.BlendEnable = TRUE;
            r.SrcBlend = r.DestBlend = r.SrcBlendAlpha = r.DestBlendAlpha = D3D11_BLEND_ONE;
            r.BlendOp = r.BlendOpAlpha = D3D11_BLEND_OP_ADD;
            hr = dev->CreateBlendState(&b, &g_p.add);
        }
    }
    if (SUCCEEDED(hr)) {
        D3D11_DEPTH_STENCIL_DESC d{};
        d.DepthEnable = FALSE;
        d.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        d.DepthFunc = D3D11_COMPARISON_ALWAYS;
        hr = dev->CreateDepthStencilState(&d, &g_p.dss);
    }
    if (SUCCEEDED(hr)) {
        D3D11_RASTERIZER_DESC r{};
        r.FillMode = D3D11_FILL_SOLID;
        r.CullMode = D3D11_CULL_NONE;
        r.DepthClipEnable = TRUE;
        hr = dev->CreateRasterizerState(&r, &g_p.rs);
    }
    if (FAILED(hr)) {
        ReleaseObjects();
        g_p.failed = true;
        Log("SSGI AO: cannot create the shaders and states (0x%08lX); off for this device", (unsigned long)hr);
        return false;
    }
    Log("SSGI AO: pipeline ready (AO and bounce %zu B, blur %zu B, history %zu B)", sizeof(kSsgiAoCS),
        sizeof(kSsgiBlurCS), sizeof(kSsgiHistoryCS));
    return true;
}

// A texture with an unordered access view and a shader resource view.
bool MakeTexture(UINT w, UINT h, DXGI_FORMAT fmt, ID3D11Texture2D*& t, ID3D11UnorderedAccessView*& uav,
                 ID3D11ShaderResourceView*& srv) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = d.ArraySize = 1;
    d.Format = fmt;
    d.SampleDesc = {1, 0};
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    HRESULT hr = g_p.dev->CreateTexture2D(&d, nullptr, &t);
    if (SUCCEEDED(hr)) hr = g_p.dev->CreateUnorderedAccessView(t, nullptr, &uav);
    if (SUCCEEDED(hr)) hr = g_p.dev->CreateShaderResourceView(t, nullptr, &srv);
    if (SUCCEEDED(hr)) return true;
    SafeRelease(srv);
    SafeRelease(uav);
    SafeRelease(t);
    Log("SSGI AO: cannot create a %u x %u texture of format %d (0x%08lX)", w, h, (int)fmt, (unsigned long)hr);
    return false;
}

bool EnsureView(View& v, UINT w, UINT h) {
    if (v.raw && v.w == w && v.h == h) return true;
    if (v.failed && v.w == w && v.h == h) return false;
    // A new size: the AO textures and the bounce go; the history (the view's own size) and the cameras stay.
    SafeRelease(v.rawSrv);
    SafeRelease(v.finSrv);
    SafeRelease(v.giSrv);
    SafeRelease(v.rawUav);
    SafeRelease(v.finUav);
    SafeRelease(v.giUav);
    SafeRelease(v.raw);
    SafeRelease(v.fin);
    SafeRelease(v.gi);
    v.giFailed = false;
    v.w = w;
    v.h = h;
    v.failed = !MakeTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, v.raw, v.rawUav, v.rawSrv) ||
               !MakeTexture(w, h, DXGI_FORMAT_R16_FLOAT, v.fin, v.finUav, v.finSrv);
    if (v.failed) {
        SafeRelease(v.rawSrv);
        SafeRelease(v.rawUav);
        SafeRelease(v.raw);
    }
    return !v.failed;
}

bool EnsureGi(View& v) {
    if (v.gi) return true;
    if (v.giFailed) return false;
    v.giFailed = !MakeTexture(v.w, v.h, DXGI_FORMAT_R16G16B16A16_FLOAT, v.gi, v.giUav, v.giSrv);
    return !v.giFailed;
}

bool EnsureHistory(View& v, UINT w, UINT h) {
    if (v.hist && v.hw == w && v.hh == h) return true;
    if (v.histFailed && v.hw == w && v.hh == h) return false;
    SafeRelease(v.histSrv);
    SafeRelease(v.histUav);
    SafeRelease(v.hist);
    v.hw = w;
    v.hh = h;
    v.histFrame = 0;
    v.histFailed = !MakeTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, v.hist, v.histUav, v.histSrv);
    return !v.histFailed;
}

// The texture behind a 2D view of the expected format (null otherwise), with its size.
ID3D11Texture2D* Texture2DOf(ID3D11ShaderResourceView* v, DXGI_FORMAT fmt, UINT& w, UINT& h) {
    if (!v) return nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    v->GetDesc(&sd);
    if (sd.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || sd.Format != fmt || sd.Texture2D.MostDetailedMip != 0)
        return nullptr;
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    ID3D11Texture2D* t = nullptr;
    if (r) r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    SafeRelease(r);
    if (!t) return nullptr;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    w = d.Width;
    h = d.Height;
    t->Release();  // the view keeps it
    return t;
}

// A typed format to read a texture through (its own, or the float / unorm one of a typeless one).
DXGI_FORMAT ReadFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        default: return f;
    }
}

// A shader resource view of a single-sample 2D texture this module does not own (null: it cannot be read): a 2D view
// (slice -1), or a 2D array view of one slice (the weight; also of a plain texture, slice 0).
ID3D11ShaderResourceView* ForeignSrv(ID3D11Texture2D* t, DXGI_FORMAT fmt, int slice = -1) {
    for (SrvEntry& e : g_srvs)
        if (e.tex == t && e.fmt == fmt && e.slice == slice) return e.srv;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    ID3D11ShaderResourceView* srv = nullptr;
    if ((d.BindFlags & D3D11_BIND_SHADER_RESOURCE) && d.SampleDesc.Count == 1 &&
        (slice < 0 ? d.ArraySize == 1 : (UINT)slice < d.ArraySize)) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = fmt;
        if (slice < 0) {
            sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D.MipLevels = 1;
        } else {
            sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sd.Texture2DArray.MipLevels = 1;
            sd.Texture2DArray.FirstArraySlice = (UINT)slice;
            sd.Texture2DArray.ArraySize = 1;
        }
        g_p.dev->CreateShaderResourceView(t, &sd, &srv);
    }
    SrvEntry& e = g_srvs[g_srvNext];
    g_srvNext = (g_srvNext + 1) % (int)(sizeof(g_srvs) / sizeof(g_srvs[0]));
    SafeRelease(e.srv);
    e.tex = t;
    e.fmt = fmt;
    e.slice = slice;
    e.srv = srv;  // null is remembered too: no new attempt per frame
    return srv;
}

// The texture and format behind the bound render target 0 (null: none or not a 2D texture view).
ID3D11Texture2D* BoundTarget(ID3D11DeviceContext* c, DXGI_FORMAT& fmt) {
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    if (!rtv) return nullptr;
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    rtv->GetDesc(&rd);
    ID3D11Resource* r = nullptr;
    rtv->GetResource(&r);
    rtv->Release();
    ID3D11Texture2D* t = nullptr;
    if (r && rd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D && rd.Texture2D.MipSlice == 0)
        r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    SafeRelease(r);
    if (!t) return nullptr;
    fmt = rd.Format;
    t->Release();  // the bound view keeps it
    return t;
}

// The overlay's target 0 as the weight (R8G8B8A8_UNORM, a 2D texture or one slice of an array), read through a 2D
// array view. Null: not readable; its details logged once.
ID3D11ShaderResourceView* WeightSrv(ID3D11DeviceContext* c) {
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    if (!rtv) return nullptr;
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    rtv->GetDesc(&rd);
    ID3D11Resource* r = nullptr;
    rtv->GetResource(&r);
    rtv->Release();
    ID3D11Texture2D* t = nullptr;
    if (r) r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    SafeRelease(r);
    int slice = -1;
    if (rd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D && rd.Texture2D.MipSlice == 0)
        slice = 0;
    else if (rd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY && rd.Texture2DArray.MipSlice == 0 &&
             rd.Texture2DArray.ArraySize == 1)
        slice = (int)rd.Texture2DArray.FirstArraySlice;
    ID3D11ShaderResourceView* srv =
        t && slice >= 0 && rd.Format == DXGI_FORMAT_R8G8B8A8_UNORM ? ForeignSrv(t, rd.Format, slice) : nullptr;
    static bool logged = false;
    if (!srv && !logged) {
        logged = true;
        D3D11_TEXTURE2D_DESC d{};
        if (t) t->GetDesc(&d);
        Log("SSGI AO: the overlay's target 0 is not readable as the weight: view dimension %d, format %d, slice %d; "
            "texture %s %ux%u, format %d, %u slices, %u samples, bind 0x%X", (int)rd.ViewDimension, (int)rd.Format,
            slice, t ? "2D" : "not 2D", d.Width, d.Height, (int)d.Format, d.ArraySize, d.SampleDesc.Count,
            d.BindFlags);
    }
    SafeRelease(t);  // the bound view keeps it
    return srv;
}

int ViewOf(int group) {
    const int e = mvrpass::EarlyEye();
    if (e >= 0) return 2 + e;
    return group >= 0 && group < kViews ? group : -1;
}

void SetPending(ID3D11DeviceContext* c, UINT slot, ID3D11ShaderResourceView* ours) {
    ID3D11ShaderResourceView* game = nullptr;
    c->PSGetShaderResources(slot, 1, &game);
    g_pending.active = true;
    g_pending.restore = true;
    g_pending.slot = slot;
    g_pending.srv = game;  // the reference PSGetShaderResources gave
    c->PSSetShaderResources(slot, 1, &ours);
}

// The compute shader's slots, saved and put back around this module's passes.
constexpr UINT kCsCbSlots[5] = {kCbGlobal, kCbParams, kCbHist, kCbProbe, kCbCommon};
struct CsState {
    ID3D11ComputeShader* cs = nullptr;
    ID3D11ShaderResourceView* srv[kCsSrvs] = {};
    ID3D11UnorderedAccessView* uav[kCsUavs] = {};
    ID3D11SamplerState* sampler[2] = {};
    ID3D11Buffer* cb[5] = {};
    UINT first[5] = {}, num[5] = {};
};

void SaveCs(ID3D11DeviceContext* c, CsState& s) {
    c->CSGetShader(&s.cs, nullptr, nullptr);
    c->CSGetShaderResources(0, kCsSrvs, s.srv);
    c->CSGetUnorderedAccessViews(0, kCsUavs, s.uav);
    c->CSGetSamplers(0, 2, s.sampler);
    for (int i = 0; i < 5; ++i) {
        if (g_ctx1)
            g_ctx1->CSGetConstantBuffers1(kCsCbSlots[i], 1, &s.cb[i], &s.first[i], &s.num[i]);
        else
            c->CSGetConstantBuffers(kCsCbSlots[i], 1, &s.cb[i]);
    }
}

void RestoreCs(ID3D11DeviceContext* c, CsState& s) {
    ID3D11ShaderResourceView* noSrv[kCsSrvs] = {};
    c->CSSetShaderResources(0, kCsSrvs, noSrv);  // ours off first: a game view may be one of our UAVs' textures
    c->CSSetShader(s.cs, nullptr, 0);
    UINT keep[kCsUavs];
    for (UINT& k : keep) k = (UINT)-1;
    c->CSSetUnorderedAccessViews(0, kCsUavs, s.uav, keep);
    c->CSSetShaderResources(0, kCsSrvs, s.srv);
    c->CSSetSamplers(0, 2, s.sampler);
    for (int i = 0; i < 5; ++i) {
        if (g_ctx1 && s.cb[i] && s.num[i])
            g_ctx1->CSSetConstantBuffers1(kCsCbSlots[i], 1, &s.cb[i], &s.first[i], &s.num[i]);
        else
            c->CSSetConstantBuffers(kCsCbSlots[i], 1, &s.cb[i]);
    }
    SafeRelease(s.cs);
    for (auto*& v : s.srv) SafeRelease(v);
    for (auto*& v : s.uav) SafeRelease(v);
    for (auto*& v : s.sampler) SafeRelease(v);
    for (auto*& b : s.cb) SafeRelease(b);
}

// The render targets, saved and put back around a pass that reads one of them or draws into another set.
struct OmState {
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
};

void SaveOm(ID3D11DeviceContext* c, OmState& s) {
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, s.rtv, &s.dsv);
}

void RestoreOm(ID3D11DeviceContext* c, OmState& s) {
    c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, s.rtv, s.dsv);
    for (auto*& v : s.rtv) SafeRelease(v);
    SafeRelease(s.dsv);
}

// One of the game's PS constant windows bound to the same CS slot (false: nothing bound).
bool BindWindow(ID3D11DeviceContext* c, UINT slot) {
    ID3D11Buffer* b = nullptr;
    UINT first = 0, num = 0;
    g_ctx1->PSGetConstantBuffers1(slot, 1, &b, &first, &num);
    if (!b) return false;
    if (num)
        g_ctx1->CSSetConstantBuffers1(slot, 1, &b, &first, &num);
    else
        c->CSSetConstantBuffers(slot, 1, &b);
    b->Release();
    return true;
}

template <class T>
bool Upload(ID3D11DeviceContext* c, ID3D11Buffer* b, const T& data) {
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(c->Map(b, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
    memcpy(m.pData, &data, sizeof(data));
    c->Unmap(b, 0);
    return true;
}

// The overlay draw: own AO (and the bounce) of its view, the AO bound in place of the game's SSAO.
bool Overlay(ID3D11DeviceContext* c, int group) {
    const int v = ViewOf(group);
    if (g_dumpFrames > 0) {
        g_dumpView = v;
        DumpLine(c, "overlay", 0);
        g_dumpDraws = 8;
    }
    if (v < 0) {
        Count(WhyGroup);
        return false;
    }
    ID3D11ShaderResourceView* srv[2] = {};
    c->PSGetShaderResources(kOverlayNormals, 2, srv);
    UINT nw = 0, nh = 0, dw = 0, dh = 0;
    ID3D11Texture2D* normals = Texture2DOf(srv[0], DXGI_FORMAT_R16G16B16A16_UNORM, nw, nh);
    ID3D11Texture2D* depth = Texture2DOf(srv[1], DXGI_FORMAT_R32_FLOAT, dw, dh);
    bool ok = false;
    View& view = g_view[v];
    if (!normals || !depth || nw != dw || nh != dh) {
        Count(WhyBindings);
    } else if (!g_ctx1) {
        Count(WhyWindows);
    } else if (!EnsureView(view, nw, nh)) {
        Count(WhyTexture);
    } else {
        D3D11_VIEWPORT vp{};
        UINT nvp = 1;
        c->RSGetViewports(&nvp, &vp);
        const UINT gx = (UINT)std::min<float>((float)nw, std::ceil(vp.TopLeftX + vp.Width));
        const UINT gy = (UINT)std::min<float>((float)nh, std::ceil(vp.TopLeftY + vp.Height));
        // The bounce: the deferred-ambient weight (the draw's target, read with the targets unbound), the history and
        // the camera of the previous frame.
        const UINT parity = (UINT)(g_frame & 1);
        bool gi = false;
        ID3D11ShaderResourceView* weight = nullptr;
        if (g_gi) {
            weight = WeightSrv(c);
            if (!weight) {
                Count(WhyWeight);
            } else if (!EnsureGi(view)) {
                Count(WhyTexture);
            } else if (view.camFrame[parity ^ 1] + 1 != g_frame || view.histFrame + 1 != g_frame || !view.histSrv) {
                Count(WhyHistory);
            } else {
                gi = true;
            }
        }
        CsState saved;
        SaveCs(c, saved);
        if (!nvp || !gx || !gy || !BindWindow(c, kCbGlobal) || !BindWindow(c, kCbCommon)) {
            Count(WhyWindows);
        } else {
            const Config& cf = Cfg();
            struct {
                float radius, thickness, power, maxRadiusPx;
                uint32_t slices, steps;
                float blurDepth;
                uint32_t giOn, camCur, camPrev;
                float giClamp, giVehicle;
                float pad[4];
            } p = {cf.ssgiRadius, cf.ssgiThickness, cf.ssgiPower, cf.ssgiMaxRadiusPx, (uint32_t)cf.ssgiSlices,
                   (uint32_t)cf.ssgiSteps, cf.ssgiBlurDepth, gi ? 1u : 0u, (uint32_t)(v * 2 + parity),
                   (uint32_t)(v * 2 + (parity ^ 1)), cf.ssgiGiClamp, cf.ssgiGiVehicle, {}};
            static_assert(sizeof(p) == 64, "Params");
            if (Upload(c, g_p.params, p)) {
                OmState om;
                if (gi) {  // the weight texture is this draw's target: off the output while the CS reads it
                    SaveOm(c, om);
                    c->OMSetRenderTargets(0, nullptr, om.dsv);
                }
                gpu::SpanBegin((gpu::Span)(gpu::AoOuterL + v));
                c->CSSetConstantBuffers(kCbParams, 1, &g_p.params);
                ID3D11SamplerState* smp[2] = {g_p.point, g_p.linear};
                c->CSSetSamplers(0, 2, smp);
                ID3D11UnorderedAccessView* u1[kCsUavs] = {nullptr, g_p.camsUav, nullptr, view.rawUav, nullptr};
                c->CSSetUnorderedAccessViews(0, kCsUavs, u1, nullptr);
                ID3D11ShaderResourceView* in[kCsSrvs] = {srv[1], srv[0], nullptr, nullptr,
                                                         gi ? weight : nullptr, gi ? view.histSrv : nullptr};
                c->CSSetShaderResources(0, kCsSrvs, in);
                c->CSSetShader(g_p.ao, nullptr, 0);
                c->Dispatch((gx + 7) / 8, (gy + 7) / 8, 1);
                ID3D11UnorderedAccessView* u2[kCsUavs] = {view.finUav, nullptr, gi ? view.giUav : nullptr, nullptr,
                                                          nullptr};
                c->CSSetUnorderedAccessViews(0, kCsUavs, u2, nullptr);  // raw's UAV off before its SRV goes on
                c->CSSetShaderResources(kTRaw, 1, &view.rawSrv);
                c->CSSetShader(g_p.blur, nullptr, 0);
                c->Dispatch((gx + 7) / 8, (gy + 7) / 8, 1);
                ID3D11UnorderedAccessView* noUav[kCsUavs] = {};
                c->CSSetUnorderedAccessViews(0, kCsUavs, noUav, nullptr);
                ID3D11ShaderResourceView* none[kCsSrvs] = {};
                c->CSSetShaderResources(0, kCsSrvs, none);
                gpu::SpanEnd((gpu::Span)(gpu::AoOuterL + v));
                if (gi) RestoreOm(c, om);
                ok = true;
            } else {
                Count(WhyTexture);
            }
        }
        RestoreCs(c, saved);
        if (ok) {
            view.frame = g_frame;
            view.camFrame[parity] = g_frame;
            view.normals = normals;
            view.vpW = vp.Width;
            view.vpH = vp.Height;
            ++g_st.passes[v];
            if (gi) {
                view.giFrame = g_frame;
                ++g_st.giPasses[v];
            }
            if (view.loggedW != nw || view.loggedH != nh) {
                view.loggedW = nw;
                view.loggedH = nh;
                Log("SSGI AO: %s: own AO at %u x %u (the G-buffer's size), overlay viewport %.0f,%.0f %.0f x %.0f%s",
                    kViewNames[v], nw, nh, vp.TopLeftX, vp.TopLeftY, vp.Width, vp.Height,
                    mvrpass::EarlyEye() >= 0 ? " (focus's SSAO early in outer's group)" : "");
            }
            SetPending(c, kOverlaySsao, g_mode == 2 ? g_p.white : view.finSrv);
        }
    }
    for (auto*& s : srv) SafeRelease(s);
    return ok;
}

// The ambient draw: the AO its view's overlay made, if that view's normals are the ones it reads; AfterDraw adds the
// view's bounce into the diffuse light target.
bool Ambient(ID3D11DeviceContext* c, int group, const AmbientSlots& slots) {
    const int v = ViewOf(group);
    if (v < 0) {
        Count(WhyGroup);
        return false;
    }
    ID3D11ShaderResourceView* n = nullptr;
    c->PSGetShaderResources(slots.normals, 1, &n);
    UINT w = 0, h = 0;
    ID3D11Texture2D* normals = Texture2DOf(n, DXGI_FORMAT_R16G16B16A16_UNORM, w, h);
    SafeRelease(n);
    const View& view = g_view[v];
    if (!normals || view.frame != g_frame || view.normals != normals || !view.finSrv) {
        Count(WhyAmbient);
        return false;
    }
    SetPending(c, slots.ssao, view.finSrv);
    g_pending.weightSlot = slots.weight;
    if (g_gi && view.giFrame == g_frame && view.giSrv) g_pending.gi = v;
    ++g_st.ambient;
    return true;
}

// The resolve draw: AfterDraw takes the history from it and draws the preview, for the view whose overlay viewport has
// this size.
bool Resolve(ID3D11DeviceContext* c, int group) {
    const bool previewAo = g_mode == 3, previewGi = g_gi == 2, capture = g_gi != 0;
    if (!previewAo && !previewGi && !capture) return false;
    const int v = ViewOf(group);
    if (v < 0) {
        Count(WhyGroup);
        return false;
    }
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    // The resolve may run in another group than the view's overlay (focus's forward pass from outer's group, B9d):
    // the view itself first, then the other kind of the same eye.
    for (int cand : {v, v ^ 2}) {
        const View& view = g_view[cand];
        if (nvp && view.frame == g_frame && view.finSrv && std::fabs(view.vpW - vp.Width) < 0.5f &&
            std::fabs(view.vpH - vp.Height) < 0.5f) {
            g_pending.active = true;
            if (capture) g_pending.capture = cand;
            if (previewGi && view.giFrame == g_frame && view.giSrv) {
                g_pending.preview = cand;
                g_pending.previewGi = true;
            } else if (previewAo) {
                g_pending.preview = cand;
            }
            return true;
        }
    }
    Count(WhyPreview);
    return false;
}

// A full-screen triangle with `ps` over the bound target, through the bound viewport, reading `srv` at `slot` over the
// view's region; the game's state back afterwards.
void FullScreen(ID3D11DeviceContext* c, const View& view, ID3D11PixelShader* ps, UINT slot,
                ID3D11ShaderResourceView* srv, ID3D11BlendState* blend, float level, float alpha) {
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    if (!nvp || vp.Width <= 0 || vp.Height <= 0) return;
    const float k[8] = {vp.TopLeftX, vp.TopLeftY, 1 / vp.Width, 1 / vp.Height, view.vpW, view.vpH, level, alpha};
    if (!Upload(c, g_p.preview, k)) return;
    ID3D11InputLayout* il = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY topo{};
    ID3D11VertexShader* vs = nullptr;
    ID3D11HullShader* hs = nullptr;
    ID3D11DomainShader* ds = nullptr;
    ID3D11GeometryShader* gs = nullptr;
    ID3D11PixelShader* gps = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11ShaderResourceView* gsrv = nullptr;
    ID3D11BlendState* bs = nullptr;
    float factor[4] = {};
    UINT mask = 0, ref = 0;
    ID3D11DepthStencilState* dss = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    c->IAGetInputLayout(&il);
    c->IAGetPrimitiveTopology(&topo);
    c->VSGetShader(&vs, nullptr, nullptr);
    c->HSGetShader(&hs, nullptr, nullptr);
    c->DSGetShader(&ds, nullptr, nullptr);
    c->GSGetShader(&gs, nullptr, nullptr);
    c->PSGetShader(&gps, nullptr, nullptr);
    c->PSGetConstantBuffers(2, 1, &cb);
    c->PSGetShaderResources(slot, 1, &gsrv);
    c->OMGetBlendState(&bs, factor, &mask);
    c->OMGetDepthStencilState(&dss, &ref);
    c->RSGetState(&rs);

    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(g_p.vs, nullptr, 0);
    c->HSSetShader(nullptr, nullptr, 0);
    c->DSSetShader(nullptr, nullptr, 0);
    c->GSSetShader(nullptr, nullptr, 0);
    c->PSSetShader(ps, nullptr, 0);
    c->PSSetConstantBuffers(2, 1, &g_p.preview);
    c->PSSetShaderResources(slot, 1, &srv);
    const float one[4] = {1, 1, 1, 1};
    c->OMSetBlendState(blend, one, 0xFFFFFFFF);
    c->OMSetDepthStencilState(g_p.dss, 0);
    c->RSSetState(g_p.rs);
    sunfx::DrawUnhooked(c, 3, 0);

    c->IASetInputLayout(il);
    c->IASetPrimitiveTopology(topo);
    c->VSSetShader(vs, nullptr, 0);
    c->HSSetShader(hs, nullptr, 0);
    c->DSSetShader(ds, nullptr, 0);
    c->GSSetShader(gs, nullptr, 0);
    c->PSSetShader(gps, nullptr, 0);
    c->PSSetConstantBuffers(2, 1, &cb);
    c->PSSetShaderResources(slot, 1, &gsrv);
    c->OMSetBlendState(bs, factor, mask);
    c->OMSetDepthStencilState(dss, ref);
    c->RSSetState(rs);
    SafeRelease(il);
    SafeRelease(vs);
    SafeRelease(hs);
    SafeRelease(ds);
    SafeRelease(gs);
    SafeRelease(gps);
    SafeRelease(cb);
    SafeRelease(gsrv);
    SafeRelease(bs);
    SafeRelease(dss);
    SafeRelease(rs);
}

// The preview: the view's AO as grey (or its bounce) over the bound target.
void Preview(ID3D11DeviceContext* c, int v, bool gi) {
    const View& view = g_view[v];
    if (gi)
        FullScreen(c, view, g_p.giPs, kTGi, view.giSrv, g_p.blend, Cfg().ssgiGiPreviewLevel, 1);
    else
        FullScreen(c, view, g_p.ps, kTAo, view.finSrv, g_p.blend, Cfg().ssgiPreviewLevel, 1);
    ++g_st.previews;
}

// After the ambient draw: the view's bounce added into the diffuse light target (render target 0 alone; the ambient
// draw's second target, the specular light, untouched). The forward pass multiplies it by the albedo.
void RunProbe(ID3D11DeviceContext* c, int v, int slot, ID3D11RenderTargetView* light) {
    const View& view = g_view[v];
    ID3D11Resource* r = nullptr;
    light->GetResource(&r);
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    light->GetDesc(&rd);
    ID3D11Texture2D* t = nullptr;
    if (r) r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    SafeRelease(r);
    ID3D11ShaderResourceView* lightSrv = t ? ForeignSrv(t, ReadFormat(rd.Format)) : nullptr;
    SafeRelease(t);  // the bound view keeps it
    const float k[4] = {view.vpW, view.vpH, 0, 0};
    ID3D11ShaderResourceView* weight = nullptr;  // referenced
    c->PSGetShaderResources(g_pending.weightSlot, 1, &weight);
    if (!lightSrv || !view.histSrv || !weight || !Upload(c, g_p.probeParams, k)) {
        SafeRelease(weight);
        return;
    }
    c->OMSetRenderTargets(0, nullptr, nullptr);  // the light target is read
    CsState saved;
    SaveCs(c, saved);
    c->CSSetConstantBuffers(kCbProbe, 1, &g_p.probeParams);
    c->CSSetSamplers(1, 1, &g_p.linear);
    ID3D11UnorderedAccessView* u[kCsUavs] = {nullptr, nullptr, nullptr, nullptr, nullptr, g_p.probeUav[slot]};
    c->CSSetUnorderedAccessViews(0, kCsUavs, u, nullptr);
    ID3D11ShaderResourceView* in[4] = {view.giSrv, lightSrv, view.histSrv, weight};
    c->CSSetShaderResources(kTProbe, 4, in);
    c->CSSetShader(g_p.probe, nullptr, 0);
    c->Dispatch(8, 8, 1);
    ID3D11UnorderedAccessView* noUav[kCsUavs] = {};
    c->CSSetUnorderedAccessViews(0, kCsUavs, noUav, nullptr);
    RestoreCs(c, saved);
    SafeRelease(weight);
    c->CopyResource(g_p.probeStaging[slot], g_p.probeBuf[slot]);
    Probe& pr = g_probe[slot];
    const Config& cf = Cfg();
    pr.wanted = false;
    pr.copied = true;
    pr.frame = g_frame;
    pr.gi = g_gi;
    pr.radius = cf.ssgiRadius;
    pr.maxPx = cf.ssgiMaxRadiusPx;
    pr.steps = cf.ssgiSteps;
    pr.strength = cf.ssgiGiStrength;
}

void Composite(ID3D11DeviceContext* c, int v) {
    const View& view = g_view[v];
    OmState om;
    SaveOm(c, om);
    const int slot = ProbeSlot(v);
    if (om.rtv[0] && slot >= 0 && g_probe[slot].wanted && !g_probe[slot].copied) RunProbe(c, v, slot, om.rtv[0]);
    if (om.rtv[0]) {
        c->OMSetRenderTargets(1, &om.rtv[0], nullptr);
        FullScreen(c, view, g_p.giPs, kTGi, view.giSrv, g_p.add, Cfg().ssgiGiStrength, 0);
        ++g_st.composites;
    }
    RestoreOm(c, om);
}

// The history of view v: the rectangle (x, y, w, h) of `src` (its final image of this frame) at half the view's size.
void Capture(ID3D11DeviceContext* c, int v, ID3D11Texture2D* src, DXGI_FORMAT fmt, float x, float y, float w,
             float h, const char* what) {
    View& view = g_view[v];
    if (view.frame != g_frame || w < 1 || h < 1) return;  // no own AO in this frame: no camera to go with it
    const UINT hw = std::max(1u, (UINT)std::ceil(view.vpW * 0.5f)), hh = std::max(1u, (UINT)std::ceil(view.vpH * 0.5f));
    ID3D11ShaderResourceView* srv = ForeignSrv(src, ReadFormat(fmt));
    if (!srv) {
        Count(WhyHistSource);
        return;
    }
    if (!EnsureHistory(view, hw, hh)) {
        Count(WhyTexture);
        return;
    }
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    const float k[8] = {x, y, w, h, 1.0f / d.Width, 1.0f / d.Height, 0, 0};
    if (!Upload(c, g_p.histParams, k)) return;
    OmState om;
    SaveOm(c, om);  // the source may be a bound target
    c->OMSetRenderTargets(0, nullptr, nullptr);
    CsState saved;
    SaveCs(c, saved);
    gpu::SpanBegin((gpu::Span)(gpu::GiHistOuterL + v));
    c->CSSetConstantBuffers(kCbHist, 1, &g_p.histParams);
    c->CSSetSamplers(1, 1, &g_p.linear);
    ID3D11UnorderedAccessView* u[kCsUavs] = {nullptr, nullptr, nullptr, nullptr, view.histUav};
    c->CSSetUnorderedAccessViews(0, kCsUavs, u, nullptr);
    c->CSSetShaderResources(kTHistSrc, 1, &srv);
    c->CSSetShader(g_p.history, nullptr, 0);
    c->Dispatch((hw + 7) / 8, (hh + 7) / 8, 1);
    ID3D11UnorderedAccessView* noUav[kCsUavs] = {};
    c->CSSetUnorderedAccessViews(0, kCsUavs, noUav, nullptr);
    gpu::SpanEnd((gpu::Span)(gpu::GiHistOuterL + v));
    RestoreCs(c, saved);
    RestoreOm(c, om);
    view.histFrame = g_frame;
    ++g_st.captures[v];
    if (view.histLoggedW != hw || view.histLoggedH != hh) {
        view.histLoggedW = hw;
        view.histLoggedH = hh;
        Log("SSGI AO: %s: history %u x %u from %s %u x %u (format %d, rectangle %.0f,%.0f %.0f x %.0f)", kViewNames[v],
            hw, hh, what, d.Width, d.Height, (int)fmt, x, y, w, h);
    }
}

// After the resolve draw: its target through its viewport.
void CaptureResolve(ID3D11DeviceContext* c, int v) {
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    ID3D11Texture2D* t = BoundTarget(c, fmt);
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    if (!t || !nvp) {
        Count(WhyHistSource);
        return;
    }
    Capture(c, v, t, fmt, vp.TopLeftX, vp.TopLeftY, vp.Width, vp.Height, "the MSAA resolve");
}

void AddStats(Stats& a, const Stats& b) {
    a.frames += b.frames;
    for (int v = 0; v < kViews; ++v) {
        a.passes[v] += b.passes[v];
        a.giPasses[v] += b.giPasses[v];
        a.captures[v] += b.captures[v];
    }
    a.ambient += b.ambient;
    a.composites += b.composites;
    a.previews += b.previews;
    for (int w = 0; w < kWhys; ++w) a.why[w] += b.why[w];
}

void LogStatsOf(const Stats& s, const char* what) {
    const double f = s.frames ? (double)s.frames : 1;
    char why[640] = "";
    size_t len = 0;
    const char* const shortWhy[kWhys] = {"outside groups", "bindings",           "constant windows",
                                         "textures",       "ambient without AO", "resolve without AO",
                                         "no weight",      "no history",         "unreadable image"};
    for (int w = 0; w < kWhys; ++w)
        if (s.why[w] && len < sizeof(why))
            len += snprintf(why + len, sizeof(why) - len, "%s%s %llu", len ? ", " : "", shortWhy[w],
                            (unsigned long long)s.why[w]);
    Log("SSGI AO stats%s: %s, ao=%d gi=%d; frames %llu; passes per frame (with the bounce, history copies): outer L "
        "%.2f (%.2f, %.2f), outer R %.2f (%.2f, %.2f), focus L %.2f (%.2f, %.2f), focus R %.2f (%.2f, %.2f); ambient "
        "draws with own AO %.2f per frame, with the bounce added %.2f; previews %llu; without: %s",
        what, g_dead ? "STOPPED" : g_mode ? "on" : "off", g_mode, g_gi, (unsigned long long)s.frames,
        s.passes[0] / f, s.giPasses[0] / f, s.captures[0] / f, s.passes[1] / f, s.giPasses[1] / f, s.captures[1] / f,
        s.passes[2] / f, s.giPasses[2] / f, s.captures[2] / f, s.passes[3] / f, s.giPasses[3] / f, s.captures[3] / f,
        s.ambient / f, s.composites / f, (unsigned long long)s.previews, len ? why : "none");
}

}  // namespace

void OnFrameStart(ID3D11DeviceContext* ctx, bool allowed) {
    Protect("OnFrameStart", [&] {
        if (g_pending.active) {  // a draw whose AfterDraw never came (cannot happen; keeps the reference count right)
            SafeRelease(g_pending.srv);
            g_pending = Pending{};
        }
        const uint32_t gen = gpu::Generation();
        if (gen != g_gen && !g_testing) {
            g_gen = gen;
            ReleaseObjects();
        }
        ++g_frame;
        int mode = g_testing ? g_testMode : Cfg().ssgiAo;
        if (!allowed || !ctx || g_dead) mode = 0;
        if (ctx != g_imm) SafeRelease(g_ctx1);
        g_imm = ctx;
        if (mode) {
            ID3D11Device* dev = nullptr;
            ctx->GetDevice(&dev);
            if (!dev || !EnsurePipeline(dev)) mode = 0;
            SafeRelease(dev);
            if (mode && !g_ctx1) ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
        }
        const int gi = mode ? (g_testing ? g_testGi : Cfg().ssgiGi) : 0;
        if (mode != g_modeLogged || gi != g_giLogged) {
            g_modeLogged = mode;
            g_giLogged = gi;
            const Config& c = Cfg();
            Log("SSGI AO: ao=%d gi=%d%s (radius %.2f m, thickness %.2f m, power %.2f, %d slices x %d steps, max %.0f "
                "px, blur depth %.3f; bounce strength %.2f, on vehicles x %.2f, luminance limit %.3g)",
                mode, gi, g_dead ? " (stopped by a fault)" : "", c.ssgiRadius, c.ssgiThickness, c.ssgiPower,
                c.ssgiSlices, c.ssgiSteps, c.ssgiMaxRadiusPx, c.ssgiBlurDepth, c.ssgiGiStrength, c.ssgiGiVehicle,
                c.ssgiGiClamp);
            if (!mode && (g_testing ? g_testGi : Cfg().ssgiGi))
                Log("SSGI AO: gi needs own AO (ao 1 to 3): no bounce with ao=0");
        }
        if (mode && (mode != g_mode || gi != g_gi)) g_dumpFrames = 2;  // the draw dump after a change to on
        else if (g_dumpFrames > 0) --g_dumpFrames;
        g_dumpDraws = 0;
        g_mode = mode;
        g_gi = gi;
        if (mode) ++g_st.frames;
        if (mode) ReadProbes();
        if (gi)
            for (Probe& pr : g_probe)
                if (!pr.wanted && !pr.copied && g_frame >= pr.next) {
                    pr.wanted = true;
                    pr.next = g_frame + kProbeFrames;
                }
    });
}

void CaptureHistory(ID3D11DeviceContext* c, int group, ID3D11Texture2D* target) {
    if (!g_gi || c != g_imm || g_dead || !target || group < 0 || group >= kViews) return;
    Protect("CaptureHistory", [&] {
        D3D11_TEXTURE2D_DESC d{};
        target->GetDesc(&d);
        Capture(c, group, target, d.Format, 0, 0, (float)d.Width, (float)d.Height, "the DLSS output");
    });
}

void PreviewInto(ID3D11DeviceContext* c, int group, ID3D11Texture2D* target) {
    if ((g_mode != 3 && g_gi != 2) || c != g_imm || g_dead || !target || group < 0 || group >= kViews) return;
    Protect("PreviewInto", [&] {
        // The view's own AO; a focus view whose SSAO ran in outer's group is still its own slot (ViewOf).
        const View& view = g_view[group];
        const bool gi = g_gi == 2 && view.giFrame == g_frame && view.giSrv;
        if (view.frame != g_frame || !view.finSrv || (!gi && g_mode != 3)) {
            Count(WhyPreview);
            return;
        }
        PreviewTarget& pt = g_previewTarget[group];
        if (pt.tex != target) {
            SafeRelease(pt.rtv);
            pt.tex = target;
            D3D11_TEXTURE2D_DESC d{};
            target->GetDesc(&d);
            if ((d.BindFlags & D3D11_BIND_RENDER_TARGET) && d.SampleDesc.Count == 1 && d.ArraySize == 1)
                g_p.dev->CreateRenderTargetView(target, nullptr, &pt.rtv);
            if (!pt.rtv)
                Log("SSGI AO: no preview into the %s DLSS output (format %d, bind 0x%X, %u samples)", kViewNames[group],
                    (int)d.Format, d.BindFlags, d.SampleDesc.Count);
        }
        if (!pt.rtv) return;
        D3D11_TEXTURE2D_DESC d{};
        target->GetDesc(&d);
        OmState om;
        SaveOm(c, om);
        D3D11_VIEWPORT vps[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
        UINT nvp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        c->RSGetViewports(&nvp, vps);
        c->OMSetRenderTargets(1, &pt.rtv, nullptr);
        const D3D11_VIEWPORT full = {0, 0, (float)d.Width, (float)d.Height, 0, 1};
        c->RSSetViewports(1, &full);
        Preview(c, group, gi);
        RestoreOm(c, om);
        c->RSSetViewports(nvp, vps);
    });
}

bool BeforeDraw(ID3D11DeviceContext* c, int kind, ID3D11PixelShader* ps, int group) {
    if (!g_mode || c != g_imm || g_dead || g_pending.active) return false;
    bool after = false;
    Protect("BeforeDraw", [&] {
        // The hook's shader bookkeeping can miss a state change it does not see (sun_fx.h CurrentPs): check it here.
        ID3D11PixelShader* bound = nullptr;
        c->PSGetShader(&bound, nullptr, nullptr);
        const bool same = bound == ps;
        SafeRelease(bound);
        if (!same) return;
        if (kind == DrawOverlay) after = Overlay(c, group);
        else if (kind == DrawAmbient) after = Ambient(c, group, kAmbient);
        else if (kind == DrawAmbientLightmap) after = Ambient(c, group, kAmbientLightmap);
        else if (kind == DrawResolve) after = Resolve(c, group);
    });
    return after;
}

void AfterDraw(ID3D11DeviceContext* c) {
    if (!g_pending.active) return;
    Protect("AfterDraw", [&] {
        if (g_pending.restore) c->PSSetShaderResources(g_pending.slot, 1, &g_pending.srv);
        SafeRelease(g_pending.srv);
        if (g_pending.gi >= 0) Composite(c, g_pending.gi);
        if (g_pending.capture >= 0) CaptureResolve(c, g_pending.capture);  // before the preview draws over it
        if (g_pending.preview >= 0) Preview(c, g_pending.preview, g_pending.previewGi);
    });
    SafeRelease(g_pending.srv);
    g_pending = Pending{};
}

bool DumpWanted() { return g_dumpDraws > 0; }

void DumpDraw(ID3D11DeviceContext* c, UINT count) {
    if (g_dumpDraws <= 0 || c != g_imm || g_dead) return;
    Protect("DumpDraw", [&] {
        char what[32];
        snprintf(what, sizeof(what), "draw %d after the overlay", 9 - g_dumpDraws);
        --g_dumpDraws;
        DumpLine(c, what, count);
    });
}

std::string Describe() {
    char buf[96];
    snprintf(buf, sizeof(buf), "own AO %s (ao=%d, gi=%d)", g_dead ? "STOPPED" : g_mode ? "on" : "off", g_mode, g_gi);
    return buf;
}

void LogStats(bool totals) {
    AddStats(g_total, g_st);
    if (g_st.frames || g_dead) LogStatsOf(g_st, "");
    g_st = Stats{};
    if (totals && g_total.frames) LogStatsOf(g_total, " (session)");
}

void Shutdown() {
    Protect("Shutdown", [&] {
        if (g_pending.active) SafeRelease(g_pending.srv);
        g_pending = Pending{};
        g_mode = 0;
        g_gi = 0;
        g_imm = nullptr;
        ReleaseObjects();
    });
}

void TestInit(ID3D11DeviceContext* ctx, int mode, int gi) {
    g_testing = true;
    g_testMode = mode;
    g_testGi = gi;
    if (g_imm != ctx) SafeRelease(g_ctx1);
    g_imm = ctx;
}

TestState TestGet() {
    Stats s = g_total;
    AddStats(s, g_st);
    TestState t;
    for (int v = 0; v < kViews; ++v) {
        t.passes += s.passes[v];
        t.giPasses += s.giPasses[v];
        t.captures += s.captures[v];
    }
    t.ambient = s.ambient;
    t.ambientMissed = s.why[WhyAmbient];
    t.composites = s.composites;
    t.previews = s.previews;
    t.noHistory = s.why[WhyHistory];
    for (int w = 0; w < kWhys; ++w) t.refused += s.why[w];
    t.refused -= s.why[WhyHistory];  // a view's first frame, not a refusal
    t.faults = g_dead ? 1 : 0;
    t.probe = g_testProbe;
    t.dumpLines = g_dumpLines;
    return t;
}

}  // namespace qvfr::ssgiao

// --mvr-pass: gate B8 (mvr_pass.h) on a real D3D11 device, without the game and without SteamVR (NVIDIA with
// Multi-View Rendering only; elsewhere the test reports a skip).
// Hooks on the context's OMSetRenderTargets, ClearRenderTargetView and ClearDepthStencilView call the module the way
// quad's and the clones' detours do; the module hooks VSSetShader and VSSetConstantBuffers itself. Fake tasks
// (vtables: Execute at +0x48, Finish at +0x38) stand for the engine's:
// - outer L's CPhase1Render: clears the targets CBeginEye bound (color, linear depth, depth), binds them again, clears
//   VS slots 0-3 (slot 3 must get the module's buffer back), draws a grid of quads at three depths with outer's
//   projection (a patched VS, with outer's TAA jitter) and one quad with a VS the patch refuses, then the tail: a mask
//   at 1/4 size from the linear depth (through the game's own view of it), then Finish;
// - focus L's CPhase1BuildCommandThread and CPhase1Render: count their runs (the module must skip both).
// Fake clones (focus size) stand for the clones module; the jitter function gives the engine's projection offset.
// Checks:
// - outer: the game's color, linear depth, depth and mask equal the same pass drawn without MVR (the mask reads the
//   linear depth after slice 0 went back); the refused VS's quad only there; VS slot 3 back to the game's binding;
// - focus: both tasks skipped, the clones hold the grid as drawn straight with the focus projection and focus's own
//   jitter (color, linear depth, depth; edge pixels may round differently) and the mask as outer's read in the
//   focus window (bilinear, computed on the CPU);
// - the module's counters and the debug layer.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hooks.h"
#include "mvr_pass.h"
#include "mvr_vs.h"

using Microsoft::WRL::ComPtr;

namespace {

const char kHlsl[] = R"(
cbuffer Cam : register(b0) { row_major float4x4 proj; };
struct VsOut { float4 pos : SV_Position; float3 vp : VIEWPOS; };
VsOut VS(uint id : SV_VertexID) {
    const float2 c[6] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1)};
    uint q = id / 6, k = id % 6;
    uint qx = q % 6, qy = q / 6;
    float z = 2.0 + (float)((qx + 2 * qy) % 3) * 0.75 + c[k].x * 0.3;  // tilted: a depth slope for the bias check
    float2 t = float2(-1.2, -1.0) + (float2(qx, qy) + c[k] * 0.9) * 0.4;
    VsOut o;
    o.vp = float3(t * z, z);
    o.pos = mul(float4(o.vp, 1), proj);
    return o;
}
struct PsOut { float4 color : SV_Target0; float lin : SV_Target1; };
PsOut PS(VsOut i) {
    PsOut o;
    o.color = float4(frac(i.vp.x * 2.0), frac(i.vp.y * 2.0), i.vp.z / 5.0, 1);
    o.lin = i.vp.z;
    return o;
}
struct ClipOut { float4 pos : SV_Position; float3 vp : VIEWPOS; float4 clip : SV_ClipDistance0; };
ClipOut VSClip(uint id : SV_VertexID) {
    float2 uv = float2(id & 1, id >> 1);
    ClipOut o;
    o.pos = float4(lerp(-0.2, 0.2, uv.x), lerp(0.2, -0.2, uv.y), 0.0, 1);
    o.vp = float3(0, 0, 0);
    o.clip = float4(1, 1, 1, 1);
    return o;
}
PsOut PSMagenta(ClipOut i) {
    PsOut o;
    o.color = float4(1, 0, 1, 1);
    o.lin = 0.5;
    return o;
}
float4 VSFull(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
Texture2D<float> lin : register(t0);
float PSMask(float4 pos : SV_Position) : SV_Target {
    float z = lin.Load(int3(int2(pos.xy) * 4 + 1, 0));
    return z > 50 ? 1.0 : z * 0.1;
}
)";

constexpr UINT kW = 96, kH = 80, kFW = 64, kFH = 48;
constexpr UINT kMW = kW / 4, kMH = kH / 4, kFMW = kFW / 4, kFMH = kFH / 4;
constexpr UINT kQuads = 30;
// Focus window in outer NDC (x right, y up).
constexpr double kX0 = -0.6, kX1 = 0.3, kY0 = -0.2, kY1 = 0.7;
const float kJitter[2] = {0.4f, -0.3f};  // the engine's projection offset (NDC = offset / render size)
// Exact in RGBA8.
const float kClear[4] = {25 / 255.0f, 51 / 255.0f, 76 / 255.0f, 1.0f};
const float kGarbage[4] = {1, 1, 0, 1};
const float kFar[4] = {100, 100, 100, 100};
constexpr uint32_t kMagenta = 0xFFFF00FFu;

ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "mvr_pass_smoke", nullptr, nullptr, entry, target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
        return nullptr;
    }
    return code;
}

// Row vectors, z forward: tangents [-1, 1] in x and y to NDC, near 0.5, far 100.
void OuterProj(float m[16]) {
    memset(m, 0, 16 * sizeof(float));
    const float n = 0.5f, f = 100.0f;
    m[0] = 1;
    m[5] = 1;
    m[10] = f / (f - n);
    m[11] = 1;
    m[14] = -n * f / (f - n);
}

// The focus projection from the outer one, as quad's FocusEyeData: columns x and y get x' = sx x + ox w.
void FocusProj(const float o[16], double sx, double ox, double sy, double oy, float m[16]) {
    memcpy(m, o, 16 * sizeof(float));
    for (int i = 0; i < 4; ++i) {
        m[i * 4 + 0] = (float)(sx * o[i * 4 + 0] + ox * o[i * 4 + 3]);
        m[i * 4 + 1] = (float)(sy * o[i * 4 + 1] + oy * o[i * 4 + 3]);
    }
}

// The engine's jitter: NDC offset (jx, jy) added per unit of w (row 2 holds z_view = w).
void Jitter(float m[16], double jx, double jy) {
    m[8] += (float)jx;
    m[9] += (float)jy;
}

struct Targets {
    ComPtr<ID3D11Texture2D> color, lin, depth, mask;
    ComPtr<ID3D11RenderTargetView> colorRtv, linRtv, maskRtv;
    ComPtr<ID3D11ShaderResourceView> linSrv;
    ComPtr<ID3D11DepthStencilView> dsv;
};

struct Scene {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ComPtr<ID3D11VertexShader> vs, vsClip, vsFull;
    ComPtr<ID3D11PixelShader> ps, psMagenta, psMask;
    ComPtr<ID3D11Buffer> camOuter, camFocus;
    Targets game, focus;  // the game's (outer size) and the fake focus clones
    ComPtr<ID3D11DepthStencilState> ds;
    ComPtr<ID3D11RasterizerState> rs;
};
Scene g_s;

using OMSetRenderTargetsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                      ID3D11DepthStencilView*);
using ClearRtvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, const FLOAT[4]);
using ClearDsvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
OMSetRenderTargetsFn o_om = nullptr;
ClearRtvFn o_clearRtv = nullptr;
ClearDsvFn o_clearDsv = nullptr;

void STDMETHODCALLTYPE D_OM(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                            ID3D11DepthStencilView* dsv) {
    ID3D11RenderTargetView* mr[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* md = nullptr;
    if (n <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && qvfr::mvrpass::MapTargets(c, n, rtv, dsv, mr, &md)) {
        rtv = mr;
        dsv = md;
    }
    o_om(c, n, rtv, dsv);
}
void STDMETHODCALLTYPE D_ClearRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]) {
    o_clearRtv(c, qvfr::mvrpass::MapClear(c, v, col), col);
}
void STDMETHODCALLTYPE D_ClearDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT f, FLOAT d, UINT8 st) {
    o_clearDsv(c, qvfr::mvrpass::MapClear(c, v, f, d, st), f, d, st);
}

bool HookContext(ID3D11DeviceContext* c) {
    void** vt = *(void***)c;
    // ID3D11DeviceContext vtable slots (d3d11.h order).
    return qvfr::hooks::CreateAt(vt[33], (void*)D_OM, (void**)&o_om, "OMSetRenderTargets") &&
           qvfr::hooks::CreateAt(vt[50], (void*)D_ClearRtv, (void**)&o_clearRtv, "ClearRenderTargetView") &&
           qvfr::hooks::CreateAt(vt[53], (void*)D_ClearDsv, (void**)&o_clearDsv, "ClearDepthStencilView");
}

void BindG(const Targets& t) {
    ID3D11RenderTargetView* rt[2] = {t.colorRtv.Get(), t.linRtv.Get()};
    g_s.ctx->OMSetRenderTargets(2, rt, t.dsv.Get());
}

// g_bindFirst (the game's order, B8 in the game): nothing is bound before outer's task, the pass binds `t` and then
// clears it. Otherwise (CBeginEye's targets bound before the task) it clears what is bound and then binds `t`.
bool g_bindFirst = false;

// The G-buffer part of a pass: clear and bind `t`, VS slots 0-3 (3 cleared), the grid, the refused quad.
void DrawG(const Targets& t, ID3D11Buffer* cam, UINT w, UINT h, bool magenta) {
    ID3D11DeviceContext* c = g_s.ctx;
    if (g_bindFirst) BindG(t);
    ID3D11RenderTargetView* rtv[2] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(2, rtv, &dsv);
    if (rtv[0]) c->ClearRenderTargetView(rtv[0], kClear);
    if (rtv[1]) c->ClearRenderTargetView(rtv[1], kFar);
    if (dsv) c->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    for (ID3D11RenderTargetView* v : rtv)
        if (v) v->Release();
    if (dsv) dsv->Release();
    if (!g_bindFirst) BindG(t);
    D3D11_VIEWPORT vp{0, 0, (float)w, (float)h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->RSSetState(g_s.rs.Get());
    ID3D11Buffer* cbs[4] = {cam, nullptr, nullptr, nullptr};
    c->VSSetConstantBuffers(0, 4, cbs);
    c->VSSetShader(g_s.vs.Get(), nullptr, 0);
    c->PSSetShader(g_s.ps.Get(), nullptr, 0);
    c->Draw(kQuads * 6, 0);
    if (!magenta) return;
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    c->VSSetShader(g_s.vsClip.Get(), nullptr, 0);
    c->PSSetShader(g_s.psMagenta.Get(), nullptr, 0);
    c->Draw(4, 0);
}

// The tail: the mask at 1/4 size from the linear depth (the game's own view of it).
void DrawMask(const Targets& t, UINT w, UINT h) {
    ID3D11DeviceContext* c = g_s.ctx;
    c->OMSetRenderTargets(1, t.maskRtv.GetAddressOf(), nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)w, (float)h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(g_s.vsFull.Get(), nullptr, 0);
    c->PSSetShader(g_s.psMask.Get(), nullptr, 0);
    c->PSSetShaderResources(0, 1, t.linSrv.GetAddressOf());
    c->Draw(3, 0);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
}

using TaskFn = void(__fastcall*)(void*);
void* g_vtOuter[32];
void* g_vtBuild[32];
struct FakeTask {
    void** vt;
};
FakeTask g_outer{g_vtOuter}, g_build{g_vtBuild};
int g_finish = 0, g_buildRuns = 0, g_focusRuns = 0;

void __fastcall FakeFinish(void*) { ++g_finish; }
void __fastcall OuterExecute(void* task) {
    DrawG(g_s.game, g_s.camOuter.Get(), kW, kH, true);
    DrawMask(g_s.game, kMW, kMH);
    if (g_bindFirst) {  // the game binds its G-buffer again after the masks, draws nothing, unbinds (run 3 of B8)
        BindG(g_s.game);
        g_s.ctx->OMSetRenderTargets(0, nullptr, nullptr);
    }
    (*(TaskFn**)task)[0x38 / 8](task);  // the engine's jmp [vtable+0x38]
}
void __fastcall BuildExecute(void* task) {
    ++g_buildRuns;
    (*(TaskFn**)task)[0x38 / 8](task);
}
void __fastcall FocusExecute(void* task) {
    ++g_focusRuns;
    DrawG(g_s.focus, g_s.camFocus.Get(), kFW, kFH, false);
    DrawMask(g_s.focus, kFMW, kFMH);
    (*(TaskFn**)task)[0x38 / 8](task);
}
void RunTask(FakeTask& t) { (*(TaskFn**)&t)[0x48 / 8](&t); }
void Finish(FakeTask& t) { (*(TaskFn**)&t)[0x38 / 8](&t); }

ComPtr<ID3D11Texture2D> Tex(UINT w, UINT h, DXGI_FORMAT f, UINT bind, UINT slices = 1) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = slices;
    d.Format = f;
    d.SampleDesc = {1, 0};
    d.BindFlags = bind;
    if (!bind) {
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    }
    ComPtr<ID3D11Texture2D> t;
    g_s.dev->CreateTexture2D(&d, nullptr, &t);
    return t;
}

bool MakeTargets(Targets& t, UINT w, UINT h) {
    ID3D11Device* dev = g_s.dev;
    t.color = Tex(w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    t.lin = Tex(w, h, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    t.depth = Tex(w, h, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL);
    t.mask = Tex(w / 4, h / 4, DXGI_FORMAT_R16_TYPELESS, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    if (!t.color || !t.lin || !t.depth || !t.mask) return false;
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    rv.Format = DXGI_FORMAT_R32_FLOAT;
    D3D11_RENDER_TARGET_VIEW_DESC mv = rv;
    mv.Format = DXGI_FORMAT_R16_FLOAT;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = DXGI_FORMAT_R32_FLOAT;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D = {0, 1};
    D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = DXGI_FORMAT_D32_FLOAT;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    return SUCCEEDED(dev->CreateRenderTargetView(t.color.Get(), nullptr, &t.colorRtv)) &&
           SUCCEEDED(dev->CreateRenderTargetView(t.lin.Get(), &rv, &t.linRtv)) &&
           SUCCEEDED(dev->CreateRenderTargetView(t.mask.Get(), &mv, &t.maskRtv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(t.lin.Get(), &sv, &t.linSrv)) &&
           SUCCEEDED(dev->CreateDepthStencilView(t.depth.Get(), &dv, &t.dsv));
}

void Spoil(const Targets& t) {
    ID3D11DeviceContext* c = g_s.ctx;
    c->ClearRenderTargetView(t.colorRtv.Get(), kGarbage);
    c->ClearRenderTargetView(t.linRtv.Get(), kGarbage);
    c->ClearRenderTargetView(t.maskRtv.Get(), kGarbage);
    c->ClearDepthStencilView(t.dsv.Get(), D3D11_CLEAR_DEPTH, 0.25f, 0);
}

// The fake clones module: the focus clone of each game target.
ID3D11Texture2D* CloneOf(ID3D11Resource* r) {
    if (r == g_s.game.color.Get()) return g_s.focus.color.Get();
    if (r == g_s.game.lin.Get()) return g_s.focus.lin.Get();
    if (r == g_s.game.depth.Get()) return g_s.focus.depth.Get();
    if (r == g_s.game.mask.Get()) return g_s.focus.mask.Get();
    return nullptr;
}
bool g_jitterOff = false;  // second frame: the engine's offset unknown (view 1 keeps outer's jitter)
bool JitterOf(float out[2]) {
    if (g_jitterOff) return false;
    out[0] = kJitter[0];
    out[1] = kJitter[1];
    return true;
}

// One slice as 32-bit values (RGBA8 or R32F) or 16-bit ones (R16F) widened.
std::vector<uint32_t> Read(ID3D11Texture2D* src, UINT slice = 0) {
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    ComPtr<ID3D11Texture2D> st = Tex(d.Width, d.Height, d.Format, 0);
    std::vector<uint32_t> px;
    if (!st) return px;
    g_s.ctx->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src, D3D11CalcSubresource(0, slice, d.MipLevels), nullptr);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(g_s.ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return px;
    const bool half = d.Format == DXGI_FORMAT_R16_TYPELESS;
    px.resize((size_t)d.Width * d.Height);
    for (UINT y = 0; y < d.Height; ++y)
        for (UINT x = 0; x < d.Width; ++x) {
            const uint8_t* row = (const uint8_t*)m.pData + (size_t)y * m.RowPitch;
            uint32_t v = 0;
            memcpy(&v, row + x * (half ? 2 : 4), half ? 2 : 4);
            px[(size_t)y * d.Width + x] = v;
        }
    g_s.ctx->Unmap(st.Get(), 0);
    return px;
}

int ChannelDiff(uint32_t a, uint32_t b) {
    int d = 0;
    for (int s = 0; s < 32; s += 8) d = std::max(d, std::abs((int)((a >> s) & 255) - (int)((b >> s) & 255)));
    return d;
}
float AsFloat(uint32_t v) {
    float f;
    memcpy(&f, &v, 4);
    return f;
}
float Half(uint32_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? m / 1024.0f * std::pow(2.0f, -14.0f) : (1 + m / 1024.0f) * std::pow(2.0f, (float)e - 15);
    return s ? -v : v;
}
uint32_t Pack(const float c[4]) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)std::lround(c[i] * 255.0f) << (8 * i);
    return v;
}

// D3D11 bilinear sampling with clamp of an R16F image at uv.
float Bilinear(const std::vector<uint32_t>& img, UINT w, UINT h, float u, float v) {
    const float x = u * w - 0.5f, y = v * h - 0.5f;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const float fx = x - x0, fy = y - y0;
    auto at = [&](int xi, int yi) {
        xi = std::clamp(xi, 0, (int)w - 1);
        yi = std::clamp(yi, 0, (int)h - 1);
        return Half(img[(size_t)yi * w + xi]);
    };
    return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

}  // namespace

int RunMvrPass() {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    const bool debugLayer = SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                        D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                                        nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    ComPtr<ID3D11InfoQueue> info;
    if (debugLayer) dev.As(&info);
    if (!qvfr::mvrvs::TestInstall(dev.Get())) {
        printf("MVR pass: the VS hook could not be installed or NVAPI is not available (not NVIDIA?): SKIPPED\n");
        return 0;
    }
    NV_QUERY_MULTIVIEW_SUPPORT_PARAMS q = {};
    q.version = NV_QUERY_MULTIVIEW_SUPPORT_PARAMS_VER;
    if (NvAPI_D3D_QueryMultiViewSupport(dev.Get(), &q) != NVAPI_OK || !q.bMultiViewSupported) {
        printf("MVR pass: no Multi-View Rendering on this GPU: SKIPPED\n");
        return 0;
    }
    g_s.dev = dev.Get();
    g_s.ctx = ctx.Get();

    ComPtr<ID3DBlob> vsCode = Compile("VS", "vs_5_0"), clipCode = Compile("VSClip", "vs_5_0"),
                     fullCode = Compile("VSFull", "vs_5_0"), psCode = Compile("PS", "ps_5_0"),
                     magCode = Compile("PSMagenta", "ps_5_0"), maskCode = Compile("PSMask", "ps_5_0");
    if (!vsCode || !clipCode || !fullCode || !psCode || !magCode || !maskCode) return 1;
    qvfr::mvrvs::TestSetEnabled(true);
    auto vsMake = [&](ID3DBlob* b, ComPtr<ID3D11VertexShader>& out) {
        return SUCCEEDED(dev->CreateVertexShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &out));
    };
    auto psMake = [&](ID3DBlob* b, ComPtr<ID3D11PixelShader>& out) {
        return SUCCEEDED(dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &out));
    };
    bool made = vsMake(vsCode.Get(), g_s.vs) && vsMake(clipCode.Get(), g_s.vsClip) && vsMake(fullCode.Get(), g_s.vsFull) &&
                psMake(psCode.Get(), g_s.ps) && psMake(magCode.Get(), g_s.psMagenta) && psMake(maskCode.Get(), g_s.psMask);
    const bool vsPatched = qvfr::mvrvs::IsPatched(g_s.vs.Get()) && !qvfr::mvrvs::IsPatched(g_s.vsClip.Get()) &&
                           qvfr::mvrvs::IsPatched(g_s.vsFull.Get());

    const double sx = 2 / (kX1 - kX0), ox = -(kX1 + kX0) / (kX1 - kX0);
    const double sy = 2 / (kY1 - kY0), oy = -(kY1 + kY0) / (kY1 - kY0);
    float po[16], pf[16];
    OuterProj(po);
    FocusProj(po, sx, ox, sy, oy, pf);
    Jitter(po, kJitter[0] / kW, kJitter[1] / kH);
    Jitter(pf, kJitter[0] / kFW, kJitter[1] / kFH);
    auto cbuf = [&](const float* m, ComPtr<ID3D11Buffer>& b) {
        D3D11_BUFFER_DESC bd{64, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA init{m, 0, 0};
        return SUCCEEDED(dev->CreateBuffer(&bd, &init, &b));
    };
    ComPtr<ID3D11Buffer> game3;  // the game's buffer in VS slot 3 before the pass
    const float zeros[64] = {};
    D3D11_BUFFER_DESC bd3{256, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
    D3D11_SUBRESOURCE_DATA init3{zeros, 0, 0};
    made = made && cbuf(po, g_s.camOuter) && cbuf(pf, g_s.camFocus) &&
           SUCCEEDED(dev->CreateBuffer(&bd3, &init3, &game3)) && MakeTargets(g_s.game, kW, kH) &&
           MakeTargets(g_s.focus, kFW, kFH);
    D3D11_DEPTH_STENCIL_DESC dsd{};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsd.DepthFunc = D3D11_COMPARISON_LESS;
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    made = made && SUCCEEDED(dev->CreateDepthStencilState(&dsd, &g_s.ds)) &&
           SUCCEEDED(dev->CreateRasterizerState(&rd, &g_s.rs));
    if (!made) {
        printf("MVR pass: D3D11 objects could not be made\n");
        return 1;
    }
    for (void*& p : g_vtOuter) p = (void*)&FakeFinish;
    for (void*& p : g_vtBuild) p = (void*)&FakeFinish;
    g_vtOuter[0x48 / 8] = (void*)&OuterExecute;
    g_vtBuild[0x48 / 8] = (void*)&BuildExecute;

    // References without the module: outer's pass and focus's own (the hooks are not in yet).
    BindG(g_s.game);
    DrawG(g_s.game, g_s.camOuter.Get(), kW, kH, true);
    DrawMask(g_s.game, kMW, kMH);
    const std::vector<uint32_t> refC = Read(g_s.game.color.Get()), refL = Read(g_s.game.lin.Get()),
                                refD = Read(g_s.game.depth.Get()), refM = Read(g_s.game.mask.Get());
    BindG(g_s.focus);
    DrawG(g_s.focus, g_s.camFocus.Get(), kFW, kFH, false);
    DrawMask(g_s.focus, kFMW, kFMH);
    const std::vector<uint32_t> refFC = Read(g_s.focus.color.Get()), refFL = Read(g_s.focus.lin.Get()),
                                refFD = Read(g_s.focus.depth.Get()), refFM = Read(g_s.focus.mask.Get());
    Spoil(g_s.game);
    Spoil(g_s.focus);

    if (!HookContext(ctx.Get())) {
        printf("MVR pass: context hooks could not be installed\n");
        return 1;
    }
    qvfr::mvrpass::TestSetup(ctx.Get(), g_vtOuter, g_vtBuild, &CloneOf, &JitterOf);
    qvfr::mvrpass::TestDepthBias(0, 0.0f, false);  // the depth as MVR draws it; the depth fix has its own frames
    // Outer L's group: CBeginEye's targets and window, then the pass.
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->VSSetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, game3.GetAddressOf());
    BindG(g_s.game);
    const bool outerSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0);
    if (!outerSkip) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    ComPtr<ID3D11Buffer> slot3;
    ctx->VSGetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, &slot3);
    const qvfr::mvrpass::TestState st = qvfr::mvrpass::TestGet();
    const std::vector<uint32_t> gC = Read(g_s.game.color.Get()), gL = Read(g_s.game.lin.Get()),
                                gD = Read(g_s.game.depth.Get()), gM = Read(g_s.game.mask.Get());
    std::vector<uint32_t> s1;
    if (ID3D11Texture2D* arr = qvfr::mvrpass::TestArray(0)) s1 = Read(arr, 1);

    // Focus L's group: its build task and its CPhase1Render (the outer's vtable: the module knows the task by it).
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    const int finishBefore = g_finish;
    const bool buildSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2);
    if (buildSkip)
        Finish(g_build);
    else
        RunTask(g_build);
    FakeTask focusAsPhase1{g_vtOuter};  // a CPhase1Render of the focus group: same class, other object
    void* focusVt[32];
    memcpy(focusVt, g_vtOuter, sizeof(focusVt));
    focusVt[0x48 / 8] = (void*)&FocusExecute;
    const bool focusSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&focusAsPhase1, 2);
    if (focusSkip) {
        Finish(focusAsPhase1);
    } else {
        focusAsPhase1.vt = focusVt;  // the engine's own run of focus's pass
        RunTask(focusAsPhase1);
    }
    const int focusFinish = g_finish - finishBefore;
    const qvfr::mvrpass::TestState st2 = qvfr::mvrpass::TestGet();
    const int buildRuns1 = g_buildRuns, focusRuns1 = g_focusRuns;  // the first frame's (later frames add to them)
    const std::vector<uint32_t> fC = Read(g_s.focus.color.Get()), fL = Read(g_s.focus.lin.Get()),
                                fD = Read(g_s.focus.depth.Get()), fM = Read(g_s.focus.mask.Get());
    qvfr::mvrpass::TestEnd();
    // Negative check, next frame: without the engine's offset view 1 keeps outer's jitter (gate B7) and focus differs.
    g_jitterOff = true;
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    BindG(g_s.game);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    const bool noJitter = !qvfr::mvrpass::TestGet().jitterKnown;
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    FakeTask focusAgain{g_vtOuter};
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2)) Finish(g_build);
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&focusAgain, 2)) Finish(focusAgain);
    const std::vector<uint32_t> nC = Read(g_s.focus.color.Get());
    qvfr::mvrpass::TestEnd();
    // The game's order (third frame): no target bound at outer's task, the MVR part starts at its first bind.
    g_jitterOff = false;
    g_bindFirst = true;
    Spoil(g_s.game);
    Spoil(g_s.focus);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    const bool lateOuterSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0);
    if (!lateOuterSkip) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    const qvfr::mvrpass::TestState st3 = qvfr::mvrpass::TestGet();
    const std::vector<uint32_t> lC = Read(g_s.game.color.Get()), lD = Read(g_s.game.depth.Get());
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    const int lateFinishBefore = g_finish, lateRunsBefore = g_buildRuns + g_focusRuns;
    FakeTask focusLate{g_vtOuter};
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2)) Finish(g_build);
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&focusLate, 2)) Finish(focusLate);
    const int lateFinish = g_finish - lateFinishBefore, lateRuns = g_buildRuns + g_focusRuns - lateRunsBefore;
    const std::vector<uint32_t> lfC = Read(g_s.focus.color.Get()), lfD = Read(g_s.focus.depth.Get());
    const qvfr::mvrpass::TestState st4 = qvfr::mvrpass::TestGet();
    qvfr::mvrpass::TestEnd();
    // Overwrite mode (fourth frame): focus runs its own tasks, its clones get the stash after its CPhase1Render.
    qvfr::mvrpass::SetMode(qvfr::mvrpass::Mode::Overwrite);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    const int owBuild = g_buildRuns, owFocus = g_focusRuns;
    FakeTask focusOw{g_vtOuter};
    const bool owSkipBuild = qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2);
    if (!owSkipBuild) RunTask(g_build);
    const bool owSkipFocus = qvfr::mvrpass::BeforeTask((uint8_t*)&focusOw, 2);
    if (!owSkipFocus) {
        focusOw.vt = focusVt;
        RunTask(focusOw);
        qvfr::mvrpass::AfterTask((uint8_t*)&focusOw);
    }
    const qvfr::mvrpass::TestState st5 = qvfr::mvrpass::TestGet();
    const std::vector<uint32_t> owC = Read(g_s.focus.color.Get());
    qvfr::mvrpass::TestEnd();
    // Depth bias (fifth frame, the game's order): focus's depth from outer moved 32 steps farther, nothing else.
    qvfr::mvrpass::SetMode(qvfr::mvrpass::Mode::Skip);
    qvfr::mvrpass::TestDepthBias(32, 0.0f, false);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    FakeTask focusBias{g_vtOuter};
    const bool biasSkipBuild = qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2);
    if (biasSkipBuild) Finish(g_build);
    const bool biasSkipFocus = qvfr::mvrpass::BeforeTask((uint8_t*)&focusBias, 2);
    if (biasSkipFocus) Finish(focusBias);
    const std::vector<uint32_t> bD = Read(g_s.focus.depth.Get());
    qvfr::mvrpass::TestEnd();
    // Slope only (sixth frame): 1 pixel's worth of the local slope, no constant steps.
    qvfr::mvrpass::TestDepthBias(0, 1.0f, false);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    FakeTask focusSlope{g_vtOuter};
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2)) Finish(g_build);
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&focusSlope, 2)) Finish(focusSlope);
    const std::vector<uint32_t> slD = Read(g_s.focus.depth.Get());
    qvfr::mvrpass::TestEnd();
    // Dilation (seventh frame): the farthest of 3 x 3 plus 32 steps (the settings' defaults for the fix).
    qvfr::mvrpass::TestDepthBias(32, 0.0f, true);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    FakeTask focusDilate{g_vtOuter};
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2)) Finish(g_build);
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&focusDilate, 2)) Finish(focusDilate);
    const std::vector<uint32_t> dlD = Read(g_s.focus.depth.Get());
    qvfr::mvrpass::TestEnd();
    qvfr::mvrpass::TestDepthBias(0, 0.0f, false);
    qvfr::mvrpass::SetMode(qvfr::mvrpass::Mode::Skip);
    g_bindFirst = false;
    // B15 (eighth frame, as the first): the corners through the stash (mvr_direct_extract off) give the same clones bit
    // for bit as straight into them (the first frame).
    qvfr::mvrpass::SetDirectExtract(false);
    Spoil(g_s.focus);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    ctx->VSSetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, game3.GetAddressOf());
    BindG(g_s.game);
    if (!qvfr::mvrpass::BeforeTask((uint8_t*)&g_outer, 0)) RunTask(g_outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&g_outer);
    const qvfr::mvrpass::TestState st8 = qvfr::mvrpass::TestGet();
    qvfr::mvrpass::FocusGroup(0, true);
    BindG(g_s.focus);
    FakeTask focusStash{g_vtOuter};
    if (qvfr::mvrpass::BeforeTask((uint8_t*)&g_build, 2)) Finish(g_build);
    const bool stashSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&focusStash, 2);
    if (stashSkip) Finish(focusStash);
    const qvfr::mvrpass::TestState st9 = qvfr::mvrpass::TestGet();
    const std::vector<uint32_t> sC = Read(g_s.focus.color.Get()), sL = Read(g_s.focus.lin.Get()),
                                sD = Read(g_s.focus.depth.Get()), sM = Read(g_s.focus.mask.Get());
    qvfr::mvrpass::TestEnd();
    qvfr::mvrpass::SetDirectExtract(true);
    if (refC.empty() || refFC.empty() || gC.empty() || fC.empty() || fM.empty() || gM.empty() || s1.empty()) {
        printf("MVR pass: read-back failed\n");
        return 1;
    }

    // Outer: as drawn without MVR.
    int outerC = 0, outerL = 0, outerD = 0, outerM = 0, magenta0 = 0, magenta1 = 0;
    for (size_t i = 0; i < refC.size(); ++i) {
        outerC += gC[i] != refC[i];
        outerL += gL[i] != refL[i];
        outerD += gD[i] != refD[i];
        magenta0 += gC[i] == kMagenta;
        magenta1 += s1[i] == kMagenta;
    }
    for (size_t i = 0; i < refM.size(); ++i) outerM += gM[i] != refM[i];
    // Focus: the grid as drawn straight with focus's projection and jitter.
    const uint32_t clear = Pack(kClear);
    int covered = 0, cOff = 0, cBig = 0, lOff = 0, dOff = 0;
    double dMax = 0;
    for (size_t i = 0; i < refFC.size(); ++i) {
        covered += refFC[i] != clear;
        const int d = ChannelDiff(fC[i], refFC[i]);
        cOff += d > 0;
        cBig += d > 2;
        lOff += std::fabs(AsFloat(fL[i]) - AsFloat(refFL[i])) > 1e-4f;
        const double dd = std::fabs((double)AsFloat(fD[i]) - (double)AsFloat(refFD[i]));
        dOff += dd > 1e-5;  // MVR's sub-pixel snapping: a few 1e-6 on the tilted quads
        if (refFD[i] != 0x3F800000u && fD[i] != 0x3F800000u) dMax = std::max(dMax, dd);
    }
    // Diagnosis: depth and linear depth bit for bit, and ours - own fitted as own's slope times a shift (px).
    int dBits = 0, lBits = 0;
    double sxx = 0, sxy = 0, syy = 0, sxd = 0, syd = 0;
    for (size_t i = 0; i < refFD.size(); ++i) {
        dBits += fD[i] != refFD[i];
        lBits += fL[i] != refFL[i];
    }
    for (UINT y = 1; y + 1 < kFH; ++y)
        for (UINT x = 1; x + 1 < kFW; ++x) {
            auto at = [&](UINT xx, UINT yy) { return (double)AsFloat(refFD[(size_t)yy * kFW + xx]); };
            const double o = at(x, y), l = at(x - 1, y), r = at(x + 1, y), up = at(x, y - 1), dn = at(x, y + 1);
            if (o >= 1 || l >= 1 || r >= 1 || up >= 1 || dn >= 1) continue;
            const double ax = r - o, bx = o - l, ay = dn - o, by = o - up;
            if (std::fabs(ax - bx) > 0.5 * (std::fabs(ax) + std::fabs(bx)) + 1e-7 ||
                std::fabs(ay - by) > 0.5 * (std::fabs(ay) + std::fabs(by)) + 1e-7)
                continue;
            const double gx = 0.5 * (ax + bx), gy = 0.5 * (ay + by);
            const double dd = (double)AsFloat(fD[(size_t)y * kFW + x]) - o;
            sxx += gx * gx;
            sxy += gx * gy;
            syy += gy * gy;
            sxd += gx * dd;
            syd += gy * dd;
        }
    const double det = sxx * syy - sxy * sxy;
    const double shiftX = det > 0 ? (sxd * syy - syd * sxy) / det : 0, shiftY = det > 0 ? (syd * sxx - sxd * sxy) / det : 0;
    printf("  diagnosis: focus depth %d px and linear depth %d px differ bit for bit; ours - own as a shift (%.5f, %.5f) "
           "px of own's slope\n", dBits, lBits, shiftX, shiftY);
    {  // row offset check: how many pixels of ours equal own at (x, y + dy) bit for bit
        int match[5] = {};
        for (int dy = -2; dy <= 2; ++dy)
            for (UINT y = 2; y + 2 < kFH; ++y)
                for (UINT x = 0; x < kFW; ++x) {
                    const uint32_t o = refFD[(size_t)(y + dy) * kFW + x], v = fD[(size_t)y * kFW + x];
                    match[dy + 2] += o == v && o != 0x3F800000u;
                }
        int matchX[5] = {};
        for (int dx = -2; dx <= 2; ++dx)
            for (UINT y = 0; y < kFH; ++y)
                for (UINT x = 2; x + 2 < kFW; ++x) {
                    const uint32_t o = refFD[(size_t)y * kFW + x + dx], v = fD[(size_t)y * kFW + x];
                    matchX[dx + 2] += o == v && o != 0x3F800000u;
                }
        printf("  diagnosis: ours(x, y) == own(x, y + dy) for dy -2..2: %d %d %d %d %d; own(x + dx, y) for dx -2..2: %d %d "
               "%d %d %d\n", match[0], match[1], match[2], match[3], match[4], matchX[0], matchX[1], matchX[2], matchX[3],
               matchX[4]);
    }
    int nOff = 0;
    for (size_t i = 0; i < refFC.size() && i < nC.size(); ++i) nOff += ChannelDiff(nC[i], refFC[i]) > 0;
    // Focus mask: outer's read in the window (bilinear) vs the module's; focus's own only reported.
    const float scaleU = (float)(1 / sx), scaleV = (float)(1 / sy);
    const float originU = (float)((1 - 1 / sx - ox / sx) / 2), originV = (float)((1 - 1 / sy + oy / sy) / 2);
    double mMax = 0, ownMax = 0;
    for (UINT y = 0; y < kFMH; ++y)
        for (UINT x = 0; x < kFMW; ++x) {
            const float u = originU + (x + 0.5f) / kFMW * scaleU, v = originV + (y + 0.5f) / kFMH * scaleV;
            const float want = Bilinear(gM, kMW, kMH, u, v);
            const size_t i = (size_t)y * kFMW + x;
            mMax = std::max(mMax, (double)std::fabs(Half(fM[i]) - want));
            ownMax = std::max(ownMax, (double)std::fabs(Half(fM[i]) - Half(refFM[i])));
        }
    const int px = (int)(kFW * kFH);
    // Third frame: outer as without MVR, focus clones as in the first frame.
    int lateOuter = 0, lateBig = 0, lateDepth = 0;
    for (size_t i = 0; i < refC.size() && i < lC.size(); ++i) lateOuter += lC[i] != refC[i] || lD[i] != refD[i];
    for (size_t i = 0; i < refFC.size() && i < lfC.size(); ++i) {
        lateBig += ChannelDiff(lfC[i], refFC[i]) > 2;
        lateDepth += std::fabs((double)AsFloat(lfD[i]) - (double)AsFloat(refFD[i])) > 1e-5;
    }

    int warnings = 0;
    if (info) {
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> buf(size);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (FAILED(info->GetMessage(i, msg, &size)) || msg->Severity > D3D11_MESSAGE_SEVERITY_WARNING) continue;
            ++warnings;
            printf("D3D11 debug layer: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
        }
    }

    const bool countsOk = vsPatched && !outerSkip && st.passes == 1 && st.stashValid && st.stashed == 3 &&
                          st.masks == 1 && st.mapFailures == 0 && st.patchedSets >= 1 && st.plainSets == 1 &&
                          st.cbRebinds == 1 && st.copiedBack == 3 && st.viewsFailures == 0 && st.jitterKnown &&
                          !st.faulted && buildSkip && focusSkip && buildRuns1 == 0 && focusRuns1 == 0 &&
                          focusFinish == 2 && st2.fills == 1 && st2.failures == 0 && st2.focusSkipped == 2;
    const bool outerOk = outerC == 0 && outerL == 0 && outerD == 0 && outerM == 0 && magenta0 > 0 && magenta1 == 0 &&
                         !slot3;
    // Edge pixels may round differently (M times the outer position against the focus projection).
    const bool focusOk = covered > px / 2 && cBig * 200 <= px && lOff * 100 <= px && dOff * 200 <= px && mMax < 0.01;
    const bool jitterOk = noJitter && nOff > 2 * cOff;
    const bool lateOk = !lateOuterSkip && st3.lateStart && st3.stashValid && st3.stashed == 3 && st3.clears >= 3 &&
                        st3.mapFailures == 0 && st3.jitterKnown && lateFinish == 2 && lateRuns == 0 &&
                        st4.focusSkipped == 2 && lC.size() == refC.size() && lfC.size() == refFC.size() &&
                        lateOuter == 0 && lateBig * 200 <= px && lateDepth * 200 <= px;
    int biasIn = 0, biasOut = 0, biasFar = 0;
    for (size_t i = 0; i < refFD.size() && i < bD.size(); ++i) {
        if (refFD[i] == 0x3F800000u) {
            biasFar += bD[i] != 0x3F800000u;
            continue;
        }
        if (fD[i] == 0x3F800000u) continue;
        const int64_t steps = (int64_t)bD[i] - (int64_t)fD[i];  // against ours without the bias (frame 1)
        (steps == 32 ? biasIn : biasOut) += 1;
    }
    const bool biasOk = biasSkipBuild && biasSkipFocus && bD.size() == refFD.size() && biasIn > 0 &&
                        biasOut * 200 <= px && biasFar == 0;
    // Slope (against ours without it): never nearer, farther where the surface slopes, at most the slope plus margin.
    int slMoved = 0, slNearer = 0, slFar = 0, slTooFar = 0;
    for (size_t i = 0; i < fD.size() && i < slD.size(); ++i) {
        if (fD[i] == 0x3F800000u) {
            slFar += slD[i] != 0x3F800000u;
            continue;
        }
        const int64_t steps = (int64_t)slD[i] - (int64_t)fD[i];
        slNearer += steps < 0;
        slMoved += steps > 2;
        float r, d;
        memcpy(&r, &fD[i], 4);
        memcpy(&d, &slD[i], 4);
        slTooFar += d - r > 0.01f;  // the grid's depth slope per pixel is far below that
    }
    const bool slopeOk = slD.size() == refFD.size() && slNearer == 0 && slFar == 0 && slTooFar == 0 &&
                         slMoved * 2 > biasIn;
    // Dilation (3 x 3 farthest + 32 steps) against focus's own depth: never nearer, so its later LESS_EQUAL passes keep all
    // of its own surfaces; own's sky stays sky.
    int dlNearer = 0, dlNearerPlain = 0, dlSkyLost = 0, dlCovered = 0;
    int64_t dlMaxSteps = 0;
    for (size_t i = 0; i < refFD.size() && i < dlD.size(); ++i) {
        if (refFD[i] == 0x3F800000u) {
            dlSkyLost += dlD[i] != 0x3F800000u;
            continue;
        }
        ++dlCovered;
        if (AsFloat(dlD[i]) < AsFloat(refFD[i])) {
            ++dlNearer;
            dlMaxSteps = std::max<int64_t>(dlMaxSteps, (int64_t)refFD[i] - (int64_t)dlD[i]);
        }
        dlNearerPlain += AsFloat(fD[i]) < AsFloat(refFD[i]);
    }
    const bool dilateOk = dlD.size() == refFD.size() && dlCovered > 0 && dlNearer == 0 && dlSkyLost == 0;
    int owBig = 0;
    for (size_t i = 0; i < refFC.size() && i < owC.size(); ++i) owBig += ChannelDiff(owC[i], refFC[i]) > 2;
    const bool owOk = !owSkipBuild && !owSkipFocus && g_buildRuns == owBuild + 1 && g_focusRuns == owFocus + 1 &&
                      st5.fills == 1 && st5.focusSkipped == 0 && owC.size() == refFC.size() && owBig * 200 <= px;
    int viaStashDiff = 0;
    const bool sizesOk = sC.size() == fC.size() && sL.size() == fL.size() && sD.size() == fD.size() && sM.size() == fM.size();
    if (sizesOk) {
        for (size_t i = 0; i < fC.size(); ++i) viaStashDiff += sC[i] != fC[i];
        for (size_t i = 0; i < fL.size(); ++i) viaStashDiff += sL[i] != fL[i];
        for (size_t i = 0; i < fD.size(); ++i) viaStashDiff += sD[i] != fD[i];
        for (size_t i = 0; i < fM.size(); ++i) viaStashDiff += sM[i] != fM[i];
    }
    const bool directOk = st.direct == 3 && st8.direct == 0 && st8.stashValid && st8.stashed == 3 && stashSkip &&
                          st9.fills == 1 && st9.failures == 0 && sizesOk && viaStashDiff == 0;
    const bool ok = countsOk && outerOk && focusOk && jitterOk && lateOk && owOk && biasOk && slopeOk && dilateOk &&
                    directOk && !warnings;
    printf("MVR pass: %s; outer pass %u, stash %s (%u targets, %u masks), binds %u, clears %u, map failures %u, VS sets "
           "patched %u / unpatched %u, slot 3 rebinds %u, copied back %u, SetMultiViewMode failures %u, jitter %s; "
           "focus: build %s, pass %s (runs %d / %d, Finish %d), fills %u (failed %u) (%s)\n",
           ok ? "PASS" : "FAIL", st.passes, st.stashValid ? "valid" : "INVALID", st.stashed, st.masks, st.binds,
           st.clears, st.mapFailures, st.patchedSets, st.plainSets, st.cbRebinds, st.copiedBack, st.viewsFailures,
           st.jitterKnown ? "moved" : "UNKNOWN", buildSkip ? "skipped" : "RUN", focusSkip ? "skipped" : "RUN",
           buildRuns1, focusRuns1, focusFinish, st2.fills, st2.failures, countsOk ? "OK" : "WRONG");
    printf("  outer vs the pass without MVR: color %d, linear depth %d, depth %d, mask %d px differ; refused VS quad "
           "%d px in the game's color, %d in slice 1; VS slot 3 %s (%s)\n",
           outerC, outerL, outerD, outerM, magenta0, magenta1, !slot3 ? "the game's binding (cleared by the pass)" : "WRONG",
           outerOk ? "OK" : "WRONG");
    printf("  focus clones vs focus's own pass (%ux%u, its jitter): %d px covered, color %d px differ (%d by > 2), "
           "linear depth %d px, depth %d px off (max %.2g); mask vs outer's in the window max %.4f (vs focus's own "
           "%.3f) (%s)\n",
           kFW, kFH, covered, cOff, cBig, lOff, dOff, dMax, mMax, ownMax, focusOk ? "OK" : "WRONG");
    printf("  B15: corners straight into the clones %u of 3 targets (first frame); through the stash %u straight, focus "
           "%s, fills %u, clones (color, linear depth, depth, mask) %d px differ from straight (%s)\n",
           st.direct, st8.direct, stashSkip ? "skipped" : "RUN", st9.fills, viaStashDiff, directOk ? "OK" : "WRONG");
    printf("  without the jitter fix (next frame, offset %s): focus color %d px differ, with it %d (%s)\n",
           noJitter ? "unknown" : "STILL KNOWN", nOff, cOff, jitterOk ? "OK" : "WRONG");
    printf("  game's order (nothing bound before outer's task): MVR part %s, stash %s (%u targets), clears %u, map "
           "failures %u; outer %d px differ; focus skipped %u, clones color %d px by > 2, depth %d px off (%s)\n",
           st3.lateStart ? "started at the first bind" : "NOT STARTED LATE", st3.stashValid ? "valid" : "INVALID",
           st3.stashed, st3.clears, st3.mapFailures, lateOuter, st4.focusSkipped, lateBig, lateDepth,
           lateOk ? "OK" : "WRONG");
    printf("  overwrite mode: focus tasks %s, fills %u, clones color %d px by > 2 (%s)\n",
           !owSkipBuild && !owSkipFocus ? "ran" : "SKIPPED", st5.fills, owBig, owOk ? "OK" : "WRONG");
    printf("  depth bias 32: focus depth %d px exactly 32 steps farther than without it, %d other, far plane changed %d (%s)\n", biasIn,
           biasOut, biasFar, biasOk ? "OK" : "WRONG");
    printf("  depth slope 1 px: %d px moved farther, %d nearer, %d too far, far plane changed %d (%s)\n", slMoved,
           slNearer, slTooFar, slFar, slopeOk ? "OK" : "WRONG");
    printf("  depth 3 x 3 farthest + 32 steps vs focus's own: nearer %d of %d px (max %lld steps; without it %d), own's sky lost %d (%s)\n",
           dlNearer, dlCovered, (long long)dlMaxSteps, dlNearerPlain, dlSkyLost, dilateOk ? "OK" : "WRONG");
    printf("  M rows [%.4f 0 0 %.4f] [0 %.4f 0 %.4f]; debug layer %s, %d warnings\n", st.rows[0][0], st.rows[0][3],
           st.rows[1][1], st.rows[1][3], debugLayer ? "on" : "not installed", warnings);
    return ok ? 0 : 1;
}

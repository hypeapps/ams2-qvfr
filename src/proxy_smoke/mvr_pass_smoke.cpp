// --mvr-pass: gate B7's test run (mvr_pass.h) on a real D3D11 device, without the game and without SteamVR (NVIDIA
// with Multi-View Rendering only; elsewhere the test reports a skip).
// Hooks on the context's OMSetRenderTargets, ClearRenderTargetView, ClearDepthStencilView and Draw call the module the
// way quad's, the clones' and sun_fx's detours do. A fake task (vtable: Execute at +0x48, Finish at +0x38) stands for
// outer L's CPhase1Render: it clears the bound color and depth targets (bound before, as by CBeginEye), binds them again,
// draws a grid of quads at three depths with the outer projection (a patched VS) and one quad with a VS the patch
// refuses, then calls Finish through its current vtable, like the engine's jmp [vtable+0x38]. VS slot 3 holds a "game"
// buffer.
// - BeforeTask arms the run, the task runs once (Finish once) with MVR into two-slice copies, AfterTask copies slice 0
//   into the game's targets and binds the game's views again.
// - The game's color equals slice 0 and the same grid drawn without MVR; the refused VS's quad is only in slice 0.
// - The corner of slice 1 (focus size) matches the same grid drawn straight with the focus projection (the outer
//   projection with the window's x' = sx x + ox w, y' = sy y + oy w, as quad's FocusEyeData), color and depth; the
//   rest of slice 1 keeps the clear color.
// - VS slot 3 holds the game's buffer again; the module's counters and the debug layer.
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
    float z = 2.0 + (float)((qx + 2 * qy) % 3) * 0.75;
    float2 t = float2(-1.2, -1.0) + (float2(qx, qy) + c[k] * 0.9) * 0.4;
    VsOut o;
    o.vp = float3(t * z, z);
    o.pos = mul(float4(o.vp, 1), proj);
    return o;
}
float4 PS(VsOut i) : SV_Target { return float4(frac(i.vp.x * 2.0), frac(i.vp.y * 2.0), i.vp.z / 5.0, 1); }
struct ClipOut { float4 pos : SV_Position; float3 vp : VIEWPOS; float4 clip : SV_ClipDistance0; };
ClipOut VSClip(uint id : SV_VertexID) {
    float2 uv = float2(id & 1, id >> 1);
    ClipOut o;
    o.pos = float4(lerp(-0.2, 0.2, uv.x), lerp(0.2, -0.2, uv.y), 0.0, 1);
    o.vp = float3(0, 0, 0);
    o.clip = float4(1, 1, 1, 1);
    return o;
}
float4 PSMagenta(ClipOut i) : SV_Target { return float4(1, 0, 1, 1); }
)";

constexpr UINT kW = 96, kH = 80, kFW = 64, kFH = 48;
constexpr UINT kQuads = 30;
// Focus window in outer NDC (x right, y up).
constexpr double kX0 = -0.6, kX1 = 0.3, kY0 = -0.2, kY1 = 0.7;
// Exact in RGBA8.
const float kClear[4] = {25 / 255.0f, 51 / 255.0f, 76 / 255.0f, 1.0f};
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

struct Scene {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ComPtr<ID3D11VertexShader> vs, vsClip;
    ComPtr<ID3D11PixelShader> ps, psMagenta;
    ComPtr<ID3D11Buffer> camOuter, camFocus;
    ComPtr<ID3D11Texture2D> color, depth;  // the "game's" targets, outer size
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11DepthStencilView> dsv;
    ComPtr<ID3D11DepthStencilState> ds;
    ComPtr<ID3D11RasterizerState> rs;
};
Scene g_s;

// What quad's, the clones' and sun_fx's detours do around the real calls.
using OMSetRenderTargetsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                      ID3D11DepthStencilView*);
using ClearRtvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, const FLOAT[4]);
using ClearDsvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
using DrawFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
OMSetRenderTargetsFn o_om = nullptr;
ClearRtvFn o_clearRtv = nullptr;
ClearDsvFn o_clearDsv = nullptr;
DrawFn o_draw = nullptr;

void STDMETHODCALLTYPE D_OM(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                            ID3D11DepthStencilView* dsv) {
    ID3D11RenderTargetView* mr[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* md = nullptr;
    if (n <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && qvfr::mvrpass::MapTargets(c, n, rtv, dsv, mr, &md)) {
        rtv = mr;
        dsv = md;
    }
    o_om(c, n, rtv, dsv);
    qvfr::mvrpass::OnSetTargets(c, n, rtv, dsv);
}
void STDMETHODCALLTYPE D_ClearRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]) {
    o_clearRtv(c, qvfr::mvrpass::MapClear(c, v), col);
}
void STDMETHODCALLTYPE D_ClearDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT f, FLOAT d, UINT8 st) {
    o_clearDsv(c, qvfr::mvrpass::MapClear(c, v), f, d, st);
}
void STDMETHODCALLTYPE D_Draw(ID3D11DeviceContext* c, UINT count, UINT start) {
    const bool after = qvfr::mvrpass::BeforeDraw(c);
    o_draw(c, count, start);
    if (after) qvfr::mvrpass::AfterDraw(c);
}

bool HookContext(ID3D11DeviceContext* c) {
    void** vt = *(void***)c;
    // ID3D11DeviceContext vtable slots (d3d11.h order).
    return qvfr::hooks::CreateAt(vt[33], (void*)D_OM, (void**)&o_om, "OMSetRenderTargets") &&
           qvfr::hooks::CreateAt(vt[50], (void*)D_ClearRtv, (void**)&o_clearRtv, "ClearRenderTargetView") &&
           qvfr::hooks::CreateAt(vt[53], (void*)D_ClearDsv, (void**)&o_clearDsv, "ClearDepthStencilView") &&
           qvfr::hooks::CreateAt(vt[13], (void*)D_Draw, (void**)&o_draw, "Draw");
}

void Draw(UINT count) { g_s.ctx->Draw(count, 0); }

void DrawGrid(ID3D11Buffer* cam, UINT w, UINT h) {
    ID3D11DeviceContext* c = g_s.ctx;
    D3D11_VIEWPORT vp{0, 0, (float)w, (float)h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->RSSetState(g_s.rs.Get());
    c->VSSetConstantBuffers(0, 1, &cam);
    c->VSSetShader(g_s.vs.Get(), nullptr, 0);
    c->PSSetShader(g_s.ps.Get(), nullptr, 0);
    Draw(kQuads * 6);
}

using TaskFn = void(__fastcall*)(void*);
void* g_vt[32];
struct FakeTask {
    void** vt;
} g_task{g_vt};
int g_finish = 0;

void __fastcall FakeFinish(void*) { ++g_finish; }

void __fastcall FakeExecute(void* task) {
    ID3D11DeviceContext* c = g_s.ctx;
    // The targets CBeginEye bound: cleared first, then bound again by the pass.
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(1, &rtv, &dsv);
    if (rtv) c->ClearRenderTargetView(rtv, kClear);
    if (dsv) c->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    if (rtv) rtv->Release();
    if (dsv) dsv->Release();
    c->OMSetRenderTargets(1, g_s.rtv.GetAddressOf(), g_s.dsv.Get());
    DrawGrid(g_s.camOuter.Get(), kW, kH);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    c->VSSetShader(g_s.vsClip.Get(), nullptr, 0);
    c->PSSetShader(g_s.psMagenta.Get(), nullptr, 0);
    Draw(4);
    (*(TaskFn**)task)[0x38 / 8](task);  // the engine's jmp [vtable+0x38]
}

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

// One slice as 32-bit values (RGBA8 or R32F).
std::vector<uint32_t> Read(ID3D11Texture2D* src, UINT slice) {
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    ComPtr<ID3D11Texture2D> st = Tex(d.Width, d.Height, d.Format, 0);
    std::vector<uint32_t> px;
    if (!st) return px;
    g_s.ctx->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src, D3D11CalcSubresource(0, slice, d.MipLevels), nullptr);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(g_s.ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return px;
    px.resize((size_t)d.Width * d.Height);
    for (UINT y = 0; y < d.Height; ++y)
        memcpy(&px[(size_t)y * d.Width], (const uint8_t*)m.pData + (size_t)y * m.RowPitch, d.Width * 4);
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

uint32_t Pack(const float c[4]) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)std::lround(c[i] * 255.0f) << (8 * i);
    return v;
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
                     psCode = Compile("PS", "ps_5_0"), magCode = Compile("PSMagenta", "ps_5_0");
    if (!vsCode || !clipCode || !psCode || !magCode) return 1;
    qvfr::mvrvs::TestSetEnabled(true);
    bool made = SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &g_s.vs)) &&
                SUCCEEDED(dev->CreateVertexShader(clipCode->GetBufferPointer(), clipCode->GetBufferSize(), nullptr,
                                                  &g_s.vsClip)) &&
                SUCCEEDED(dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &g_s.ps)) &&
                SUCCEEDED(dev->CreatePixelShader(magCode->GetBufferPointer(), magCode->GetBufferSize(), nullptr,
                                                 &g_s.psMagenta));
    const bool vsPatched = qvfr::mvrvs::IsPatched(g_s.vs.Get()) && !qvfr::mvrvs::IsPatched(g_s.vsClip.Get());

    const double sx = 2 / (kX1 - kX0), ox = -(kX1 + kX0) / (kX1 - kX0);
    const double sy = 2 / (kY1 - kY0), oy = -(kY1 + kY0) / (kY1 - kY0);
    float po[16], pf[16];
    OuterProj(po);
    FocusProj(po, sx, ox, sy, oy, pf);
    auto cbuf = [&](const float* m, ComPtr<ID3D11Buffer>& b) {
        D3D11_BUFFER_DESC bd{64, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA init{m, 0, 0};
        return SUCCEEDED(dev->CreateBuffer(&bd, &init, &b));
    };
    ComPtr<ID3D11Buffer> game3;  // the game's buffer in VS slot 3
    const float zeros[64] = {};
    D3D11_BUFFER_DESC bd3{256, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
    D3D11_SUBRESOURCE_DATA init3{zeros, 0, 0};
    made = made && cbuf(po, g_s.camOuter) && cbuf(pf, g_s.camFocus) &&
           SUCCEEDED(dev->CreateBuffer(&bd3, &init3, &game3));
    g_s.color = Tex(kW, kH, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    g_s.depth = Tex(kW, kH, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL);
    ComPtr<ID3D11Texture2D> refColor = Tex(kFW, kFH, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_RENDER_TARGET);
    ComPtr<ID3D11Texture2D> refDepth = Tex(kFW, kFH, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL);
    D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = DXGI_FORMAT_D32_FLOAT;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    ComPtr<ID3D11RenderTargetView> refRtv;
    ComPtr<ID3D11DepthStencilView> refDsv;
    D3D11_DEPTH_STENCIL_DESC dsd{};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsd.DepthFunc = D3D11_COMPARISON_LESS;
    made = made && g_s.color && g_s.depth && refColor && refDepth &&
           SUCCEEDED(dev->CreateRenderTargetView(g_s.color.Get(), nullptr, &g_s.rtv)) &&
           SUCCEEDED(dev->CreateDepthStencilView(g_s.depth.Get(), &dv, &g_s.dsv)) &&
           SUCCEEDED(dev->CreateRenderTargetView(refColor.Get(), nullptr, &refRtv)) &&
           SUCCEEDED(dev->CreateDepthStencilView(refDepth.Get(), &dv, &refDsv)) &&
           SUCCEEDED(dev->CreateDepthStencilState(&dsd, &g_s.ds));
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    made = made && SUCCEEDED(dev->CreateRasterizerState(&rd, &g_s.rs));
    if (!made) {
        printf("MVR pass: D3D11 objects could not be created\n");
        return 1;
    }

    // Reference: focus L drawn straight with the focus projection at the focus size.
    ctx->OMSetRenderTargets(1, refRtv.GetAddressOf(), refDsv.Get());
    ctx->ClearRenderTargetView(refRtv.Get(), kClear);
    ctx->ClearDepthStencilView(refDsv.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    DrawGrid(g_s.camFocus.Get(), kFW, kFH);
    const std::vector<uint32_t> refC = Read(refColor.Get(), 0), refD = Read(refDepth.Get(), 0);

    // The fake CPhase1Render of outer L.
    for (void*& p : g_vt) p = (void*)&FakeFinish;
    g_vt[0x48 / 8] = (void*)&FakeExecute;
    g_vt[0x38 / 8] = (void*)&FakeFinish;
    ctx->VSSetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, game3.GetAddressOf());
    if (!HookContext(ctx.Get())) {
        printf("MVR pass: context hooks could not be installed\n");
        return 1;
    }
    qvfr::mvrpass::TestSetup(ctx.Get(), g_vt);
    qvfr::mvrpass::TestArm(kW, kH, kFW, kFH, sx, ox, sy, oy);
    uint8_t* task = (uint8_t*)&g_task;
    ctx->OMSetRenderTargets(1, g_s.rtv.GetAddressOf(), g_s.dsv.Get());  // as CBeginEye leaves them
    qvfr::mvrpass::BeforeTask(task, 0);  // arms the run with MVR
    (*(TaskFn**)task)[0x48 / 8](task);  // the engine's run
    qvfr::mvrpass::AfterTask(task);     // slice 0 back into the game's targets
    ID3D11RenderTargetView* boundRtv = nullptr;
    ID3D11DepthStencilView* boundDsv = nullptr;
    ctx->OMGetRenderTargets(1, &boundRtv, &boundDsv);
    const bool restored = boundRtv == g_s.rtv.Get() && boundDsv == g_s.dsv.Get();
    if (boundRtv) boundRtv->Release();
    if (boundDsv) boundDsv->Release();
    const qvfr::mvrpass::TestState st = qvfr::mvrpass::TestGet();
    ID3D11Texture2D* arrC = qvfr::mvrpass::TestArray(0);
    ID3D11Texture2D* arrD = qvfr::mvrpass::TestArray(1);
    const std::vector<uint32_t> game = Read(g_s.color.Get(), 0);
    std::vector<uint32_t> s0, s1, d1;
    // The same pass without MVR into the game's targets, for slice 0.
    ctx->OMSetRenderTargets(1, g_s.rtv.GetAddressOf(), g_s.dsv.Get());
    ctx->ClearRenderTargetView(g_s.rtv.Get(), kClear);
    ctx->ClearDepthStencilView(g_s.dsv.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    DrawGrid(g_s.camOuter.Get(), kW, kH);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ctx->VSSetShader(g_s.vsClip.Get(), nullptr, 0);
    ctx->PSSetShader(g_s.psMagenta.Get(), nullptr, 0);
    ctx->Draw(4, 0);
    const std::vector<uint32_t> plain = Read(g_s.color.Get(), 0);
    if (arrC && arrD) {
        s0 = Read(arrC, 0);
        s1 = Read(arrC, 1);
        d1 = Read(arrD, 1);
    }
    ComPtr<ID3D11Buffer> slot3;
    ctx->VSGetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, &slot3);
    qvfr::mvrpass::TestEnd();
    if (refC.empty() || refD.empty() || plain.empty() || game.empty() || s0.empty() || s1.empty() || d1.empty()) {
        printf("MVR pass: read-back failed (copies %p %p)\n", (void*)arrC, (void*)arrD);
        return 1;
    }

    const uint32_t clear = Pack(kClear);
    int plainDiff = 0, slice0Diff = 0, magenta0 = 0, magenta1 = 0, outside = 0, cornerOff = 0, cornerBig = 0,
        covered = 0, depthOff = 0;
    double depthMax = 0;
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) {
            const size_t i = (size_t)y * kW + x;
            plainDiff += s0[i] != plain[i];
            slice0Diff += s0[i] != game[i];
            magenta0 += s0[i] == kMagenta;
            magenta1 += s1[i] == kMagenta;
            if (x >= kFW || y >= kFH) {
                outside += s1[i] != clear;
                continue;
            }
            const size_t r = (size_t)y * kFW + x;
            covered += refC[r] != clear;
            const int d = ChannelDiff(s1[i], refC[r]);
            cornerOff += d > 0;
            cornerBig += d > 2;
            const double dd = std::fabs((double)AsFloat(d1[i]) - (double)AsFloat(refD[r]));
            if (dd > 1e-6) ++depthOff;
            if (refD[r] != 0x3F800000u && d1[i] != 0x3F800000u) depthMax = std::max(depthMax, dd);
        }
    const int cornerPx = (int)(kFW * kFH);

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

    const bool countsOk = vsPatched && st.ran && !st.faulted && st.targets == 2 && st.binds == 2 && st.clears == 2 &&
                          st.mapFailures == 0 && st.patchedDraws == 1 && st.plainDraws == 1 && st.cbSwaps == 1 &&
                          st.copiedBack == 2 && st.viewsFailures == 0 && g_finish == 1;
    const bool gameOk = plainDiff == 0 && slice0Diff == 0 && magenta0 > 0 && slot3.Get() == game3.Get() && restored;
    // Edge pixels may round differently (M times the outer position against the focus projection).
    const bool viewOk = magenta1 == 0 && outside == 0 && covered > cornerPx / 2 && cornerBig * 200 <= cornerPx &&
                        depthOff * 200 <= cornerPx;
    const bool ok = countsOk && gameOk && viewOk && !warnings;
    printf("MVR pass: %s; counters: targets %u, binds %u, clears %u, map failures %u, draws patched %u / unpatched %u, "
           "slot 3 swaps %u, copied back %u, SetMultiViewMode failures %u; Finish calls %d (%s)\n",
           ok ? "PASS" : "FAIL", st.targets, st.binds, st.clears, st.mapFailures, st.patchedDraws, st.plainDraws,
           st.cbSwaps, st.copiedBack, st.viewsFailures, g_finish, countsOk ? "OK" : "WRONG");
    printf("  game's targets: %s after the run, game's color vs slice 0 %d px differ, slice 0 vs the pass without MVR "
           "%d px differ, refused VS quad %d px in slice 0, VS slot 3 %s (%s)\n",
           restored ? "bound again" : "NOT bound again", slice0Diff, plainDiff, magenta0,
           slot3.Get() == game3.Get() ? "the game's buffer again" : "WRONG", gameOk ? "OK" : "WRONG");
    printf("  view 1: corner %ux%u vs the focus projection: %d px covered, %d px differ (%d by > 2), depth %d px off "
           "(max %.2g); outside the corner %d px not the clear color; refused VS quad %d px in slice 1 (%s)\n",
           kFW, kFH, covered, cornerOff, cornerBig, depthOff, depthMax, outside, magenta1, viewOk ? "OK" : "WRONG");
    printf("  debug layer %s, %d warnings\n", debugLayer ? "on" : "not installed", warnings);
    return ok ? 0 : 1;
}

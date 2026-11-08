// --mvr-forward: gate B9c (mvr_pass.h, the forward test) on a real D3D11 device, without the game and without SteamVR
// (NVIDIA with Multi-View Rendering only; elsewhere the test reports a skip).
// The modules run as in the game: the sun effects module's hooks (CreatePixelShader for the PS patch, Draw), the VS and
// PS patches, the pass module's own VS and draw hooks; hooks on OMSetRenderTargets and the clears call the module the
// way quad's detour does. A fake CPhase3Render task of outer L draws a forward-like pass: a grid of tilted quads (a
// patched VS with outer's projection and TAA jitter, screen coordinates as the game's VS give them) whose PS reads
// globalTexture0 by Load at the screen input's pixel and gathers globalTexture2 at SV_Position (as the game's forward
// PS do, B2), with LESS_EQUAL against the depth the earlier tasks left. Fake focus clones (focus size) hold focus's
// state before its forward pass: another color, a depth with an occluder in one corner, other screen textures.
// As in the game (run of 2026-11-07 12:44), the task first binds its targets, clears a 1/2 size target and its depth,
// draws into them, and then binds the outer-size targets again for the forward draws: MVR must pause there.
// Checks:
// - outer: the game's color and depth equal the same pass drawn without MVR, bit for bit, the 1/2 size target too
//   (drawn with one view into the game's own target);
// - view 1: the corner of slice 1 (the module's stash) equals focus's own pass drawn straight into the clones with the
//   focus projection, focus's jitter and its screen textures (edge pixels may round differently): the copy-in of the
//   clones (color and depth), the screen inputs' map, the screen textures in t21..t25 and their UV scale;
// - the clones are not changed by outer's pass; the comparison after focus's own pass runs;
// - the module's counters and the debug layer (its linkage report for SV_RenderTargetArrayIndex is expected).
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hooks.h"
#include "mvr_pass.h"
#include "mvr_ps.h"
#include "mvr_vs.h"
#include "sun_fx.h"

using Microsoft::WRL::ComPtr;

namespace {

const char kHlsl[] = R"(
cbuffer Cam : register(b0) { row_major float4x4 proj; };
cbuffer G : register(b1) { float4 viewportPixelDimensions; float4 screenSpaceDimensions; };
struct VsOut { float4 pos : SV_Position; float4 scr : TEXCOORD0; float3 vp : VIEWPOS; };
VsOut VS(uint id : SV_VertexID) {
    const float2 c[6] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1)};
    uint q = id / 6, k = id % 6;
    uint qx = q % 6, qy = q / 6;
    float z = 2.0 + (float)((qx + 2 * qy) % 3) * 0.75 + c[k].x * 0.3;
    float2 t = float2(-1.2, -1.0) + (float2(qx, qy) + c[k] * 0.9) * 0.4;
    VsOut o;
    o.vp = float3(t * z, z);
    o.pos = mul(float4(o.vp, 1), proj);
    o.scr = float4(0.5 * (o.pos.x + o.pos.w), 0.5 * (o.pos.w - o.pos.y), o.pos.z, o.pos.w);
    return o;
}
Texture2D globalTexture0 : register(t3);
Texture2D<float> globalTexture2 : register(t5);
SamplerState pointSampler : register(s0);
// B12: a PS constant buffer bound at an offset (PSSetConstantBuffers1, like the game's _GLOBAL_ of the sky dome): zero at
// the buffer's start, so a restore without the offset gives a black red channel.
cbuffer Tint : register(b0) { float4 tint; };
float4 Fwd(VsOut i) : SV_Target {
    float2 uv = i.scr.xy / i.scr.w;
    int2 p = int2(min(uv * viewportPixelDimensions.zw, viewportPixelDimensions.zw - 1));
    float4 a = globalTexture0.Load(int3(p, 0));
    float4 g = globalTexture2.GatherRed(pointSampler, i.pos.xy * screenSpaceDimensions.zw);
    return float4(a.x * tint.x, a.y, dot(g, 0.25), frac(i.vp.x));
}
float4 VSOcc(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2(id & 1, id >> 1);
    return float4(lerp(-1.0, 0.0, uv.x), lerp(1.0, 0.0, uv.y), 0.05, 1);
}
float4 Zero(float4 p : SV_Position) : SV_Target { return 0; }
// B13: a G-buffer PS like the game's (PS 82705): the live track map at SV_Position * ssd.zw, extrapolated by the
// screen input.
Texture2D GLOBAL_liveTrackTextureMap_tex : register(t7);
float4 Gbuf(VsOut i) : SV_Target {
    float2 s = i.pos.xy * screenSpaceDimensions.zw;
    float2 uv = s + 2.1 * (i.scr.xy / i.scr.w - s);
    float4 m = GLOBAL_liveTrackTextureMap_tex.SampleLevel(pointSampler, uv, 0);
    return float4(m.xy, 0, 1);
}
// B12: a multisampled texture's samples side by side (pixel (x, y), sample s at (x * 8 + s, y)) for the read-back.
float4 VSFull(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
Texture2DMSArray<float4> msColor : register(t0);
Texture2DMSArray<float> msDepth : register(t0);
float4 UnpackC(float4 p : SV_Position) : SV_Target {
    uint2 q = uint2(p.xy);
    return msColor.Load(int3(q.x / 8, q.y, 0), q.x % 8);
}
float UnpackD(float4 p : SV_Position) : SV_Target {
    uint2 q = uint2(p.xy);
    return msDepth.Load(int3(q.x / 8, q.y, 0), q.x % 8);
}
)";
constexpr UINT kMs = 8;  // B12: MSAA 8x as the game's High

constexpr UINT kW = 96, kH = 80, kFW = 64, kFH = 48;
constexpr UINT kQuads = 30;
constexpr UINT kSlot0 = 3, kSlot2 = 5;  // the PS's registers of globalTexture0 and globalTexture2
constexpr double kX0 = -0.6, kX1 = 0.3, kY0 = -0.2, kY1 = 0.7;  // the focus window in outer NDC (y up)
const float kJitter[2] = {0.4f, -0.3f};
const float kOuterColor[4] = {0.1f, 0.2f, 0.3f, 1.0f};  // what the earlier tasks left in the forward target
const float kFocusColor[4] = {0.7f, 0.6f, 0.5f, 1.0f};
constexpr float kDepthLeft = 0.9f;

ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "mvr_forward_smoke", nullptr, nullptr, entry, target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
        return nullptr;
    }
    return code;
}

void OuterProj(float m[16]) {
    memset(m, 0, 16 * sizeof(float));
    const float n = 0.5f, f = 100.0f;
    m[0] = 1;
    m[5] = 1;
    m[10] = f / (f - n);
    m[11] = 1;
    m[14] = -n * f / (f - n);
}
void FocusProj(const float o[16], double sx, double ox, double sy, double oy, float m[16]) {
    memcpy(m, o, 16 * sizeof(float));
    for (int i = 0; i < 4; ++i) {
        m[i * 4 + 0] = (float)(sx * o[i * 4 + 0] + ox * o[i * 4 + 3]);
        m[i * 4 + 1] = (float)(sy * o[i * 4 + 1] + oy * o[i * 4 + 3]);
    }
}
void Jitter(float m[16], double jx, double jy) {
    m[8] += (float)jx;
    m[9] += (float)jy;
}

struct Set {  // one size: the forward target, its depth, the two screen textures, a 1/2 size target and depth
    ComPtr<ID3D11Texture2D> color, depth, gt0, gt2, half, halfDepth;
    ComPtr<ID3D11RenderTargetView> rtv, halfRtv;
    ComPtr<ID3D11DepthStencilView> dsv, halfDsv;
    ComPtr<ID3D11ShaderResourceView> srv0, srv2;
    ComPtr<ID3D11Buffer> cam, g;
    UINT w = 0, h = 0;
};

struct Scene {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ComPtr<ID3D11VertexShader> vs, vsOcc;
    ComPtr<ID3D11PixelShader> fwd, zero, gbuf;
    ComPtr<ID3D11Texture2D> live;  // B13: outer's live track map, texel (x, y) = its UV
    ComPtr<ID3D11ShaderResourceView> liveSrv;
    ComPtr<ID3D11DepthStencilState> ds;
    ComPtr<ID3D11RasterizerState> rs;
    ComPtr<ID3D11SamplerState> point;
    ComPtr<ID3D11Buffer> cam12;  // the game's camera constants in VS slot 12 at an offset (the jitter probe reads them)
    ComPtr<ID3D11Buffer> obj13;  // object constants in VS slot 13: worldViewProj = worldView * (1..16)
    ComPtr<ID3D11Buffer> tint0;  // PS slot 0 at byte 256 (1, 1, 1, 1), zero before it (B12)
    Set game, focus;
    ComPtr<ID3D11Texture2D> init[4];  // the start of the pass: game color, depth, focus color, depth
};
Scene g_s;
void BindTint(ID3D11DeviceContext* c);

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
    return qvfr::hooks::CreateAt(vt[33], (void*)D_OM, (void**)&o_om, "OMSetRenderTargets") &&
           qvfr::hooks::CreateAt(vt[50], (void*)D_ClearRtv, (void**)&o_clearRtv, "ClearRenderTargetView") &&
           qvfr::hooks::CreateAt(vt[53], (void*)D_ClearDsv, (void**)&o_clearDsv, "ClearDepthStencilView");
}

ComPtr<ID3D11Texture2D> Tex(UINT w, UINT h, DXGI_FORMAT f, UINT bind, const void* data = nullptr, UINT pitch = 0) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = f;
    d.SampleDesc = {1, 0};
    d.BindFlags = bind;
    if (!bind) {
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    }
    D3D11_SUBRESOURCE_DATA sd{data, pitch, 0};
    ComPtr<ID3D11Texture2D> t;
    g_s.dev->CreateTexture2D(&d, data ? &sd : nullptr, &t);
    return t;
}

// Screen textures with distinct values per texel and per set (`k` = 0 outer, 1 focus).
bool MakeSet(Set& s, UINT w, UINT h, int k, const float proj[16]) {
    ID3D11Device* dev = g_s.dev;
    s.w = w;
    s.h = h;
    std::vector<float> p0((size_t)w * h * 4), p2((size_t)w * h);
    for (UINT y = 0; y < h; ++y)
        for (UINT x = 0; x < w; ++x) {
            float* p = &p0[((size_t)y * w + x) * 4];
            p[0] = 0.01f * x + 0.5f * k;
            p[1] = 0.01f * y + 0.25f * k;
            p[2] = p[3] = 0;
            p2[(size_t)y * w + x] = 0.003f * x + 0.007f * y + 0.6f * k;
        }
    s.color = Tex(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    s.depth = Tex(w, h, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL);
    s.gt0 = Tex(w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_BIND_SHADER_RESOURCE, p0.data(), w * 16);
    s.gt2 = Tex(w, h, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, p2.data(), w * 4);
    s.half = Tex(w / 2, h / 2, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_RENDER_TARGET);
    s.halfDepth = Tex(w / 2, h / 2, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL);
    if (!s.color || !s.depth || !s.gt0 || !s.gt2 || !s.half || !s.halfDepth) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = DXGI_FORMAT_D32_FLOAT;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    const float g[8] = {0, 0, (float)w, (float)h, 1, 1, 1.0f / w, 1.0f / h};
    D3D11_BUFFER_DESC bd{64, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
    D3D11_SUBRESOURCE_DATA cam{proj, 0, 0};
    D3D11_BUFFER_DESC gd{32, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
    D3D11_SUBRESOURCE_DATA gi{g, 0, 0};
    return SUCCEEDED(dev->CreateRenderTargetView(s.color.Get(), nullptr, &s.rtv)) &&
           SUCCEEDED(dev->CreateDepthStencilView(s.depth.Get(), &dv, &s.dsv)) &&
           SUCCEEDED(dev->CreateRenderTargetView(s.half.Get(), nullptr, &s.halfRtv)) &&
           SUCCEEDED(dev->CreateDepthStencilView(s.halfDepth.Get(), &dv, &s.halfDsv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(s.gt0.Get(), nullptr, &s.srv0)) &&
           SUCCEEDED(dev->CreateShaderResourceView(s.gt2.Get(), nullptr, &s.srv2)) &&
           SUCCEEDED(dev->CreateBuffer(&bd, &cam, &s.cam)) && SUCCEEDED(dev->CreateBuffer(&gd, &gi, &s.g));
}

bool g_halfDraw = true;  // the live mode's pass draws nothing into the 1/2 size target (as the game's, B9c run 2)

// B11: the velocity fill source stands in for mvr_motion: it notes what it got and clears the target to a marker.
const float kVelMark[4] = {0.25f, -0.5f, 0, 0};
constexpr uint64_t kVelMarkBits = 0xB8003400;  // (0.25, -0.5) as R16G16_FLOAT
struct FillCall {
    int calls = 0, eye = -1;
    ID3D11Texture2D *depth = nullptr, *target = nullptr;  // compared only
    ID3D11RenderTargetView* rtv = nullptr;
};
FillCall g_fill;
bool FakeFill(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* depth, ID3D11Texture2D* target, const char** why) {
    ++g_fill.calls;
    g_fill.eye = eye;
    g_fill.depth = depth;
    g_fill.target = target;
    if (g_fill.rtv) c->ClearRenderTargetView(g_fill.rtv, kVelMark);
    *why = nullptr;
    return true;
}

// The forward pass of one set into its targets (the game binds them inside the task).
void DrawForward(const Set& s) {
    ID3D11DeviceContext* c = g_s.ctx;
    ID3D11RenderTargetView* rtv = s.rtv.Get();
    c->OMSetRenderTargets(1, &rtv, s.dsv.Get());
    const float black[4] = {0, 0, 0, 0};
    c->ClearRenderTargetView(s.halfRtv.Get(), black);
    c->ClearDepthStencilView(s.halfDsv.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    ID3D11RenderTargetView* hrtv = s.halfRtv.Get();
    c->OMSetRenderTargets(1, &hrtv, s.halfDsv.Get());
    D3D11_VIEWPORT hvp{0, 0, (float)(s.w / 2), (float)(s.h / 2), 0, 1};
    c->RSSetViewports(1, &hvp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    c->RSSetState(g_s.rs.Get());
    ID3D11Buffer* vcb[4] = {s.cam.Get(), nullptr, nullptr, nullptr};
    c->VSSetConstantBuffers(0, 4, vcb);
    ComPtr<ID3D11DeviceContext1> c1;
    if (SUCCEEDED(c->QueryInterface(IID_PPV_ARGS(&c1)))) {
        const UINT first = 16, num = 64;  // byte 256 of the buffer, 1024 bytes
        c1->VSSetConstantBuffers1(12, 1, g_s.cam12.GetAddressOf(), &first, &num);
        c->VSSetConstantBuffers(13, 1, g_s.obj13.GetAddressOf());
    }
    ID3D11Buffer* pcb[2] = {nullptr, s.g.Get()};
    c->PSSetConstantBuffers(0, 2, pcb);
    BindTint(c);
    c->PSSetShaderResources(kSlot0, 1, s.srv0.GetAddressOf());
    c->PSSetShaderResources(kSlot2, 1, s.srv2.GetAddressOf());
    c->PSSetSamplers(0, 1, g_s.point.GetAddressOf());
    c->VSSetShader(g_s.vs.Get(), nullptr, 0);
    c->PSSetShader(g_s.fwd.Get(), nullptr, 0);
    if (g_halfDraw) c->Draw(kQuads * 6, 0);  // the 1/2 size target (one view)
    c->OMSetRenderTargets(1, &rtv, s.dsv.Get());
    D3D11_VIEWPORT vp{0, 0, (float)s.w, (float)s.h, 0, 1};
    c->RSSetViewports(1, &vp);
    // The grid through the sun effects module's Draw hook, its first quad once more (same depth, LESS_EQUAL) through
    // the pass module's own DrawInstanced hook (SV_VertexID does not include the start vertex).
    c->Draw(kQuads * 6, 0);
    c->DrawInstanced(6, 1, 0, 0);
    ID3D11ShaderResourceView* none[8] = {};
    c->PSSetShaderResources(0, 8, none);
    c->OMSetRenderTargets(0, nullptr, nullptr);
}

// The state before the forward pass: outer's color and depth as left, focus's clones with another color and an
// occluder in the top left quarter of its depth.
void MakeStart() {
    ID3D11DeviceContext* c = g_s.ctx;
    c->ClearRenderTargetView(g_s.game.rtv.Get(), kOuterColor);
    c->ClearDepthStencilView(g_s.game.dsv.Get(), D3D11_CLEAR_DEPTH, kDepthLeft, 0);
    c->ClearRenderTargetView(g_s.focus.rtv.Get(), kFocusColor);
    c->ClearDepthStencilView(g_s.focus.dsv.Get(), D3D11_CLEAR_DEPTH, kDepthLeft, 0);
    c->OMSetRenderTargets(0, nullptr, g_s.focus.dsv.Get());
    D3D11_VIEWPORT vp{0, 0, (float)kFW, (float)kFH, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->RSSetState(g_s.rs.Get());
    c->VSSetShader(g_s.vsOcc.Get(), nullptr, 0);
    c->PSSetShader(nullptr, nullptr, 0);
    c->Draw(4, 0);
    c->OMSetRenderTargets(0, nullptr, nullptr);
}

void Save() {
    ID3D11Texture2D* src[4] = {g_s.game.color.Get(), g_s.game.depth.Get(), g_s.focus.color.Get(), g_s.focus.depth.Get()};
    for (int i = 0; i < 4; ++i) {
        D3D11_TEXTURE2D_DESC d{};
        src[i]->GetDesc(&d);
        g_s.dev->CreateTexture2D(&d, nullptr, &g_s.init[i]);
        g_s.ctx->CopyResource(g_s.init[i].Get(), src[i]);
    }
}
void Restore() {
    ID3D11Texture2D* dst[4] = {g_s.game.color.Get(), g_s.game.depth.Get(), g_s.focus.color.Get(), g_s.focus.depth.Get()};
    for (int i = 0; i < 4; ++i) g_s.ctx->CopyResource(dst[i], g_s.init[i].Get());
}

// B12: the forward pass with MSAA 8x: its color (RGBA16F with a shader view, as the game's 15454) and depth (R32
// typeless; outer's without a shader view like the engine's, focus's clone without one either).
struct MsSet {
    ComPtr<ID3D11Texture2D> color, depth;
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11DepthStencilView> dsv;
};
MsSet g_msGame, g_msFocus;

// The fake clones module.
ID3D11Texture2D* CloneOf(ID3D11Resource* r) {
    if (r && r == g_msGame.color.Get()) return g_msFocus.color.Get();
    if (r && r == g_msGame.depth.Get()) return g_msFocus.depth.Get();
    if (r == g_s.game.color.Get()) return g_s.focus.color.Get();
    if (r == g_s.game.depth.Get()) return g_s.focus.depth.Get();
    if (r == g_s.game.gt0.Get()) return g_s.focus.gt0.Get();
    if (r == g_s.game.gt2.Get()) return g_s.focus.gt2.Get();
    if (r == g_s.game.half.Get()) return g_s.focus.half.Get();
    if (r == g_s.game.halfDepth.Get()) return g_s.focus.halfDepth.Get();
    return nullptr;
}
bool JitterOf(float out[2]) {
    out[0] = kJitter[0];
    out[1] = kJitter[1];
    return true;
}

// One texture (subresource `sub`) as raw texels (8 bytes for RGBA16F, 4 for R32).
std::vector<uint64_t> Read(ID3D11Texture2D* src, UINT sub = 0) {
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    ComPtr<ID3D11Texture2D> st = Tex(d.Width, d.Height, d.Format, 0);
    std::vector<uint64_t> px;
    if (!st) return px;
    g_s.ctx->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src, sub, nullptr);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(g_s.ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return px;
    const size_t texel = d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
    px.resize((size_t)d.Width * d.Height);
    for (UINT y = 0; y < d.Height; ++y)
        for (UINT x = 0; x < d.Width; ++x) {
            uint64_t v = 0;
            memcpy(&v, (const uint8_t*)m.pData + (size_t)y * m.RowPitch + x * texel, texel);
            px[(size_t)y * d.Width + x] = v;
        }
    g_s.ctx->Unmap(st.Get(), 0);
    return px;
}

float Half(uint32_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? m / 1024.0f * std::pow(2.0f, -14.0f) : (1 + m / 1024.0f) * std::pow(2.0f, (float)e - 15);
    return s ? -v : v;
}
float MaxHalfDiff(uint64_t a, uint64_t b) {
    float d = 0;
    for (int i = 0; i < 4; ++i)
        d = std::max(d, std::fabs(Half((uint32_t)(a >> (16 * i)) & 0xFFFF) - Half((uint32_t)(b >> (16 * i)) & 0xFFFF)));
    return d;
}
float AsFloat(uint64_t v) {
    float f;
    const uint32_t u = (uint32_t)v;
    memcpy(&f, &u, 4);
    return f;
}

// B13: outer's G-buffer pass (the game binds and clears inside the task): the grid with the live track PS.
const float kGbufClear[4] = {0, 0, 0, 0};
void DrawGbuf(const Set& s) {
    ID3D11DeviceContext* c = g_s.ctx;
    ID3D11RenderTargetView* rtv = s.rtv.Get();
    c->OMSetRenderTargets(1, &rtv, s.dsv.Get());
    c->ClearRenderTargetView(rtv, kGbufClear);
    c->ClearDepthStencilView(s.dsv.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    D3D11_VIEWPORT vp{0, 0, (float)s.w, (float)s.h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    c->RSSetState(g_s.rs.Get());
    ID3D11Buffer* vcb[4] = {s.cam.Get(), nullptr, nullptr, nullptr};
    c->VSSetConstantBuffers(0, 4, vcb);
    ID3D11Buffer* pcb[2] = {nullptr, s.g.Get()};
    c->PSSetConstantBuffers(0, 2, pcb);
    c->PSSetShaderResources(7, 1, g_s.liveSrv.GetAddressOf());
    c->PSSetSamplers(0, 1, g_s.point.GetAddressOf());
    c->VSSetShader(g_s.vs.Get(), nullptr, 0);
    c->PSSetShader(g_s.gbuf.Get(), nullptr, 0);
    c->Draw(kQuads * 6, 0);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(7, 1, &none);
    c->OMSetRenderTargets(0, nullptr, nullptr);
}

using TaskFn = void(__fastcall*)(void*);
void* g_vtPhase1[32];  // B13: outer's CPhase1Render
void* g_vtPhase3[32];
void* g_vtFocus[32];
void* g_vtDummy[32];
void* g_vtBuild3[32];  // focus's CPhase3BuildCommandThread (live mode)
struct FakeTask {
    void** vt;
};
int g_finish = 0;
void __fastcall FakeFinish(void*) { ++g_finish; }
void __fastcall OuterExecute(void* task) {
    DrawForward(g_s.game);
    (*(TaskFn**)task)[0x38 / 8](task);
}
void __fastcall FocusExecute(void* task) {
    DrawForward(g_s.focus);
    (*(TaskFn**)task)[0x38 / 8](task);
}
void __fastcall GbufExecute(void* task) {
    DrawGbuf(g_s.game);
    (*(TaskFn**)task)[0x38 / 8](task);
}
void RunTask(FakeTask& t) { (*(TaskFn**)&t)[0x48 / 8](&t); }

// ---------------------------------------------------------------- B12: MSAA

ComPtr<ID3D11Texture2D> MsTex(UINT w, UINT h, DXGI_FORMAT f, UINT bind, UINT slices) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = slices;
    d.Format = f;
    d.SampleDesc = {kMs, 0};
    d.BindFlags = bind;
    ComPtr<ID3D11Texture2D> t;
    g_s.dev->CreateTexture2D(&d, nullptr, &t);
    return t;
}

// `slices` 2: the game's own two-slice arrays (B17), with views of the engine's kind (they see slice 0).
bool MakeMsSet(MsSet& m, UINT w, UINT h, UINT depthBind, UINT slices) {
    m.color = MsTex(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, slices);
    m.depth = MsTex(w, h, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_DEPTH_STENCIL | depthBind, slices);
    if (!m.color || !m.depth) return false;
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
    D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = DXGI_FORMAT_D32_FLOAT;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
    return SUCCEEDED(g_s.dev->CreateRenderTargetView(m.color.Get(), &rv, &m.rtv)) &&
           SUCCEEDED(g_s.dev->CreateDepthStencilView(m.depth.Get(), &dv, &m.dsv));
}

struct MsShaders {
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> color, depth;
};
MsShaders g_msSh;

// B12: PS slot 0 at an offset, as the game binds its constants (the module's passes must restore the range).
void BindTint(ID3D11DeviceContext* c) {
    ComPtr<ID3D11DeviceContext1> c1;
    if (FAILED(c->QueryInterface(IID_PPV_ARGS(&c1)))) return;
    const UINT first = 16, num = 16;
    c1->PSSetConstantBuffers1(0, 1, g_s.tint0.GetAddressOf(), &first, &num);
}

// How the multisampled pass starts: 0 over what the earlier tasks left, 1 binds its targets and clears all of them first
// (as a game's forward pass may), 2 clears only its depth and first draws an opaque full-screen background without depth
// writes (AMS2's forward pass with MSAA: the sky dome; mvr_ms_color_skip).
int g_msMode = 0;
const float kMsClearColor[4] = {0.05f, 0.15f, 0.25f, 1.0f};
ComPtr<ID3D11DepthStencilState> g_msBgDs;  // depth test LESS_EQUAL, no writes (the sky dome's)
ComPtr<ID3D11PixelShader> g_msBgPs;

// The forward pass of `s` (projection, screen textures) into the multisampled targets `m`.
void DrawForwardMs(const Set& s, const MsSet& m) {
    ID3D11DeviceContext* c = g_s.ctx;
    ID3D11RenderTargetView* rtv = m.rtv.Get();
    c->OMSetRenderTargets(1, &rtv, m.dsv.Get());
    if (g_msMode == 1) c->ClearRenderTargetView(rtv, kMsClearColor);
    if (g_msMode != 0) c->ClearDepthStencilView(m.dsv.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    D3D11_VIEWPORT vp{0, 0, (float)s.w, (float)s.h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    if (g_msMode == 2) {  // the background over every sample (its old color must not matter)
        c->OMSetDepthStencilState(g_msBgDs.Get(), 0);
        c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        c->RSSetState(g_s.rs.Get());
        c->VSSetShader(g_msSh.vs.Get(), nullptr, 0);
        c->PSSetShader(g_msBgPs.Get(), nullptr, 0);
        c->Draw(3, 0);
    }
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    c->RSSetState(g_s.rs.Get());
    ID3D11Buffer* vcb[4] = {s.cam.Get(), nullptr, nullptr, nullptr};
    c->VSSetConstantBuffers(0, 4, vcb);
    ComPtr<ID3D11DeviceContext1> c1;
    if (SUCCEEDED(c->QueryInterface(IID_PPV_ARGS(&c1)))) {
        const UINT first = 16, num = 64;
        c1->VSSetConstantBuffers1(12, 1, g_s.cam12.GetAddressOf(), &first, &num);
        c->VSSetConstantBuffers(13, 1, g_s.obj13.GetAddressOf());
    }
    ID3D11Buffer* pcb[2] = {nullptr, s.g.Get()};
    c->PSSetConstantBuffers(0, 2, pcb);
    BindTint(c);
    c->PSSetShaderResources(kSlot0, 1, s.srv0.GetAddressOf());
    c->PSSetShaderResources(kSlot2, 1, s.srv2.GetAddressOf());
    c->PSSetSamplers(0, 1, g_s.point.GetAddressOf());
    c->VSSetShader(g_s.vs.Get(), nullptr, 0);
    c->PSSetShader(g_s.fwd.Get(), nullptr, 0);
    c->Draw(kQuads * 6, 0);
    c->DrawInstanced(6, 1, 0, 0);
    ID3D11ShaderResourceView* none[8] = {};
    c->PSSetShaderResources(0, 8, none);
    c->OMSetRenderTargets(0, nullptr, nullptr);
}
void __fastcall OuterExecuteMs(void* task) {
    DrawForwardMs(g_s.game, g_msGame);
    (*(TaskFn**)task)[0x38 / 8](task);
}
void* g_vtPhase3Ms[32];

// Outer's targets as the earlier tasks left them, focus's clones with another color and the occluder in its depth.
void MakeStartMs() {
    ID3D11DeviceContext* c = g_s.ctx;
    c->ClearRenderTargetView(g_msGame.rtv.Get(), kOuterColor);
    c->ClearDepthStencilView(g_msGame.dsv.Get(), D3D11_CLEAR_DEPTH, kDepthLeft, 0);
    c->ClearRenderTargetView(g_msFocus.rtv.Get(), kFocusColor);
    c->ClearDepthStencilView(g_msFocus.dsv.Get(), D3D11_CLEAR_DEPTH, kDepthLeft, 0);
    c->OMSetRenderTargets(0, nullptr, g_msFocus.dsv.Get());
    D3D11_VIEWPORT vp{0, 0, (float)kFW, (float)kFH, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(g_s.ds.Get(), 0);
    c->RSSetState(g_s.rs.Get());
    c->VSSetShader(g_s.vsOcc.Get(), nullptr, 0);
    c->PSSetShader(nullptr, nullptr, 0);
    c->Draw(4, 0);
    c->OMSetRenderTargets(0, nullptr, nullptr);
}



// Slice `slice` of a multisampled texture, its samples side by side (kMs * width x height texels).
std::vector<uint64_t> ReadMs(ID3D11Texture2D* t, UINT slice = 0) {
    std::vector<uint64_t> none;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    const bool depth = (d.BindFlags & D3D11_BIND_DEPTH_STENCIL) != 0;
    ComPtr<ID3D11Texture2D> src = t;
    if (!(d.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        src.Reset();
        if (FAILED(g_s.dev->CreateTexture2D(&sd, nullptr, &src))) return none;
        g_s.ctx->CopyResource(src.Get(), t);
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = depth ? DXGI_FORMAT_R32_FLOAT : d.Format;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
    sv.Texture2DMSArray = {slice, 1};
    ComPtr<ID3D11ShaderResourceView> srv;
    const DXGI_FORMAT outF = depth ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
    ComPtr<ID3D11Texture2D> out = Tex(d.Width * kMs, d.Height, outF, D3D11_BIND_RENDER_TARGET);
    ComPtr<ID3D11RenderTargetView> rtv;
    if (!out || FAILED(g_s.dev->CreateShaderResourceView(src.Get(), &sv, &srv)) ||
        FAILED(g_s.dev->CreateRenderTargetView(out.Get(), nullptr, &rtv)))
        return none;
    ID3D11DeviceContext* c = g_s.ctx;
    ID3D11RenderTargetView* r = rtv.Get();
    c->OMSetRenderTargets(1, &r, nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)(d.Width * kMs), (float)d.Height, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->IASetInputLayout(nullptr);
    c->OMSetDepthStencilState(nullptr, 0);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    c->RSSetState(g_s.rs.Get());
    c->VSSetShader(g_msSh.vs.Get(), nullptr, 0);
    c->PSSetShader(depth ? g_msSh.depth.Get() : g_msSh.color.Get(), nullptr, 0);
    c->PSSetShaderResources(0, 1, srv.GetAddressOf());
    c->Draw(3, 0);
    ID3D11ShaderResourceView* nsrv = nullptr;
    c->PSSetShaderResources(0, 1, &nsrv);
    c->OMSetRenderTargets(0, nullptr, nullptr);
    return Read(out.Get());
}

struct MsRun {
    qvfr::mvrpass::TestLiveState lv;
    bool ran = false;
    std::vector<uint64_t> outerC, outerD, focusC, focusD;
};
struct MsCheck {
    bool ok = false, skipped = false;
    std::string text;
};

// B12: the live forward pass (B9d) with multisampled targets, three times: (1) the module's copies, the pass draws over
// what the earlier tasks left (the corner of slice 1 gets focus's clones sample by sample); (2) outer's targets as the
// game's own two-slice arrays (B17 for MSAA) and a pass that binds and then clears all of them (no copy-in); (3) the
// copies and a pass that clears only its depth and draws a full-screen background first (the color gets no copy-in,
// mvr_ms_color_skip). Each: outer's samples bit for bit as the pass without the module, focus's clones per sample as
// focus's own pass (edges may round).
MsCheck RunMsaa(double sx, double ox, double sy, double oy) {
    MsCheck r;
    ID3D11Device* dev = g_s.dev;
    ID3D11DeviceContext* ctx = g_s.ctx;
    ComPtr<ID3DBlob> vsCode = Compile("VSFull", "vs_5_0"), cCode = Compile("UnpackC", "ps_5_0"),
                     dCode = Compile("UnpackD", "ps_5_0"), bgCode = Compile("Zero", "ps_5_0");
    D3D11_DEPTH_STENCIL_DESC bgd{};
    bgd.DepthEnable = TRUE;
    bgd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    bgd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    UINT q = 0;
    if (FAILED(dev->CheckMultisampleQualityLevels(DXGI_FORMAT_R16G16B16A16_FLOAT, kMs, &q)) || !q) {
        r.ok = r.skipped = true;
        r.text = "no MSAA 8x for RGBA16F: SKIPPED";
        return r;
    }
    MsSet plainGame, arrGame;
    bool made = vsCode && cCode && dCode &&
                SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr,
                                                  &g_msSh.vs)) &&
                SUCCEEDED(dev->CreatePixelShader(cCode->GetBufferPointer(), cCode->GetBufferSize(), nullptr,
                                                 &g_msSh.color)) &&
                SUCCEEDED(dev->CreatePixelShader(dCode->GetBufferPointer(), dCode->GetBufferSize(), nullptr,
                                                 &g_msSh.depth)) &&
                bgCode && SUCCEEDED(dev->CreatePixelShader(bgCode->GetBufferPointer(), bgCode->GetBufferSize(), nullptr,
                                                           &g_msBgPs)) &&
                SUCCEEDED(dev->CreateDepthStencilState(&bgd, &g_msBgDs)) &&
                MakeMsSet(plainGame, kW, kH, 0, 1) && MakeMsSet(arrGame, kW, kH, D3D11_BIND_SHADER_RESOURCE, 2) &&
                MakeMsSet(g_msFocus, kFW, kFH, 0, 1);
    if (!made) {
        r.text = "D3D11 objects could not be made: FAIL";
        return r;
    }
    g_msGame = plainGame;
    MakeStartMs();
    ComPtr<ID3D11Texture2D> init[4];
    ID3D11Texture2D* src[4] = {g_msGame.color.Get(), g_msGame.depth.Get(), g_msFocus.color.Get(), g_msFocus.depth.Get()};
    for (int i = 0; i < 4; ++i) {
        D3D11_TEXTURE2D_DESC d{};
        src[i]->GetDesc(&d);
        dev->CreateTexture2D(&d, nullptr, &init[i]);
        ctx->CopyResource(init[i].Get(), src[i]);
    }
    // The start of the pass in `g` (slice 0 of the arrays) and focus's clones.
    auto restore = [&](MsSet& g) {
        ctx->CopySubresourceRegion(g.color.Get(), 0, 0, 0, 0, init[0].Get(), 0, nullptr);
        ctx->CopySubresourceRegion(g.depth.Get(), 0, 0, 0, 0, init[1].Get(), 0, nullptr);
        ctx->CopyResource(g_msFocus.color.Get(), init[2].Get());
        ctx->CopyResource(g_msFocus.depth.Get(), init[3].Get());
    };
    // References without the module, for both kinds of pass.
    constexpr int kRuns = 3;
    std::vector<uint64_t> refOC[kRuns], refOD[kRuns], refFC[kRuns], refFD[kRuns];
    for (int k = 0; k < kRuns; ++k) {
        g_msMode = k;
        restore(g_msGame);
        DrawForwardMs(g_s.game, g_msGame);
        DrawForwardMs(g_s.focus, g_msFocus);
        refOC[k] = ReadMs(g_msGame.color.Get());
        refOD[k] = ReadMs(g_msGame.depth.Get());
        refFC[k] = ReadMs(g_msFocus.color.Get());
        refFD[k] = ReadMs(g_msFocus.depth.Get());
    }
    memcpy(g_vtPhase3Ms, g_vtPhase3, sizeof(g_vtPhase3Ms));
    g_vtPhase3Ms[0x48 / 8] = (void*)&OuterExecuteMs;
    static ID3D11Resource* s_arr[2] = {};
    s_arr[0] = arrGame.color.Get();
    s_arr[1] = arrGame.depth.Get();
    MsRun run[kRuns];
    for (int k = 0; k < kRuns; ++k) {
        g_msMode = k;
        g_msGame = k == 1 ? arrGame : plainGame;
        restore(g_msGame);
        const qvfr::mvrpass::NativeFn isArr = [](ID3D11Resource* x) { return x && (x == s_arr[0] || x == s_arr[1]); };
        qvfr::mvrpass::TestNative(k == 1 ? isArr : nullptr, nullptr);
        qvfr::mvrpass::OnFrameStart(true);
        qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
        qvfr::mvrpass::TestForwardLive(g_vtPhase3Ms);
        FakeTask t{g_vtPhase3Ms};
        run[k].ran = !qvfr::mvrpass::BeforeTask((uint8_t*)&t, 0);
        if (run[k].ran) RunTask(t);
        qvfr::mvrpass::AfterTask((uint8_t*)&t);
        run[k].lv = qvfr::mvrpass::TestLiveGet();
        run[k].outerC = ReadMs(g_msGame.color.Get());
        run[k].outerD = ReadMs(g_msGame.depth.Get());
        run[k].focusC = ReadMs(g_msFocus.color.Get());
        run[k].focusD = ReadMs(g_msFocus.depth.Get());
        qvfr::mvrpass::TestEnd();
        qvfr::mvrpass::TestNative(nullptr, nullptr);
    }
    g_msMode = 0;
    g_msGame = MsSet{};
    const int samples = (int)(kFW * kFH * kMs);
    bool ok = true;
    char line[1024];
    std::string text;
    for (int k = 0; k < kRuns; ++k) {
        const MsRun& m = run[k];
        const qvfr::mvrpass::TestLiveState& lv = m.lv;
        int outer = 0, cOff = 0, cBig = 0, dOff = 0, covered = 0;
        float cMax = 0;
        bool sizes = m.outerC.size() == refOC[k].size() && m.outerD.size() == refOD[k].size() &&
                     m.focusC.size() == (size_t)samples && m.focusD.size() == (size_t)samples &&
                     refFC[k].size() == (size_t)samples && refFD[k].size() == (size_t)samples && !refOC[k].empty();
        if (sizes) {
            for (size_t i = 0; i < refOC[k].size(); ++i) outer += m.outerC[i] != refOC[k][i] || m.outerD[i] != refOD[k][i];
            for (int i = 0; i < samples; ++i) {
                const float d = MaxHalfDiff(m.focusC[i], refFC[k][i]);
                cOff += d > 0;
                cBig += d > 0.02f;
                cMax = std::max(cMax, d);
                dOff += std::fabs(AsFloat(m.focusD[i]) - AsFloat(refFD[k][i])) > 1e-5f;
                covered += AsFloat(refFD[k][i]) != (k != 0 ? 1.0f : kDepthLeft);
            }
        }
        const bool counts =
            m.ran && lv.passes == 1 && lv.ok == 1 && lv.msTargets == 2 &&
            (k == 0   ? lv.copyIns == 2 && lv.clearFills == 0 && lv.colorSkips == 0 && lv.native == 0 && lv.copiedBack == 2
             : k == 1 ? lv.copyIns == 0 && lv.clearFills == 2 && lv.colorSkips == 0 && lv.native == 2 && lv.copiedBack == 0
                      : lv.copyIns == 0 && lv.clearFills == 1 && lv.colorSkips == 1 && lv.native == 0 &&
                            lv.copiedBack == 2);
        const bool runOk = sizes && counts && outer == 0 && covered > samples / 3 && cBig * 100 <= samples &&
                           dOff * 100 <= samples;
        ok = ok && runOk;
        snprintf(line, sizeof(line),
                 "  MSAA %ux (B12) %s: live pass %u, into the clones %u%s%s; targets multisampled %u, copied in %u, "
                 "cleared first %u, color without copy-in %u, as the game's arrays %u, copied back %u; outer vs without "
                 "MVR %d samples differ; "
                 "clones vs focus's own pass (%d samples, %d covered): color %d differ (%d by > 0.02, max %.4f), depth "
                 "%d off (%s)\n",
                 kMs,
                 k == 0   ? "with copies"
                 : k == 1 ? "as the game's arrays, cleared first"
                          : "with copies, depth cleared and a background first",
                 lv.passes, lv.ok, lv.why.empty() ? "" : ": ", lv.why.c_str(), lv.msTargets, lv.copyIns, lv.clearFills,
                 lv.colorSkips, lv.native,
                 lv.copiedBack, outer, samples, covered, cOff, cBig, cMax, dOff, runOk ? "OK" : "WRONG");
        text += line;
    }
    r.ok = ok;
    r.text = text;
    return r;
}

}  // namespace

int RunMvrForward() {
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
    if (!qvfr::sunfx::TestInstall(dev.Get(), ctx.Get()) || !qvfr::mvrvs::TestInstall(dev.Get())) {
        printf("MVR forward: the hooks could not be installed or NVAPI is not available (not NVIDIA?): SKIPPED\n");
        return 0;
    }
    if (!qvfr::mvrvs::DeviceHasMvr(dev.Get())) {
        printf("MVR forward: no Multi-View Rendering on this GPU: SKIPPED\n");
        return 0;
    }
    if (!qvfr::mvrps::TestInstall()) {
        printf("MVR forward: FAIL (the PS module could not be installed)\n");
        return 1;
    }
    g_s.dev = dev.Get();
    g_s.ctx = ctx.Get();
    qvfr::mvrvs::TestSetEnabled(true);
    qvfr::mvrps::TestSetEnabled(true);
    ComPtr<ID3DBlob> vsCode = Compile("VS", "vs_5_0"), occCode = Compile("VSOcc", "vs_5_0"),
                     fwdCode = Compile("Fwd", "ps_5_0"), zeroCode = Compile("Zero", "ps_5_0"),
                     gbufCode = Compile("Gbuf", "ps_5_0");
    if (!vsCode || !occCode || !fwdCode || !zeroCode || !gbufCode) return 1;
    bool made =
        SUCCEEDED(dev->CreatePixelShader(zeroCode->GetBufferPointer(), zeroCode->GetBufferSize(), nullptr, &g_s.zero)) &&
        SUCCEEDED(dev->CreatePixelShader(gbufCode->GetBufferPointer(), gbufCode->GetBufferSize(), nullptr, &g_s.gbuf)) &&
        SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &g_s.vs)) &&
        SUCCEEDED(dev->CreateVertexShader(occCode->GetBufferPointer(), occCode->GetBufferSize(), nullptr, &g_s.vsOcc)) &&
        SUCCEEDED(dev->CreatePixelShader(fwdCode->GetBufferPointer(), fwdCode->GetBufferSize(), nullptr, &g_s.fwd));
    const qvfr::mvrps::Info psInfo = qvfr::mvrps::InfoOf(g_s.fwd.Get());
    const bool shadersOk = made && qvfr::mvrvs::IsPatched(g_s.vs.Get()) && psInfo.kind == qvfr::mvrps::Kind::Patched &&
                           psInfo.slot[0] == kSlot0 && psInfo.slot[2] == kSlot2 &&
                           psInfo.slot[1] == qvfr::mvrps::kNoSlot;
    const qvfr::mvrps::Info gbufInfo = qvfr::mvrps::InfoOf(g_s.gbuf.Get());
    bool gbufPsOk = made && gbufInfo.kind == qvfr::mvrps::Kind::Patched && gbufInfo.slot[qvfr::mvrps::kLiveTrack] == 7;
    for (unsigned k = 0; k < qvfr::mvrps::kLiveTrack; ++k) gbufPsOk = gbufPsOk && gbufInfo.slot[k] == qvfr::mvrps::kNoSlot;

    const double sx = 2 / (kX1 - kX0), ox = -(kX1 + kX0) / (kX1 - kX0);
    const double sy = 2 / (kY1 - kY0), oy = -(kY1 + kY0) / (kY1 - kY0);
    float po[16], pf[16];
    OuterProj(po);
    FocusProj(po, sx, ox, sy, oy, pf);
    Jitter(po, kJitter[0] / kW, kJitter[1] / kH);
    Jitter(pf, kJitter[0] / kFW, kJitter[1] / kFH);
    D3D11_DEPTH_STENCIL_DESC dsd{};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    D3D11_SAMPLER_DESC smp{};
    smp.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    smp.AddressU = smp.AddressV = smp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    smp.MaxLOD = D3D11_FLOAT32_MAX;
    made = made && MakeSet(g_s.game, kW, kH, 0, po) && MakeSet(g_s.focus, kFW, kFH, 1, pf) &&
           SUCCEEDED(dev->CreateDepthStencilState(&dsd, &g_s.ds)) && SUCCEEDED(dev->CreateRasterizerState(&rd, &g_s.rs)) &&
           SUCCEEDED(dev->CreateSamplerState(&smp, &g_s.point));
    {  // camera constants at byte 256: projMatrix 1..16 at +128, cameraJitter (0.25, -0.5) at +944
        std::vector<float> cam(1024, 0.0f);
        for (int i = 0; i < 16; ++i) cam[(256 + 128) / 4 + i] = (float)(i + 1);
        cam[(256 + 944) / 4] = 0.25f;
        cam[(256 + 944) / 4 + 1] = -0.5f;
        D3D11_BUFFER_DESC cd{4096, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA ci{cam.data(), 0, 0};
        made = made && SUCCEEDED(dev->CreateBuffer(&cd, &ci, &g_s.cam12));
        // worldView: a translation (1, 2, 3) (row vectors); worldViewProj = worldView * P with P = 1..16, so the probe's
        // worldView^-1 * worldViewProj must give 1..16 again.
        const float wv[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1};
        std::vector<float> obj(116, 0.0f);
        for (int r = 0; r < 4; ++r)
            for (int k = 0; k < 4; ++k) {
                float sum = 0;
                for (int j = 0; j < 4; ++j) sum += wv[r * 4 + j] * (float)(j * 4 + k + 1);
                obj[r * 4 + k] = sum;
            }
        memcpy(&obj[144 / 4], wv, sizeof(wv));
        D3D11_BUFFER_DESC od{464, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA oi{obj.data(), 0, 0};
        made = made && SUCCEEDED(dev->CreateBuffer(&od, &oi, &g_s.obj13));
        std::vector<float> tint(128, 0.0f);
        for (int i = 0; i < 4; ++i) tint[256 / 4 + i] = 1.0f;
        D3D11_BUFFER_DESC td{512, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA ti{tint.data(), 0, 0};
        made = made && SUCCEEDED(dev->CreateBuffer(&td, &ti, &g_s.tint0));
    }
    {  // B13: outer's live track map, a ramp (texel centres hold their UV: bilinear reads give the UV exactly)
        std::vector<float> lt((size_t)kW * kH * 4);
        for (UINT y = 0; y < kH; ++y)
            for (UINT x = 0; x < kW; ++x) {
                float* p = &lt[((size_t)y * kW + x) * 4];
                p[0] = (x + 0.5f) / kW;
                p[1] = (y + 0.5f) / kH;
                p[2] = 0;
                p[3] = 1;
            }
        g_s.live = Tex(kW, kH, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_BIND_SHADER_RESOURCE, lt.data(), kW * 16);
        made = made && g_s.live && SUCCEEDED(dev->CreateShaderResourceView(g_s.live.Get(), nullptr, &g_s.liveSrv));
    }
    if (!made) {
        printf("MVR forward: D3D11 objects could not be made\n");
        return 1;
    }

    // References without the module (MVR off: the patched shaders draw as the originals).
    MakeStart();
    Save();
    const std::vector<uint64_t> startF = Read(g_s.focus.color.Get()), startFD = Read(g_s.focus.depth.Get());
    DrawForward(g_s.game);
    const std::vector<uint64_t> refC = Read(g_s.game.color.Get()), refD = Read(g_s.game.depth.Get()),
                                refH = Read(g_s.game.half.Get());
    DrawForward(g_s.focus);
    const std::vector<uint64_t> refFC = Read(g_s.focus.color.Get()), refFD = Read(g_s.focus.depth.Get());
    Restore();

    if (!HookContext(ctx.Get())) {
        printf("MVR forward: context hooks could not be installed\n");
        return 1;
    }
    for (void*& p : g_vtPhase3) p = (void*)&FakeFinish;
    for (void*& p : g_vtDummy) p = (void*)&FakeFinish;
    g_vtPhase3[0x48 / 8] = (void*)&OuterExecute;
    memcpy(g_vtFocus, g_vtPhase3, sizeof(g_vtFocus));
    g_vtFocus[0x48 / 8] = (void*)&FocusExecute;
    memcpy(g_vtPhase1, g_vtDummy, sizeof(g_vtPhase1));
    g_vtPhase1[0x48 / 8] = (void*)&GbufExecute;
    qvfr::mvrpass::TestSetup(ctx.Get(), g_vtPhase1, g_vtDummy, &CloneOf, &JitterOf);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    qvfr::mvrpass::TestForward(g_vtPhase3);
    // Outer L's CPhase3Render: nothing bound before it (the game binds inside the task).
    FakeTask outer{g_vtPhase3};
    const bool outerSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&outer, 0);
    if (!outerSkip) RunTask(outer);
    qvfr::mvrpass::AfterTask((uint8_t*)&outer);
    const qvfr::mvrpass::TestForwardState st = qvfr::mvrpass::TestForwardGet();
    const std::vector<uint64_t> gC = Read(g_s.game.color.Get()), gD = Read(g_s.game.depth.Get()),
                                gH = Read(g_s.game.half.Get());
    const std::vector<uint64_t> fAfter = Read(g_s.focus.color.Get()), fdAfter = Read(g_s.focus.depth.Get());
    std::vector<uint64_t> oursC, oursD;
    for (int k = 0; k < st.stashed; ++k) {
        ID3D11Texture2D* t = qvfr::mvrpass::TestForwardStash(k);
        if (!t) continue;
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        (d.Format == DXGI_FORMAT_R32_TYPELESS ? oursD : oursC) = Read(t);
    }
    // Focus L's own CPhase3Render (the same class, another object): the module compares after it.
    FakeTask focus{g_vtPhase3};
    const bool focusSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&focus, 2);
    if (!focusSkip) {
        focus.vt = g_vtFocus;
        RunTask(focus);
    }
    qvfr::mvrpass::AfterTask((uint8_t*)&focus);
    const qvfr::mvrpass::TestForwardState st2 = qvfr::mvrpass::TestForwardGet();
    const std::vector<uint64_t> ownC = Read(g_s.focus.color.Get());
    qvfr::mvrpass::TestEnd();
    if (refC.empty() || refFC.empty() || gC.empty() || oursC.size() != refFC.size() || oursD.size() != refFD.size()) {
        printf("MVR forward: FAIL (read-back: ours %zu / %zu px, stashed %d, why: %s)\n", oursC.size(), oursD.size(),
               st.stashed, st.why.c_str());
        return 1;
    }

    int outerC = 0, outerD = 0, outerH = 0, clonesChanged = 0;
    for (size_t i = 0; i < refC.size(); ++i) {
        outerC += gC[i] != refC[i];
        outerD += gD[i] != refD[i];
    }
    for (size_t i = 0; i < refH.size() && i < gH.size(); ++i) outerH += gH[i] != refH[i];
    for (size_t i = 0; i < startF.size(); ++i) clonesChanged += fAfter[i] != startF[i] || fdAfter[i] != startFD[i];
    // View 1 vs focus's own: covered pixels, the occluder's quarter (the copied-in depth), edge rounding.
    const uint64_t focusStart = startF[0];  // the copied-in color (top left: the occluder keeps it)
    int covered = 0, occluded = 0, occOurs = 0, cOff = 0, cBig = 0, dOff = 0, ownOff = 0;
    float cMax = 0;
    for (UINT y = 0; y < kFH; ++y)
        for (UINT x = 0; x < kFW; ++x) {
            const size_t i = (size_t)y * kFW + x;
            covered += refFC[i] != focusStart;
            if (x < kFW / 2 && y < kFH / 2) {
                occluded += refFC[i] == focusStart;
                occOurs += oursC[i] == focusStart;
            }
            const float d = MaxHalfDiff(oursC[i], refFC[i]);
            cOff += d > 0;
            cBig += d > 0.02f;
            cMax = std::max(cMax, d);
            dOff += std::fabs(AsFloat(oursD[i]) - AsFloat(refFD[i])) > 1e-5f;
            ownOff += ownC[i] != refFC[i];
        }

    // Live mode (B9d): the same pass, view 1 straight into focus's clones (the depth as MVR drew it here, to compare it
    // with focus's own), the 1/2 size clears repeated on their clones, focus's CPhase3BuildCommandThread and
    // CPhase3Render skipped; the clones it wrote noted for focus R's twins.
    Restore();
    const float mark[4] = {0.3f, 0.9f, 0.1f, 1.0f};
    ctx->ClearRenderTargetView(g_s.focus.halfRtv.Get(), mark);
    ctx->ClearDepthStencilView(g_s.focus.halfDsv.Get(), D3D11_CLEAR_DEPTH, 0.25f, 0);
    g_halfDraw = false;
    for (void*& p : g_vtBuild3) p = (void*)&FakeFinish;
    qvfr::mvrpass::SetPhase3BuildVtable((uintptr_t)g_vtBuild3);
    qvfr::mvrpass::TestDepthBias(0, 0.0f, false);
    qvfr::mvrpass::OnFrameStart(true);
    qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
    qvfr::mvrpass::TestForwardLive(g_vtPhase3);
    FakeTask outerL{g_vtPhase3};
    const bool liveOuterSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&outerL, 0);
    if (!liveOuterSkip) RunTask(outerL);
    qvfr::mvrpass::AfterTask((uint8_t*)&outerL);
    const std::vector<uint64_t> lgC = Read(g_s.game.color.Get()), lgD = Read(g_s.game.depth.Get());
    const std::vector<uint64_t> lC = Read(g_s.focus.color.Get()), lD = Read(g_s.focus.depth.Get()),
                                lH = Read(g_s.focus.half.Get()), lHD = Read(g_s.focus.halfDepth.Get());
    // Mode 3 (default): focus's velocity pass (an R16G16 target with the depth clone) gets the conservative copy.
    ComPtr<ID3D11Texture2D> velTex = Tex(kFW, kFH, DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_RENDER_TARGET);
    ComPtr<ID3D11RenderTargetView> velRtv;
    if (velTex) dev->CreateRenderTargetView(velTex.Get(), nullptr, &velRtv);
    qvfr::mvrpass::FocusGroupIn(0);
    bool velOk = false, plainOk = false;
    if (velRtv) {
        ID3D11RenderTargetView* vr = velRtv.Get();
        ctx->OMSetRenderTargets(1, &vr, g_s.focus.dsv.Get());
        ComPtr<ID3D11DepthStencilView> got;
        ctx->OMGetRenderTargets(0, nullptr, &got);
        ComPtr<ID3D11Resource> gr;
        if (got) got->GetResource(&gr);
        velOk = gr && gr.Get() != (ID3D11Resource*)g_s.focus.depth.Get();
        ID3D11RenderTargetView* fr = g_s.focus.rtv.Get();  // another target with the same depth: as it is
        ctx->OMSetRenderTargets(1, &fr, g_s.focus.dsv.Get());
        got.Reset();
        gr.Reset();
        ctx->OMGetRenderTargets(0, nullptr, &got);
        if (got) got->GetResource(&gr);
        plainOk = gr.Get() == (ID3D11Resource*)g_s.focus.depth.Get();
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
    }
    // B11: the velocity pass's first draw (the linear depth, R32 at the target's size, in PS t0) gets the fill and
    // both draws are dropped; after a bind without the depth clone the next draw goes into the target (top left
    // quarter, the PS gives 0 there).
    int velQuarter = 0, velMarked = 0, velOther = 0;
    if (velRtv) {
        g_fill = FillCall{};
        g_fill.rtv = velRtv.Get();
        qvfr::mvrpass::SetVelocityFill(&FakeFill);
        const float zero[4] = {0, 0, 0, 0};
        ctx->ClearRenderTargetView(velRtv.Get(), zero);
        ID3D11RenderTargetView* vr = velRtv.Get();
        ctx->OMSetRenderTargets(1, &vr, g_s.focus.dsv.Get());
        ID3D11ShaderResourceView* ld = g_s.focus.srv2.Get();
        ctx->PSSetShaderResources(0, 1, &ld);
        const D3D11_VIEWPORT vp{0, 0, (float)kFW, (float)kFH, 0, 1};
        ctx->RSSetViewports(1, &vp);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        ctx->VSSetShader(g_s.vsOcc.Get(), nullptr, 0);
        ctx->PSSetShader(g_s.zero.Get(), nullptr, 0);
        ctx->OMSetDepthStencilState(nullptr, 0);
        ctx->RSSetState(g_s.rs.Get());
        ctx->Draw(4, 0);
        ctx->Draw(4, 0);
        ctx->OMSetRenderTargets(1, &vr, nullptr);
        ctx->Draw(4, 0);
        ID3D11ShaderResourceView* none = nullptr;
        ctx->PSSetShaderResources(0, 1, &none);
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        qvfr::mvrpass::SetVelocityFill(nullptr);
        const std::vector<uint64_t> v = Read(velTex.Get());
        for (UINT y = 0; y < kFH && v.size() == (size_t)kFW * kFH; ++y)
            for (UINT x = 0; x < kFW; ++x) {
                const uint64_t p = v[(size_t)y * kFW + x];
                if (x < kFW / 2 && y < kFH / 2)
                    velQuarter += p == 0;
                else if (p == kVelMarkBits)
                    ++velMarked;
                else
                    ++velOther;
            }
    }
    FakeTask build{g_vtBuild3};
    const int finishBefore = g_finish;
    const bool buildSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&build, 2);
    if (buildSkip) (*(TaskFn**)&build)[0x38 / 8](&build);
    FakeTask focusL{g_vtPhase3};  // the same class as outer's
    const bool liveFocusSkip = qvfr::mvrpass::BeforeTask((uint8_t*)&focusL, 2);
    if (liveFocusSkip) {
        (*(TaskFn**)&focusL)[0x38 / 8](&focusL);
    } else {
        focusL.vt = g_vtFocus;
        RunTask(focusL);
    }
    qvfr::mvrpass::AfterTask((uint8_t*)&focusL);
    const int finishes = g_finish - finishBefore;
    // The focus clones the pass wrote (color, depth, the 1/2 size target and its depth) are noted for focus R's twins
    // (the twins themselves: qvfr_proxy_smoke --clones).
    const qvfr::mvrpass::TestLiveState lv = qvfr::mvrpass::TestLiveGet();
    qvfr::mvrpass::TestEnd();
    // B17: the same live pass with outer's depth as the game's own two-slice array (the G-buffer's depth, which the
    // clones module makes one in the game; the color stays a single texture): no copy of the depth's slice 0 in or back,
    // outer's color and depth (slice 0) and focus's clones as in the live run above, bit for bit.
    ComPtr<ID3D11Texture2D> arrDepth;
    ComPtr<ID3D11DepthStencilView> arrDsv;
    {
        D3D11_TEXTURE2D_DESC ad{};
        g_s.game.depth->GetDesc(&ad);
        ad.ArraySize = 2;
        ad.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = DXGI_FORMAT_D32_FLOAT;
        dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        if (SUCCEEDED(dev->CreateTexture2D(&ad, nullptr, &arrDepth))) dev->CreateDepthStencilView(arrDepth.Get(), &dv, &arrDsv);
    }
    static ID3D11Resource* s_arrDepth = nullptr;
    s_arrDepth = arrDepth.Get();
    qvfr::mvrpass::TestLiveState nlv{};
    std::vector<uint64_t> ngC, ngD, nC, nD;
    if (arrDsv) {
        Restore();
        ctx->CopySubresourceRegion(arrDepth.Get(), 0, 0, 0, 0, g_s.init[1].Get(), 0, nullptr);
        const ComPtr<ID3D11Texture2D> plainDepth = g_s.game.depth;
        const ComPtr<ID3D11DepthStencilView> plainDsv = g_s.game.dsv;
        g_s.game.depth = arrDepth;
        g_s.game.dsv = arrDsv;
        qvfr::mvrpass::TestNative([](ID3D11Resource* r) { return r && r == s_arrDepth; }, nullptr);
        qvfr::mvrpass::SetForward(true);
        qvfr::mvrpass::OnFrameStart(true);
        qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
        qvfr::mvrpass::TestForwardLive(g_vtPhase3);
        FakeTask outerN{g_vtPhase3};
        if (!qvfr::mvrpass::BeforeTask((uint8_t*)&outerN, 0)) RunTask(outerN);
        qvfr::mvrpass::AfterTask((uint8_t*)&outerN);
        nlv = qvfr::mvrpass::TestLiveGet();
        ngC = Read(g_s.game.color.Get());
        ngD = Read(arrDepth.Get(), 0);
        nC = Read(g_s.focus.color.Get());
        nD = Read(g_s.focus.depth.Get());
        qvfr::mvrpass::TestEnd();
        qvfr::mvrpass::TestNative(nullptr, nullptr);
        g_s.game.depth = plainDepth;
        g_s.game.dsv = plainDsv;
    }
    auto differ = [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        if (a.empty() || a.size() != b.size()) return 1 << 30;
        int d = 0;
        for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
        return d;
    };
    const int nOuter = differ(ngC, lgC) + differ(ngD, lgD), nFocus = differ(nC, lC) + differ(nD, lD);
    const bool nativeOk = arrDsv && nlv.passes == 1 && nlv.ok == 1 && nlv.native == 1 && nlv.copiedBack == 1 &&
                          lv.native == 0 && lv.copiedBack == 2 && nOuter == 0 && nFocus == 0;
    const MsCheck msaa = RunMsaa(sx, ox, sy, oy);  // B12
    qvfr::mvrpass::SetForward(false);
    qvfr::mvrpass::TestDepthBias(-1, 0.0f, false);
    const int lpx = (int)(kFW * kFH);
    int lOuter = 0, lC1 = 0, lCBig = 0, lD1 = 0, lHalf = 0, lHalfD = 0;
    float lMax = 0;
    for (size_t i = 0; i < refC.size() && i < lgC.size() && i < lgD.size(); ++i)
        lOuter += lgC[i] != refC[i] || lgD[i] != refD[i];
    for (size_t i = 0; i < lC.size() && i < refFC.size(); ++i) {
        const float d = MaxHalfDiff(lC[i], refFC[i]);
        lC1 += d > 0;
        lCBig += d > 0.02f;
        lMax = std::max(lMax, d);
        lD1 += std::fabs(AsFloat(lD[i]) - AsFloat(refFD[i])) > 1e-5f;
    }
    for (uint64_t v : lH) lHalf += v != 0;  // the pass cleared the 1/2 size target to 0
    for (uint64_t v : lHD) lHalfD += AsFloat(v) != 1.0f;
    const bool liveOk = lv.passes == 1 && lv.ok == 1 && !liveOuterSkip && buildSkip && liveFocusSkip && finishes == 2 &&
                        lv.focusSkipped == 1 && lv.buildSkipped == 1 && lv.otherClears == 2 &&
                        lv.otherClearMisses == 0 && lC.size() == (size_t)lpx && lOuter == 0 && lCBig * 100 <= lpx &&
                        lD1 * 100 <= lpx && lHalf == 0 && lHalfD == 0 && lH.size() == (size_t)(kFW / 2) * (kFH / 2) &&
                        lv.twinEntries == 4 && velOk && plainOk && lv.velBinds == 2;
    const int velPx = (int)(kFW * kFH);
    const bool velSkipOk = g_fill.calls == 1 && g_fill.eye == 0 && g_fill.depth == g_s.focus.gt2.Get() &&
                           g_fill.target == velTex.Get() && lv.velFilled == 1 && lv.velSkippedDraws == 2 &&
                           lv.velDrawn == 0 && velQuarter == velPx / 4 && velMarked == velPx - velPx / 4 &&
                           velOther == 0;

    // B13: outer L's G-buffer pass (CPhase1Render) with a PS that reads the live track map like the game's: view 1 (the
    // corner of slice 1 of the color copy) must read outer's map in the focus window (texel = outer UV of the focus
    // pixel) with mvr_gbuffer_ps, and reads it elsewhere without it; slice 0 equals the pass without the module.
    struct GbufRun {
        qvfr::mvrpass::TestState st;
        std::vector<uint64_t> s1, game;
        bool ran = false;
    };
    auto runGbuf = [&](bool on) {
        GbufRun r;
        qvfr::mvrpass::SetGbufferPs(on);
        qvfr::mvrpass::OnFrameStart(true);
        qvfr::mvrpass::SetWindow(0, true, sx, ox, sy, oy, kW, kH);
        FakeTask g1{g_vtPhase1};
        r.ran = !qvfr::mvrpass::BeforeTask((uint8_t*)&g1, 0);
        if (r.ran) RunTask(g1);
        qvfr::mvrpass::AfterTask((uint8_t*)&g1);
        r.st = qvfr::mvrpass::TestGet();
        r.game = Read(g_s.game.color.Get());
        for (int k = 0; k < 8; ++k) {
            ID3D11Texture2D* a = qvfr::mvrpass::TestArray(k);
            if (!a) continue;
            D3D11_TEXTURE2D_DESC d{};
            a->GetDesc(&d);
            if (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && d.Width == kW && d.Height == kH && d.ArraySize == 2 &&
                d.SampleDesc.Count == 1)  // not the MSAA run's copies (B12)
                r.s1 = Read(a, 1);
        }
        qvfr::mvrpass::TestEnd();
        return r;
    };
    const GbufRun gOff = runGbuf(false), gOn = runGbuf(true);
    qvfr::mvrpass::SetGbufferPs(true);
    DrawGbuf(g_s.game);  // the reference: outside any pass the module leaves the draws alone
    const std::vector<uint64_t> refG = Read(g_s.game.color.Get());
    // A covered pixel has alpha 1; the expected texel is outer's UV of the focus pixel's centre.
    auto gbufCheck = [&](const GbufRun& r, int& covered, int& wrong, float& maxErr) {
        covered = wrong = 0;
        maxErr = 0;
        if (r.s1.size() != (size_t)kW * kH) return;
        for (UINT y = 0; y < kFH; ++y)
            for (UINT x = 0; x < kFW; ++x) {
                const uint64_t p = r.s1[(size_t)y * kW + x];
                if (Half((uint32_t)(p >> 48) & 0xFFFF) != 1.0f) continue;
                ++covered;
                const float eu = (float)((1 + kX0) / 2 + (x + 0.5) / kFW * (kX1 - kX0) / 2);
                const float ev = (float)((1 - kY1) / 2 + (y + 0.5) / kFH * (kY1 - kY0) / 2);
                const float e = std::max(std::fabs(Half((uint32_t)p & 0xFFFF) - eu),
                                         std::fabs(Half((uint32_t)(p >> 16) & 0xFFFF) - ev));
                maxErr = std::max(maxErr, e);
                wrong += e > 1e-3f;
            }
    };
    int gCovOn = 0, gWrongOn = 0, gCovOff = 0, gWrongOff = 0, gOuterOn = 0, gOuterOff = 0;
    float gMaxOn = 0, gMaxOff = 0;
    gbufCheck(gOn, gCovOn, gWrongOn, gMaxOn);
    gbufCheck(gOff, gCovOff, gWrongOff, gMaxOff);
    for (size_t i = 0; i < refG.size(); ++i) {
        gOuterOn += i >= gOn.game.size() || gOn.game[i] != refG[i];
        gOuterOff += i >= gOff.game.size() || gOff.game[i] != refG[i];
    }
    const int gpx = (int)(kFW * kFH);
    const bool gbufOk = gbufPsOk && gOn.ran && gOff.ran && gOn.st.stashValid && gOn.st.gbufPsDraws == 1 &&
                        gOn.st.gbufPsKept == 0 && gOn.st.liveTrack && gOn.st.srvMisses == 0 &&
                        gOff.st.gbufPsDraws == 0 && !gOff.st.liveTrack && gCovOn > gpx / 3 && gCovOff == gCovOn &&
                        gWrongOn == 0 && gWrongOff > gCovOff / 2 && gOuterOn == 0 && gOuterOff == 0 &&
                        refG.size() == (size_t)kW * kH;

    int warnings = 0, expected = 0;
    if (info) {
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> buf(size);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (FAILED(info->GetMessage(i, msg, &size)) || msg->Severity > D3D11_MESSAGE_SEVERITY_WARNING) continue;
            const std::string text(msg->pDescription, strnlen(msg->pDescription, msg->DescriptionByteLength));
            if (text.find("(SV_RenderTargetArrayIndex,0) as input, but it is not provided") != std::string::npos) {
                ++expected;
                continue;
            }
            ++warnings;
            printf("D3D11 debug layer: %s\n", text.c_str());
        }
    }

    const int px = (int)(kFW * kFH);
    const bool countsOk = shadersOk && !outerSkip && st.outerDone && !st.failed && st.stashed == 2 &&
                          st.draws == 3 && st.plainDraws == 1 && st.psPatchedDraws == 2 && st.srvMisses == 0 &&
                          st.pauses == 1 && st.pausedDraws == 1 && st.pausedClears == 2 &&
                          st.copyIns == 2 && st.mapFailures == 0 && st.viewsFailures == 0 && st.copiedBack == 2 &&
                          !focusSkip && st2.focusDone && ownOff == 0;
    const bool outerOk = outerC == 0 && outerD == 0 && outerH == 0 && gH.size() == refH.size();
    const bool focusOk = clonesChanged == 0 && covered > px / 3 && occluded > px / 8 && occOurs == occluded &&
                         cBig * 100 <= px && dOff * 100 <= px;
    const bool ok = countsOk && outerOk && focusOk && liveOk && velSkipOk && gbufOk && nativeOk && msaa.ok && !warnings;
    printf("%s", msaa.text.c_str());
    printf("  B17: live pass with outer's depth as the game's own two-slice array: %s, targets as the arrays %u (before "
           "%u), copied back %u (before %u); outer %d px, focus's clones %d px differ from the live run (%s)\n",
           nlv.ok ? "ok" : "FAILED", nlv.native, lv.native, nlv.copiedBack, lv.copiedBack, nOuter, nFocus,
           nativeOk ? "OK" : "WRONG");
    printf("MVR forward: %s; PS %s (globalTexture0 in t%u, globalTexture2 in t%u), VS %s; outer pass %s%s%s, stash %d "
           "targets, binds %u, copy-ins %u, map failures %u, draws %u (one view %u, patched PS %u, PS slot 3 rebinds "
           "%u, screen textures without a clone %u), copied back %u, SetMultiViewMode failures %u; paused %u times "
           "(draws %u, clears %u) (%s)\n",
           ok ? "PASS" : "FAIL", psInfo.kind == qvfr::mvrps::Kind::Patched ? "patched" : "NOT PATCHED", psInfo.slot[0],
           psInfo.slot[2], qvfr::mvrvs::IsPatched(g_s.vs.Get()) ? "patched" : "NOT PATCHED",
           st.outerDone ? "done" : "NOT RUN", st.why.empty() ? "" : ": ", st.why.c_str(), st.stashed, st.binds,
           st.copyIns, st.mapFailures, st.draws, st.plainDraws, st.psPatchedDraws, st.psCbRebinds, st.srvMisses,
           st.copiedBack, st.viewsFailures, st.pauses, st.pausedDraws, st.pausedClears, countsOk ? "OK" : "WRONG");
    printf("  outer vs the pass without MVR: color %d, depth %d, 1/2 size target %d px differ (%s)\n", outerC, outerD,
           outerH, outerOk ? "OK" : "WRONG");
    printf("  view 1 vs focus's own pass (%ux%u): %d px covered, occluder quarter %d px kept in own, %d in ours; color "
           "%d px differ (%d by > 0.02, max %.4f), depth %d px off; clones changed by outer's pass %d px; focus's own "
           "run %s, %d px off its reference (%s)\n",
           kFW, kFH, covered, occluded, occOurs, cOff, cBig, cMax, dOff, clonesChanged,
           st2.focusDone ? "compared" : "NOT COMPARED", ownOff, focusOk ? "OK" : "WRONG");
    printf("  live (B9d): outer pass %u, into the clones %u%s%s; outer vs without MVR %d px; clones vs focus's own pass: "
           "color %d px differ (%d by > 0.02, max %.4f), depth %d px off; 1/2 size clone cleared %s (%d / %d px off), "
           "clears repeated %u (shared %u); focus build %s, CPhase3Render %s (finishes %d); clones noted for focus R's "
           "twins %u; velocity pass on the copy %s (binds %u), other targets on the clone %s (%s)\n",
           lv.passes, lv.ok, lv.why.empty() ? "" : ": ", lv.why.c_str(), lOuter, lC1, lCBig, lMax, lD1,
           lHalf || lHalfD ? "WRONG" : "ok", lHalf, lHalfD, lv.otherClears, lv.otherClearMisses,
           buildSkip ? "skipped" : "RUN", liveFocusSkip ? "skipped" : "RUN", finishes, lv.twinEntries,
           velOk ? "yes" : "NO", lv.velBinds, plainOk ? "yes" : "NO", liveOk ? "OK" : "WRONG");
    printf("  velocity pass with outer's vectors (B11): fill calls %d (eye %d, linear depth %s, target %s), passes "
           "filled %u, draws skipped %u, passes drawn %u; target: drawn quarter %d / %d px, marker %d / %d px, other "
           "%d px (%s)\n",
           g_fill.calls, g_fill.eye, g_fill.depth == g_s.focus.gt2.Get() ? "ok" : "WRONG",
           g_fill.target == velTex.Get() ? "ok" : "WRONG", lv.velFilled, lv.velSkippedDraws, lv.velDrawn, velQuarter,
           velPx / 4, velMarked, velPx - velPx / 4, velOther, velSkipOk ? "OK" : "WRONG");
    printf("  G-buffer PS with the live track map (B13): PS %s (map in t%u); on: pass %s, stash %s, draws with view 1's "
           "rows %u, kept %u, map read %s, misses %u, view 1 %d px covered, %d off (max %.5f), slice 0 %d px off; off: "
           "draws %u, view 1 %d px covered, %d off (max %.3f), slice 0 %d px off (%s)\n",
           gbufInfo.kind == qvfr::mvrps::Kind::Patched ? "patched" : "NOT PATCHED", gbufInfo.slot[qvfr::mvrps::kLiveTrack],
           gOn.ran ? "ran" : "SKIPPED", gOn.st.stashValid ? "valid" : "INVALID", gOn.st.gbufPsDraws,
           gOn.st.gbufPsKept, gOn.st.liveTrack ? "yes" : "NO", gOn.st.srvMisses, gCovOn, gWrongOn, gMaxOn, gOuterOn,
           gOff.st.gbufPsDraws, gCovOff, gWrongOff, gMaxOff, gOuterOff, gbufOk ? "OK" : "WRONG");
    printf("  debug layer %s, %d warnings, %d expected linkage reports\n", debugLayer ? "on" : "not installed", warnings,
           expected);
    return ok ? 0 : 1;
}

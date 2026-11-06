// --mvr-vs: the proxy's MVR vertex shader module (mvr_vs.h) on a real D3D11 device, without SteamVR and without the
// proxy DLL (NVIDIA with Multi-View Rendering only; elsewhere the test reports a skip).
// - A vertex shader created with the module off (original) and on (patched through the hook and NVAPI).
// - Both draw the same rectangle into slice 0 of a 2-slice target with Multi-View Rendering off (as in the game at
//   gate B6): pixels must be equal and slice 1 untouched.
// - With Multi-View Rendering on (2 views, slices 0 and 1) and the module's constant buffer in slot 3 (view 1: M = a
//   shift by 0.25 in x, clip planes keep x >= 0 of view 0), slice 1 must hold the shifted right half of the rectangle.
// - A vertex shader that already has a clip distance output is refused and created as the original.
// Checks the module's counters and the debug layer.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "mvr_vs.h"

using Microsoft::WRL::ComPtr;

namespace {

const char kHlsl[] = R"(
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VsOut VS(uint id : SV_VertexID) {
    float2 uv = float2(id & 1, id >> 1);
    VsOut o;
    o.pos = float4(lerp(-0.6, 0.6, uv.x), lerp(0.5, -0.5, uv.y), 0.5, 1);
    o.uv = uv;
    return o;
}
struct ClipOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 clip : SV_ClipDistance0; };
ClipOut VSClip(uint id : SV_VertexID) {
    VsOut v = VS(id);
    ClipOut o;
    o.pos = v.pos;
    o.uv = v.uv;
    o.clip = float4(1, 1, 1, 1);
    return o;
}
float4 PS(VsOut i) : SV_Target { return float4(i.uv, 0.5, 1); }
)";

constexpr UINT kW = 64, kH = 48;
constexpr float kShift = 0.25f;  // view 1: x + 0.25 * w in clip space

ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "mvr_vs_smoke", nullptr, nullptr, entry, target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
        return nullptr;
    }
    return code;
}

struct Target {
    ComPtr<ID3D11Texture2D> tex, staging;
    ComPtr<ID3D11RenderTargetView> rtv;  // both slices
};

bool MakeTarget(ID3D11Device* dev, Target& t) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = kW;
    d.Height = kH;
    d.MipLevels = 1;
    d.ArraySize = 2;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc = {1, 0};
    d.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &t.tex))) return false;
    d.BindFlags = 0;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &t.staging))) return false;
    D3D11_RENDER_TARGET_VIEW_DESC r{};
    r.Format = d.Format;
    r.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
    r.Texture2DArray.ArraySize = 2;
    return SUCCEEDED(dev->CreateRenderTargetView(t.tex.Get(), &r, &t.rtv));
}

void Draw(ID3D11DeviceContext* ctx, Target& t, ID3D11VertexShader* vs, ID3D11PixelShader* ps) {
    const float clear[4] = {0, 0, 0, 0};
    ctx->ClearRenderTargetView(t.rtv.Get(), clear);
    ctx->OMSetRenderTargets(1, t.rtv.GetAddressOf(), nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)kW, (float)kH, 0, 1};
    ctx->RSSetViewports(1, &vp);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ctx->IASetInputLayout(nullptr);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->Draw(4, 0);
}

// Both slices as RGBA8, slice after slice.
std::vector<uint32_t> Read(ID3D11DeviceContext* ctx, Target& t) {
    ctx->CopyResource(t.staging.Get(), t.tex.Get());
    std::vector<uint32_t> px(2 * kW * kH);
    for (UINT s = 0; s < 2; ++s) {
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(t.staging.Get(), s, D3D11_MAP_READ, 0, &m))) return {};
        for (UINT y = 0; y < kH; ++y)
            memcpy(&px[(s * kH + y) * kW], (const uint8_t*)m.pData + y * m.RowPitch, kW * 4);
        ctx->Unmap(t.staging.Get(), s);
    }
    return px;
}

bool SetViews(ID3D11DeviceContext* ctx, NvU32 views) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = views;
    for (NvU32 i = 0; i < 4; ++i) p.renderTargetIndexOffset[i] = i;
    return NvAPI_D3D_SetMultiViewMode(ctx, &p) == NVAPI_OK;
}

}  // namespace

int RunMvrVs() {
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
        printf("MVR VS: the hook could not be installed or NVAPI is not available (not NVIDIA?): SKIPPED\n");
        return 0;
    }
    NV_QUERY_MULTIVIEW_SUPPORT_PARAMS q = {};
    q.version = NV_QUERY_MULTIVIEW_SUPPORT_PARAMS_VER;
    if (NvAPI_D3D_QueryMultiViewSupport(dev.Get(), &q) != NVAPI_OK || !q.bMultiViewSupported) {
        printf("MVR VS: no Multi-View Rendering on this GPU: SKIPPED\n");
        return 0;
    }

    ComPtr<ID3DBlob> vsCode = Compile("VS", "vs_5_0"), clipCode = Compile("VSClip", "vs_5_0"),
                     psCode = Compile("PS", "ps_5_0");
    if (!vsCode || !clipCode || !psCode) return 1;
    ComPtr<ID3D11VertexShader> vsOrig, vsPatched, vsClip;
    ComPtr<ID3D11PixelShader> ps;
    qvfr::mvrvs::TestSetEnabled(false);
    bool made = SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vsOrig));
    const qvfr::mvrvs::TestState off = qvfr::mvrvs::TestGet();
    qvfr::mvrvs::TestSetEnabled(true);
    made = made &&
           SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vsPatched));
    const qvfr::mvrvs::TestState afterPatch = qvfr::mvrvs::TestGet();
    made = made &&
           SUCCEEDED(dev->CreateVertexShader(clipCode->GetBufferPointer(), clipCode->GetBufferSize(), nullptr, &vsClip));
    const qvfr::mvrvs::TestState afterClip = qvfr::mvrvs::TestGet();
    made = made && SUCCEEDED(dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &ps));
    Target t;
    made = made && MakeTarget(dev.Get(), t);
    // Constant buffer of the patch (mvr_patch.h): view 1 = M_1 (4 rows), then its 4 clip planes in view 0's clip space.
    const float rows[8][4] = {
        {1, 0, 0, kShift}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},  // M_1
        {1, 0, 0, 0},      {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1},  // x >= 0 (w > 0 for the rest)
    };
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(rows);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA init{rows, 0, 0};
    ComPtr<ID3D11Buffer> cb;
    made = made && SUCCEEDED(dev->CreateBuffer(&bd, &init, &cb));
    if (!made) {
        printf("MVR VS: D3D11 objects could not be created\n");
        return 1;
    }
    ctx->VSSetConstantBuffers(qvfr::mvrvs::kCbSlot, 1, cb.GetAddressOf());

    // Multi-View Rendering off: the original and the patched shader draw the same pixels into slice 0.
    Draw(ctx.Get(), t, vsOrig.Get(), ps.Get());
    const std::vector<uint32_t> ref = Read(ctx.Get(), t);
    Draw(ctx.Get(), t, vsPatched.Get(), ps.Get());
    const std::vector<uint32_t> pat = Read(ctx.Get(), t);
    // The refused shader (original) draws like the first one.
    Draw(ctx.Get(), t, vsClip.Get(), ps.Get());
    const std::vector<uint32_t> clipImg = Read(ctx.Get(), t);
    // Multi-View Rendering on (test only; the game stays at one view in gate B6).
    const bool mvrSet = SetViews(ctx.Get(), 2);
    Draw(ctx.Get(), t, vsPatched.Get(), ps.Get());
    const bool mvrOff = SetViews(ctx.Get(), 1);
    const std::vector<uint32_t> mvr = Read(ctx.Get(), t);
    if (ref.empty() || pat.empty() || clipImg.empty() || mvr.empty()) {
        printf("MVR VS: read-back failed\n");
        return 1;
    }

    int covered = 0, diffOff = 0, slice1Off = 0, diffClip = 0, diffMvr0 = 0, diffMvr1 = 0, rightHalf = 0;
    const int shiftPx = (int)(kShift * 0.5f * kW + 0.5f);
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) {
            const size_t i0 = y * kW + x, i1 = (kH + y) * kW + x;
            covered += ref[i0] != 0;
            diffOff += pat[i0] != ref[i0];
            slice1Off += pat[i1] != 0;
            diffClip += clipImg[i0] != ref[i0];
            diffMvr0 += mvr[i0] != ref[i0];
            // Slice 1: view 0's pixel shifted right, only where view 0's x >= 0 (the right half of the target).
            const int sx = (int)x - shiftPx;
            uint32_t expect = 0;
            if (sx >= (int)kW / 2 && sx < (int)kW) expect = ref[y * kW + sx];
            rightHalf += expect != 0;
            diffMvr1 += mvr[i1] != expect;
        }
    const qvfr::mvrvs::TestState s = qvfr::mvrvs::TestGet();

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

    const bool countsOk = off.made == 0 && afterPatch.made == 1 && afterPatch.patched == 1 && afterClip.made == 2 &&
                          afterClip.refused == 1 && afterClip.patched == 1 && s.nvapiFailed == 0 && s.faults == 0;
    const bool imageOk = covered > 0 && diffOff == 0 && slice1Off == 0 && diffClip == 0;
    const bool mvrOk = mvrSet && mvrOff && rightHalf > 0 && diffMvr0 == 0 && diffMvr1 == 0;
    const bool ok = countsOk && imageOk && mvrOk && !warnings;
    printf("MVR VS: %s; counters: made %llu, patched %llu, refused %llu, NVAPI failures %llu, skipped %llu, faults %llu, "
           "largest output register o%u (%s)\n",
           ok ? "PASS" : "FAIL", s.made, s.patched, s.refused, s.nvapiFailed, s.skipped, s.faults, s.maxOutputReg,
           countsOk ? "OK" : "WRONG");
    printf("  MVR off: rectangle %d px, patched vs original %d px differ, slice 1 written %d px, refused shader %d px "
           "differ (%s)\n",
           covered, diffOff, slice1Off, diffClip, imageOk ? "OK" : "WRONG");
    printf("  MVR on (2 views, test only): set %s, view 0 %d px differ, view 1 (shifted right half, %d px) %d px differ "
           "(%s)\n",
           mvrSet && mvrOff ? "ok" : "FAILED", diffMvr0, rightHalf, diffMvr1, mvrOk ? "OK" : "WRONG");
    printf("  debug layer %s, %d warnings\n", debugLayer ? "on" : "not installed", warnings);
    return ok ? 0 : 1;
}

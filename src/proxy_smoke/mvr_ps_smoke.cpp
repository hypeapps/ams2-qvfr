// --mvr-ps: the proxy's MVR pixel shader module (mvr_ps.h) on a real D3D11 device, without SteamVR and without the
// proxy DLL (NVIDIA with Multi-View Rendering only; elsewhere the test reports a skip). The patch itself, with MVR on,
// is checked by qvfr_mvr_test --ps-patch.
// - The module runs in the sun effects module's CreatePixelShader hook, as in the game.
// - A forward-like pixel shader (a screen input divided by its w, Load and Gather of globalTexture0/2) is patched, one
//   without screen reads is left as it is (not needed), one that already reads SV_RenderTargetArrayIndex is refused.
// - The patched shader draws exactly what the original draws (MVR off, gate B9b).
// - Geometry shaders writing SV_RenderTargetArrayIndex are counted.
// Checks the module's counters and the debug layer (its linkage report for SV_RenderTargetArrayIndex is expected).
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mvr_ps.h"
#include "mvr_vs.h"
#include "sun_fx.h"

using Microsoft::WRL::ComPtr;

namespace {

const char kHlsl[] = R"(
struct VsOut { float4 pos : SV_Position; float4 scr : TEXCOORD0; float2 uv : TEXCOORD1; };
VsOut VS(uint id : SV_VertexID) {
    float2 uv = float2(id & 1, id >> 1);
    float w = lerp(1.0, 3.0, uv.y);  // perspective: w varies over the quad
    VsOut o;
    o.pos = float4(lerp(-0.8, 0.8, uv.x) * w, lerp(0.7, -0.7, uv.y) * w, 0.5 * w, w);
    o.scr = float4(0.5 * (o.pos.x + o.pos.w), 0.5 * (o.pos.w - o.pos.y), o.pos.z, o.pos.w);
    o.uv = uv;
    return o;
}
Texture2D globalTexture0 : register(t3);
Texture2D globalTexture2 : register(t5);
SamplerState pointSampler : register(s0);
cbuffer G : register(b1) { float4 viewportPixelDimensions; float4 screenSpaceDimensions; };
float4 Fwd(VsOut i) : SV_Target {
    float2 uv = i.scr.xy / i.scr.w;
    int2 p = int2(min(uv * viewportPixelDimensions.zw, viewportPixelDimensions.zw - 1));
    float4 a = globalTexture0.Load(int3(p, 0));
    float4 g = globalTexture2.GatherRed(pointSampler, i.pos.xy * screenSpaceDimensions.zw);
    return float4(a.xy, dot(g, 1), i.uv.x);
}
float4 Plain(VsOut i) : SV_Target { return float4(i.uv, 0.5, 1); }
float4 Layered(VsOut i, uint rt : SV_RenderTargetArrayIndex) : SV_Target {
    return float4(i.scr.xy / i.scr.w, rt, 1);
}
struct GsOut { float4 pos : SV_Position; uint rt : SV_RenderTargetArrayIndex; };
[maxvertexcount(3)] void GSLayered(triangle VsOut v[3], inout TriangleStream<GsOut> s) {
    for (int k = 0; k < 3; ++k) { GsOut o; o.pos = v[k].pos; o.rt = 1; s.Append(o); }
}
[maxvertexcount(3)] void GSPlain(triangle VsOut v[3], inout TriangleStream<VsOut> s) {
    for (int k = 0; k < 3; ++k) s.Append(v[k]);
}
)";

constexpr UINT kW = 64, kH = 48;

ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "mvr_ps_smoke", nullptr, nullptr, entry, target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
        return nullptr;
    }
    return code;
}

const char* KindName(qvfr::mvrps::Kind k) {
    using K = qvfr::mvrps::Kind;
    return k == K::Patched ? "patched" : k == K::NotNeeded ? "not needed" : k == K::Original ? "original" : "unknown";
}

}  // namespace

int RunMvrPs() {
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
        printf("MVR PS: the hooks could not be installed or NVAPI is not available (not NVIDIA?): SKIPPED\n");
        return 0;
    }
    if (!qvfr::mvrvs::DeviceHasMvr(dev.Get())) {
        printf("MVR PS: no Multi-View Rendering on this GPU: SKIPPED\n");
        return 0;
    }
    if (!qvfr::mvrps::TestInstall()) {
        printf("MVR PS: FAIL (the module could not be installed)\n");
        return 1;
    }

    ComPtr<ID3DBlob> vsCode = Compile("VS", "vs_5_0"), fwdCode = Compile("Fwd", "ps_5_0"),
                     plainCode = Compile("Plain", "ps_5_0"), layCode = Compile("Layered", "ps_5_0"),
                     gsLayCode = Compile("GSLayered", "gs_5_0"), gsPlainCode = Compile("GSPlain", "gs_5_0");
    if (!vsCode || !fwdCode || !plainCode || !layCode || !gsLayCode || !gsPlainCode) return 1;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> fwdOrig, fwd, plain, lay;
    ComPtr<ID3D11GeometryShader> gsLay, gsPlain;
    qvfr::mvrps::TestSetEnabled(false);
    bool made = SUCCEEDED(dev->CreatePixelShader(fwdCode->GetBufferPointer(), fwdCode->GetBufferSize(), nullptr, &fwdOrig));
    const qvfr::mvrps::TestState off = qvfr::mvrps::TestGet();
    qvfr::mvrps::TestSetEnabled(true);
    made = made && SUCCEEDED(dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vs));
    made = made && SUCCEEDED(dev->CreatePixelShader(fwdCode->GetBufferPointer(), fwdCode->GetBufferSize(), nullptr, &fwd));
    made = made &&
           SUCCEEDED(dev->CreatePixelShader(plainCode->GetBufferPointer(), plainCode->GetBufferSize(), nullptr, &plain));
    made = made && SUCCEEDED(dev->CreatePixelShader(layCode->GetBufferPointer(), layCode->GetBufferSize(), nullptr, &lay));
    made = made && SUCCEEDED(dev->CreateGeometryShader(gsLayCode->GetBufferPointer(), gsLayCode->GetBufferSize(),
                                                       nullptr, &gsLay));
    made = made && SUCCEEDED(dev->CreateGeometryShader(gsPlainCode->GetBufferPointer(), gsPlainCode->GetBufferSize(),
                                                       nullptr, &gsPlain));

    // Screen textures (texel coordinates) and constants of a kW x kH target.
    auto tex = [&](float scale, ComPtr<ID3D11ShaderResourceView>& srv) {
        std::vector<float> px(kW * kH * 4);
        for (UINT y = 0; y < kH; ++y)
            for (UINT x = 0; x < kW; ++x) {
                float* p = &px[(y * kW + x) * 4];
                p[0] = x * scale;
                p[1] = y * scale;
                p[2] = p[3] = 0;
            }
        D3D11_TEXTURE2D_DESC d{};
        d.Width = kW;
        d.Height = kH;
        d.MipLevels = d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.SampleDesc = {1, 0};
        d.Usage = D3D11_USAGE_IMMUTABLE;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA sd{px.data(), kW * 16, 0};
        ComPtr<ID3D11Texture2D> t;
        return SUCCEEDED(dev->CreateTexture2D(&d, &sd, &t)) && SUCCEEDED(dev->CreateShaderResourceView(t.Get(), nullptr, &srv));
    };
    ComPtr<ID3D11ShaderResourceView> t0, t2;
    made = made && tex(1.0f / 64, t0) && tex(1.0f / 256, t2);
    const float g[8] = {0, 0, kW, kH, 0, 0, 1.0f / kW, 1.0f / kH};
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(g);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA init{g, 0, 0};
    ComPtr<ID3D11Buffer> cb;
    made = made && SUCCEEDED(dev->CreateBuffer(&bd, &init, &cb));
    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ComPtr<ID3D11SamplerState> smp;
    made = made && SUCCEEDED(dev->CreateSamplerState(&sd, &smp));
    D3D11_TEXTURE2D_DESC td{};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc = {1, 0};
    td.BindFlags = D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> rt, staging;
    ComPtr<ID3D11RenderTargetView> rtv;
    made = made && SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &rt)) &&
           SUCCEEDED(dev->CreateRenderTargetView(rt.Get(), nullptr, &rtv));
    td.BindFlags = 0;
    td.Usage = D3D11_USAGE_STAGING;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    made = made && SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &staging));
    if (!made) {
        printf("MVR PS: D3D11 objects could not be created\n");
        return 1;
    }

    auto draw = [&](ID3D11PixelShader* ps) {
        const float clear[4] = {-1, -1, -1, -1};
        ctx->ClearRenderTargetView(rtv.Get(), clear);
        ctx->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
        D3D11_VIEWPORT vp{0, 0, (float)kW, (float)kH, 0, 1};
        ctx->RSSetViewports(1, &vp);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        ctx->IASetInputLayout(nullptr);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ID3D11ShaderResourceView* srv[6] = {nullptr, nullptr, nullptr, t0.Get(), nullptr, t2.Get()};
        ctx->PSSetShaderResources(0, 6, srv);
        ctx->PSSetConstantBuffers(1, 1, cb.GetAddressOf());
        ctx->PSSetSamplers(0, 1, smp.GetAddressOf());
        ctx->Draw(4, 0);
        ctx->CopyResource(staging.Get(), rt.Get());
        std::vector<float> px(kW * kH * 4);
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m))) return std::vector<float>{};
        for (UINT y = 0; y < kH; ++y) memcpy(&px[y * kW * 4], (const uint8_t*)m.pData + y * m.RowPitch, kW * 16);
        ctx->Unmap(staging.Get(), 0);
        return px;
    };
    const std::vector<float> ref = draw(fwdOrig.Get()), pat = draw(fwd.Get());
    int covered = 0, differ = 0;
    if (ref.empty() || pat.size() != ref.size()) {
        printf("MVR PS: read-back failed\n");
        return 1;
    }
    for (size_t i = 0; i < ref.size(); i += 4) {
        covered += ref[i + 3] >= 0;
        differ += memcmp(&ref[i], &pat[i], 16) != 0;
    }

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

    using K = qvfr::mvrps::Kind;
    const qvfr::mvrps::TestState s = qvfr::mvrps::TestGet();
    const K kOrig = qvfr::mvrps::KindOf(fwdOrig.Get()), kFwd = qvfr::mvrps::KindOf(fwd.Get()),
            kPlain = qvfr::mvrps::KindOf(plain.Get()), kLay = qvfr::mvrps::KindOf(lay.Get());
    const bool kindsOk = kOrig == K::Unknown && kFwd == K::Patched && kPlain == K::NotNeeded && kLay == K::Original;
    const bool countsOk = off.made == 0 && s.made == 3 && s.patched == 1 && s.notNeeded == 1 && s.refused == 1 &&
                          s.failed == 0 && s.gsMade == 2 && s.gsLayered == 1 && s.faults == 0;
    const bool imageOk = covered > 0 && differ == 0;
    const bool ok = kindsOk && countsOk && imageOk && !warnings;
    printf("MVR PS: %s; kinds: original %s, forward-like %s, plain %s, layered %s (%s)\n", ok ? "PASS" : "FAIL",
           KindName(kOrig), KindName(kFwd), KindName(kPlain), KindName(kLay), kindsOk ? "OK" : "WRONG");
    printf("  counters: made %llu, patched %llu, not needed %llu, refused %llu, creation failed %llu, geometry shaders "
           "%llu (layered %llu), faults %llu (%s)\n",
           s.made, s.patched, s.notNeeded, s.refused, s.failed, s.gsMade, s.gsLayered, s.faults,
           countsOk ? "OK" : "WRONG");
    printf("  MVR off: %d px drawn, patched vs original %d px differ, bit-exact (%s)\n", covered, differ,
           imageOk ? "OK" : "WRONG");
    printf("  debug layer %s, %d warnings, %d expected linkage reports\n", debugLayer ? "on" : "not installed", warnings,
           expected);
    return ok ? 0 : 1;
}

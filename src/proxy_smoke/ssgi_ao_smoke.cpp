// --ssgi-ao: the proxy's own AO module (ssgi_ao.h) on a real D3D11 device, without SteamVR and without the proxy DLL.
// A synthetic G-buffer as the game's (R16G16B16A16_UNORM spheremap view normals, R32_FLOAT linear depth / far) of a
// floor 1 m below the eye meeting a wall 6 m ahead, and the game's constant windows (tangents, far, the view's share of
// the targets) bound with PSSetConstantBuffers1 as the game binds them. Draws registered in sun_fx's table as the
// overlay, the ambient pass and the MSAA resolve copy the SSAO texture they get into a target. Frames: ao=1 (own AO in
// both, the slot back after the draw, an ambient draw with other normals keeps the game's texture), ao=2 (the overlay
// gets 1), ao=3 (the preview over the resolve), ao=0 (the game's texture). Then the bounce (gi=1, gi=2): the overlay
// draws into a deferred-ambient weight target (R8G8B8A8_UNORM, the left half 0 as a vehicle), the view's history is a
// radiance image (the wall red, the floor green) given as DLSS's output after each frame, the camera stays (viewI the
// identity). Checks the AO at the floor-wall corner and on open surfaces, the bounce there (red on the floor by the
// wall, green on the wall by the floor, nothing on open surfaces and in the frame without a history), the module's
// counters and the debug layer.
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ssgi_ao.h"
#include "sun_fx.h"

using Microsoft::WRL::ComPtr;

namespace {

const char kHlsl[] = R"(
Texture2D<float> t1 : register(t1);
Texture2D<float> t2 : register(t2);
Texture2D<float> t3 : register(t3);
SamplerState pointClamp : register(s0);
static const float2 kInvSize = 1 / float2(128, 96);
float4 VS(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
// As the game's shaders: the SSAO texture at the view's UV (screenSpaceDimensions.xy = 1 here).
float4 Overlay(float4 p : SV_Position) : SV_Target { return t1.SampleLevel(pointClamp, p.xy * kInvSize, 0).xxxx; }
float4 Ambient(float4 p : SV_Position) : SV_Target { return t2.SampleLevel(pointClamp, p.xy * kInvSize, 0).xxxx; }
// The ambient pass of tracks with a lightmap: the SSAO at t3 (normals at t5).
float4 AmbientLm(float4 p : SV_Position) : SV_Target { return t3.SampleLevel(pointClamp, p.xy * kInvSize, 0).xxxx; }
float4 Resolve(float4 p : SV_Position) : SV_Target { return float4(0.25, 0.25, 0.25, 1); }
)";

constexpr UINT kW = 128, kH = 96;
constexpr float kFar = 100;

uint16_t Unorm16(float v) { return (uint16_t)std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 65535); }

// The scene at a pixel: view depth and normal (view space x right, y up, z forward).
void Scene(UINT x, UINT y, float& z, float n[3]) {
    const float rx = -1 + 2 * (x + 0.5f) / kW, ry = 1 - 2 * (y + 0.5f) / kH;  // tangents 1 on every side
    const float wall = 6, floorT = ry < 0 ? -1 / ry : 1e9f;
    (void)rx;
    if (floorT < wall) {
        z = floorT;
        n[0] = 0, n[1] = 1, n[2] = 0;
    } else {
        z = wall;
        n[0] = 0, n[1] = 0, n[2] = -1;
    }
}

// The game's spheremap encoding (the inverse of its decode, docs/ssgi-plan.md "Ustalenia G0").
void Encode(const float n[3], uint16_t out[4]) {
    const float w = (n[2] + 1) * 0.5f;
    float ex = 0, ey = 0;
    if (w > 1e-6f) {
        const float s = 2 * std::sqrt(w);
        ex = n[0] / s;
        ey = n[1] / s;
    } else {
        ex = 1;  // n = (0, 0, -1): any point of the unit circle
    }
    out[0] = Unorm16(ex * 0.5f + 0.5f);
    out[1] = Unorm16(ey * 0.5f + 0.5f);
    out[2] = 0;
    out[3] = 0;
}

template <class T>
bool Read(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* t, std::vector<T>& out) {
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> s;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &s))) return false;
    ctx->CopyResource(s.Get(), t);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(ctx->Map(s.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;
    out.resize((size_t)d.Width * d.Height);
    for (UINT y = 0; y < d.Height; ++y)
        memcpy(&out[(size_t)y * d.Width], (uint8_t*)m.pData + y * m.RowPitch, d.Width * sizeof(T));
    ctx->Unmap(s.Get(), 0);
    return true;
}

float Half(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e ? std::ldexp(1.0f + m / 1024.0f, (int)e - 15) : std::ldexp(m / 1024.0f, -14);
    return s ? -v : v;
}

}  // namespace

int RunSsgiAo() {
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
    ComPtr<ID3D11DeviceContext1> ctx1;
    D3D11_FEATURE_DATA_D3D11_OPTIONS opt{};
    dev->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &opt, sizeof(opt));
    if (FAILED(ctx.As(&ctx1)) || !opt.ConstantBufferOffsetting) {
        printf("No ID3D11DeviceContext1 with constant buffer offsets\n");
        return 1;
    }
    if (!qvfr::sunfx::TestInstall(dev.Get(), ctx.Get())) {
        printf("Sun effects hooks could not be installed\n");
        return 1;
    }

    auto blob = [&](const char* entry, const char* target) {
        ID3DBlob *code = nullptr, *err = nullptr;
        if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "ssgi_ao_smoke", nullptr, nullptr, entry, target, 0, 0, &code,
                              &err))) {
            printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
            code = nullptr;
        }
        if (err) err->Release();
        return code;
    };
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> overlay, ambient, resolve, ambientLm;
    {
        ID3DBlob* c[5] = {blob("VS", "vs_5_0"), blob("Overlay", "ps_5_0"), blob("Ambient", "ps_5_0"),
                          blob("Resolve", "ps_5_0"), blob("AmbientLm", "ps_5_0")};
        for (ID3DBlob* b : c)
            if (!b) return 1;
        dev->CreateVertexShader(c[0]->GetBufferPointer(), c[0]->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(c[1]->GetBufferPointer(), c[1]->GetBufferSize(), nullptr, &overlay);
        dev->CreatePixelShader(c[2]->GetBufferPointer(), c[2]->GetBufferSize(), nullptr, &ambient);
        dev->CreatePixelShader(c[3]->GetBufferPointer(), c[3]->GetBufferSize(), nullptr, &resolve);
        dev->CreatePixelShader(c[4]->GetBufferPointer(), c[4]->GetBufferSize(), nullptr, &ambientLm);
        for (ID3DBlob* b : c) b->Release();
    }
    qvfr::sunfx::TestRegister(overlay.Get(), qvfr::sunfx::KindAoOverlay);
    qvfr::sunfx::TestRegister(ambient.Get(), qvfr::sunfx::KindAoAmbient);
    qvfr::sunfx::TestRegister(resolve.Get(), qvfr::sunfx::KindAoResolve);
    qvfr::sunfx::TestRegister(ambientLm.Get(), qvfr::sunfx::KindAoAmbientLightmap);

    // The G-buffer.
    std::vector<uint16_t> nrm((size_t)kW * kH * 4);
    std::vector<float> dep((size_t)kW * kH);
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) {
            float z, n[3];
            Scene(x, y, z, n);
            dep[(size_t)y * kW + x] = z / kFar;
            Encode(n, &nrm[((size_t)y * kW + x) * 4]);
        }
    auto texture = [&](DXGI_FORMAT f, UINT bind, const void* data, UINT pitch, ComPtr<ID3D11Texture2D>& t) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = kW;
        d.Height = kH;
        d.MipLevels = d.ArraySize = 1;
        d.Format = f;
        d.SampleDesc = {1, 0};
        d.BindFlags = bind;
        const D3D11_SUBRESOURCE_DATA init = {data, pitch, 0};
        return SUCCEEDED(dev->CreateTexture2D(&d, data ? &init : nullptr, &t));
    };
    ComPtr<ID3D11Texture2D> normals, normals2, depth, gameSsao, outO, outA, outR, outW, weightInit, outG, radiance;
    std::vector<uint16_t> half(kW * kH, 0x3800);  // 0.5
    const UINT srv = D3D11_BIND_SHADER_RESOURCE, rt = D3D11_BIND_RENDER_TARGET;
    bool made = texture(DXGI_FORMAT_R16G16B16A16_UNORM, srv, nrm.data(), kW * 8, normals) &&
                texture(DXGI_FORMAT_R16G16B16A16_UNORM, srv, nrm.data(), kW * 8, normals2) &&
                texture(DXGI_FORMAT_R32_FLOAT, srv, dep.data(), kW * 4, depth) &&
                texture(DXGI_FORMAT_R16_FLOAT, srv, half.data(), kW * 2, gameSsao) &&
                texture(DXGI_FORMAT_R32_FLOAT, rt, nullptr, 0, outO) &&
                texture(DXGI_FORMAT_R32_FLOAT, rt, nullptr, 0, outA) &&
                texture(DXGI_FORMAT_R16G16B16A16_FLOAT, rt | srv, nullptr, 0, outR);
    // The bounce: the weight (1 the world, 0 the vehicle on the left), the radiance (wall red, floor green).
    std::vector<uint32_t> wgt((size_t)kW * kH);
    std::vector<uint64_t> rad((size_t)kW * kH);
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) {
            float z, n[3];
            Scene(x, y, z, n);
            wgt[(size_t)y * kW + x] = x < kW / 2 ? 0xFF000000u : 0xFFFFFFFFu;
            const uint64_t one = 0x3C00, alphaOne = one << 48;
            rad[(size_t)y * kW + x] = alphaOne | (n[1] > 0.5f ? one << 16 : one);  // floor: green, wall: red
        }
    made = made && texture(DXGI_FORMAT_R8G8B8A8_UNORM, rt | srv, nullptr, 0, outW) &&
           texture(DXGI_FORMAT_R8G8B8A8_UNORM, srv, wgt.data(), kW * 4, weightInit) &&
           texture(DXGI_FORMAT_R16G16B16A16_FLOAT, rt | srv, nullptr, 0, outG) &&
           texture(DXGI_FORMAT_R16G16B16A16_FLOAT, srv | rt, rad.data(), kW * 8, radiance);
    ComPtr<ID3D11ShaderResourceView> normalsSrv, normals2Srv, depthSrv, ssaoSrv, weightSrv;
    ComPtr<ID3D11RenderTargetView> rtvO, rtvA, rtvR, rtvW, rtvG;
    made = made && SUCCEEDED(dev->CreateShaderResourceView(normals.Get(), nullptr, &normalsSrv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(normals2.Get(), nullptr, &normals2Srv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(depth.Get(), nullptr, &depthSrv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(gameSsao.Get(), nullptr, &ssaoSrv)) &&
           SUCCEEDED(dev->CreateShaderResourceView(weightInit.Get(), nullptr, &weightSrv)) &&
           SUCCEEDED(dev->CreateRenderTargetView(outO.Get(), nullptr, &rtvO)) &&
           SUCCEEDED(dev->CreateRenderTargetView(outA.Get(), nullptr, &rtvA)) &&
           SUCCEEDED(dev->CreateRenderTargetView(outR.Get(), nullptr, &rtvR)) &&
           SUCCEEDED(dev->CreateRenderTargetView(outW.Get(), nullptr, &rtvW)) &&
           SUCCEEDED(dev->CreateRenderTargetView(outG.Get(), nullptr, &rtvG));
    // The game's constant buffer: COMMON_BUFFER2's window at constant 16, the overlay's _GLOBAL_ at constant 256.
    ComPtr<ID3D11Buffer> cb;
    {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = 65536;
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        made = made && SUCCEEDED(dev->CreateBuffer(&d, nullptr, &cb));
    }
    ComPtr<ID3D11SamplerState> sampler;
    {
        D3D11_SAMPLER_DESC s{};
        s.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        s.AddressU = s.AddressV = s.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        s.MaxLOD = D3D11_FLOAT32_MAX;
        made = made && SUCCEEDED(dev->CreateSamplerState(&s, &sampler));
    }
    if (!made) {
        printf("Test resources could not be made\n");
        return 1;
    }
    {
        D3D11_MAPPED_SUBRESOURCE m{};
        ctx->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
        memset(m.pData, 0, 65536);
        float* common = (float*)((uint8_t*)m.pData + 16 * 16);
        const float ssd[4] = {1, 1, 1.0f / kW, 1.0f / kH}, vpd[4] = {1.0f / kW, 1.0f / kH, kW, kH};
        memcpy(common + 24 * 4, ssd, sizeof(ssd));  // screenSpaceDimensions, offset 384
        memcpy(common + 29 * 4, vpd, sizeof(vpd));  // viewportPixelDimensions, offset 464
        for (int i = 0; i < 4; ++i) common[(4 + i) * 4 + i] = 1;  // viewI, offset 64: the identity
        float* global = (float*)((uint8_t*)m.pData + 256 * 16);
        const float tan[4] = {1, 1, 1, 1};
        memcpy(global, tan, sizeof(tan));  // cProjectionTanAngles, offset 0
        global[20 * 4] = kFar;             // frustumDepth, offset 320
        ctx->Unmap(cb.Get(), 0);
    }

    auto draw = [&](ID3D11PixelShader* ps, ID3D11RenderTargetView* rtv, ID3D11ShaderResourceView* t1,
                    ID3D11ShaderResourceView* t2, ID3D11ShaderResourceView* t3, ID3D11ShaderResourceView* t4,
                    ID3D11ShaderResourceView* t5 = nullptr) {
        ID3D11Buffer* b = cb.Get();
        UINT first = 16, num = 64;
        ctx1->PSSetConstantBuffers1(12, 1, &b, &first, &num);
        first = 256;
        num = 32;
        ctx1->PSSetConstantBuffers1(0, 1, &b, &first, &num);
        ID3D11ShaderResourceView* s[6] = {nullptr, t1, t2, t3, t4, t5};
        ctx->PSSetShaderResources(0, 6, s);
        ID3D11SamplerState* smp = sampler.Get();
        ctx->PSSetSamplers(0, 1, &smp);
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        const D3D11_VIEWPORT vp = {0, 0, (float)kW, (float)kH, 0, 1};
        ctx->RSSetViewports(1, &vp);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ctx->Draw(3, 0);
        ID3D11RenderTargetView* none = nullptr;
        ctx->OMSetRenderTargets(1, &none, nullptr);
    };
    // One frame in eye group 0 (outer L): the overlay, the ambient pass of the same normals, one of other normals,
    // the resolve. Returns whether the game's SSAO view is back in slot 1 after the overlay.
    auto frame = [&](int mode) {
        qvfr::ssgiao::TestInit(ctx.Get(), mode);
        qvfr::sunfx::OnFrameStart(ctx.Get(), false);
        qvfr::ssgiao::OnFrameStart(ctx.Get(), true);
        qvfr::sunfx::BeginGroup(0, 0, 0, 1, 1);
        draw(overlay.Get(), rtvO.Get(), ssaoSrv.Get(), normalsSrv.Get(), depthSrv.Get(), nullptr);
        ID3D11ShaderResourceView* back = nullptr;
        ctx->PSGetShaderResources(1, 1, &back);
        const bool restored = back == ssaoSrv.Get();
        if (back) back->Release();
        draw(ambient.Get(), rtvA.Get(), nullptr, ssaoSrv.Get(), nullptr, normalsSrv.Get());
        draw(ambient.Get(), rtvA.Get(), nullptr, ssaoSrv.Get(), nullptr, normals2Srv.Get());  // other normals
        draw(resolve.Get(), rtvR.Get(), nullptr, nullptr, nullptr, nullptr);
        qvfr::sunfx::EndGroup();
        return restored;
    };
    auto at = [](const std::vector<float>& v, UINT x, UINT y) { return v[(size_t)y * kW + x]; };

    // ao=1: own AO in the overlay and the ambient pass. The frame's last ambient draw (other normals) leaves the game's
    // texture in the ambient target, so the ambient pass with this view's normals runs once more alone and is read.
    std::vector<float> o1, a1, o2, o0;
    std::vector<uint64_t> r3;
    const bool restored1 = frame(1);
    Read(dev.Get(), ctx.Get(), outO.Get(), o1);
    // The ambient pass with this view's normals: draw it once more alone and read it.
    qvfr::sunfx::BeginGroup(0, 0, 0, 1, 1);
    draw(ambient.Get(), rtvA.Get(), nullptr, ssaoSrv.Get(), nullptr, normalsSrv.Get());
    qvfr::sunfx::EndGroup();
    Read(dev.Get(), ctx.Get(), outA.Get(), a1);
    const qvfr::ssgiao::TestState st1 = qvfr::ssgiao::TestGet();
    frame(2);
    Read(dev.Get(), ctx.Get(), outO.Get(), o2);
    frame(3);
    Read(dev.Get(), ctx.Get(), outR.Get(), r3);
    frame(0);
    Read(dev.Get(), ctx.Get(), outO.Get(), o0);
    const qvfr::ssgiao::TestState st = qvfr::ssgiao::TestGet();
    // The lightmap variant of the ambient pass (SSAO t3, normals t5): own AO there too.
    std::vector<float> aLm;
    qvfr::ssgiao::TestInit(ctx.Get(), 1);
    qvfr::sunfx::OnFrameStart(ctx.Get(), false);
    qvfr::ssgiao::OnFrameStart(ctx.Get(), true);
    qvfr::sunfx::BeginGroup(0, 0, 0, 1, 1);
    draw(overlay.Get(), rtvO.Get(), ssaoSrv.Get(), normalsSrv.Get(), depthSrv.Get(), nullptr);
    draw(ambientLm.Get(), rtvA.Get(), nullptr, nullptr, ssaoSrv.Get(), nullptr, normalsSrv.Get());
    qvfr::sunfx::EndGroup();
    Read(dev.Get(), ctx.Get(), outA.Get(), aLm);
    const unsigned long long lmAmbient = qvfr::ssgiao::TestGet().ambient - st.ambient;

    // The bounce: frame 1 without a history (nothing added), frame 2 with it (read), frame 3 its preview.
    auto giFrame = [&](int gi) {
        ctx->CopyResource(outW.Get(), weightInit.Get());  // the overlay below writes into it
        qvfr::ssgiao::TestInit(ctx.Get(), 1, gi);
        qvfr::sunfx::OnFrameStart(ctx.Get(), false);
        qvfr::ssgiao::OnFrameStart(ctx.Get(), true);
        qvfr::sunfx::BeginGroup(0, 0, 0, 1, 1);
        draw(overlay.Get(), rtvW.Get(), ssaoSrv.Get(), normalsSrv.Get(), depthSrv.Get(), nullptr);
        draw(ambient.Get(), rtvG.Get(), nullptr, ssaoSrv.Get(), weightSrv.Get(), normalsSrv.Get());  // t3 the weight
        draw(resolve.Get(), rtvR.Get(), nullptr, nullptr, nullptr, nullptr);  // a history from the resolve too
        qvfr::sunfx::EndGroup();
        qvfr::ssgiao::CaptureHistory(ctx.Get(), 0, radiance.Get());  // as DLSS's output, after the resolve's
    };
    std::vector<uint64_t> g1, g2, r6;
    giFrame(1);
    Read(dev.Get(), ctx.Get(), outG.Get(), g1);
    giFrame(1);
    Read(dev.Get(), ctx.Get(), outG.Get(), g2);
    giFrame(2);
    Read(dev.Get(), ctx.Get(), outR.Get(), r6);
    const qvfr::ssgiao::TestState sg = qvfr::ssgiao::TestGet();
    // The probe: asked for by a stats line, made at the next frame's bounce, read 3 frames later.
    qvfr::ssgiao::LogStats(false);
    for (int i = 0; i < 5; ++i) {  // Present flushes in the game; here by hand
        giFrame(1);
        ctx->Flush();
        Sleep(20);
    }
    const std::string probe = qvfr::ssgiao::TestGet().probe;
    const unsigned long long dumpLines = qvfr::ssgiao::TestGet().dumpLines;  // the draw dump after each change to on
    qvfr::ssgiao::Shutdown();
    qvfr::sunfx::Shutdown();

    // The wall just above the corner (floor 6 cm below), the wall high up, the floor near the eye.
    const float corner = at(o1, kW / 2, 55), wallHigh = at(o1, kW / 2, 10), floorNear = at(o1, kW / 2, 90);
    float maxDiffA = 0, minO2 = 1, maxErrR = 0, maxErr0 = 0;
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) {
            maxDiffA = std::fmax(maxDiffA, std::fabs(at(o1, x, y) - at(a1, x, y)));
            minO2 = std::fmin(minO2, at(o2, x, y));
            maxErr0 = std::fmax(maxErr0, std::fabs(at(o0, x, y) - 0.5f));
            const float r = Half((uint16_t)(r3[(size_t)y * kW + x] & 0xFFFF));
            maxErrR = std::fmax(maxErrR, std::fabs(r - 500 * at(o1, x, y)) / 500);
        }
    const bool aoOk = corner < 0.85f && wallHigh > 0.99f && floorNear > 0.99f;
    float maxDiffLm = 0;
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x) maxDiffLm = std::fmax(maxDiffLm, std::fabs(at(o1, x, y) - at(aLm, x, y)));
    const bool lmOk = lmAmbient == 1 && maxDiffLm < 1e-6f;
    // The bounce: the ambient target minus the AO the ambient draw wrote there (the same in every channel).
    auto bounce = [&](const std::vector<uint64_t>& g, UINT x, UINT y, int ch) {
        return Half((uint16_t)(g[(size_t)y * kW + x] >> (16 * ch))) - at(o1, x, y);
    };
    float maxG1 = 0;
    for (UINT y = 0; y < kH; ++y)
        for (UINT x = 0; x < kW; ++x)
            for (int ch = 0; ch < 3; ++ch) maxG1 = std::fmax(maxG1, std::fabs(bounce(g1, x, y, ch)));
    // Floor 0.35 m and wall 0.44 m from the corner, on the world side (x 96) and the vehicle side (x 32).
    float floorRed[2], floorGreen[2], wallRed[2], wallGreen[2];
    const UINT xs[2] = {96, 32};
    for (int i = 0; i < 2; ++i) {
        floorRed[i] = bounce(g2, xs[i], 56, 0);
        floorGreen[i] = bounce(g2, xs[i], 56, 1);
        wallRed[i] = bounce(g2, xs[i], 52, 0);
        wallGreen[i] = bounce(g2, xs[i], 52, 1);
    }
    float openMax = 0;
    for (UINT x = 8; x < kW - 8; ++x)
        for (int ch = 0; ch < 3; ++ch)
            openMax = std::fmax(openMax, std::fmax(std::fabs(bounce(g2, x, 10, ch)), std::fabs(bounce(g2, x, 90, ch))));
    bool giOk = maxG1 < 1e-3f && openMax < 0.01f;
    for (int i = 0; i < 2; ++i)
        giOk = giOk && floorRed[i] > 0.02f && floorRed[i] > 2 * std::fabs(floorGreen[i]) && wallGreen[i] > 0.02f &&
               wallGreen[i] > 2 * std::fabs(wallRed[i]);
    // The preview (gi=2, level 2) over the resolve: 2 x the bounce of frame 3 (as frame 2's: the same history).
    const float previewRed = Half((uint16_t)(r6[(size_t)56 * kW + 96] & 0xFFFF));
    giOk = giOk && std::fabs(previewRed - 2 * floorRed[0]) < 0.01f + 0.05f * floorRed[0];
    // 3 frames: 2 passes with the bounce, 2 history copies each (resolve, DLSS output), 2 added, 1 without history.
    const bool giCountsOk = sg.giPasses == 2 && sg.captures == 6 && sg.composites == 2 && sg.noHistory == 1 &&
                            sg.previews == st.previews + 1 && sg.refused == st.refused && sg.faults == 0 &&
                            probe.find("samples") != std::string::npos && dumpLines > 0;
    // Frame 1 (ao=1): 1 pass, 2 ambient draws with own AO (the frame's and the extra one), 1 missed (other normals).
    // Frames 2 and 3 add a pass, an ambient draw and a miss each, frame 3 the preview; frame 4 (ao=0) nothing.
    const bool countsOk = st1.passes == 1 && st1.ambient == 2 && st1.ambientMissed == 1 && st.passes == 3 &&
                          st.ambient == 4 && st.ambientMissed == 3 && st.previews == 1 && st.faults == 0;

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
    const bool ok = aoOk && restored1 && maxDiffA < 1e-6f && minO2 > 0.999f && maxErrR < 0.01f && maxErr0 < 1e-3f &&
                    countsOk && giOk && giCountsOk && lmOk && !warnings;
    printf("SSGI AO check %s: AO at the floor-wall corner %.3f, wall high up %.3f, floor near the eye %.3f; game's "
           "slot back after the overlay %d; ambient = overlay (max difference %.6f); ao=2 overlay min %.4f; ao=3 preview "
           "max error %.4f; ao=0 game's texture max error %.4f; passes %llu, ambient with own AO %llu, missed %llu, "
           "previews %llu, refused %llu, faults %llu; debug layer %s, %d warnings\n",
           ok ? "OK" : "FAILED", corner, wallHigh, floorNear, restored1, maxDiffA, minO2, maxErrR, maxErr0, st.passes,
           st.ambient, st.ambientMissed, st.previews, st.refused, st.faults, debugLayer ? "on" : "not installed",
           warnings);
    printf("SSGI GI check %s: without a history max %.5f; floor by the wall red %.3f green %.3f (vehicle side red %.3f "
           "green %.3f); wall by the floor green %.3f red %.3f (vehicle side green %.3f red %.3f); open surfaces max "
           "%.4f; preview red %.3f; passes with the bounce %llu, history copies %llu, added %llu, without history %llu, "
           "previews %llu, refused %llu" "\n",
           giOk && giCountsOk ? "OK" : "FAILED", maxG1, floorRed[0], floorGreen[0], floorRed[1], floorGreen[1],
           wallGreen[0], wallRed[0], wallGreen[1], wallRed[1], openMax, previewRed, sg.giPasses, sg.captures,
           sg.composites, sg.noHistory, sg.previews, sg.refused);
    printf("SSGI lightmap ambient check %s: own AO there %llu, max difference %.6f\n", lmOk ? "OK" : "FAILED", lmAmbient,
           maxDiffLm);
    printf("%s; draw dump lines %llu\n", probe.empty() ? "SSGI GI probe: none" : probe.c_str(), dumpLines);
    return ok ? 0 : 7;
}

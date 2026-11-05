// --taa-fix: the TAA viewportScale fix of taa_views.h on a real D3D11 device, without SteamVR and without the proxy DLL.
// The game keeps the TAA resolve's and locks' cb0 (1/w, 1/h, w, h, accumulation rates) at one fixed place for all
// views. Here that place holds another view's size (1814 x 1522); a pass shaped like the locks (render target R8,
// t3 R16G16_FLOAT) and one shaped like the resolve (R11G11B10, t3 R16G16_FLOAT, t5 R8) write what their cb0 says.
// Checks that both read the size of their own render target and the rest of the game's cb0, that the game's binding
// (buffer, first constant, count) comes back after each draw, the fix counter and the debug layer.
// Then the object velocity fix: a vertex shader reads prevProjMatrix and prevViewProj from COMMON_BUFFER2 (slot 12) and
// writes them into an R16G16_FLOAT target. As in a focus group it must read the focus view's previous matrices; outside
// one, the game's.
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "sun_fx.h"
#include "taa_views.h"

namespace {

const char kHlsl[] = R"(
cbuffer G : register(b0) { float4 viewportScale; float velocityRate; float constantRate; float rotation; float pad; };
float4 VS(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 Locks(float4 p : SV_Position) : SV_Target { return float4(viewportScale.z / 4096, 0, 0, 1); }
float4 Resolve(float4 p : SV_Position) : SV_Target {
    return float4(viewportScale.z / 4096, viewportScale.y * 16, constantRate, 1);
}
cbuffer COMMON_BUFFER2 : register(b12) { float4 common2[61]; };  // prevProjMatrix at 51, prevViewProj at 55
struct VelOut { float4 p : SV_Position; float2 v : TEXCOORD0; };
VelOut VelVS(uint id : SV_VertexID) {
    VelOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    o.v = float2(common2[51].x, common2[56].y);
    return o;
}
float4 VelPS(VelOut i) : SV_Target { return float4(i.v, 0, 1); }
)";

constexpr UINT kW = 64, kH = 48;
constexpr UINT kFirst = 32;  // the game's cb0 at constant 32 of its big buffer (a multiple of 16)

float Small(uint32_t bits, int mant) {  // R11G11B10 channel: 5-bit exponent, no sign
    const uint32_t e = bits >> mant, m = bits & ((1u << mant) - 1);
    return e ? std::ldexp(1.0f + (float)m / (1u << mant), (int)e - 15) : std::ldexp((float)m / (1u << mant), -14);
}

}  // namespace

int RunTaaFix() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const bool debugLayer = SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                        D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                                        nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);
    ID3D11DeviceContext1* c1 = nullptr;
    ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&c1);
    // The DrawIndexed detour that calls the fix belongs to the sun effects module.
    if (!c1 || !qvfr::sunfx::TestInstall(dev, ctx)) {
        printf("TAA fix: no D3D11.1 context or the draw hooks could not be installed\n");
        return 1;
    }
    qvfr::taaviews::TestEnableFix(ctx, true);

    auto blob = [&](const char* entry, const char* target) {
        ID3DBlob *code = nullptr, *err = nullptr;
        if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "taa_fix_smoke", nullptr, nullptr, entry, target, 0, 0, &code,
                              &err))) {
            printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
            code = nullptr;
        }
        if (err) err->Release();
        return code;
    };
    ID3DBlob* vsCode = blob("VS", "vs_5_0");
    ID3DBlob* locksCode = blob("Locks", "ps_5_0");
    ID3DBlob* resolveCode = blob("Resolve", "ps_5_0");
    ID3DBlob* velVsCode = blob("VelVS", "vs_5_0");
    ID3DBlob* velPsCode = blob("VelPS", "ps_5_0");
    if (!vsCode || !locksCode || !resolveCode || !velVsCode || !velPsCode) return 1;
    ID3D11VertexShader* velVs = nullptr;
    ID3D11PixelShader* velPs = nullptr;
    dev->CreateVertexShader(velVsCode->GetBufferPointer(), velVsCode->GetBufferSize(), nullptr, &velVs);
    dev->CreatePixelShader(velPsCode->GetBufferPointer(), velPsCode->GetBufferSize(), nullptr, &velPs);
    velVsCode->Release();
    velPsCode->Release();
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader *locksPs = nullptr, *resolvePs = nullptr;
    dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(locksCode->GetBufferPointer(), locksCode->GetBufferSize(), nullptr, &locksPs);
    dev->CreatePixelShader(resolveCode->GetBufferPointer(), resolveCode->GetBufferSize(), nullptr, &resolvePs);
    for (ID3DBlob* b : {vsCode, locksCode, resolveCode}) b->Release();

    // Targets and inputs with the formats of the game's TAA passes.
    auto tex = [&](DXGI_FORMAT f, UINT bind, ID3D11Texture2D** t) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = kW;
        d.Height = kH;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = f;
        d.SampleDesc = {1, 0};
        d.BindFlags = bind;
        return SUCCEEDED(dev->CreateTexture2D(&d, nullptr, t));
    };
    ID3D11Texture2D *locksRt = nullptr, *resolveRt = nullptr, *vel = nullptr, *locksIn = nullptr;
    ID3D11RenderTargetView *locksRtv = nullptr, *resolveRtv = nullptr;
    ID3D11ShaderResourceView *velSrv = nullptr, *locksSrv = nullptr;
    bool made = tex(DXGI_FORMAT_R8_UNORM, D3D11_BIND_RENDER_TARGET, &locksRt) &&
                tex(DXGI_FORMAT_R11G11B10_FLOAT, D3D11_BIND_RENDER_TARGET, &resolveRt) &&
                tex(DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_SHADER_RESOURCE, &vel) &&
                tex(DXGI_FORMAT_R8_UNORM, D3D11_BIND_SHADER_RESOURCE, &locksIn) &&
                SUCCEEDED(dev->CreateRenderTargetView(locksRt, nullptr, &locksRtv)) &&
                SUCCEEDED(dev->CreateRenderTargetView(resolveRt, nullptr, &resolveRtv)) &&
                SUCCEEDED(dev->CreateShaderResourceView(vel, nullptr, &velSrv)) &&
                SUCCEEDED(dev->CreateShaderResourceView(locksIn, nullptr, &locksSrv));
    // The game's big dynamic constant buffer; its cb0 says 1814 x 1522 (another view's size).
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 64 * 1024;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ID3D11Buffer* big = nullptr;
    made = made && SUCCEEDED(dev->CreateBuffer(&bd, nullptr, &big));
    const UINT idx[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    D3D11_BUFFER_DESC ib{};
    ib.ByteWidth = sizeof(idx);
    ib.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA id{idx};
    ID3D11Buffer* ibuf = nullptr;
    made = made && SUCCEEDED(dev->CreateBuffer(&ib, &id, &ibuf));
    if (!made) {
        printf("TAA fix: could not make the test objects\n");
        return 1;
    }
    D3D11_MAPPED_SUBRESOURCE m{};
    ctx->Map(big, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    memset(m.pData, 0, bd.ByteWidth);
    const float game[8] = {1.0f / 1814, 1.0f / 1522, 1814, 1522, 0, 0.95f, 0.5f, 0};
    memcpy((uint8_t*)m.pData + kFirst * 16, game, sizeof(game));
    ctx->Unmap(big, 0);

    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetIndexBuffer(ibuf, DXGI_FORMAT_R32_UINT, 0);
    ctx->VSSetShader(vs, nullptr, 0);
    const D3D11_VIEWPORT vp{0, 0, (float)kW, (float)kH, 0, 1};
    ctx->RSSetViewports(1, &vp);
    UINT first = kFirst, num = 16;
    c1->PSSetConstantBuffers1(0, 1, &big, &first, &num);
    int restored = 0;
    auto checkBinding = [&] {
        ID3D11Buffer* b = nullptr;
        UINT f = 0, n = 0;
        c1->PSGetConstantBuffers1(0, 1, &b, &f, &n);
        restored += b == big && f == kFirst && n == 16;
        if (b) b->Release();
    };
    // Locks: R8 target, t3 velocity.
    ID3D11ShaderResourceView* locksViews[4] = {nullptr, nullptr, nullptr, velSrv};
    ctx->PSSetShaderResources(0, 4, locksViews);
    ctx->OMSetRenderTargets(1, &locksRtv, nullptr);
    ctx->PSSetShader(locksPs, nullptr, 0);
    ctx->DrawIndexed(6, 0, 0);
    checkBinding();
    // Resolve: R11G11B10 target, t3 velocity, t5 locks.
    ID3D11ShaderResourceView* resolveViews[6] = {nullptr, nullptr, nullptr, velSrv, nullptr, locksSrv};
    ctx->OMSetRenderTargets(1, &resolveRtv, nullptr);
    ctx->PSSetShaderResources(0, 6, resolveViews);
    ctx->PSSetShader(resolvePs, nullptr, 0);
    ctx->DrawIndexed(6, 0, 0);
    checkBinding();
    ID3D11RenderTargetView* none = nullptr;
    ctx->OMSetRenderTargets(1, &none, nullptr);

    // Object velocity: the game's COMMON_BUFFER2 at constant 64 says prevProjMatrix[0].x 0.5, prevViewProj[1].y 0.25.
    constexpr UINT kCommon2 = 64;
    ctx->Map(big, 0, D3D11_MAP_WRITE_NO_OVERWRITE, 0, &m);
    float* c2 = (float*)((uint8_t*)m.pData + kCommon2 * 16);
    c2[51 * 4] = 0.5f;
    c2[56 * 4 + 1] = 0.25f;
    ctx->Unmap(big, 0);
    ID3D11Texture2D *velRt[2] = {};
    ID3D11RenderTargetView* velRtv[2] = {};
    for (int i = 0; i < 2; ++i)
        made = made && tex(DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_RENDER_TARGET, &velRt[i]) &&
               SUCCEEDED(dev->CreateRenderTargetView(velRt[i], nullptr, &velRtv[i]));
    if (!made) {
        printf("TAA fix: could not make the velocity targets\n");
        return 1;
    }
    UINT c2First = kCommon2, c2Num = 64;
    c1->VSSetConstantBuffers1(12, 1, &big, &c2First, &c2Num);
    ctx->VSSetShader(velVs, nullptr, 0);
    ctx->PSSetShader(velPs, nullptr, 0);
    ID3D11ShaderResourceView* noViews[6] = {};
    ctx->PSSetShaderResources(0, 6, noViews);
    float focusProj[16] = {}, focusViewProj[16] = {};
    focusProj[0] = 2.0f;
    focusViewProj[5] = 3.0f;
    int velRestored = 0;
    for (int i = 0; i < 2; ++i) {
        qvfr::taaviews::TestSetPrevFocus(i == 0 ? focusProj : nullptr, focusViewProj);
        ctx->OMSetRenderTargets(1, &velRtv[i], nullptr);
        qvfr::taaviews::OnSetTargets(ctx, 1, &velRtv[i]);  // as quad's OMSetRenderTargets detour does
        ctx->DrawIndexed(9, 0, 0);
        ID3D11Buffer* b = nullptr;
        UINT f = 0, n = 0;
        c1->VSGetConstantBuffers1(12, 1, &b, &f, &n);
        velRestored += b == big && f == kCommon2 && n == 64;
        if (b) b->Release();
    }
    qvfr::taaviews::TestSetPrevFocus(nullptr, nullptr);
    ctx->OMSetRenderTargets(1, &none, nullptr);

    // Read back the center pixel of both targets.
    auto read = [&](ID3D11Texture2D* t, uint32_t& out) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* s = nullptr;
        if (FAILED(dev->CreateTexture2D(&d, nullptr, &s))) return false;
        ctx->CopyResource(s, t);
        D3D11_MAPPED_SUBRESOURCE r{};
        const bool ok = SUCCEEDED(ctx->Map(s, 0, D3D11_MAP_READ, 0, &r));
        if (ok) {
            const uint8_t* row = (const uint8_t*)r.pData + (kH / 2) * r.RowPitch;
            if (d.Format == DXGI_FORMAT_R8_UNORM)
                out = row[kW / 2];
            else
                memcpy(&out, row + (kW / 2) * 4, 4);
            ctx->Unmap(s, 0);
        }
        s->Release();
        return ok;
    };
    uint32_t lp = 0, rp = 0, vp0 = 0, vp1 = 0;
    const bool readOk = read(locksRt, lp) && read(resolveRt, rp) && read(velRt[0], vp0) && read(velRt[1], vp1);
    auto half = [](uint32_t h) {
        const uint32_t e = (h >> 10) & 31, mm = h & 1023;
        const float v = e ? std::ldexp(1.0f + mm / 1024.0f, (int)e - 15) : std::ldexp(mm / 1024.0f, -14);
        return (h >> 15) & 1 ? -v : v;
    };
    const float focusX = half(vp0 & 0xFFFF), focusY = half(vp0 >> 16), gameX = half(vp1 & 0xFFFF),
                gameY = half(vp1 >> 16);
    const bool velOk = focusX == 2.0f && focusY == 3.0f && gameX == 0.5f && gameY == 0.25f && velRestored == 2 &&
                       qvfr::taaviews::TestObjectDraws() == 1;
    const float locksW = lp / 255.0f * 4096;
    const float resolveW = Small(rp & 0x7FF, 6) * 4096, resolveInvH = Small((rp >> 11) & 0x7FF, 6) / 16,
                resolveRate = Small(rp >> 22, 5);
    const bool locksOk = std::fabs(locksW - kW) < 9;  // 8-bit output: 16 per step
    const bool resolveOk = std::fabs(resolveW - kW) < 2 && std::fabs(resolveInvH - 1.0f / kH) < 0.001f &&
                           std::fabs(resolveRate - 0.95f) < 0.02f;
    const uint64_t fixed = qvfr::taaviews::TestFixedDraws();
    UINT64 warnings = 0;
    if (info) {
        const UINT64 n = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < n; ++i) {
            SIZE_T len = 0;
            info->GetMessage(i, nullptr, &len);
            D3D11_MESSAGE* msg = (D3D11_MESSAGE*)malloc(len);
            if (msg && SUCCEEDED(info->GetMessage(i, msg, &len)) && msg->Severity <= D3D11_MESSAGE_SEVERITY_WARNING) {
                ++warnings;
                printf("  debug layer: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
            }
            free(msg);
        }
    }
    const bool ok = readOk && locksOk && resolveOk && restored == 2 && fixed == 2 && velOk && warnings == 0;
    printf("TAA fix check %s: locks read width %.0f (target %u, the game's cb0 1814), resolve width %.1f, 1/height "
           "%.5f, constant rate %.3f (0.95 in the game's cb0); game binding back after %d of 2 draws; %llu draws "
           "fixed; object velocity in a focus group (%.2f, %.2f) (focus 2, 3), outside (%.2f, %.2f) (game 0.5, 0.25), "
           "game binding back after %d of 2, %llu draws fixed; debug layer %s, %llu warnings\n",
           ok ? "OK" : "FAILED", locksW, kW, resolveW, resolveInvH, resolveRate, restored,
           (unsigned long long)fixed, focusX, focusY, gameX, gameY, velRestored,
           (unsigned long long)qvfr::taaviews::TestObjectDraws(), info ? "on" : "off", (unsigned long long)warnings);
    qvfr::taaviews::TestEnableFix(nullptr, false);
    return ok ? 0 : 1;
}

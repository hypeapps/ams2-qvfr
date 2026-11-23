// --sr-carry: the DLSS SR carry (dlaa.cpp Carry, shaders/sr.hlsl PSColor) of an R16G16B16A16_TYPELESS texture that the
// game reads as UNORM (the TAA resolve's normal/stencil, stencil bits in .w), stretched 2x on a real D3D11 device.
// - through UNORM views from the nearest texel (the game's view format) every output texel is its source texel bit for
//   bit;
// - through FLOAT views with the linear filter (the carry's choice for a typeless texture without a hint) texels change:
//   UNORM values read as halfs (0xFFFF is a NaN) and neighbours blended, stencil bits too.
#include <windows.h>
#include <d3d11.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "sr_color_ps.h"  // kSrColorPS
#include "sr_vs.h"        // kSrVS

namespace {

constexpr UINT kSW = 4, kSH = 2, kDW = 8, kDH = 4;

struct Texel {
    uint16_t c[4];
};

// Normals near 1.0 and 0.5 (halfs NaN, Inf and negative) next to small ones, stencil classes 0x88 and 0.
Texel Source(UINT x, UINT y) {
    if ((x + y) & 1) return {{0x0000, 0x1234, 0x7FFF, 0x0000}};
    return {{0xFFFF, 0x8000, 0x7C01, 0x0088}};
}

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// The carry's draw from src into dst through views of `fmt` with `filter`; dst read back.
bool CarryOnce(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* src, DXGI_FORMAT fmt, D3D11_FILTER filter,
               std::vector<Texel>& out) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = kDW;
    d.Height = kDH;
    d.MipLevels = d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R16G16B16A16_TYPELESS;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D *dst = nullptr, *st = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11SamplerState* smp = nullptr;
    D3D11_SAMPLER_DESC sd{};
    sd.Filter = filter;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = fmt;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels = 1;
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = fmt;
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    D3D11_TEXTURE2D_DESC sdesc = d;
    sdesc.Usage = D3D11_USAGE_STAGING;
    sdesc.BindFlags = 0;
    sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    bool ok = SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &dst)) && SUCCEEDED(dev->CreateTexture2D(&sdesc, nullptr, &st)) &&
              SUCCEEDED(dev->CreateShaderResourceView(src, &sv, &srv)) &&
              SUCCEEDED(dev->CreateRenderTargetView(dst, &rv, &rtv)) &&
              SUCCEEDED(dev->CreateVertexShader(kSrVS, sizeof(kSrVS), nullptr, &vs)) &&
              SUCCEEDED(dev->CreatePixelShader(kSrColorPS, sizeof(kSrColorPS), nullptr, &ps)) &&
              SUCCEEDED(dev->CreateSamplerState(&sd, &smp));
    if (ok) {
        const D3D11_VIEWPORT vp{0, 0, (float)kDW, (float)kDH, 0, 1};
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->IASetInputLayout(nullptr);
        ctx->VSSetShader(vs, nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ctx->PSSetSamplers(0, 1, &smp);
        ctx->PSSetShaderResources(0, 1, &srv);
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        ctx->RSSetViewports(1, &vp);
        ctx->Draw(3, 0);
        ID3D11ShaderResourceView* none = nullptr;
        ctx->PSSetShaderResources(0, 1, &none);
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        ctx->CopyResource(st, dst);
        D3D11_MAPPED_SUBRESOURCE m{};
        ok = SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m));
        if (ok) {
            out.resize(kDW * kDH);
            for (UINT y = 0; y < kDH; ++y)
                memcpy(&out[y * kDW], (const uint8_t*)m.pData + y * m.RowPitch, kDW * sizeof(Texel));
            ctx->Unmap(st, 0);
        }
    }
    Rel(smp);
    Rel(ps);
    Rel(vs);
    Rel(rtv);
    Rel(srv);
    Rel(st);
    Rel(dst);
    return ok;
}

}  // namespace

int RunSrCarry() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    std::vector<Texel> data(kSW * kSH);
    for (UINT y = 0; y < kSH; ++y)
        for (UINT x = 0; x < kSW; ++x) data[y * kSW + x] = Source(x, y);
    D3D11_TEXTURE2D_DESC d{};
    d.Width = kSW;
    d.Height = kSH;
    d.MipLevels = d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R16G16B16A16_TYPELESS;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA init{data.data(), kSW * sizeof(Texel), 0};
    ID3D11Texture2D* src = nullptr;
    int fails = 0;
    if (FAILED(dev->CreateTexture2D(&d, &init, &src))) {
        printf("FAIL: source texture\n");
        ++fails;
    } else {
        std::vector<Texel> exact, old;
        const bool a = CarryOnce(dev, ctx, src, DXGI_FORMAT_R16G16B16A16_UNORM, D3D11_FILTER_MIN_MAG_MIP_POINT, exact);
        const bool b = CarryOnce(dev, ctx, src, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_FILTER_MIN_MAG_MIP_LINEAR, old);
        int exactBad = 0, oldBad = 0, oldStencil = 0, oldNan = 0;
        for (UINT y = 0; a && b && y < kDH; ++y)
            for (UINT x = 0; x < kDW; ++x) {
                const Texel s = Source(x / 2, y / 2), e = exact[y * kDW + x], o = old[y * kDW + x];
                if (memcmp(&s, &e, sizeof(Texel))) ++exactBad;
                if (memcmp(&s, &o, sizeof(Texel))) ++oldBad;
                if (o.c[3] != s.c[3]) ++oldStencil;
                if (s.c[0] == 0xFFFF && o.c[0] != 0xFFFF) ++oldNan;
            }
        printf("%s: UNORM views from the nearest texel: %d of %u texels differ from their source texel\n",
               a && b && !exactBad ? "PASS" : "FAIL", exactBad, kDW * kDH);
        printf("%s: FLOAT views with the linear filter: %d of %u texels changed, %d of them in the stencil bits, "
               "%d values 0xFFFF lost (first changed texel %04X %04X %04X %04X)\n",
               a && b && oldBad ? "PASS" : "FAIL", oldBad, kDW * kDH, oldStencil, oldNan, b ? old[0].c[0] : 0,
               b ? old[0].c[1] : 0, b ? old[0].c[2] : 0, b ? old[0].c[3] : 0);
        if (!a || !b || exactBad || !oldBad) ++fails;
    }
    Rel(src);
    Rel(ctx);
    Rel(dev);
    printf("%s\n", fails ? "SR CARRY: FAIL" : "SR CARRY: PASS");
    return fails ? 1 : 0;
}

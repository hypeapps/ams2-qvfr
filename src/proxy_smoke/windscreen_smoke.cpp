// --windscreen-depth: the windscreen glass's depth (shaders/windscreen.hlsl CSDilate) on a real D3D11 device, without
// SteamVR and without the proxy DLL. 64 x 48, a viewport at (4, 3) of 56 x 40, radius 1 and 2. Checks:
// - in the viewport the ring within the radius around a near square made far, its inside left near;
// - outside the viewport and in the other slice the depth as it was;
// - the debug layer stays quiet.
#include <windows.h>
#include <d3d11.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "windscreen_dilate_cs.h"  // kWindscreenDilateCS

namespace {

constexpr UINT kW = 64, kH = 48, kX0 = 4, kY0 = 3, kVW = 56, kVH = 40;

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

}  // namespace

int RunWindscreenDepth() {
    printf("Windscreen glass depth smoke (CSDilate, %ux%u, viewport %u,%u %ux%u)\n", kW, kH, kX0, kY0, kVW, kVH);
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
    int failures = 0;
    // The glass's depth (CSDilate), as windscreen.cpp makes it: slice 1 of a two-slice D32 array holds a near square
    // (0.3) on the far plane (1.0) inside the viewport (4, 3, 56 x 40) and 0.7 outside it; slice 0 holds 0.2. The copy
    // goes into the unordered-access texture (other slices kept), the shader writes slice 1, the result is copied into a
    // depth-stencil texture and read back. Radius 1 and 2. Expected: in the viewport the ring within the radius around
    // the square far, its inside near; outside the viewport and in slice 0 the depth as it was.
    for (int radius = 1; radius <= 2; ++radius) {
        constexpr UINT kSq0 = 20, kSq1 = 35;  // the square, inclusive
        auto isNear = [&](int x, int y) { return x >= (int)kSq0 && x <= (int)kSq1 && y >= (int)kSq0 && y <= (int)kSq1; };
        auto inVp = [&](UINT x, UINT y) { return x >= kX0 && x < kX0 + kVW && y >= kY0 && y < kY0 + kVH; };
        std::vector<float> src((size_t)kW * kH * 2, 0.2f);
        for (UINT y = 0; y < kH; ++y)
            for (UINT x = 0; x < kW; ++x)
                src[(size_t)kW * kH + (size_t)y * kW + x] = !inVp(x, y) ? 0.7f : isNear((int)x, (int)y) ? 0.3f : 1.0f;
        D3D11_TEXTURE2D_DESC d{};
        d.Width = kW;
        d.Height = kH;
        d.MipLevels = 1;
        d.ArraySize = 2;
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.SampleDesc.Count = 1;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_TEXTURE2D_DESC md = d;
        md.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        D3D11_TEXTURE2D_DESC w = d;
        w.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        D3D11_TEXTURE2D_DESC sd = d;
        sd.BindFlags = 0;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        D3D11_SUBRESOURCE_DATA si[2] = {{src.data(), kW * 4, 0}, {src.data() + (size_t)kW * kH, kW * 4, 0}};
        ID3D11Texture2D *st = nullptr, *mt = nullptr, *dt = nullptr, *rb = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        ID3D11UnorderedAccessView* uav = nullptr;
        ID3D11ComputeShader* cs = nullptr;
        ID3D11Buffer* cb = nullptr;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = DXGI_FORMAT_R32_FLOAT;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        sv.Texture2DArray.MipLevels = 1;
        sv.Texture2DArray.ArraySize = 2;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = DXGI_FORMAT_R32_FLOAT;
        uv.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
        uv.Texture2DArray.ArraySize = 2;
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = 32;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const bool ok = SUCCEEDED(dev->CreateTexture2D(&d, si, &st)) && SUCCEEDED(dev->CreateTexture2D(&md, nullptr, &mt)) &&
                        SUCCEEDED(dev->CreateTexture2D(&w, nullptr, &dt)) &&
                        SUCCEEDED(dev->CreateTexture2D(&sd, nullptr, &rb)) &&
                        SUCCEEDED(dev->CreateShaderResourceView(st, &sv, &srv)) &&
                        SUCCEEDED(dev->CreateUnorderedAccessView(mt, &uv, &uav)) &&
                        SUCCEEDED(dev->CreateComputeShader(kWindscreenDilateCS, sizeof(kWindscreenDilateCS), nullptr, &cs)) &&
                        SUCCEEDED(dev->CreateBuffer(&bd, nullptr, &cb));
        if (!ok) {
            printf("  depth radius %d: setup failed\n", radius);
            ++failures;
        } else {
            struct {
                uint32_t origin[2], size[2], tex[2], slice;
                int32_t radius;
            } p{{kX0, kY0}, {kVW, kVH}, {kW, kH}, 1, radius};
            ctx->UpdateSubresource(cb, 0, nullptr, &p, 0, 0);
            ctx->CopyResource(mt, st);
            ctx->CSSetShader(cs, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &cb);
            ctx->CSSetShaderResources(0, 1, &srv);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch((kW + 7) / 8, (kH + 7) / 8, 1);
            ID3D11ShaderResourceView* noSrv = nullptr;
            ID3D11UnorderedAccessView* noUav = nullptr;
            ctx->CSSetShaderResources(0, 1, &noSrv);
            ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
            ctx->CopyResource(dt, mt);
            ctx->CopyResource(rb, dt);
            int bad = 0, ring = 0;
            for (UINT sl = 0; sl < 2; ++sl) {
                D3D11_MAPPED_SUBRESOURCE m{};
                if (FAILED(ctx->Map(rb, sl, D3D11_MAP_READ, 0, &m))) {
                    printf("  depth radius %d: map failed\n", radius);
                    ++bad;
                    continue;
                }
                for (UINT y = 0; y < kH; ++y)
                    for (UINT x = 0; x < kW; ++x) {
                        const float got = ((const float*)((const uint8_t*)m.pData + (size_t)y * m.RowPitch))[x];
                        float want = sl == 0 ? 0.2f : 0.7f;
                        if (sl == 1 && inVp(x, y)) {
                            bool inner = true;  // the square and every pixel within the radius in it
                            for (int dy = -radius; dy <= radius; ++dy)
                                for (int dx = -radius; dx <= radius; ++dx)
                                    inner = inner && isNear((int)x + dx, (int)y + dy);
                            want = inner ? 0.3f : 1.0f;
                            ring += (isNear((int)x, (int)y) && !inner) ? 1 : 0;
                        }
                        if (std::fabs(got - want) > 1e-6f) {
                            if (bad < 4)
                                printf("  depth radius %d: slice %u (%u, %u) %.3f, expected %.3f\n", radius, sl, x, y, got,
                                       want);
                            ++bad;
                        }
                    }
                ctx->Unmap(rb, sl);
            }
            printf("  depth radius %d: %d texels off, the square's outer ring made far %d (%s)\n", radius, bad, ring,
                   bad == 0 && ring > 0 ? "OK" : "WRONG");
            if (bad || ring == 0) ++failures;
        }
        Rel(cb);
        Rel(cs);
        Rel(uav);
        Rel(srv);
        Rel(rb);
        Rel(dt);
        Rel(mt);
        Rel(st);
    }
    UINT64 msgs = 0;
    if (info) {
        msgs = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < msgs && i < 8; ++i) {
            SIZE_T len = 0;
            info->GetMessage(i, nullptr, &len);
            std::vector<char> buf(len);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (SUCCEEDED(info->GetMessage(i, msg, &len))) printf("  debug layer: %s\n", msg->pDescription);
        }
    }
    printf("  debug layer: %s, %llu message(s)\n", debugLayer ? "on" : "not available", (unsigned long long)msgs);
    if (msgs) ++failures;
    Rel(info);
    Rel(ctx);
    Rel(dev);
    printf("Windscreen glass depth smoke: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}

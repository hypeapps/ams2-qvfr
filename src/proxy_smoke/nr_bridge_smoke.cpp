// qvfr_proxy_smoke --nr-bridge: the NR bridge (nr_bridge.h) without the game. Two R11G11B10_FLOAT images in the focus
// size with random valid values (denormals included, no Inf/NaN) go through the D3D12 device and back many times on
// two slots; they must come back bit for bit, without D3D11 debug layer errors.
#include <windows.h>

#include <d3d11.h>
#include <d3d11sdklayers.h>
#include <dxgi.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "nr_bridge.h"

namespace {

constexpr UINT kW = 2117, kH = 1775;  // the focus size in the user's quad views
constexpr int kRounds = 100;          // per slot
constexpr int kAtlasRounds = 20;      // slot 4: two regions' guides, then one evaluation

// Random R11G11B10_FLOAT: each channel's exponent below 31 (31 is Inf/NaN), mantissa anything.
uint32_t RandomPacked(std::mt19937& rng) {
    auto ch = [&](int mantBits) {
        const uint32_t e = rng() % 31, m = rng() & ((1u << mantBits) - 1);
        return (e << mantBits) | m;
    };
    return ch(6) | (ch(6) << 11) | (ch(5) << 22);
}

IDXGIAdapter1* NvidiaAdapter() {
    IDXGIFactory1* f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&f))) return nullptr;
    IDXGIAdapter1* found = nullptr;
    for (UINT i = 0; !found; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (f->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.VendorId == 0x10DE) found = a;
        else a->Release();
    }
    f->Release();
    return found;
}

}  // namespace

int RunNrBridge() {
    IDXGIAdapter1* adapter = NvidiaAdapter();
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const D3D_DRIVER_TYPE type = adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
    const D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1;
    bool debugLayer = SUCCEEDED(D3D11CreateDevice(adapter, type, nullptr, D3D11_CREATE_DEVICE_DEBUG, &fl, 1,
                                                  D3D11_SDK_VERSION, &dev, nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(adapter, type, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr,
                                                &ctx))) {
        printf("NR bridge: D3D11CreateDevice failed\n");
        if (adapter) adapter->Release();
        return 1;
    }
    if (adapter) adapter->Release();
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);
    printf("NR bridge: D3D11 device %s, debug layer %s\n", adapter ? "on the NVIDIA adapter" : "on the default adapter",
           debugLayer ? "on" : "off");

    std::mt19937 rng(1234);
    std::vector<uint32_t> ref[2];
    ID3D11Texture2D* tex[2] = {};
    ID3D11ShaderResourceView* srv[2] = {};
    ID3D11UnorderedAccessView* uav[2] = {};
    D3D11_TEXTURE2D_DESC d{};
    d.Width = kW;
    d.Height = kH;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    for (int i = 0; i < 2; ++i) {
        ref[i].resize((size_t)kW * kH);
        for (uint32_t& p : ref[i]) p = RandomPacked(rng);
        D3D11_SUBRESOURCE_DATA init{ref[i].data(), kW * 4, 0};
        if (FAILED(dev->CreateTexture2D(&d, &init, &tex[i])) ||
            FAILED(dev->CreateShaderResourceView(tex[i], nullptr, &srv[i])) ||
            FAILED(dev->CreateUnorderedAccessView(tex[i], nullptr, &uav[i]))) {
            printf("NR bridge: test textures failed\n");
            return 1;
        }
    }

    int fails = 0;
    {
        qvfr::nrbridge::Bridge bridge;
        if (!bridge.Init(dev)) {
            printf("NR bridge: Init failed: %s\n", bridge.why().c_str());
            return 1;
        }
        printf("NR bridge: D3D12 device on %s\n", bridge.adapter().c_str());
        // Slot 0: plain passthrough. Slot 1: the model's shape (input_scale 0.002, Reinhard and gamma, transfer ratio)
        // around the copy; an unchanged image must also come back bit for bit.
        qvfr::nrbridge::Settings plain, shaped;
        shaped.testShaders = true;
        shaped.inputScale = 0.002f;
        shaped.tonemap = 2;
        shaped.transfer = 1;
        for (int r = 0; r < kRounds; ++r)
            for (int i = 0; i < 2; ++i) {
                bridge.Configure(i == 0 ? plain : shaped);
                if (!bridge.Process(ctx, i, srv[i], uav[i], nullptr, nullptr, kW, kH, r == 0)) {
                    printf("NR bridge: Process failed in round %d slot %d: %s\n", r, i, bridge.why().c_str());
                    return 1;
                }
            }

        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        for (int i = 0; i < 2; ++i) {
            ID3D11Texture2D* st = nullptr;
            if (FAILED(dev->CreateTexture2D(&sd, nullptr, &st))) return 1;
            ctx->CopyResource(st, tex[i]);
            D3D11_MAPPED_SUBRESOURCE map{};
            if (FAILED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &map))) return 1;
            size_t diff = 0;
            for (UINT y = 0; y < kH; ++y) {
                const uint32_t* row = (const uint32_t*)((const uint8_t*)map.pData + (size_t)y * map.RowPitch);
                for (UINT x = 0; x < kW; ++x) {
                    const uint32_t want = ref[i][(size_t)y * kW + x];
                    if (row[x] == want) continue;
                    if (diff < 5) printf("  slot %d (%u, %u): 0x%08X, want 0x%08X\n", i, x, y, row[x], want);
                    ++diff;
                }
            }
            ctx->Unmap(st, 0);
            st->Release();
            printf("NR bridge: slot %d (%s) after %d round trips: %zu of %u pixels differ\n", i,
                   i == 0 ? "plain" : "scale, Reinhard, gamma, transfer ratio", kRounds, diff, kW * kH);
            if (diff) ++fails;
        }
        // mode=final: a finished image (R8G8B8A8_TYPELESS through UNORM views, alpha random) on slots 2 and 3, guides
        // first, then the color with guidesReady; slot 3 with strength 3. Without the model the edit is zero, so both
        // must come back bit for bit, alpha included.
        {
            D3D11_TEXTURE2D_DESC fd = d;
            fd.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            std::vector<uint32_t> fref((size_t)kW * kH);
            for (uint32_t& p : fref) p = rng();
            ID3D11Texture2D* ft[2] = {};
            ID3D11ShaderResourceView* fs[2] = {};
            ID3D11UnorderedAccessView* fu[2] = {};
            D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
            sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sv.Texture2D.MipLevels = 1;
            D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};
            uv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uv.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            for (int i = 0; i < 2; ++i) {
                D3D11_SUBRESOURCE_DATA init{fref.data(), kW * 4, 0};
                if (FAILED(dev->CreateTexture2D(&fd, &init, &ft[i])) ||
                    FAILED(dev->CreateShaderResourceView(ft[i], &sv, &fs[i])) ||
                    FAILED(dev->CreateUnorderedAccessView(ft[i], &uv, &fu[i]))) {
                    printf("NR bridge: final-image test textures failed\n");
                    return 1;
                }
            }
            qvfr::nrbridge::Settings fin;
            fin.testShaders = true;
            qvfr::nrbridge::Settings fin3 = fin;
            fin3.strength = 3.0f;
            for (int r = 0; r < kRounds; ++r)
                for (int i = 0; i < 2; ++i) {
                    bridge.Configure(i == 0 ? fin : fin3);
                    if (!bridge.Guides(ctx, 2 + i, nullptr, nullptr, kW, kH) ||
                        !bridge.Process(ctx, 2 + i, fs[i], fu[i], nullptr, nullptr, kW, kH, r == 0, true)) {
                        printf("NR bridge: final image failed in round %d slot %d: %s\n", r, 2 + i,
                               bridge.why().c_str());
                        return 1;
                    }
                }
            // guidesReady at a size without guides: nothing written, not a failure.
            if (bridge.Process(ctx, 2, fs[0], fu[0], nullptr, nullptr, kW - 1, kH, false, true) || bridge.failed()) {
                printf("NR bridge: guidesReady without guides of the size did not refuse\n");
                ++fails;
            }
            // The atlas (slot 4): the guides of two images go into the halves of slot 0's image, then the whole image
            // goes through once (it must stay bit for bit too). A region outside the atlas is refused.
            for (int r = 0; r < kAtlasRounds; ++r) {
                bridge.Configure(fin);
                const UINT half = kW / 2;
                if (!bridge.Guides(ctx, 4, nullptr, nullptr, half, kH, 0, 0, kW, kH) ||
                    !bridge.Guides(ctx, 4, nullptr, nullptr, kW - half, kH, half, 0, kW, kH) ||
                    !bridge.Process(ctx, 4, fs[0], fu[0], nullptr, nullptr, kW, kH, r == 0, true)) {
                    printf("NR bridge: atlas failed in round %d: %s\n", r, bridge.why().c_str());
                    return 1;
                }
            }
            if (bridge.Guides(ctx, 4, nullptr, nullptr, kW, kH, 1, 0, kW, kH) || bridge.failed()) {
                printf("NR bridge: a region outside the atlas was not refused\n");
                ++fails;
            }
            D3D11_TEXTURE2D_DESC fsd = fd;
            fsd.Usage = D3D11_USAGE_STAGING;
            fsd.BindFlags = 0;
            fsd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            for (int i = 0; i < 2; ++i) {
                ID3D11Texture2D* st = nullptr;
                if (FAILED(dev->CreateTexture2D(&fsd, nullptr, &st))) return 1;
                ctx->CopyResource(st, ft[i]);
                D3D11_MAPPED_SUBRESOURCE map{};
                if (FAILED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &map))) return 1;
                size_t diff = 0;
                for (UINT y = 0; y < kH; ++y) {
                    const uint32_t* row = (const uint32_t*)((const uint8_t*)map.pData + (size_t)y * map.RowPitch);
                    for (UINT x = 0; x < kW; ++x) {
                        const uint32_t want = fref[(size_t)y * kW + x];
                        if (row[x] == want) continue;
                        if (diff < 5) printf("  slot %d (%u, %u): 0x%08X, want 0x%08X\n", 2 + i, x, y, row[x], want);
                        ++diff;
                    }
                }
                ctx->Unmap(st, 0);
                st->Release();
                printf("NR bridge: slot %d (final image RGBA8%s, guides first) after %d round trips: %zu of %u pixels "
                       "differ\n",
                       2 + i, i ? ", strength 3" : ", then the atlas of two regions in slot 4", kRounds, diff, kW * kH);
                if (diff) ++fails;
            }
            for (int i = 0; i < 2; ++i) {
                fu[i]->Release();
                fs[i]->Release();
                ft[i]->Release();
            }
        }

        const qvfr::nrbridge::Stats s = bridge.stats();
        printf("NR bridge: %llu calls, CPU %.3f ms per call (with the Flush), D3D12 GPU %.3f ms per call (%llu samples)\n",
               (unsigned long long)s.calls, s.calls ? s.cpuMs / (double)s.calls : 0.0,
               s.gpuSamples ? s.gpuMs / (double)s.gpuSamples : 0.0, (unsigned long long)s.gpuSamples);
        if (s.calls != 4u * kRounds + kAtlasRounds) ++fails;
    }  // the bridge is released here, while the device lives

    if (info) {
        UINT64 errors = 0;
        const UINT64 n = info->GetNumStoredMessages();
        for (UINT64 k = 0; k < n; ++k) {
            SIZE_T len = 0;
            info->GetMessage(k, nullptr, &len);
            std::vector<char> buf(len);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (FAILED(info->GetMessage(k, msg, &len))) continue;
            if (msg->Severity <= D3D11_MESSAGE_SEVERITY_ERROR) {
                if (errors < 5) printf("  D3D11 debug: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
                ++errors;
            }
        }
        printf("NR bridge: D3D11 debug layer errors: %llu\n", (unsigned long long)errors);
        if (errors) ++fails;
        info->Release();
    }
    for (int i = 0; i < 2; ++i) {
        uav[i]->Release();
        srv[i]->Release();
        tex[i]->Release();
    }
    ctx->Release();
    dev->Release();
    printf("NR bridge: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

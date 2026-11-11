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
constexpr int kAtlasRounds = 20;      // slot 4: two images into their regions, one evaluation, both back out

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
                       2 + i, i ? ", strength 3" : "", kRounds, diff, kW * kH);
                if (diff) ++fails;
            }
            for (int i = 0; i < 2; ++i) {
                fu[i]->Release();
                fs[i]->Release();
                ft[i]->Release();
            }
        }
        // The atlas path (slots 4 and 5): images straight into their regions of the slot's input (PackRegion), the
        // slot through once (Evaluate), each image straight back out into its own texture (UnpackRegion).
        {
            D3D11_TEXTURE2D_DESC td{};
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
            sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sv.Texture2D.MipLevels = 1;
            D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};
            uv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uv.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            struct Img {
                ID3D11Texture2D* tex = nullptr;
                ID3D11ShaderResourceView* srv = nullptr;
                ID3D11UnorderedAccessView* uav = nullptr;
            };
            std::vector<Img> made;
            auto make = [&](UINT w, UINT h, const std::vector<uint32_t>* px) {
                Img m;
                td.Width = w;
                td.Height = h;
                D3D11_SUBRESOURCE_DATA init{px ? px->data() : nullptr, w * 4, 0};
                if (FAILED(dev->CreateTexture2D(&td, px ? &init : nullptr, &m.tex)) ||
                    FAILED(dev->CreateShaderResourceView(m.tex, &sv, &m.srv)) ||
                    FAILED(dev->CreateUnorderedAccessView(m.tex, &uv, &m.uav)))
                    m = {};
                made.push_back(m);
                return m;
            };
            auto read = [&](ID3D11Texture2D* tex, UINT w, UINT h, std::vector<uint32_t>& out) {
                D3D11_TEXTURE2D_DESC sd = td;
                sd.Width = w;
                sd.Height = h;
                sd.Usage = D3D11_USAGE_STAGING;
                sd.BindFlags = 0;
                sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                ID3D11Texture2D* st = nullptr;
                if (FAILED(dev->CreateTexture2D(&sd, nullptr, &st))) return false;
                ctx->CopyResource(st, tex);
                D3D11_MAPPED_SUBRESOURCE map{};
                if (FAILED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &map))) return false;
                out.resize((size_t)w * h);
                for (UINT y = 0; y < h; ++y)
                    memcpy(&out[(size_t)y * w], (const uint8_t*)map.pData + (size_t)y * map.RowPitch, w * 4);
                ctx->Unmap(st, 0);
                st->Release();
                return true;
            };
            qvfr::nrbridge::Settings fin;
            fin.testShaders = true;
            fin.transfer = 0;  // as [nr] mode=final
            bridge.Configure(fin);

            // Two full-size images side by side in slot 4 (16 px apart): both back bit for bit, alpha included.
            constexpr UINT kGap = 16;
            const UINT aw = 2 * kW + kGap;
            std::vector<uint32_t> pa((size_t)kW * kH), pb((size_t)kW * kH), got;
            for (uint32_t& p : pa) p = rng();
            for (uint32_t& p : pb) p = rng();
            const Img ia = make(kW, kH, &pa), ib = make(kW, kH, &pb), oa = make(kW, kH, nullptr),
                      ob = make(kW, kH, nullptr);
            if (!ia.uav || !ib.uav || !oa.uav || !ob.uav) {
                printf("NR bridge: atlas test textures failed\n");
                return 1;
            }
            using qvfr::nrbridge::Mark;
            for (int r = 0; r < kAtlasRounds; ++r) {
                bridge.TimeMark(ctx, Mark::Guides);
                const bool g0 = bridge.Guides(ctx, 4, nullptr, nullptr, kW, kH, 0, 0, aw, kH);
                bridge.TimeMark(ctx, Mark::Other);
                bridge.TimeMark(ctx, Mark::Guides);
                const bool g1 = bridge.Guides(ctx, 4, nullptr, nullptr, kW, kH, kW + kGap, 0, aw, kH);
                bridge.TimeMark(ctx, Mark::Pack);
                if (!g0 || !g1 || !bridge.PackRegion(ctx, 4, ia.srv, 0, 0, kW, kH) ||
                    !bridge.PackRegion(ctx, 4, ib.srv, kW + kGap, 0, kW, kH) || !bridge.Evaluate(ctx, 4, r == 0) ||
                    !bridge.UnpackRegion(ctx, 4, ia.srv, oa.uav, 0, 0, kW, kH, 1, kW, kH) ||
                    !bridge.UnpackRegion(ctx, 4, ib.srv, ob.uav, kW + kGap, 0, kW, kH, 1, kW, kH)) {
                    printf("NR bridge: atlas failed in round %d: %s\n", r, bridge.why().c_str());
                    return 1;
                }
                bridge.TimeFrameEnd(ctx);
            }
            // The D3D11 timing of the atlas frames: results a few frames later, without flushing.
            ctx->Flush();
            for (int k = 0; k < 200 && bridge.stats().d11Frames < (uint64_t)kAtlasRounds; ++k) {
                Sleep(5);
                bridge.TimeFrameEnd(ctx);
            }
            {
                const qvfr::nrbridge::Stats t = bridge.stats();
                const double f = t.d11Frames ? (double)t.d11Frames : 1.0;
                const bool sane = t.d11Frames >= (uint64_t)kAtlasRounds / 4 && t.d11GuidesMs > 0 && t.d11PackMs > 0 &&
                                  t.d11HandoffMs > 0 && t.d11UnpackMs > 0;
                printf("NR bridge: D3D11 timing of %llu of %d atlas frames: guides %.3f, pack %.3f, hand-off %.3f, "
                       "unpack %.3f ms per frame%s\n",
                       (unsigned long long)t.d11Frames, kAtlasRounds, t.d11GuidesMs / f, t.d11PackMs / f,
                       t.d11HandoffMs / f, t.d11UnpackMs / f, sane ? "" : " (MISSING)");
                if (!sane) ++fails;
            }
            size_t diffA = 0, diffB = 0;
            if (!read(oa.tex, kW, kH, got)) return 1;
            for (size_t i = 0; i < got.size(); ++i) diffA += got[i] != pa[i];
            if (!read(ob.tex, kW, kH, got)) return 1;
            for (size_t i = 0; i < got.size(); ++i) diffB += got[i] != pb[i];
            printf("NR bridge: atlas of two images in slot 4 after %d round trips: %zu and %zu of %u pixels differ\n",
                   kAtlasRounds, diffA, diffB, kW * kH);
            if (diffA || diffB) ++fails;
            // Refused, not failures: a region outside the atlas (guides and color).
            if (bridge.Guides(ctx, 4, nullptr, nullptr, kW, kH, 1, 0, kW, kH) ||
                bridge.PackRegion(ctx, 4, ia.srv, kW + kGap + 1, 0, kW, kH) || bridge.failed()) {
                printf("NR bridge: a region outside the atlas was not refused\n");
                ++fails;
            }

            // An image at half resolution in slot 5: its region holds the 2x2 averages (odd sizes clamp at the edge;
            // read back through a step-1 unpack of the region alone); unpacked at its step without an edit it comes
            // back bit for bit.
            constexpr UINT kStep = 2, kX = 24, kY = 8;
            const UINT sw = (kW + kStep - 1) / kStep, sh = (kH + kStep - 1) / kStep;
            const UINT sAw = sw + kX + 8, sAh = sh + kY + 8;
            std::vector<uint32_t> img((size_t)kW * kH), regionPx;
            for (uint32_t& p : img) p = rng();
            const Img src = make(kW, kH, &img), part = make(sw, sh, nullptr), res = make(kW, kH, nullptr);
            if (!src.uav || !part.uav || !res.uav) {
                printf("NR bridge: half-resolution test textures failed\n");
                return 1;
            }
            if (!bridge.Guides(ctx, 5, nullptr, nullptr, sw, sh, kX, kY, sAw, sAh, kStep) ||
                !bridge.PackRegion(ctx, 5, src.srv, kX, kY, sw, sh, kStep) || !bridge.Evaluate(ctx, 5, true) ||
                !bridge.UnpackRegion(ctx, 5, src.srv, part.uav, kX, kY, sw, sh, 1, sw, sh) ||
                !bridge.UnpackRegion(ctx, 5, src.srv, res.uav, kX, kY, sw, sh, kStep, kW, kH)) {
                printf("NR bridge: half-resolution region failed: %s\n", bridge.why().c_str());
                return 1;
            }
            if (!read(part.tex, sw, sh, regionPx) || !read(res.tex, kW, kH, got)) return 1;
            size_t badAvg = 0, diff = 0;
            for (UINT y = 0; y < sh; ++y)
                for (UINT x = 0; x < sw; ++x)
                    for (int ch = 0; ch < 3; ++ch) {  // the region's alpha is not the image's
                        int sum = 0;
                        for (UINT j = 0; j < kStep; ++j)
                            for (UINT i = 0; i < kStep; ++i) {
                                const UINT px = (std::min)(x * kStep + i, kW - 1);
                                const UINT py = (std::min)(y * kStep + j, kH - 1);
                                sum += (int)((img[(size_t)py * kW + px] >> (8 * ch)) & 0xFF);
                            }
                        const int want = sum / (int)(kStep * kStep);
                        const int v = (int)((regionPx[(size_t)y * sw + x] >> (8 * ch)) & 0xFF);
                        if (v < want - 1 || v > want + 1) ++badAvg;
                    }
            for (size_t i = 0; i < got.size(); ++i) diff += got[i] != img[i];
            printf("NR bridge: half-resolution region %ux%u of %ux%u: %zu channels off the 2x2 average, the image back "
                   "without an edit: %zu of %u pixels differ\n",
                   sw, sh, kW, kH, badAvg, diff, kW * kH);
            if (badAvg || diff) ++fails;

            // A uniform edit: with debug=1 the model's output reads as 0.5 (no change: mid grey), so an image of 60/255
            // gets the edit 0.5 - 60/255 and comes back at 127 or 128 (alpha stays). Then only on a centre part (half
            // of each axis from (kW/4, kH/4), fading out over kFade pixels): outside the part the image stays bit for
            // bit, past the band the edit is whole, in the band in between.
            qvfr::nrbridge::Settings dbg = fin;
            dbg.debug = 1;
            bridge.Configure(dbg);
            std::vector<uint32_t> flat((size_t)kW * kH);
            for (uint32_t& p : flat) p = (rng() & 0xFF000000u) | 0x003C3C3Cu;
            const Img flatImg = make(kW, kH, &flat);
            if (!flatImg.uav) return 1;
            auto lifted = [](int v) { return v == 127 || v == 128; };
            if (!bridge.PackRegion(ctx, 5, flatImg.srv, kX, kY, sw, sh, kStep) || !bridge.Evaluate(ctx, 5, false) ||
                !bridge.UnpackRegion(ctx, 5, flatImg.srv, res.uav, kX, kY, sw, sh, kStep, kW, kH) ||
                !read(res.tex, kW, kH, got))
                return 1;
            size_t badLift = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                for (int ch = 0; ch < 3; ++ch) badLift += !lifted((int)((got[i] >> (8 * ch)) & 0xFF));
                badLift += (got[i] >> 24) != (flat[i] >> 24);
            }
            printf("NR bridge: a uniform edit enlarged onto the image: %zu channels off\n", badLift);
            if (badLift) ++fails;
            constexpr UINT kFade = 32;
            const UINT px0 = kW / 4, py0 = kH / 4, pw = kW / 2, ph = kH / 2;
            const UINT rw = (pw + kStep - 1) / kStep, rh = (ph + kStep - 1) / kStep;
            if (!bridge.Guides(ctx, 5, nullptr, nullptr, rw, rh, kX, kY, sAw, sAh, kStep, px0, py0) ||
                !bridge.PackRegion(ctx, 5, flatImg.srv, kX, kY, rw, rh, kStep, px0, py0) ||
                !bridge.Evaluate(ctx, 5, false) ||
                !bridge.UnpackRegion(ctx, 5, flatImg.srv, res.uav, kX, kY, rw, rh, kStep, kW, kH, px0, py0, kFade) ||
                !read(res.tex, kW, kH, got))
                return 1;
            size_t outsideOff = 0, insideOff = 0, bandOff = 0;
            for (UINT y = 0; y < kH; ++y)
                for (UINT x = 0; x < kW; ++x) {
                    const size_t i = (size_t)y * kW + x;
                    const int lx = (int)x - (int)px0, ly = (int)y - (int)py0;
                    const int inside = (std::min)((std::min)(lx, (int)pw - 1 - lx), (std::min)(ly, (int)ph - 1 - ly));
                    if (inside < 0) {
                        outsideOff += got[i] != flat[i];
                        continue;
                    }
                    for (int ch = 0; ch < 3; ++ch) {
                        const int v = (int)((got[i] >> (8 * ch)) & 0xFF);
                        if (inside >= (int)kFade)
                            insideOff += !lifted(v);
                        else
                            bandOff += v < 60 || v > 128;
                    }
                }
            printf("NR bridge: the edit on a centre part: outside %zu pixels, inside %zu, in the fade band %zu "
                   "channels off\n",
                   outsideOff, insideOff, bandOff);
            if (outsideOff || insideOff || bandOff) ++fails;
            for (Img& m : made) {
                if (m.uav) m.uav->Release();
                if (m.srv) m.srv->Release();
                if (m.tex) m.tex->Release();
            }
        }

        const qvfr::nrbridge::Stats s = bridge.stats();
        printf("NR bridge: %llu calls, CPU %.3f ms per call (with the Flush), D3D12 GPU %.3f ms per call (%llu samples)\n",
               (unsigned long long)s.calls, s.calls ? s.cpuMs / (double)s.calls : 0.0,
               s.gpuSamples ? s.gpuMs / (double)s.gpuSamples : 0.0, (unsigned long long)s.gpuSamples);
        if (s.calls != 4u * kRounds + kAtlasRounds + 3u) ++fails;
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

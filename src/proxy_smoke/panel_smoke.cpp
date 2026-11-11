// qvfr_proxy_smoke --panel: the tuning panel's picture (panel_ui.h) without the game. RmlUi draws the rows into the
// texture on a D3D11 device with the debug layer; the picture must hold the background, the red accent and text, change
// with the selection and not be drawn again without a change, and the "game's" state (a render target, a viewport, a
// pixel shader resource) must be the same after drawing. With QVFR_PANEL_BMP=<path> the picture is saved there.
#include <windows.h>

#include <d3d11.h>
#include <d3d11sdklayers.h>
#include <dxgi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "panel_ui.h"

namespace {

constexpr unsigned kW = 1320, kH = 1440;
constexpr char kFooter[] = "Up/Down choose  \xC2\xB7  Left/Right change  \xC2\xB7  Enter switch  \xC2\xB7  Esc close";

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

bool ReadBack(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* tex, std::vector<uint32_t>& out) {
    D3D11_TEXTURE2D_DESC d{};
    tex->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* st = nullptr;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &st))) return false;
    ctx->CopyResource(st, tex);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        st->Release();
        return false;
    }
    out.resize((size_t)d.Width * d.Height);
    for (UINT y = 0; y < d.Height; ++y)
        memcpy(&out[(size_t)y * d.Width], (const uint8_t*)m.pData + (size_t)y * m.RowPitch, d.Width * 4);
    ctx->Unmap(st, 0);
    st->Release();
    return true;
}

void SaveBmp(const char* path, const std::vector<uint32_t>& px, unsigned w, unsigned h) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    BITMAPFILEHEADER fh{};
    BITMAPINFOHEADER ih{};
    fh.bfType = 0x4D42;
    fh.bfOffBits = sizeof(fh) + sizeof(ih);
    fh.bfSize = fh.bfOffBits + w * h * 4;
    ih.biSize = sizeof(ih);
    ih.biWidth = (LONG)w;
    ih.biHeight = -(LONG)h;  // top-down
    ih.biPlanes = 1;
    ih.biBitCount = 32;
    fwrite(&fh, sizeof(fh), 1, f);
    fwrite(&ih, sizeof(ih), 1, f);
    for (uint32_t p : px) {  // RGBA -> BGRA
        const uint32_t bgra = (p & 0xFF00FF00u) | ((p & 0xFFu) << 16) | ((p >> 16) & 0xFFu);
        fwrite(&bgra, 4, 1, f);
    }
    fclose(f);
}

}  // namespace

int RunPanel() {
    IDXGIAdapter1* adapter = NvidiaAdapter();
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const D3D_DRIVER_TYPE type = adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
    const D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1;
    const bool debugLayer = SUCCEEDED(D3D11CreateDevice(adapter, type, nullptr, D3D11_CREATE_DEVICE_DEBUG, &fl, 1,
                                                        D3D11_SDK_VERSION, &dev, nullptr, &ctx));
    if (!debugLayer &&
        FAILED(D3D11CreateDevice(adapter, type, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("Panel: D3D11CreateDevice failed\n");
        if (adapter) adapter->Release();
        return 1;
    }
    if (adapter) adapter->Release();
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);
    printf("Panel: D3D11 device, debug layer %s\n", debugLayer ? "on" : "off");

    // The "game's" state: a render target, a viewport and a pixel shader resource that must survive the panel.
    D3D11_TEXTURE2D_DESC gd{};
    gd.Width = 64;
    gd.Height = 64;
    gd.MipLevels = 1;
    gd.ArraySize = 1;
    gd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    gd.SampleDesc.Count = 1;
    gd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D *gameRt = nullptr, *gameTex = nullptr;
    ID3D11RenderTargetView* gameRtv = nullptr;
    ID3D11ShaderResourceView* gameSrv = nullptr;
    if (FAILED(dev->CreateTexture2D(&gd, nullptr, &gameRt)) || FAILED(dev->CreateTexture2D(&gd, nullptr, &gameTex)) ||
        FAILED(dev->CreateRenderTargetView(gameRt, nullptr, &gameRtv)) ||
        FAILED(dev->CreateShaderResourceView(gameTex, nullptr, &gameSrv))) {
        printf("Panel: game state textures failed\n");
        return 1;
    }
    const D3D11_VIEWPORT gameVp{3, 5, 40, 30, 0, 1};
    ctx->OMSetRenderTargets(1, &gameRtv, nullptr);
    ctx->RSSetViewports(1, &gameVp);
    ctx->PSSetShaderResources(0, 1, &gameSrv);

    int fails = 0;
    {
        qvfr::panelui::Ui ui;
        if (!ui.Init(dev, kW, kH)) {
            printf("Panel: Init failed: %s\n", ui.why().c_str());
            return 1;
        }
        using R = qvfr::panelui::Row;
        std::vector<R> rows;
        auto add = [&](R::Kind k, const char* label, const char* value, float fill, bool on, const char* hint) {
            R r;
            r.kind = k;
            r.label = label;
            r.value = value;
            r.fill = fill;
            r.on = on;
            r.hint = hint;
            rows.push_back(r);
        };
        add(R::Head, "DLSS 5 Neural Rendering", "", -1, false, "");
        add(R::Toggle, "Neural Rendering", "ON", -1, true, "A/B switch, the same as right Ctrl + F8");
        add(R::Value, "Intensity", "1.00", 1.0f, false, "How much of the model's change reaches the image");
        add(R::Value, "Style", "2  Cinematic", -1, false, "The model's look; Cinematic: softer, photographic light");
        add(R::Value, "Preset", "Default", -1, false, "The model's variant; #1 costs about twice the GPU time");
        add(R::Value, "Strength  *", "1.5x", 0.5f, false,
            "Multiplies the model's change; above 1 stronger than the model");
        add(R::Value, "Local tone", "1.00", 0.5f, false, "Light and shadow in small areas (local contrast)");
        add(R::Value, "Local structure", "1.00", 0.5f, false, "Fine detail and texture of surfaces");
        add(R::Toggle, "Auto mask", "ON", -1, true, "Leaves flat text and displays as they are");
        add(R::Head, "DLSS 5 Foveation and performance", "", -1, false, "");
        add(R::Toggle, "NR on the outer views", "OFF", -1, false,
            "The periphery too; hides the tone step at the focus edge, +0.5 ms");
        add(R::Value, "Outer resolution", "1/2", -1, false,
            "Lower is cheaper; only the model's change is enlarged back");
        rows.back().dim = true;
        add(R::Value, "Outer area", "100%", 1.0f, false,
            "Only this centre part of the periphery goes through the model");
        rows.back().dim = true;
        add(R::Value, "Outer evaluation", "With the focus", -1, false,
            "Separate: the periphery does not change the focus, +0.9 ms");
        rows.back().dim = true;
        add(R::Value, "Focus edge blend", "0.20", 0.2f, false,
            "Width of the blend between the focus and the periphery");
        add(R::Head, "Settings", "", -1, false, "");
        add(R::Button, "Save to qvfr.ini", "Saved 3", -1, false,
            "Writes the options marked * to qvfr.ini");
        add(R::Button, "Back to the ini", "Enter", -1, false, "Undoes the changes not saved yet");
        std::vector<qvfr::panelui::Stat> stats = {{"90 fps", "frame rate, 11.1 ms"},
                                                  {"6.42 ms", "CPU frame (render thread)"},
                                                  {"11.09 ms", "GPU frame"},
                                                  {"3.95 ms", "NR on the GPU"}};
        const std::vector<qvfr::panelui::Stat> parts = {{"3.31 ms", "Model (D3D12)"}, {"0.34 ms", "Queue switches"},
                                                        {"0.12 ms", "Guides"},        {"0.07 ms", "Pack"},
                                                        {"0.09 ms", "Unpack"},        {"2.9 MP", "Atlas"}};
        const char* const head = "Measured, last second";
        const char* const partsHead = "NR cost parts";
        const char* const note = "";

        ui.Set(rows, 1, "DLSS 5 tuning", kFooter, stats, head, parts, partsHead, note);
        std::vector<uint32_t> a, b;
        if (!ui.Render(ctx) || !ReadBack(dev, ctx, ui.Texture(), a)) {
            printf("Panel: first render failed\n");
            return 1;
        }
        // Background (dark), the accent (red) and text (light) must all be there.
        size_t dark = 0, red = 0, light = 0;
        for (uint32_t p : a) {
            const int r = p & 0xFF, g = (p >> 8) & 0xFF, bl = (p >> 16) & 0xFF;
            if (r < 40 && g < 40 && bl < 40) ++dark;
            if (r > 180 && g < 70 && bl < 70) ++red;
            if (r > 200 && g > 200 && bl > 200) ++light;
        }
        printf("Panel: %ux%u picture: dark %zu, red %zu, light %zu pixels\n", kW, kH, dark, red, light);
        if (dark < a.size() / 2 || red < 500 || light < 500) ++fails;
        if (const char* bmp = getenv("QVFR_PANEL_BMP")) {
            SaveBmp(bmp, a, kW, kH);
            printf("Panel: picture saved to %s\n", bmp);
        }
        // No change: no new picture. A new measured value: a new picture. Another selection: a different picture.
        ui.Set(rows, 1, "DLSS 5 tuning", kFooter, stats, head, parts, partsHead, note);
        const bool again = ui.Render(ctx);
        stats[0].value = "45 fps";
        ui.Set(rows, 1, "DLSS 5 tuning", kFooter, stats, head, parts, partsHead, note);
        const bool statDrawn = ui.Render(ctx);
        stats[0].value = "90 fps";
        ui.Set(rows, 3, "DLSS 5 tuning", kFooter, stats, head, parts, partsHead, note);
        if (!ui.Render(ctx) || !ReadBack(dev, ctx, ui.Texture(), b)) {
            printf("Panel: second render failed\n");
            return 1;
        }
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        printf("Panel: unchanged rows drawn again: %s; a new measured value drawn: %s; another selection changed %zu "
               "pixels\n",
               again ? "yes" : "no", statDrawn ? "yes" : "no", diff);
        if (again || !statDrawn || diff < 1000) ++fails;
        // The game's state after drawing.
        ID3D11RenderTargetView* rtv = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        D3D11_VIEWPORT vp{};
        UINT nvp = 1;
        ctx->OMGetRenderTargets(1, &rtv, nullptr);
        ctx->RSGetViewports(&nvp, &vp);
        ctx->PSGetShaderResources(0, 1, &srv);
        const bool same = rtv == gameRtv && srv == gameSrv && nvp == 1 && vp.TopLeftX == gameVp.TopLeftX &&
                          vp.Width == gameVp.Width && vp.Height == gameVp.Height;
        printf("Panel: the game's render target, viewport and shader resource after drawing: %s\n",
               same ? "unchanged" : "CHANGED");
        if (!same) ++fails;
        if (rtv) rtv->Release();
        if (srv) srv->Release();
    }  // the panel is released here, while the device lives

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
        printf("Panel: D3D11 debug layer errors: %llu\n", (unsigned long long)errors);
        if (errors) ++fails;
        info->Release();
    }
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    ctx->PSSetShaderResources(0, 0, nullptr);
    gameSrv->Release();
    gameRtv->Release();
    gameTex->Release();
    gameRt->Release();
    ctx->Release();
    dev->Release();
    printf("Panel: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

#include "fps_hud.h"
#include <windows.h>
#include <cstdio>
#include <mutex>
#include "common.h"
#include "diag.h"

namespace qvfr::hud {
namespace {
std::mutex g_settingsMutex;
Settings g_configured, g_live;
bool g_keyDown = false;

bool Scene(const XrFrameEndInfo* info) {
    if (!info || !info->layers) return false;
    for (uint32_t i = 0; i < info->layerCount; ++i)
        if (info->layers[i] && info->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) return true;
    return false;
}

// GDI renders only into a private memory bitmap. Never captures a screen or changes a game render target.
class Canvas {
public:
    HDC dc = nullptr;
    HBITMAP bitmap = nullptr;
    HGDIOBJ previous = nullptr;
    HFONT fontLarge = nullptr, fontSmall = nullptr;
    uint32_t* pixels = nullptr;
    Canvas() {
        dc = CreateCompatibleDC(nullptr);
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = kWidth;
        bi.bmiHeader.biHeight = -(LONG)kHeight;
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        bitmap = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, (void**)&pixels, nullptr, 0);
        if (dc && bitmap) previous = SelectObject(dc, bitmap);
        fontLarge = CreateFontW(-70, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Consolas");
        fontSmall = CreateFontW(-25, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Consolas");
    }
    ~Canvas() {
        if (previous) SelectObject(dc, previous);
        if (bitmap) DeleteObject(bitmap);
        if (fontLarge) DeleteObject(fontLarge);
        if (fontSmall) DeleteObject(fontSmall);
        if (dc) DeleteDC(dc);
    }
    bool Ok() const { return dc && bitmap && pixels && fontLarge && fontSmall; }
    void Text(const wchar_t* text, int x, int y, HFONT font, COLORREF color) {
        auto old = SelectObject(dc, font);
        SetTextColor(dc, color);
        TextOutW(dc, x, y, text, (int)wcslen(text));
        SelectObject(dc, old);
    }
};
}  // namespace

void Configure(const Settings& value) {
    const Settings s = Sanitize(value);
    std::lock_guard lock(g_settingsMutex);
    const bool wasEnabled = g_live.enabled;
    g_live = s;
    // A shortcut toggle survives unrelated slider changes and repeated publication of the same INI.
    if (s.enabled == g_configured.enabled) g_live.enabled = wasEnabled;
    g_configured = s;
}

void PollHotkey() {
    std::lock_guard lock(g_settingsMutex);
    const bool down = g_live.hotkey && (GetAsyncKeyState(VK_RCONTROL) & 0x8000) && (GetAsyncKeyState(VK_INSERT) & 0x8000);
    if (down && !g_keyDown) {
        g_live.enabled = !g_live.enabled;
        Log("FPS HUD: %s (right Ctrl + Insert)", g_live.enabled ? "on" : "off");
    }
    g_keyDown = down;
}

Settings Current() { std::lock_guard lock(g_settingsMutex); return g_live; }

std::vector<uint32_t> Render(const Reading& value, const Settings& settings) {
    // Reuse the memory DC and fonts on the render thread; only a tiny bitmap is uploaded, at most four times/s.
    thread_local Canvas c;
    if (!c.Ok()) return {};
    std::fill_n(c.pixels, kWidth * kHeight, 0x0010151c);
    SetBkMode(c.dc, TRANSPARENT);
    COLORREF color = RGB(240, 244, 248);
    if (value.valid && value.runtimeHz > 0) {
        const double ratio = value.fps / value.runtimeHz;
        color = ratio >= 0.95 ? RGB(102, 235, 153) : ratio >= 0.75 ? RGB(255, 205, 100) : RGB(255, 115, 110);
    }
    wchar_t text[64];
    if (value.valid) swprintf_s(text, L"%.1f FPS", value.fps);
    else wcscpy_s(text, L"--.- FPS");
    c.Text(text, 22, 12, c.fontLarge, color);
    if (settings.details) {
        if (value.valid) swprintf_s(text, L"FRAME %5.1f ms", value.frameMs);
        else wcscpy_s(text, L"FRAME   --.- ms");
        c.Text(text, 26, 99, c.fontSmall, RGB(230, 233, 238));
        if (value.runtimeHz > 0) swprintf_s(text, L"XR %5.1f Hz   |   APP FPS", value.runtimeHz);
        else wcscpy_s(text, L"XR   --.- Hz   |   APP FPS");
        c.Text(text, 26, 136, c.fontSmall, RGB(160, 173, 190));
    }
    GdiFlush();
    std::vector<uint32_t> rgba(kWidth * kHeight);
    for (size_t i = 0; i < rgba.size(); ++i) {
        const uint32_t p = c.pixels[i];  // GDI BGRA -> RGBA, straight alpha (the layer declares unpremultiplied)
        rgba[i] = 0xeb000000u | ((p & 0xff) << 16) | (p & 0xff00) | ((p >> 16) & 0xff);
    }
    return rgba;
}

void Overlay::Start(XrSession session, XrSpace space, ID3D11Device* device, uint32_t maxLayers, const Dispatch& api) {
    Stop();
    api_ = api;
    session_ = session;
    space_ = space;
    device_ = device;
    maxLayers_ = maxLayers;
    if (device_) { device_->AddRef(); device_->GetImmediateContext(&context_); }
}

void Overlay::Stop() {
    if (swapchain_ && api_.destroy) api_.destroy(swapchain_);
    swapchain_ = XR_NULL_HANDLE;
    images_.clear();
    if (context_) context_->Release();
    if (device_) device_->Release();
    context_ = nullptr;
    device_ = nullptr;
    session_ = XR_NULL_HANDLE;
    space_ = XR_NULL_HANDLE;
    acquired_ = ready_ = failed_ = logged_ = false;
    paintedAt_ = -1;
    meter_.Reset();
}

void Overlay::Disable(const char* reason) {
    if (!failed_) {
        Log("FPS HUD disabled for this session: %s (the scene passes through)", reason);
        diag::Set("FPS HUD", std::string("unavailable: ") + reason);
    }
    failed_ = true;
}

bool Overlay::Create() {
    if (swapchain_) return true;
    uint32_t count = 0;
    if (XR_FAILED(api_.formats(session_, 0, &count, nullptr)) || !count || count > 1024) return false;
    std::vector<int64_t> formats(count);
    if (XR_FAILED(api_.formats(session_, count, &count, formats.data()))) return false;
    int64_t format = 0;
    for (auto candidate : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM})
        if (std::find(formats.begin(), formats.end(), (int64_t)candidate) != formats.end()) { format = candidate; break; }
    if (!format) return false;
    XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    ci.format = format;
    ci.sampleCount = ci.faceCount = ci.arraySize = ci.mipCount = 1;
    ci.width = kWidth;
    ci.height = kHeight;
    if (XR_FAILED(api_.create(session_, &ci, &swapchain_))) return false;
    if (XR_FAILED(api_.images(swapchain_, 0, &count, nullptr)) || !count || count > 64) return false;
    images_.assign(count, {XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR});
    if (XR_FAILED(api_.images(swapchain_, count, &count, (XrSwapchainImageBaseHeader*)images_.data()))) return false;
    for (const auto& image : images_) if (!image.texture) return false;
    Log("FPS HUD: %u x %u D3D11 OpenXR swapchain, format %lld, %u images; head locked, app FPS", kWidth, kHeight, format, count);
    diag::Set("FPS HUD", "available: app FPS, frame interval and runtime Hz; right Ctrl + Insert");
    return true;
}

bool Overlay::Paint(double now, const Settings& settings, const Reading& value) {
    if (!acquired_) {
        XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        if (XR_FAILED(api_.acquire(swapchain_, &ai, &acquiredIndex_))) { Disable("acquire image failed"); return false; }
        acquired_ = true;
    }
    // Never block the game for the counter. A timed-out acquire remains pending and is retried next frame.
    XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wi.timeout = 0;
    const XrResult waited = api_.wait(swapchain_, &wi);
    if (waited == XR_TIMEOUT_EXPIRED) return ready_;
    if (waited != XR_SUCCESS || acquiredIndex_ >= images_.size()) { Disable("wait image failed"); return false; }
    const auto pixels = Render(value, settings);
    if (!pixels.empty()) {
        context_->UpdateSubresource(images_[acquiredIndex_].texture, 0, nullptr, pixels.data(), kWidth * 4, 0);
        context_->Flush();  // submit the tiny upload before handing the image back; no CPU wait for the GPU
    }
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    const XrResult released = api_.release(swapchain_, &ri);
    acquired_ = false;
    if (released != XR_SUCCESS || pixels.empty()) { Disable("render/release image failed"); return false; }
    ready_ = true;
    paintedAt_ = now;
    paintedSettings_ = settings;
    return true;
}

bool Overlay::Prepare(const XrFrameEndInfo* input, XrFrameEndInfo& output, XrCompositionLayerQuad& quad,
                      std::vector<const XrCompositionLayerBaseHeader*>& layers, double now, double periodMs, bool shouldRender) {
    const Settings settings = Current();
    if (!settings.enabled || !shouldRender || !Scene(input) || failed_) return false;
    if (!device_ || !context_ || !space_ || !api_.Complete()) { Disable("requires OpenXR with D3D11 and VIEW space"); return false; }
    if (input->layerCount >= maxLayers_) return false;  // never exceed the runtime's layer limit
    if (!Create()) { Disable("no compatible overlay swapchain"); return false; }
    if (acquired_ || !ready_ || now - paintedAt_ >= 250 || settings != paintedSettings_)
        if (!Paint(now, settings, meter_.Read(now, periodMs))) return false;
    if (!ready_ || failed_) return false;
    quad = {XR_TYPE_COMPOSITION_LAYER_QUAD};
    quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
    quad.space = space_;
    quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    quad.subImage.swapchain = swapchain_;
    quad.subImage.imageRect.extent = {(int32_t)kWidth, settings.details ? (int32_t)kHeight : 100};
    quad.pose.orientation.w = 1;
    constexpr double rad = 3.14159265358979323846 / 180;
    quad.pose.position = {(float)std::tan(settings.xDegrees * rad), (float)std::tan(settings.yDegrees * rad), -1.0f};
    quad.size.width = 0.36f * settings.scale;
    quad.size.height = quad.size.width * quad.subImage.imageRect.extent.height / kWidth;
    layers.assign(input->layers, input->layers + input->layerCount);
    layers.push_back((const XrCompositionLayerBaseHeader*)&quad);
    output = *input;  // preserve displayTime, environmentBlendMode and the entire next chain
    output.layerCount = (uint32_t)layers.size();
    output.layers = layers.data();
    if (!logged_) { Log("FPS HUD: first overlay submitted (scene layers %u)", input->layerCount); logged_ = true; }
    return true;
}

void Overlay::Ended(const XrFrameEndInfo* original, XrResult result, double now, bool added, bool shouldRender) {
    if (added && XR_FAILED(result)) Disable("runtime rejected frame with HUD; disabled on following frames");
    if (XR_SUCCEEDED(result) && shouldRender && Scene(original)) meter_.Submit(now);
}

}  // namespace qvfr::hud

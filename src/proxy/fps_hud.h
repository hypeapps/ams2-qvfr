// Native OpenXR composition-layer HUD. Independent of the game's eye textures, TAA/DLAA and quad view sizes.
#pragma once
#include <d3d11.h>
#include <openxr/openxr.h>
#ifndef XR_USE_GRAPHICS_API_D3D11
#define XR_USE_GRAPHICS_API_D3D11
#endif
#include <openxr/openxr_platform.h>
#include <vector>
#include "fps_hud_data.h"

namespace qvfr::hud {

// The frame thread publishes live settings; xrEndFrame reads a consistent snapshot.
void Configure(const Settings& s);
void PollHotkey();  // right Ctrl + Insert, edge triggered
Settings Current();

struct Dispatch {
    PFN_xrEnumerateSwapchainFormats formats = nullptr;
    PFN_xrCreateSwapchain create = nullptr;
    PFN_xrDestroySwapchain destroy = nullptr;
    PFN_xrEnumerateSwapchainImages images = nullptr;
    PFN_xrAcquireSwapchainImage acquire = nullptr;
    PFN_xrWaitSwapchainImage wait = nullptr;
    PFN_xrReleaseSwapchainImage release = nullptr;
    bool Complete() const { return formats && create && destroy && images && acquire && wait && release; }
};

inline constexpr uint32_t kWidth = 512, kHeight = 176;
// RGBA8 straight alpha; used by the overlay and by its image/texture smoke test.
std::vector<uint32_t> Render(const Reading& value, const Settings& settings);

// All calls serialized by the existing OpenXR layer mutex. It owns the session and VIEW space;
// this class owns only its small swapchain and its device/context references.
class Overlay {
    Dispatch api_;
    XrSession session_ = XR_NULL_HANDLE;
    XrSpace space_ = XR_NULL_HANDLE;
    XrSwapchain swapchain_ = XR_NULL_HANDLE;
    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    std::vector<XrSwapchainImageD3D11KHR> images_;
    uint32_t maxLayers_ = 0, acquiredIndex_ = 0;
    bool acquired_ = false, ready_ = false, failed_ = false, logged_ = false;
    double paintedAt_ = -1;
    Settings paintedSettings_;
    Meter meter_;
    bool Create();
    bool Paint(double now, const Settings& settings, const Reading& value);
public:
    void Start(XrSession session, XrSpace viewSpace, ID3D11Device* device, uint32_t maxLayers, const Dispatch& api);
    void Stop();
    void Disable(const char* reason);
    bool Prepare(const XrFrameEndInfo* input, XrFrameEndInfo& output,
                 XrCompositionLayerQuad& quad, std::vector<const XrCompositionLayerBaseHeader*>& layers,
                 double now, double periodMs, bool shouldRender);
    void Ended(const XrFrameEndInfo* original, XrResult result, double now, bool added, bool shouldRender);
    Reading Read(double now, double periodMs) const { return meter_.Read(now, periodMs); }
};

}  // namespace qvfr::hud

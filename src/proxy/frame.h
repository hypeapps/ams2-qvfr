// Frame logic: SteamVR timings to CSV, benchmark (hotkey, constant head pose), Submit analysis, gaze reading.
// Called from the IVRCompositor and IVRSystem wrappers (wrappers.cpp).
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::frame {

void NoteSystem(vr::IVRSystem* real);  // after VR_Init: the backend's IVRSystem (eye frusta for the OpenXR gaze)
void OnWaitGetPosesBegin();
void OnWaitGetPosesEnd(vr::IVRCompositor* real, vr::TrackedDevicePose_t* render, uint32_t nRender,
                       vr::TrackedDevicePose_t* game, uint32_t nGame, vr::EVRCompositorError err, double blockMs);
// Benchmark with a constant pose: also replaces the HMD pose (index 0) in these calls.
void FilterLastPoses(vr::TrackedDevicePose_t* render, uint32_t nRender, vr::TrackedDevicePose_t* game, uint32_t nGame);
void FilterDevicePoses(vr::ETrackingUniverseOrigin origin, vr::TrackedDevicePose_t* poses, uint32_t n);
void OnSetTrackingSpace(vr::ETrackingUniverseOrigin origin);

void OnSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds, vr::EVRSubmitFlags flags,
              vr::EVRCompositorError err, double durMs);

void NoteRecommendedSize(uint32_t realW, uint32_t realH, uint32_t returnedW, uint32_t returnedH);
std::string TextureInfo(void* handle, int type);  // D3D11 texture description for the log (size, format, samples)
void OnShutdown();

}  // namespace qvfr::frame

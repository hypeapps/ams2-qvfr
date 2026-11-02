// OpenXR API layer inside the proxy DLL (M8.1 spike, path=openxr). OpenComposite's built-in OpenXR loader finds it
// through the process environment that runtime::Prepare sets (XR_API_LAYER_PATH, XR_ENABLE_API_LAYERS), so it sits
// between the implicit layers and the runtime. It enables XR_EXT_eye_gaze_interaction, adds its own action set with
// the gaze pose to OpenComposite's xrAttachSessionActionSets and xrSyncActions, and after every xrWaitFrame locates the
// gaze in VIEW space at the predicted display time. It also logs how OpenComposite drives the session.
// The loader entry point xrNegotiateLoaderApiLayerInterface is exported from xr_layer.cpp.
#pragma once

#include <cstdint>

namespace qvfr::xrlayer {

inline constexpr char kLayerName[] = "XR_APILAYER_AMS2QVFR_eye_gaze";

struct Gaze {
    bool ok = false;      // the action is active and the orientation is valid
    float dir[3] = {};    // unit gaze direction relative to the head (VIEW space: x right, y up, -z forward)
    uint64_t seq = 0;     // number of the xrWaitFrame that located it
};

bool Loaded();            // a loader negotiated with the layer in this process
bool GazeAttached();      // the gaze action set is attached to the current session
bool ReadGaze(Gaze& out); // the newest located gaze; false when the layer has none (not loaded, not attached)

}  // namespace qvfr::xrlayer

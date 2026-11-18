// Which implicit OpenXR API layers have to sit below the mod's layer on the OpenXR path.
//
// The OpenXR loader puts implicit layers closest to the application and the layers from XR_ENABLE_API_LAYERS (ours)
// after them, so a layer that provides the eye gaze as an implicit layer (VIVE OpenXR Facial Tracking, OpenXR-Eye-Trackers)
// never sees the mod's gaze calls: they go from our layer straight to the runtime. For the game's process only, such a
// layer can be turned off as an implicit layer (its disable_environment variable) and enabled again as an explicit layer
// listed after ours. MakePlan decides which layers qualify; runtime.cpp applies the result to the environment.
//
// The checks repeat what the loader requires of a manifest (manifest_file.cpp, CreateIfValid): a name in
// XR_ENABLE_API_LAYERS that the loader cannot find makes xrCreateInstance fail, so a layer is moved only when the loader
// is certain to accept its manifest, and never when the folder would give it a second 64-bit copy of the same layer.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace qvfr::xrorder {

struct Layer {
    std::string name;        // the layer's name, for XR_ENABLE_API_LAYERS
    std::string disableEnv;  // its disable_environment variable
    std::string reason;      // why it is moved (for the log)
    std::wstring manifest;   // its manifest file
    std::wstring folder;     // to add to XR_API_LAYER_PATH; empty = the loader already searches a folder with it
    bool declaresGaze = false;  // the manifest lists XR_EXT_eye_gaze_interaction
};

struct Input {
    std::vector<std::wstring> enabledManifests;  // enabled implicit layers, in the loader's order (HKLM, then HKCU)
    std::wstring setting;                        // [runtime] openxr_gaze_layers
    std::wstring layerPath;                      // XR_API_LAYER_PATH set outside the mod
    std::string ownLayer;                        // the mod's layer name, never moved
    std::function<bool(const std::string&)> envSet;  // is this environment variable set?
};

struct Plan {
    std::vector<Layer> move;         // in the loader's order, which the chain keeps
    std::vector<std::string> notes;  // layers left alone and why (for the log)
};

Plan MakePlan(const Input& in);

}  // namespace qvfr::xrorder

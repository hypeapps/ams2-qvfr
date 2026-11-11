// OpenVR 0.9.12 interface wrappers that the proxy hands to the game instead of the real ones.
#pragma once

#include <openvr.h>  // 0.9.12

namespace qvfr::wrap {

// Return the wrapper for a given real object (one per object), or nullptr for nullptr.
vr::IVRSystem* System(vr::IVRSystem* real);
vr::IVRCompositor* Compositor(vr::IVRCompositor* real);
vr::IVROverlay* Overlay(vr::IVROverlay* real);

// Wrapper for an interface obtained through VR_GetGenericInterface, if it is one of the supported versions.
void* Generic(const char* version, void* real);

// The backend's own overlay interface (not the wrapper), or nullptr before VR_Init.
vr::IVROverlay* RealOverlay();

}  // namespace qvfr::wrap

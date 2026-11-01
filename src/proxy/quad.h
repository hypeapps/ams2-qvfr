// Quad mode ([quad] mode=quad): two more eye groups in the engine's VR task graph render the focus views.
// The builder loop runs for views 2, 3 and 4 (4-byte patch); views 3 and 4 do not fit the render context arrays,
// so their tasks get view 2, and the focus groups swap the view 2 fields of the context for their own camera data
// while they run. M4.3: the game renders at the focus size with the full projection (outer), and the focus groups get
// the focus window of their eye (projection in the eye camera and in the VR device eye data). M4.4: at Submit the proxy
// composes each eye from outer and focus into a grid-size image (compose.h); shots compare the focus views with their
// window in the outer views and in the composed eyes.
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::quad {

// All functions guard themselves (SEH); call them outside Guarded.
void Install();       // VR_Init: engine hooks and the builder patch (only on the supported exe)
// The first call of a VR session places the focus window; with the hooks in place the game gets the focus size.
void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h);
bool Engaged();       // the game got the focus size this session: the wrappers must stay in place
void OnFrameStart();  // WaitGetPoses entry (render thread): frame bookkeeping, shot hotkey, shot read-back
void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex);  // the first Submit gives the D3D11 device
// With the focus groups running: a Texture_t (kept in `storage`) pointing to the composed eye, or without the
// composition to outer at the focus size (right eye: this frame's outer R copy); otherwise `tex` unchanged.
const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                   vr::Texture_t& storage);
void LogStats(bool totals);
std::string Describe();   // one line for summary.txt
const char* RunSuffix();  // "_quad" while the focus groups run, else ""
void Shutdown();          // VR_Shutdown: hooks off, builder patch removed

}  // namespace qvfr::quad

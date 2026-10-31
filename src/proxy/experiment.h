// M2.5 experiment without engine hooks: the proxy changes the eye size and projection the game gets,
// and in focus mode places the game image back into a full-FOV texture before Submit.
// The game reads size and projection once per VR session, so the mode is fixed per session ([experiment] in qvfr.ini).
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::experiment {

// All Adjust* functions take the real (unwrapped) IVRSystem; the first call of a session computes the focus window.
void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h);
void AdjustProjectionRaw(vr::IVRSystem* real, vr::EVREye eye, float* l, float* r, float* t, float* b);
void AdjustProjectionMatrix(vr::IVRSystem* real, vr::EVREye eye, vr::HmdMatrix44_t& m);

// Focus mode with pad=1: copies the game texture into the eye's full-FOV texture and returns a Texture_t
// (kept in `storage`) that points to it. Otherwise returns `tex` unchanged.
const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, vr::Texture_t& storage);

std::string Describe();  // one line for summary.txt and the log
const char* RunSuffix();  // "" (off or inactive), "_focus" or "_outer": benchmark folder suffix
void Shutdown();          // VR_Shutdown: releases textures, the next session computes everything again

}  // namespace qvfr::experiment

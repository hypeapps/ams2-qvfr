// M2.5 experiment without engine hooks: the proxy changes the eye size and projection the game gets,
// and in focus mode places the game image back into a full-FOV texture before Submit.
// The game reads size and projection once per VR session, so the mode is fixed per session ([experiment] in qvfr.ini).
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::experiment {

// Every function guards itself (SEH) and does not depend on the diagnostics kill-switch, so the size, projection
// and Submit stay consistent for the whole session. Call them outside Guarded.

// All Adjust* functions take the real (unwrapped) IVRSystem; the first call of a session computes the focus window.
void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h);
void AdjustProjectionRaw(vr::IVRSystem* real, vr::EVREye eye, float* l, float* r, float* t, float* b);
void AdjustProjectionMatrix(vr::IVRSystem* real, vr::EVREye eye, vr::HmdMatrix44_t& m);

// Focus mode with pad=1: copies the game texture into the eye's full-FOV texture and returns a Texture_t
// (kept in `storage`) that points to it. Otherwise (including Submit with partial bounds) returns `tex` unchanged.
const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                   vr::Texture_t& storage);

bool Engaged();           // the game got a replaced size this session: the wrappers must stay in place
void MarkRunStart();      // benchmark start: Describe() counts padded Submits from here
std::string Describe();   // one line for summary.txt and the log
const char* RunSuffix();  // "" (off or inactive), "_focus", "_focus_nopad" or "_outer": benchmark folder suffix
void Shutdown();          // VR_Shutdown: releases textures, the next session computes everything again

}  // namespace qvfr::experiment

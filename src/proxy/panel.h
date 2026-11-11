// The tuning panel in the headset (dlaa-plan D7.P): the DLSS 5 (NR) options as in other DLSS 5 mods plus the mod's
// foveation options, changed live, and the measured frame rate, GPU frame and NR cost of the last second. [panel]
// modifier + key (right Ctrl + F2) opens and closes it; while it is open the arrow keys choose a row and change it,
// Enter switches toggles and presses buttons, Esc closes it, and the game does not get these keys (panel_input.h,
// its DirectInput keyboard). The picture
// (panel_ui.h) goes to an overlay of the backend's IVROverlay (SteamVR, or OpenComposite on path=openxr, which makes it
// a quad layer), placed in the seated space like the game's HUD, so the game's image and NR's cost stay as they are.
// Changes go to the NR knobs (nr.h) until "Save to qvfr.ini" (the changed options into the ini, comments kept) or "Back
// to the ini". Render thread only; a fault turns the panel off for the session (logged), nothing else.
#pragma once

#include <openvr.h>  // 0.9.12

namespace qvfr::panel {

// Every Submit, before the real one: the hotkey and keys, and with the panel open its picture and overlay (once per
// frame, at the right eye).
void OnSubmit(vr::EVREye eye);
void Shutdown();  // VR_Shutdown: the overlay and the picture go (before gpu::Release)

}  // namespace qvfr::panel

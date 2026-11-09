// Quad mode ([quad] mode=quad): two more eye groups in the engine's VR task graph render the focus views.
// The builder loop runs for views 2, 3 and 4 (4-byte patch); views 3 and 4 do not fit the render context arrays,
// so their tasks get view 2, and the focus groups swap the view 2 fields of the context for their own camera data
// while they run. M4.3: the game renders at the focus size with the full projection (outer), and the focus groups get
// the focus window of their eye (projection in the eye camera and in the VR device eye data). M4.4: at Submit the proxy
// composes each eye from outer and focus into a grid-size image (compose.h); shots compare the focus views with their
// window in the outer views and in the composed eyes. M5: the focus window follows the gaze every frame (DFR), with
// the fixed center (FFR) as the fallback.
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::quad {

// All functions guard themselves (SEH); call them outside Guarded.
void Install();       // VR_Init: engine hooks and the builder patch (only on the supported exe)
// The first call of a VR session places the focus window; with the hooks in place the game gets the focus size.
void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h);
bool Engaged();       // the game got the focus size this session: the wrappers must stay in place
// WaitGetPoses entry (render thread): frame bookkeeping, hotkeys (shot, overlay, gaze freeze, quad <-> outer only), shot
// read-back. M4.5: in outer-only mode (hotkey or watchdog) the focus groups' tasks are not executed and SteamVR gets outer.
void OnFrameStart();
// M5 (DFR): where the frame's focus window comes from and the left eye's window position in the 1:1 grid.
enum WindowSource { SrcNone = -1, SrcLive, SrcHeld, SrcFixed, SrcFrozen };
struct FrameWindow {
    int src = SrcNone;  // SrcNone: no focus window (quad mode off or not set up)
    uint32_t x0 = 0, y0 = 0;
};
const char* SourceName(int src);  // "live", "held", "fixed", "frozen", "" for SrcNone
// WaitGetPoses exit (render thread), every frame: the gaze (NDC per eye of the full projection, x right, y up; kept
// while frozen) places this frame's focus window, used by the focus groups and the composition. `benchmark`: a
// benchmark counts down or records, so the window stays at the fixed center unless the gaze is frozen.
FrameWindow SetGaze(bool ok, const float ndcL[2], const float ndcR[2], bool benchmark);
void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex);  // the first Submit gives the D3D11 device
// With the focus groups running: a Texture_t (kept in `storage`) pointing to the composed eye, or without the
// composition to outer at the focus size (right eye: this frame's outer R copy); otherwise `tex` unchanged.
const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                   vr::Texture_t& storage);
void LogStats(bool totals);
// The eye groups' CPU per full frame (ms; 0 = none) of the latest stats interval (a benchmark's window after its end).
void LastWindowCpu(double groupMs[4]);
// Benchmark sequence ([benchmark] sequence, frame.cpp): the live switches by name, the same as their hotkeys or live
// ini keys (mvr, focus_cull, sun_effects, taa_views, taa_motion_fix, outer_only, task_profile, mvr_early, mvr_mode,
// mvr_forward, mvr_forward_depth, mvr_ms_color_skip, mvr_livetrack_skip, mvr_velocity, mvr_velocity_skip,
// mvr_gbuffer_ps, mvr_direct_extract, mvr_native_targets). False: no such switch, or it is unavailable (`why`).
bool GetSwitch(const std::string& name, int& value, std::string* why = nullptr);
bool SetSwitch(const std::string& name, int value, std::string* why = nullptr);
std::string Describe();   // one line for summary.txt
const char* RunSuffix();  // "_quad" while the focus groups run, else ""
void Shutdown();          // VR_Shutdown: hooks off, builder patch removed

}  // namespace qvfr::quad

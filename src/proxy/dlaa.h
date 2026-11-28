// DLAA prototype ([dlaa] mode=dlaa): NVIDIA DLSS Super Resolution at render size replaces the game's TAA resolve.
// The engine computes everything DLSS needs for its own TAA (CTAAControl, docs/ams2-renderer.md): HDR color before
// tonemapping, linear depth, per-object motion vectors (uv now - uv before) and an 8-phase Halton jitter in the
// projection. CTAAControl's resolve only records into the engine's command buffers, so a DrawIndexed hook recognizes the
// replayed TAA draw by what is bound, runs DLSS instead of it and copies its output into the target; the rest of the
// game's pipeline is unchanged. The resolve hook gives the render context (jitter index) and skips the locks pass.
// Needs TAA on in the game, the supported exe and a proxy built with the DLSS SDK (QVFR_HAVE_DLSS); anything missing
// or failing leaves the game's TAA.
// Quad views (dlaa-plan D4): one DLSS feature per eye group (outer L/R, focus L/R), each at its own size; the focus
// views get their own TAA targets and motion vectors with the window's motion from taa_views (D2). The quad module
// owns the D3D11 DrawIndexed hook then (sun_fx), and DLAA's check runs from it.
#pragma once

#include <cstdint>
#include <string>

namespace qvfr::dlaa {

// All functions guard themselves (SEH); call them outside Guarded.
void Install();       // VR_Init: engine hooks (only on the supported exe with [dlaa] mode=dlaa)
void OnFrameStart();  // WaitGetPoses entry (render thread): hotkey, live settings, frame counter
// Quad eye groups (render thread): 0 outer L, 1 outer R, 2 focus L, 3 focus R; each gets its own DLSS history.
void BeginGroup(int group);
void EndGroup();
// Quad frame start: true in a shot frame (right Ctrl + F10); each view's TAA resolve is then measured once (log).
void SetShotFrame(bool shot);
// DLAA's frame counter (advanced in OnFrameStart; mvr_motion keys outer's copies by it).
uint64_t Frame();
// Any thread (MVR gate B8): the projection offset the render tasks set in the engine's current frame (SetJitter hook,
// size-independent: the jitter is offset / render size in NDC). False: not set in this frame or no hook.
bool FrameJitter(float out[2]);
// Logs the next `calls` SetJitter calls (any thread): context, offset, frame counter, caller (MVR jitter diagnosis).
void TraceJitter(int calls);
// DLSS SR (sr.h, [dlaa] mode=dlss in stereo): DLSS can run now, so the scene may render at the render size.
bool SrWanted();
// Render thread: DLSS SR's render sizes changed while the game runs: the histories start over, the state names the scales
// (focus's, or the eye's in stereo, and outer's).
void OnSrScalesChanged(float focus, float outer);
// Benchmark sequence switches (quad.cpp): the DLSS preset of all views (0), of outer's (1), of focus's (2) over the ini's,
// as NVSDK_NGX_DLSS_Hint_Render_Preset (10 J, 11 K, 12 L, 13 M); -1 = the ini's.
int GetPresetSwitch(int which);
void SetPresetSwitch(int which, int preset);
// Benchmark sequence switch dlss_game_taa: 1 = the game's TAA, 0 = DLSS (as the [dlaa] key hotkey); applied at the next
// frame start.
int GetGameTaaSwitch();
void SetGameTaaSwitch(int on);
// Benchmark sequence switch dlss_exposure: the exposure hotkey's step, 0 = the ini's, 1 auto, 2 0.005, 3 0.002,
// 4 0.001.
constexpr int kExposureSteps = 5;
int GetExposureSwitch();
void SetExposureSwitch(int step);
// Benchmark sequence switch dlss_jitter: the jitter sequence while DLSS runs (jitter_seq.h: 0 the game's, 1 full range
// with the game's 8 phases, 2 full range with NVIDIA's phase count); -1 = [dlaa] jitter. Latched per engine frame.
int GetJitterSwitch();
void SetJitterSwitch(int mode);
// DLSS replaces the game's TAA in this frame (setting, hotkey, no failure); its evaluation stores outer's motion
// vectors for focus (mvr_motion.h StoreOuter), so without it focus's velocity pass gets none from outer.
bool DlssNow();
void LogStats(bool totals);
std::string Describe();  // one line for summary.txt
void Shutdown();         // VR_Shutdown: DLSS features and NGX released

}  // namespace qvfr::dlaa

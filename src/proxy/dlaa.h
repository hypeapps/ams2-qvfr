// DLAA prototype ([dlaa] mode=dlaa): NVIDIA DLSS Super Resolution at render size replaces the game's TAA resolve.
// The engine computes everything DLSS needs for its own TAA (CTAAControl, docs/ams2-renderer.md): HDR color before
// tonemapping, linear depth, per-object motion vectors (uv now - uv before) and an 8-phase Halton jitter in the
// projection. CTAAControl's resolve only records into the engine's command buffers, so a DrawIndexed hook recognizes the
// replayed TAA draw by what is bound, runs DLSS instead of it and copies its output into the target; the rest of the
// game's pipeline is unchanged. The resolve hook gives the render context (jitter index) and skips the locks pass.
// Needs TAA on in the game, the supported exe and a proxy built with the DLSS SDK (QVFR_HAVE_DLSS); anything missing
// or failing leaves the game's TAA.
#pragma once

#include <string>

namespace qvfr::dlaa {

// All functions guard themselves (SEH); call them outside Guarded.
void Install();       // VR_Init: engine hooks (only on the supported exe with [dlaa] mode=dlaa)
void OnFrameStart();  // WaitGetPoses entry (render thread): hotkey, live settings, frame counter
void LogStats(bool totals);
std::string Describe();  // one line for summary.txt
void Shutdown();         // VR_Shutdown: DLSS features and NGX released

}  // namespace qvfr::dlaa

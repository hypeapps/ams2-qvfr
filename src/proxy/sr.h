// DLSS Super Resolution in stereo ([dlaa] mode=dlss, [quad] mode=off; docs/dlaa-plan.md S1 prototype). The game gets
// the eye size from SteamVR as usual (O). The scene phase of each eye group renders into clones of the engine's
// eye-size targets (clones.h, outer set) at render_scale x O; at the TAA resolve's draw DLSS scales the scene to O
// (dlaa.cpp), the textures the passes after TAA need are carried from their clones into the originals, and the clones go
// out, so the rest of the group (windscreen, particles, bloom, tonemapping, Submit) runs at O.
// This module drives the clones in stereo: the VR task graph's eye groups (CBeginEye, CEndEye; not the menus'), the
// D3D11 render target binds (bind fix and learning), the sizes and the frame start.
#pragma once

#include <cstdint>

struct ID3D11DeviceContext;
struct ID3D11Resource;

namespace qvfr::sr {

// All functions guard themselves (SEH).
void Install();                          // VR_Init: engine hooks (supported exe, [dlaa] mode=dlss, [quad] mode=off)
bool On();                               // installed and not failed
void NoteGameSize(uint32_t w, uint32_t h);  // GetRecommendedRenderTargetSize: the eye size the game renders at
// WaitGetPoses entry (render thread): clones' frame start; `wanted` = DLSS can run now (else the game renders at O).
void OnFrameStart(bool wanted);
// Shot ([quad] shot_modifier + shot_key, testers: right Ctrl + F10): in one frame, per eye group the DLSS input (scene
// color at the render size) and output (eye size) at the TAA draw, and the two Submit textures, as PNG files in
// x64\qvfr\shots\ (shots.h).
void ShotDlss(ID3D11DeviceContext* c, ID3D11Resource* in, ID3D11Resource* out);  // dlaa.cpp, right after DLSS SR
void OnSubmit(int eye, void* d3dTexture);  // IVRCompositor::Submit, before the real call (render thread)
void Shutdown();                         // VR_Shutdown

}  // namespace qvfr::sr

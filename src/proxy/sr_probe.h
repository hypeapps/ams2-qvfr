// DLSS SR in quad views: a probe of the eye groups' constant buffers (docs/dlaa-plan.md S2.1, test 1: car lights seen
// through other cars, the sun wrong).
//
// The passes after the TAA resolve are recorded on worker threads while the scene set's sizes are in the engine
// objects (S1 probe), so a constant they compute from the screen size may say the render size R on targets of the view
// size O. The probe answers which draws hold which size: on the hotkey ([debug] sr_probe_modifier + sr_probe_key) it
// copies the VS and PS constant buffers bound at every draw of the next frame's eye groups with DLSS SR's scene sets,
// and at the frame start after it looks in them for R, O and the game size G (and their 1/2, 1/4, 1/8: width, height,
// their reciprocals, half and double reciprocals). The log lists each place (draw caller, stage, slot, float index)
// with the size found, the phase and the number of draws; a render size after TAA, or a view or game size before it, is
// marked SUSPECT. Draw and DrawIndexed come through sun_fx's hooks, the instanced draws through sr_cb's (after its fix,
// so the probe sees the constants the draw gets).
#pragma once

#include <d3d11.h>

namespace qvfr::srprobe {

// Quad views with DLSS SR and a probe key: ready (the draw hooks are sun_fx's and sr_cb's).
void Install(ID3D11DeviceContext* imm);
// Any thread, lock-free: this frame is probed.
bool Armed();
// Render thread, before a draw of the immediate context (`draw`: the method's name, `caller`: the engine's return
// address). Guards itself.
void OnDraw(ID3D11DeviceContext* c, void* caller, const char* draw);
// Render thread, frame start: the report of the probed frame before, then the hotkey.
void OnFrameStart(ID3D11DeviceContext* imm);
void Shutdown();
// Smoke test: arm the probe for the draws from now on (no hotkey, no hooks); report them at once: the places found
// (`suspects`: the SUSPECT ones).
void TestArm(ID3D11DeviceContext* c);
size_t TestReport(ID3D11DeviceContext* c, size_t* suspects);

}  // namespace qvfr::srprobe

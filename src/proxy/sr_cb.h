// DLSS SR in quad views: the constants of the passes after the TAA resolve set to the view size (docs/dlaa-plan.md
// S2.1, probe run 3).
//
// The engine records the passes after the TAA resolve on worker threads while the scene set's render size R is in its
// objects, so their per-view constants hold R (cb12[116..119] the size, cb12[98..99] its reciprocal, cb11[88..89] in
// focus, cb10[1014..1015]) where the targets have the view size O. Soft particles and light glows read the scene depth
// at SV_Position / R and saw it 1.5 times too far out: car lights showed through the cars in front.
// After the TAA resolve no render-size value is right, so before each draw of that phase this module copies the windows
// bound to VS and PS constant buffer slots 10..13 (as the game binds them with *SetConstantBuffers1), replaces in them
// every pair of floats at an even index that is a render-size pair (width and height at 1, 1/2, 1/4, 1/8, rounded down
// or up, and 0.5, 1 or 2 over them) by the view-size pair (a compute shader on the GPU, shaders/sr.hlsl CSFixSizes),
// and binds the fixed copy in its place. A window is fixed once per eye group. [debug] sr_cb_fix (live) turns it off,
// sr_probe_modifier + sr_cb_fix_key toggles it. The instanced draw hooks live here too (they also feed the SR probe).
#pragma once

#include <d3d11.h>

#include <cstdint>

namespace qvfr::srcb {

// Quad views with DLSS SR, first frame with a context: the shader, ID3D11DeviceContext1 and the instanced draw hooks
// (hookDraws false: mvr_pass hooked them already and its detours call OnDraw and the probe).
void Install(ID3D11DeviceContext* imm, bool hookDraws = true);
// Render thread, before a draw of the immediate context (Draw and DrawIndexed from sun_fx's hooks, the instanced ones from
// this module's or mvr_pass's): in the phase after the TAA resolve of an SR group, the bound windows fixed. Guards itself.
void OnDraw(ID3D11DeviceContext* c);
// Render thread, frame start: the setting and the hotkey.
void OnFrameStart();
void LogStats(bool totals);
void Shutdown();
// Smoke test: the module on `c` without hooks; the windows fixed since then.
void TestInit(ID3D11DeviceContext* c);
uint64_t TestWindowsFixed();

}  // namespace qvfr::srcb

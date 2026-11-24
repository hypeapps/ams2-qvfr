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
// and binds the fixed copy in its place. Each copy and compute step interrupts the GPU's graphics work, so windows of
// the engine's dynamic constant buffer share blocks: a block is fixed once per frame and pair of sizes, each window binds
// at its distance from the block's start, and a Map of that buffer drops its blocks. [debug] sr_cb_fix (live) turns it off,
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
// From the Map hook, before the real call: blocks copied from `r` no longer hold its data.
void OnBufferWrite(ID3D11DeviceContext* c, ID3D11Resource* r);
// Render thread, frame start: the setting and the hotkey.
void OnFrameStart();
void LogStats(bool totals);
void Shutdown();
// The passes after TAA also hold the view size's own fractions (targets without a scene clone keep the view size while
// the scene set is in). Near half the view size some render-size pairs equal them (R = O / 2 is also O / 2 of the
// view), and the fix would rewrite those too. Ambiguous: some pair of the table for these sizes equals a view-size
// value (O / 1..64 rounded down or up per axis, and 0.5, 1 or 2 over it).
bool Ambiguous(uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh);
// The render size grown by whole pixels on both axes (up to 64, below the view size) until it is not ambiguous. False:
// unchanged (not ambiguous, or no such size).
bool AvoidAmbiguous(uint32_t& rw, uint32_t& rh, uint32_t ow, uint32_t oh);
// Smoke test: the module on `c` without hooks; the blocks fixed since then.
void TestInit(ID3D11DeviceContext* c);
uint64_t TestBlocksFixed();

}  // namespace qvfr::srcb

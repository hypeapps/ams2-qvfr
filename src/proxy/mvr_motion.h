// MVR (docs/mvr.md B10): the focus views' motion vectors from outer's, for DLSS.
//
// With MVR's live forward (mvr_pass.h, B9d) a focus view's color, linear depth and G-buffer are outer's geometry drawn
// through the window map (view 1), but its velocity pass still draws focus's own geometry. The engine's object matrices
// of the focus groups differ from outer's by float rounding (world coordinates of a few hundred metres), up to 0.4 px
// near the cockpit and changing from frame to frame while the car moves: DLSS then reprojects the history by motion
// that does not match the image, and fine detail near the cockpit sparkles ("snow"), more with speed.
// This module gives DLSS motion vectors of the geometry the focus image shows:
// - at outer's TAA resolve of an eye, outer's motion vectors and linear depth in that eye's focus window are copied aside
//   (DLAA's typed copies are shared with the focus views, and the engine's linear depth is shared by all groups);
// - at the focus view's resolve of that eye, when its forward pass came from outer in this frame, a compute pass moves
//   them into focus's uv (mvr_motion.hlsl): this frame's window map, the last frame's map for the previous position,
//   texels picked by depth at silhouettes (outer's texel depth against the focus pixel's). DLSS gets the result in place
//   of the engine's motion vectors.
// [debug] mvr_velocity (live, hotkey mvr_modifier + mvr_velocity_key) switches it. In a shot frame the module computes
// its vectors also when they are not used and logs how they differ from the engine's.
// B11 (docs/mvr.md): with [debug] mvr_velocity_skip, the focus group's velocity pass gets the same vectors written into
// its target at its first draw (FillFocus, from mvr_pass) and skips its draws; the game's locks pass, the game's TAA
// (cockpit_taa) and DLSS then all read them, and ForFocus leaves that view's vectors as they are.
// All functions run on the render thread. StoreOuter and ForFocus run inside DLAA's evaluation (the CS state is DLAA's
// own context state); FillFocus runs in the game's pass and puts back the CS state it changes.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::mvrmotion {

// One axis pair of a window map in uv: focus uv = scale * outer uv + offset (u and v), from mvr_pass's clip-space map
// focus = (sx x + ox w, sy y + oy w): su = sx, ou = (1 - sx + ox) / 2, sv = sy, ov = (1 - sy - oy) / 2.
struct Map {
    float su = 1, ou = 0, sv = 1, ov = 0;
};
Map FromClip(double sx, double ox, double sy, double oy);

// The source of the windows (quad: mvr_pass). eye 0 left, 1 right. False: no focus window in this frame. cur and prev:
// this frame's map and the last frame's (prev = cur when the last frame had none). fromOuter: the focus group of the
// eye skipped its own forward pass in this frame (its image is view 1 of outer's pass).
using WindowFn = bool (*)(int eye, Map* cur, Map* prev, bool* fromOuter);
void SetWindowSource(WindowFn fn);

void SetEnabled(bool on);  // the setting and the hotkey
bool Enabled();

// Outer's TAA resolve of the eye: `motion` = outer's motion vectors (R16G16, typed or typeless), `depth` = its linear
// depth (R32), `frame` = DLAA's frame counter. Copies the parts in this frame's window when the module needs them.
void StoreOuter(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* motion, ID3D11Texture2D* depth, uint64_t frame,
                bool shot);
// The focus view's TAA resolve of the eye: `depth` = its linear depth as DLAA hands it to DLSS (R32_FLOAT, typed, with
// shader resource binding), `motion` = the engine's motion vectors of the view (typed, for the shot comparison), w x h =
// the view's size. Returns the motion vectors for DLSS (this module's texture, typed R16G16_FLOAT) or null (use the
// engine's; the reason is counted). In a shot frame it also compares them with `motion` (log).
ID3D11Texture2D* ForFocus(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* depth, ID3D11Texture2D* motion, UINT w,
                          UINT h, uint64_t frame, bool shot);
// The focus group's velocity pass of the eye, at its first draw (B11): `depth` = the focus view's linear depth (R32),
// `target` = the velocity pass's target (R16G16, the same size). Writes this module's vectors into `target` when they
// are on, the eye's focus image comes from outer in this frame and outer's copy is from `frame` (DLAA's counter).
// False: nothing written (the velocity pass must draw); the reason goes to *why (static text).
bool FillFocus(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* depth, ID3D11Texture2D* target, uint64_t frame,
               const char** why);

std::string Describe();
void LogStats(bool totals);
void Release();  // device change or shutdown

// qvfr_proxy_smoke --mvr-motion: the compute pass alone on a test context (no windows source, no frame checks).
ID3D11Texture2D* TestRun(ID3D11DeviceContext* c, ID3D11Texture2D* outerMotion, ID3D11Texture2D* outerDepth,
                         ID3D11Texture2D* focusDepth, const Map& cur, const Map& prev, UINT w, UINT h, bool nearestOnly);

}  // namespace qvfr::mvrmotion

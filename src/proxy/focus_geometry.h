// Focus window on the 1:1 pixel grid (docs/architecture.md, R5): the SteamVR size is the grid, the focus covers
// `size` of each eye's tangent span around a view direction, and its edges sit on grid pixels.
// Used by the size and projection experiment ([experiment]) and by quad mode ([quad]).
#pragma once

#include <openvr.h>  // 0.9.12

#include <cstdint>

namespace qvfr::focus {

// Tangent conventions follow the OpenVR projection matrix (DirectX):
//   m00 = 2/(r-l), m02 = (r+l)/(r-l), m11 = 2/(b-t), m12 = (b+t)/(b-t).
// From it, texture column 0 (NDC x = -1) is tangent l and row 0 (NDC y = +1) is tangent b. So in "y up" terms
// b is the upper edge and t the lower edge, whatever the OpenVR names say.
struct Eye {
    float l = 0, r = 0, t = 0, b = 0;      // real frustum (GetProjectionRaw)
    float fl = 0, fr = 0, ft = 0, fb = 0;  // focus frustum, edges on the 1:1 pixel grid
    uint32_t x0 = 0, y0 = 0;               // focus position in the grid (pixels)
};

struct Geometry {
    uint32_t gridW = 0, gridH = 0;    // SteamVR size = 1:1 grid
    uint32_t focusW = 0, focusH = 0;  // focus window in grid pixels
    Eye eye[2];
};

// Reads the SteamVR size and both eye frusta from the real IVRSystem and places the focus window (size: fraction of the
// tangent span, both axes; cx, cy: center tangents, right and up are +). Checks that GetProjectionMatrix matches the
// formula above, so a replaced matrix and the window always agree. `who` prefixes the log lines. False = unusable.
bool Compute(vr::IVRSystem* real, float size, float cx, float cy, const char* who, Geometry& g);
// Places the focus window of one eye (its frustum l, r, t, b set) around the center tangents cx, cy, clamped so the
// whole window stays in the grid (M5: once per frame for the gaze).
void Place(const Geometry& g, Eye& e, float cx, float cy);
// Center tangents of a point given as NDC of the eye's full projection (x right, y up), e.g. the gaze from
// GetEyeTrackedFoveationCenter.
void NdcToTangent(const Eye& e, float ndcX, float ndcY, float& cx, float& cy);

}  // namespace qvfr::focus

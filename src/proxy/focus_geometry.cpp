#include "focus_geometry.h"

#include <algorithm>
#include <cmath>

#include "common.h"

namespace qvfr::focus {

namespace {

bool Near(float a, float b) { return std::fabs(a - b) <= 1e-3f; }

}  // namespace

bool Compute(vr::IVRSystem* real, float size, float cx, float cy, const char* who, Geometry& g) {
    g = Geometry{};
    if (!real) return false;
    real->GetRecommendedRenderTargetSize(&g.gridW, &g.gridH);
    if (g.gridW < 64 || g.gridH < 64) {
        Log("%s: SteamVR size %u x %u is not usable", who, g.gridW, g.gridH);
        return false;
    }
    g.focusW = std::max(16u, (uint32_t)std::lround(size * g.gridW));
    g.focusH = std::max(16u, (uint32_t)std::lround(size * g.gridH));
    g.focusW = std::min(g.focusW, g.gridW);
    g.focusH = std::min(g.focusH, g.gridH);

    for (int i = 0; i < 2; ++i) {
        vr::EVREye ev = i ? vr::Eye_Right : vr::Eye_Left;
        Eye& e = g.eye[i];
        real->GetProjectionRaw(ev, &e.l, &e.r, &e.t, &e.b);
        if (!(e.r > e.l) || !(e.b > e.t)) {
            Log("%s: unexpected projection (%s eye) l %.5f r %.5f t %.5f b %.5f", who, i ? "right" : "left", e.l, e.r, e.t,
                e.b);
            return false;
        }
        // The game uses GetProjectionMatrix; a replacement only works if it matches the formula above.
        vr::HmdMatrix44_t m = real->GetProjectionMatrix(ev, 0.1f, 100.0f, vr::API_DirectX);
        float m00 = 2 / (e.r - e.l), m02 = (e.r + e.l) / (e.r - e.l), m11 = 2 / (e.b - e.t), m12 = (e.b + e.t) / (e.b - e.t);
        if (!Near(m.m[0][0], m00) || !Near(m.m[0][2], m02) || !Near(m.m[1][1], m11) || !Near(m.m[1][2], m12)) {
            Log("%s: GetProjectionMatrix (%s eye) does not match GetProjectionRaw: matrix %.5f %.5f %.5f %.5f, "
                "formula %.5f %.5f %.5f %.5f",
                who, i ? "right" : "left", m.m[0][0], m.m[0][2], m.m[1][1], m.m[1][2], m00, m02, m11, m12);
            return false;
        }

        const double densX = g.gridW / (double)(e.r - e.l);  // pixels per tangent unit
        const double densY = g.gridH / (double)(e.b - e.t);
        const double ccx = std::clamp((double)cx, (double)e.l, (double)e.r);
        const double ccy = std::clamp((double)cy, (double)e.t, (double)e.b);
        long x0 = std::lround((ccx - e.l) * densX - g.focusW / 2.0);
        long y0 = std::lround((e.b - ccy) * densY - g.focusH / 2.0);
        x0 = std::clamp(x0, 0L, (long)(g.gridW - g.focusW));
        y0 = std::clamp(y0, 0L, (long)(g.gridH - g.focusH));
        e.x0 = (uint32_t)x0;
        e.y0 = (uint32_t)y0;
        e.fl = (float)(e.l + x0 / densX);
        e.fr = (float)(e.l + (x0 + g.focusW) / densX);
        e.fb = (float)(e.b - y0 / densY);
        e.ft = (float)(e.b - (y0 + g.focusH) / densY);
    }
    return true;
}

}  // namespace qvfr::focus

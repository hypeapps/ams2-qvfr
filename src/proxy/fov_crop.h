// FOV crop: fractions removed from the tangent-space image, not angular percentages.
// The same retained fractions scale the recommended dimensions, preserving pixel density.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace qvfr::fovcrop {

inline float Limit(float v) { return std::isfinite(v) ? std::clamp(v, 0.0f, 0.4f) : 0.0f; }

struct Settings {
    bool enabled = false;
    float top = 0, bottom = 0, inner = 0, outer = 0;
    bool operator==(const Settings&) const = default;
    bool Active() const { return enabled && (top > 0 || bottom > 0 || inner > 0 || outer > 0); }
    float Width() const { return Active() ? 1 - inner - outer : 1; }
    float Height() const { return Active() ? 1 - top - bottom : 1; }
};

inline uint32_t Dimension(uint32_t n, float retained) {
    if (!n || retained == 1) return n;
    return std::clamp(static_cast<uint32_t>(std::round(double(n) * retained)), 1u, n);
}

// Fov is XrFovf (kept independent of the OpenXR headers for config and math tests).
// OpenXR: left/down negative, right/up positive. Inner/outer mirror between the two eyes.
template <class Fov>
bool Apply(const Settings& s, uint32_t eye, const Fov& in, Fov& out) {
    out = in;
    if (!s.Active()) return true;
    const float angles[] = {in.angleLeft, in.angleRight, in.angleUp, in.angleDown};
    for (float a : angles)
        if (!std::isfinite(a) || std::fabs(a) >= 1.57079f) return false;
    const float l = std::tan(in.angleLeft), r = std::tan(in.angleRight);
    const float d = std::tan(in.angleDown), u = std::tan(in.angleUp);
    if (r <= l || u <= d || eye > 1) return false;
    const float left = eye == 0 ? s.outer : s.inner;
    const float right = eye == 0 ? s.inner : s.outer;
    // Preserve untouched edges exactly, including the disabled/zero-crop case.
    if (left > 0) out.angleLeft = std::atan(l + (r - l) * left);
    if (right > 0) out.angleRight = std::atan(r - (r - l) * right);
    if (s.top > 0) out.angleUp = std::atan(u - (u - d) * s.top);
    if (s.bottom > 0) out.angleDown = std::atan(d + (u - d) * s.bottom);
    return true;
}

}  // namespace qvfr::fovcrop

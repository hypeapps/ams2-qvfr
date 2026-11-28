#pragma once
// The TAA jitter the engine puts into its projections, and the sequences the proxy can give it instead while DLSS
// replaces the game's TAA ([dlaa] jitter, benchmark switch dlss_jitter).
//
// Units are the engine's projection offset o (CTAAControl slot 0 GetJitter, SetJitter): the projection moves by
// (o.x / width, o.y / height) in NDC, so the image moves by (o.x / 2, -o.y / 2) pixels of the view's render target.
// The engine's table: o = (Halton(55 + (frame & 7), 2 and 3) - 0.5) * 1.19, at most 0.502, i.e. +-0.25 px in 8 phases.
// NVIDIA's DLSS Programming Guide asks for +-0.5 px and about 8 * (output / render)^2 phases per sequence.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace qvfr::jitter {

enum Mode : int {
    kGame = 0,    // the engine's table
    kFull8 = 1,   // the engine's 8 phases scaled by 2 / 1.19 to the full range, up to +-0.42 px (the amplitude alone)
    kNvidia = 2,  // Halton (2, 3) over +-0.5 px with NVIDIA's phase count for the render scale
};
constexpr int kModeCount = 3;

constexpr float kGameScale = 1.19f;  // the engine's (h - 0.5) * 1.19 (CTAAControl ctor)
constexpr float kFullScale = 2.0f;   // (h - 0.5) * 2: +-1 offset = +-0.5 px
constexpr int kGamePhases = 8;

inline float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index) {
        f /= (float)base;
        r += f * (float)(index % base);
        index /= base;
    }
    return r;
}

// Phases for a render scale (render size / output size per axis): ceil(8 / s^2), at least the game's 8.
inline int NvidiaPhases(float scale) {
    const float s = std::clamp(scale, 0.25f, 1.0f);
    return std::max(kGamePhases, (int)std::ceil(8.0f / (s * s) - 1e-4f));
}

// The engine's own offset for a frame counter (as its table holds it).
inline void GameOffset(uint32_t frame, float out[2]) {
    const uint32_t i = 55 + (frame & 7);
    out[0] = (Halton(i, 2) - 0.5f) * kGameScale;
    out[1] = (Halton(i, 3) - 0.5f) * kGameScale;
}

// Replaces the engine's offset `o` of frame `frame` for a mode; kGame leaves it. The Halton index starts at 1, so
// the offset is never (0, 0) (Halton base 3 is never 0.5), which the proxy reads as "no jitter".
inline void Apply(int mode, int phases, uint32_t frame, float o[2]) {
    if (mode == kFull8) {
        o[0] *= kFullScale / kGameScale;
        o[1] *= kFullScale / kGameScale;
    } else if (mode == kNvidia) {
        const uint32_t i = 1 + frame % (uint32_t)std::max(phases, 1);
        o[0] = (Halton(i, 2) - 0.5f) * kFullScale;
        o[1] = (Halton(i, 3) - 0.5f) * kFullScale;
    }
}

}  // namespace qvfr::jitter

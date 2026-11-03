// [quad] preset values (M8.5), shared by the proxy and the settings GUI. Each sets peripheral_multiplier and the four
// *_section keys; focus_multiplier stays 1.0 (the focus at the density of the SteamVR resolution). balanced = the
// defaults of the mbucchia Quad-Views-Foveated layer. A key given in qvfr.ini wins over the preset.
#pragma once

namespace qvfr::presets {

struct Preset {
    const char* name;
    float peripheral;      // peripheral_multiplier
    float focusX, focusY;  // horizontal/vertical_focus_section (with eye tracking)
    float fixedX, fixedY;  // horizontal/vertical_fixed_section (without it)
};

inline constexpr Preset kPresets[] = {
    {"quality", 0.6f, 0.45f, 0.45f, 0.6f, 0.55f},
    {"balanced", 0.5f, 0.35f, 0.35f, 0.5f, 0.45f},
    {"performance", 0.4f, 0.3f, 0.3f, 0.4f, 0.4f},
};
inline constexpr int kDefaultPreset = 1;

}  // namespace qvfr::presets

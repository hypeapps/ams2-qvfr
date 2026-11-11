// The player settings of qvfr.ini as the GUI shows them: percentages like QuadViews Companion, VectorXR and the LMU
// launcher (docs/architecture.md, settings GUI). Values come from the ini; the five preset keys fall back to the
// preset when they are empty, like in the proxy (common.cpp ApplyPreset).
#pragma once

#include <string>
#include <vector>

#include "ini_file.h"

namespace qvfr::config {

struct Settings {
    bool openxr = false;          // [runtime] path
    bool quad = true;             // [quad] mode=quad (anything else: off for the GUI)
    std::string modeText = "quad";
    int preset = 1;               // index in presets::kPresets, -1 = unknown name
    std::string presetText = "balanced";
    // Effective values (given in the ini, or from the preset) and whether each was given.
    float peripheral = 0.5f, focusX = 0.35f, focusY = 0.35f, fixedX = 0.5f, fixedY = 0.45f;
    bool givenPeripheral = false, givenFocusX = false, givenFocusY = false, givenFixedX = false, givenFixedY = false;
    float focusMultiplier = 1.0f;
    float focusOffsetX = 0, focusOffsetY = 0, fixedOffsetX = 0, fixedOffsetY = 0;
    float smoothen = 0.2f;
    bool followGaze = true;
    float gazeHoldS = 0.6f;
    int gazeKey = 0x75, gazeMod = 0xA3;
    bool warningBox = true;
    bool overlay = false;  // [debug] overlay: the gaze and the blend edges drawn in the game
    // [mvr] (docs/mvr.md): NVIDIA Multi-View Rendering of the focus views in outer's passes
    bool mvr = true;                 // mode=mvr, anything but off (restart)
    // [dlaa] (docs/dlaa.md): NVIDIA DLAA instead of the game's TAA
    bool dlaa = false;               // mode=dlaa (restart)
    std::string dlaaPreset = "k";    // j, k, l, m ("default" reads as k, NVIDIA's DLAA default)
    bool dlaaAutoExposure = false;
    float dlaaExposure = 0.002f;     // fixed DLSS exposure with auto off
    // [nr] (dlaa-plan D7): DLSS 5 neural rendering on the finished image; runs on DLAA's motion vectors and depth, so
    // it needs [dlaa] mode=dlaa. All live. The same options as the tuning panel in the headset (proxy panel.cpp).
    bool nr = false;                 // mode=final (or the older focus)
    float nrIntensity = 1.0f, nrStrength = 1.0f, nrLocalTone = 1.0f, nrLocalStructure = 1.0f;
    int nrStyle = 0, nrPreset = 0;   // 0..3
    bool nrAutoMask = true;
    bool nrOuter = true;             // the outer views through the model too
    int nrOuterStep = 2;             // outer_resolution 1/step: 1 full, 2 half, 3 a third
    float nrOuterArea = 1.0f;        // 0.5..1
    bool nrOuterSeparate = false;    // outer_atlas=separate
    // Testers
    bool csvAlways = false;
    int benchKey = 0;
    int overlayKey = 0, freezeKey = 0, outerKey = 0, clonesKey = 0, shotKey = 0, cullKey = 0, sunKey = 0,
        taaKey = 0, dlaaKey = 0, dlaaExpKey = 0, dlaaPresetKey = 0;

    bool Custom() const { return givenPeripheral || givenFocusX || givenFocusY || givenFixedX || givenFixedY; }
};

Settings Read(const IniFile& ini);

// Writers: each changes the document (the caller saves). Values are written the way a person would type them.
std::string Num(float v);
void WritePreset(IniFile& ini, int preset);        // preset=<name> and the five preset keys emptied
void ClearPresetOverrides(IniFile& ini);
// DLSS 5 on ([nr] mode=final, and [dlaa] mode=dlaa, which it runs on) or off ([nr] mode=off; DLAA stays as it is).
void WriteNr(IniFile& ini, bool on);

// Keys that only take effect at the next game start (the rest work while the game runs, about 1 s after saving).
struct KeyRef {
    const char* section;
    const char* key;
    const char* label;  // for the "restart needed" note
};
const std::vector<KeyRef>& RestartKeys();
// Labels of the restart keys whose values differ between two documents (empty = none).
std::vector<std::string> ChangedRestartKeys(const IniFile& before, const IniFile& now);

// The default qvfr.ini (the same text the proxy writes).
const char* DefaultIniText();

}  // namespace qvfr::config

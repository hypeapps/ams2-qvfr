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
    // Testers
    bool csvAlways = false;
    int benchKey = 0;
    int overlayKey = 0, freezeKey = 0, outerKey = 0, clonesKey = 0, shotKey = 0;

    bool Custom() const { return givenPeripheral || givenFocusX || givenFocusY || givenFixedX || givenFixedY; }
};

Settings Read(const IniFile& ini);

// Writers: each changes the document (the caller saves). Values are written the way a person would type them.
std::string Num(float v);
void WritePreset(IniFile& ini, int preset);        // preset=<name> and the five preset keys emptied
void ClearPresetOverrides(IniFile& ini);

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

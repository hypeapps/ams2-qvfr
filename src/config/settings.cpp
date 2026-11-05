#include "settings.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "presets.h"

namespace qvfr::config {

namespace {

const char kDefaultIni[] =
#include "default_ini.inc"
    ;

int ParseInt(const std::string& s, int def) {
    if (s.empty()) return def;
    return (int)strtol(s.c_str(), nullptr, 0);  // 0x.. or decimal, like the proxy
}

float ParseFloat(const std::string& s, float def) { return s.empty() ? def : (float)atof(s.c_str()); }

// A key with a non-empty value.
bool Given(const IniFile& ini, const char* sec, const char* key, std::string& v) { return ini.Get(sec, key, v) && !v.empty(); }

float Clamp(float v, float lo, float hi) { return !(v >= lo) ? lo : v > hi ? hi : v; }

const char* const kPresetKeys[] = {"peripheral_multiplier", "horizontal_focus_section", "vertical_focus_section",
                                   "horizontal_fixed_section", "vertical_fixed_section"};

}  // namespace

Settings Read(const IniFile& ini) {
    Settings s;
    s.openxr = ini.GetOr("runtime", "path", "openxr") != "openvr";  // openxr is the default
    s.modeText = ini.GetOr("quad", "mode", "off");
    s.quad = s.modeText == "quad";
    s.presetText = ini.GetOr("quad", "preset", "balanced");
    if (s.presetText.empty()) s.presetText = "balanced";
    s.preset = -1;
    for (int i = 0; i < (int)std::size(presets::kPresets); ++i)
        if (s.presetText == presets::kPresets[i].name) s.preset = i;
    const presets::Preset& p = presets::kPresets[s.preset >= 0 ? s.preset : presets::kDefaultPreset];
    std::string v;
    s.peripheral = (s.givenPeripheral = Given(ini, "quad", "peripheral_multiplier", v)) ? ParseFloat(v, p.peripheral) : p.peripheral;
    s.focusX = (s.givenFocusX = Given(ini, "quad", "horizontal_focus_section", v)) ? ParseFloat(v, p.focusX) : p.focusX;
    s.focusY = (s.givenFocusY = Given(ini, "quad", "vertical_focus_section", v)) ? ParseFloat(v, p.focusY) : p.focusY;
    s.fixedX = (s.givenFixedX = Given(ini, "quad", "horizontal_fixed_section", v)) ? ParseFloat(v, p.fixedX) : p.fixedX;
    s.fixedY = (s.givenFixedY = Given(ini, "quad", "vertical_fixed_section", v)) ? ParseFloat(v, p.fixedY) : p.fixedY;
    // The proxy's ranges (common.cpp ReadQuadView).
    s.peripheral = Clamp(s.peripheral, 0.1f, 1e6f);
    s.focusX = Clamp(s.focusX, 0.1f, 0.9f);
    s.focusY = Clamp(s.focusY, 0.1f, 0.9f);
    s.fixedX = Clamp(s.fixedX, 0.1f, 0.9f);
    s.fixedY = Clamp(s.fixedY, 0.1f, 0.9f);
    s.focusMultiplier = Clamp(ParseFloat(ini.GetOr("quad", "focus_multiplier", ""), 1.0f), 0.1f, 1e6f);
    s.focusOffsetX = Clamp(ParseFloat(ini.GetOr("quad", "horizontal_focus_offset", ""), 0), -0.5f, 0.5f);
    s.focusOffsetY = Clamp(ParseFloat(ini.GetOr("quad", "vertical_focus_offset", ""), 0), -0.5f, 0.5f);
    s.fixedOffsetX = Clamp(ParseFloat(ini.GetOr("quad", "horizontal_fixed_offset", ""), 0), -0.5f, 0.5f);
    s.fixedOffsetY = Clamp(ParseFloat(ini.GetOr("quad", "vertical_fixed_offset", ""), 0), -0.5f, 0.5f);
    s.smoothen = Clamp(ParseFloat(ini.GetOr("quad", "smoothen_focus_view_edges", ""), 0.2f), 0, 0.5f);
    s.followGaze = ParseInt(ini.GetOr("quad", "follow_gaze", ""), 1) != 0;
    s.gazeHoldS = Clamp(ParseFloat(ini.GetOr("quad", "gaze_hold_s", ""), 0.6f), 0, 10);
    s.gazeKey = ParseInt(ini.GetOr("debug", "gaze_key", ""), 0x75);
    s.gazeMod = ParseInt(ini.GetOr("debug", "gaze_modifier", ""), 0xA3);
    s.warningBox = ParseInt(ini.GetOr("log", "warning_box", ""), 1) != 0;
    s.overlay = ParseInt(ini.GetOr("debug", "overlay", ""), 0) != 0;
    s.csvAlways = ParseInt(ini.GetOr("timing", "csv_always", ""), 0) != 0;
    s.benchKey = ParseInt(ini.GetOr("benchmark", "key", ""), 0);
    s.overlayKey = ParseInt(ini.GetOr("debug", "overlay_key", ""), 0);
    s.freezeKey = ParseInt(ini.GetOr("debug", "freeze_key", ""), 0);
    s.outerKey = ParseInt(ini.GetOr("debug", "outer_key", ""), 0);
    s.clonesKey = ParseInt(ini.GetOr("debug", "clones_key", ""), 0);
    s.cullKey = ParseInt(ini.GetOr("debug", "cull_key", ""), 0);
    s.sunKey = ParseInt(ini.GetOr("debug", "sun_effects_key", ""), 0);
    s.shotKey = ParseInt(ini.GetOr("quad", "shot_key", ""), 0);
    return s;
}

std::string Num(float v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", v);
    std::string s = buf;
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s += '0';
    if (s == "-0.0") s = "0.0";
    return s;
}

void ClearPresetOverrides(IniFile& ini) {
    for (const char* k : kPresetKeys) ini.Set("quad", k, "");
}

void WritePreset(IniFile& ini, int preset) {
    if (preset < 0 || preset >= (int)std::size(presets::kPresets)) return;
    ini.Set("quad", "preset", presets::kPresets[preset].name);
    ClearPresetOverrides(ini);
}

const std::vector<KeyRef>& RestartKeys() {
    static const std::vector<KeyRef> k = {
        {"runtime", "path", "VR API (OpenVR / OpenXR)"},
        {"runtime", "opencomposite", "OpenComposite DLL"},
        {"runtime", "openxr_layer", "OpenXR layer"},
        {"proxy", "wrap", "proxy wrap"},
        {"quad", "mode", "quad views on/off"},
        {"quad", "preset", "preset"},
        {"quad", "peripheral_multiplier", "peripheral resolution"},
        {"quad", "focus_multiplier", "focus resolution"},
        {"quad", "horizontal_focus_section", "focus width"},
        {"quad", "vertical_focus_section", "focus height"},
        {"quad", "horizontal_fixed_section", "focus width without eye tracking"},
        {"quad", "vertical_fixed_section", "focus height without eye tracking"},
        {"timing", "csv_always", "CSV for the whole session"},
        {"timing", "gpu", "GPU timestamps"},
        {"benchmark", "auto_start_s", "benchmark auto start"},
    };
    return k;
}

std::vector<std::string> ChangedRestartKeys(const IniFile& before, const IniFile& now) {
    std::vector<std::string> out;
    for (const KeyRef& k : RestartKeys()) {
        std::string a, b;
        before.Get(k.section, k.key, a);
        now.Get(k.section, k.key, b);
        if (a != b) out.push_back(k.label);
    }
    return out;
}

const char* DefaultIniText() { return kDefaultIni; }

}  // namespace qvfr::config

#include "common.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace qvfr {

std::atomic<bool> g_diagDisabled{false};

namespace {

std::mutex g_logMutex;
FILE* g_log = nullptr;
bool g_logTried = false;
fs::path g_logFile;
Config g_cfg;
fs::path g_base;

const auto g_t0 = std::chrono::steady_clock::now();

void OpenLogLocked() {
    if (g_logTried) return;
    g_logTried = true;
    std::error_code ec;
    fs::create_directories(LogDir(), ec);
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring name = L"qvfr_" + Widen(LocalTimeStamp()) + L"_" + fs::path(exe).stem().wstring() + L".log";
    g_logFile = LogDir() / name;
    g_log = _wfopen(g_logFile.c_str(), L"wb");
}

const char kDefaultIni[] =
    "; ams2-qvfr settings: quad views foveated rendering for Automobilista 2 in VR. The file is created on the first game\n"
    "; start; delete it to get the defaults back. Most changes take effect after restarting the game; the ones that work\n"
    "; while the game runs are listed at the end. What the mod detected at the last start: the \"Detected\" block in the\n"
    "; newest log in x64\\qvfr\\logs\\.\n"
    "\n"
    "[runtime]\n"
    "; openvr = the Valve openvr_api.dll (x64\\qvfr\\openvr\\) talks to SteamVR;\n"
    "; openxr = OpenComposite (x64\\qvfr\\opencomposite\\) talks to the OpenXR runtime that is active in the system\n"
    "; (choose it in SteamVR, Pimax Play, Meta, Varjo Base...; the mod never changes it).\n"
    "path=openvr\n"
    "; openxr: another OpenComposite DLL (full path, or relative to x64\\); empty = the default above.\n"
    "opencomposite=\n"
    "; openxr: 1 = the proxy's OpenXR API layer reads the gaze (XR_EXT_eye_gaze_interaction) under OpenComposite.\n"
    "openxr_layer=1\n"
    "\n"
    "[quad]\n"
    "; quad = quad views: the game renders the whole field of view at a lower density (outer) and a smaller area at\n"
    ";        full density (focus) that follows your gaze, and the mod puts both together for each eye;\n"
    "; off  = the mod changes nothing (the game renders as without it).\n"
    "; Works only with AMS2AVX.exe of the build this mod version knows; otherwise the game renders as without the mod.\n"
    "mode=quad\n"
    "; Size of the focus area and density of the outer view:\n"
    ";   quality     = focus 45% x 45% of the field of view (60% x 55% without eye tracking), outer at 0.6\n"
    ";   balanced    = focus 35% x 35% (50% x 45% without eye tracking), outer at 0.5 (the mbucchia defaults)\n"
    ";   performance = focus 30% x 30% (40% x 40% without eye tracking), outer at 0.4\n"
    "; Restart the game after a change.\n"
    "preset=balanced\n"
    "; The next keys have the names, meaning and defaults of the Quad-Views-Foveated OpenXR layer by mbucchia, so its\n"
    "; guides apply (https://github.com/mbucchia/Quad-Views-Foveated/wiki/Advanced-Configuration).\n"
    "; Empty = the value of the preset; a number here wins over the preset.\n"
    "; Pixel density of the outer view (the whole field of view) and of the focus view, as a multiplier of each\n"
    "; dimension of the resolution set in SteamVR (1.0 = that resolution, above 1 = supersampling).\n"
    "; Restart the game after a change.\n"
    "peripheral_multiplier=\n"
    "focus_multiplier=1.0\n"
    "; Focus area as a fraction of the field of view in each direction (0.1..0.9): with eye tracking (follow_gaze=1)\n"
    "; and without it (follow_gaze=0). The game session keeps the area of the follow_gaze value it started with; the\n"
    "; DFR <-> FFR hotkey only moves the window. Restart the game after a change.\n"
    "horizontal_focus_section=\n"
    "vertical_focus_section=\n"
    "horizontal_fixed_section=\n"
    "vertical_fixed_section=\n"
    "; Shift of the focus area from the gaze point (focus) and from straight ahead (fixed), in screen units of the eye\n"
    "; (-1..1 across the field of view), -0.5..0.5. Horizontal: + moves both eyes' areas outward (away from the nose).\n"
    "; Vertical: + is up.\n"
    "horizontal_focus_offset=0\n"
    "vertical_focus_offset=0\n"
    "horizontal_fixed_offset=0\n"
    "vertical_fixed_offset=0\n"
    "; Soft edge between focus and outer: blend band on each side as a fraction of the focus area (0..0.5, 0 = hard\n"
    "; edge). As in mbucchia, the focus keeps at least half weight up to its edge.\n"
    "smoothen_focus_view_edges=0.2\n"
    "; 1 = the focus area follows your gaze every frame (dynamic foveated rendering, DFR; needs eye tracking);\n"
    "; 0 = it stays straight ahead plus the fixed offsets (fixed foveated rendering, FFR). It also goes there when the\n"
    "; gaze is not valid for longer than gaze_hold_s (until then it stays at the last valid gaze, e.g. while blinking)\n"
    "; and while a benchmark runs.\n"
    "follow_gaze=1\n"
    "gaze_hold_s=0.6\n"
    "; Advanced: 1 = the mod composes each eye from outer (upscaled to the SteamVR resolution) and focus (in its\n"
    "; window); 0 = outer only, at the game's render size (diagnostics).\n"
    "compose=1\n"
    "; Advanced (not in mbucchia): the outermost margin_px pixels of the focus area are not shown (guard band for\n"
    "; screen-space effects that break at the image edge); the soft edge starts inside them. 0 = the whole focus area.\n"
    "margin_px=0\n"
    "; Testers: hotkey for a shot of the four views (outer L/R, focus L/R) and the composed eyes: PNG files in\n"
    "; x64\\qvfr\\shots\\ and pixel differences in the log. 0xA3 = right Ctrl, 0x79 = F10; shot_key=0 = no hotkey.\n"
    "shot_modifier=0xA3\n"
    "shot_key=0\n"
    "; Testers: > 0 = one shot by itself this many seconds after the first frame with focus views.\n"
    "shot_auto_s=0\n"
    "; Developers ([quad] mode=probe): frames logged in full, hooks in the game engine and in D3D11.\n"
    "probe_frames=3\n"
    "probe_engine=1\n"
    "probe_d3d=1\n"
    "\n"
    "[debug]\n"
    "; Hotkeys: modifier + key (VK codes, 0xA3 = right Ctrl; key=0 = no hotkey).\n"
    "; Focus area: follow the gaze (DFR) <-> fixed straight ahead (FFR), like [quad] follow_gaze. 0x75 = F6.\n"
    "gaze_modifier=0xA3\n"
    "gaze_key=0x75\n"
    "; The keys below are tools for testers and are off for players (the values in brackets are the tester hotkeys).\n"
    "; Debug overlay in the composed image: yellow = where the focus starts to blend in, green = from where it is\n"
    "; fully sharp, red ring = gaze point (blue when frozen). overlay=1 = on at start. (0x77 = F8)\n"
    "overlay=0\n"
    "overlay_modifier=0xA3\n"
    "overlay_key=0\n"
    "; Freeze the gaze point. (0x76 = F7)\n"
    "freeze_modifier=0xA3\n"
    "freeze_key=0\n"
    "; Quad <-> outer only: the focus views are not rendered and the headset gets outer alone, at the game's smaller\n"
    "; render size (not the full-density image of [quad] mode=off). The mod also switches to outer only by itself when\n"
    "; quad mode fails. (0x7A = F11)\n"
    "outer_modifier=0xA3\n"
    "outer_key=0\n"
    "; Clones: 1 = the views whose size differs from the game's render size (with the presets the focus views) render\n"
    "; into their own copies of the engine's eye-size targets in their own size, 0 = into the game-size targets of\n"
    "; outer. The hotkey switches while the game runs. (0x74 = F5)\n"
    "clones=1\n"
    "clones_modifier=0xA3\n"
    "clones_key=0\n"
    "; 1 = a viewport the game sets for a full-size target while its smaller copy is bound is scaled to the copy.\n"
    "clones_viewport_fix=1\n"
    "; 1 = a pass that binds a full-size target by its own D3D11 view while the copies are in gets the copy instead.\n"
    "clones_bind_fix=1\n"
    "; 1 = each shot also saves every copy of the focus views (the game stops for about 3 s).\n"
    "shot_clones=0\n"
    "\n"
    "[log]\n"
    "; 1 = serious problems found at the start (for example TAA on, Pimax LibMagic loaded, an unsupported game build)\n"
    "; also show in a Windows message box; the game keeps running. 0 = only in the log.\n"
    "warning_box=1\n"
    "; How many first calls of key methods (projection, size, Submit) to log with values.\n"
    "first_calls=3\n"
    "; Interval in seconds for call counters and gaze state in the log (0 = only on game exit).\n"
    "stats_interval_s=60\n"
    "\n"
    "[timing]\n"
    "; Testers: 1 = frame timing CSV for the whole session (x64\\qvfr\\bench\\<date>_session\\), 0 = only during a benchmark.\n"
    "csv_always=0\n"
    "; 1 = GPU timestamps per frame phase (outer, focus, composition, whole frame) in the CSV, the benchmark summary and\n"
    "; the stats lines in the log. Costs a few microseconds per frame.\n"
    "gpu=1\n"
    "\n"
    "[benchmark]\n"
    "; Testers: a measurement of the frame times into x64\\qvfr\\bench\\. Hotkey: modifier + key (VK codes; 0xA3 = right\n"
    "; Ctrl, 0x78 = F9; modifier=0 = the key alone, key=0 = no hotkey). Pressing it during the countdown or the recording\n"
    "; aborts the measurement.\n"
    "modifier=0xA3\n"
    "key=0\n"
    "; Seconds from key press to recording start (time to rest your hands and keep your head still).\n"
    "delay_s=5\n"
    "; Recording length in seconds.\n"
    "duration_s=30\n"
    "; 1 = the game gets a constant head pose while recording (the image in the headset sticks to your head).\n"
    "freeze_pose=1\n"
    "; Beeps: 1 short = countdown, 2 high = recording start, 1 long low = end.\n"
    "beep=1\n"
    "; > 0: the benchmark starts by itself this many seconds after the first frame (for tests, normally 0).\n"
    "auto_start_s=0\n"
    "\n"
    "[gaze]\n"
    "; 1 = read the gaze every frame (SteamVR on path=openvr, the OpenXR layer on path=openxr); 0 = no gaze (FFR only).\n"
    "read=1\n"
    "\n"
    "[proxy]\n"
    "; 1 = the proxy wraps IVRSystem, IVRCompositor and IVROverlay (needed for quad views, logs, timings, benchmark).\n"
    "; 0 = the game gets the real interfaces and the proxy only logs startup (fallback).\n"
    "wrap=1\n"
    "\n"
    "[experiment]\n"
    "; Developers: size and projection test without engine hooks ([quad] mode=off). The game reads both once per VR\n"
    "; session, so restart the game after a change. The size reported by SteamVR is treated as the 1:1 grid.\n"
    ";   off   = no change (reference)\n"
    ";   focus = the game renders only the focus window (focus_size of the tangent span) at 1:1 density\n"
    ";   outer = the game renders the full FOV into a texture of the focus size\n"
    "mode=off\n"
    "; Focus window as a fraction of the eye's tangent span, both axes (0.5 = 50% x 50%).\n"
    "focus_size=0.5\n"
    "; Focus center as view direction tangents, the same for both eyes (0,0 = optical axis; right and up are +).\n"
    "center_x=0\n"
    "center_y=0\n"
    "; focus only: 1 = copy the game image 1:1 into a black full-FOV texture, so it shows in its real place and scale.\n"
    "; 0 = submit the game texture as is (the compositor stretches it over the whole field of view).\n"
    "pad=1\n"
    "\n"
    "[trace]\n"
    "; Developers: 1 = log where in the game the key OpenVR calls come from (size, projection, eye transforms, poses,\n"
    "; Submit, overlays): every unique call stack once as module+RVA, hit counts in the stats. Normally 0.\n"
    "callers=0\n"
    "; Game frames per stack (1..48).\n"
    "depth=12\n"
    "\n"
    "; Settings that work while the game runs (the file is checked about once a second): [quad] the four *_offset keys,\n"
    "; smoothen_focus_view_edges, follow_gaze, gaze_hold_s, compose, margin_px, the shot keys and shot_auto_s; all of\n"
    "; [debug]; [log] warning_box and stats_interval_s; [benchmark] all but auto_start_s; [gaze] read. Everything else\n"
    "; needs a game restart.\n";

std::string IniString(const fs::path& ini, const wchar_t* sec, const wchar_t* key, const wchar_t* def) {
    wchar_t buf[64];
    GetPrivateProfileStringW(sec, key, def, buf, 64, ini.c_str());
    return Narrow(buf);
}

int IniInt(const fs::path& ini, const wchar_t* sec, const wchar_t* key, int def) {
    wchar_t buf[64];
    GetPrivateProfileStringW(sec, key, L"", buf, 64, ini.c_str());
    if (!buf[0]) return def;
    return (int)wcstol(buf, nullptr, 0);  // 0x.. or decimal
}

float IniFloat(const fs::path& ini, const wchar_t* sec, const wchar_t* key, float def) {
    wchar_t buf[64];
    GetPrivateProfileStringW(sec, key, L"", buf, 64, ini.c_str());
    return buf[0] ? (float)_wtof(buf) : def;
}

std::wstring IniPathValue(const fs::path& ini, const wchar_t* sec, const wchar_t* key) {
    wchar_t buf[1024];
    GetPrivateProfileStringW(sec, key, L"", buf, 1024, ini.c_str());
    return buf;
}

bool IniHas(const fs::path& ini, const wchar_t* sec, const wchar_t* key) {
    wchar_t buf[8];
    GetPrivateProfileStringW(sec, key, L"", buf, 8, ini.c_str());
    return buf[0] != 0;
}

// Clamps like the mbucchia layer does, and logs a changed value.
float Ranged(const char* key, float v, float lo, float hi) {
    const float r = !(v >= lo) ? lo : v > hi ? hi : v;
    if (r != v) Log("Settings: [quad] %s=%.3f outside %.2f..%.2f, using %.3f", key, v, lo, hi, r);
    return r;
}

// The [quad] keys of M7 (mbucchia's), after the pre-M7 keys that still work with a warning.
// M8.5 presets: the focus area with and without eye tracking and the outer density; focus_multiplier stays 1.0 (1:1
// in the focus). balanced = the mbucchia defaults.
struct Preset {
    const char* name;
    float peripheral, focusX, focusY, fixedX, fixedY;
};
const Preset kPresets[] = {
    {"quality", 0.6f, 0.45f, 0.45f, 0.6f, 0.55f},
    {"balanced", 0.5f, 0.35f, 0.35f, 0.5f, 0.45f},
    {"performance", 0.4f, 0.3f, 0.3f, 0.4f, 0.4f},
};

void ApplyPreset(const fs::path& ini, Config& c) {
    std::string name = IniString(ini, L"quad", L"preset", L"balanced");
    if (name.empty()) name = "balanced";
    const Preset* p = nullptr;
    for (const Preset& k : kPresets)
        if (name == k.name) p = &k;
    if (!p) {
        Log("Settings: unknown [quad] preset=\"%s\", using balanced", name.c_str());
        p = &kPresets[1];
    }
    c.preset = p->name;
    c.peripheralMultiplier = p->peripheral;
    c.focusSectionX = p->focusX;
    c.focusSectionY = p->focusY;
    c.fixedSectionX = p->fixedX;
    c.fixedSectionY = p->fixedY;
    std::string given;
    for (const wchar_t* k : {L"peripheral_multiplier", L"horizontal_focus_section", L"vertical_focus_section",
                             L"horizontal_fixed_section", L"vertical_fixed_section"})
        if (IniHas(ini, L"quad", k)) given += (given.empty() ? "" : ", ") + Narrow(k);
    Log("Settings: [quad] preset=%s (peripheral_multiplier %.2f, focus section %.2f x %.2f, fixed section %.2f x %.2f)%s%s",
        p->name, p->peripheral, p->focusX, p->focusY, p->fixedX, p->fixedY, given.empty() ? "" : "; given in the ini: ",
        given.c_str());
}

void ReadQuadView(const fs::path& ini, Config& c) {
    const wchar_t* q = L"quad";
    ApplyPreset(ini, c);
    if (IniHas(ini, q, L"focus_size")) {
        const float s = Ranged("focus_size", IniFloat(ini, q, L"focus_size", 0.5f), 0.1f, 0.9f);
        Log("Settings: WARNING: [quad] focus_size is a pre-M7 key; it sets peripheral_multiplier and the four *_section "
            "keys that are not given to %.3f (the old meaning). Use those keys instead", s);
        if (!IniHas(ini, q, L"peripheral_multiplier")) c.peripheralMultiplier = s;
        c.focusSectionX = c.focusSectionY = c.fixedSectionX = c.fixedSectionY = s;
    }
    if (IniHas(ini, q, L"center_x") || IniHas(ini, q, L"center_y")) {
        c.quadCenterX = IniFloat(ini, q, L"center_x", 0.0f);
        c.quadCenterY = IniFloat(ini, q, L"center_y", 0.0f);
        Log("Settings: WARNING: [quad] center_x/center_y are pre-M7 keys (view direction tangents, added to the fixed "
            "center: %.3f, %.3f); use horizontal_fixed_offset/vertical_fixed_offset instead", c.quadCenterX,
            c.quadCenterY);
    }
    if (IniHas(ini, q, L"feather_px")) {
        if (IniHas(ini, q, L"smoothen_focus_view_edges")) {
            Log("Settings: WARNING: [quad] feather_px (pre-M7) ignored: smoothen_focus_view_edges is given");
        } else {
            c.legacyEdge = true;
            c.quadFeatherPx = IniInt(ini, q, L"feather_px", 128);
            if (c.quadFeatherPx < 0 || c.quadFeatherPx > 1024) {
                Log("Settings: [quad] feather_px=%d outside 0..1024, using 128", c.quadFeatherPx);
                c.quadFeatherPx = 128;
            }
            Log("Settings: WARNING: [quad] feather_px is a pre-M7 key: the old soft edge (%d px after the margin, from 0 "
                "to full focus); use smoothen_focus_view_edges instead", c.quadFeatherPx);
        }
    }
    c.peripheralMultiplier = Ranged("peripheral_multiplier",
                                    IniFloat(ini, q, L"peripheral_multiplier", c.peripheralMultiplier), 0.1f, 1e6f);
    c.focusMultiplier = Ranged("focus_multiplier", IniFloat(ini, q, L"focus_multiplier", c.focusMultiplier), 0.1f, 1e6f);
    c.focusSectionX = Ranged("horizontal_focus_section",
                             IniFloat(ini, q, L"horizontal_focus_section", c.focusSectionX), 0.1f, 0.9f);
    c.focusSectionY = Ranged("vertical_focus_section",
                             IniFloat(ini, q, L"vertical_focus_section", c.focusSectionY), 0.1f, 0.9f);
    c.fixedSectionX = Ranged("horizontal_fixed_section",
                             IniFloat(ini, q, L"horizontal_fixed_section", c.fixedSectionX), 0.1f, 0.9f);
    c.fixedSectionY = Ranged("vertical_fixed_section",
                             IniFloat(ini, q, L"vertical_fixed_section", c.fixedSectionY), 0.1f, 0.9f);
    c.focusOffsetX = Ranged("horizontal_focus_offset", IniFloat(ini, q, L"horizontal_focus_offset", 0.0f), -0.5f, 0.5f);
    c.focusOffsetY = Ranged("vertical_focus_offset", IniFloat(ini, q, L"vertical_focus_offset", 0.0f), -0.5f, 0.5f);
    c.fixedOffsetX = Ranged("horizontal_fixed_offset", IniFloat(ini, q, L"horizontal_fixed_offset", 0.0f), -0.5f, 0.5f);
    c.fixedOffsetY = Ranged("vertical_fixed_offset", IniFloat(ini, q, L"vertical_fixed_offset", 0.0f), -0.5f, 0.5f);
    c.smoothen = Ranged("smoothen_focus_view_edges", IniFloat(ini, q, L"smoothen_focus_view_edges", c.smoothen), 0.0f,
                        0.5f);
}

}  // namespace

double NowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
}

std::string LocalTimeStamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

// Called from code paths outside Guarded, so it must never let an exception reach the game.
void LogRaw(const char* text) {
    try {
        std::lock_guard lock(g_logMutex);
        OpenLogLocked();
        if (g_log) {
            fputs(text, g_log);
            fputc('\n', g_log);
            fflush(g_log);
        }
    } catch (...) {
    }
}

void Log(const char* fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    char line[4200];
    snprintf(line, sizeof(line), "[%10.1f] %s", NowMs(), buf);
    LogRaw(line);
}

std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

fs::path ModuleDir() {
    HMODULE m = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&ModuleDir), &m);
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(m, buf, MAX_PATH);
    return n > 0 && n < MAX_PATH ? fs::path(buf).parent_path() : fs::path();
}

namespace {

bool g_baseFallback = false;

// A file can be created in `dir` (the game folder may sit where the user cannot write).
bool Writable(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    HANDLE h = CreateFileW((dir / L".qvfr_write_test").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    CloseHandle(h);
    return true;
}

}  // namespace

// Never logs: the log itself opens through it.
fs::path BaseDir() {
    if (g_base.empty()) {
        wchar_t env[MAX_PATH] = {};
        DWORD n = GetEnvironmentVariableW(L"QVFR_HOME", env, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            g_base = env;
        } else {
            g_base = ModuleDir() / L"qvfr";
            if (ModuleDir().empty() || !Writable(g_base)) {
                std::error_code ec;
                g_base = fs::temp_directory_path(ec) / L"ams2-qvfr";
                g_baseFallback = true;
            }
        }
    }
    return g_base;
}

bool BaseDirIsFallback() {
    BaseDir();
    return g_baseFallback;
}

fs::path LogDir() { return BaseDir() / L"logs"; }

fs::path LogFile() {
    std::lock_guard lock(g_logMutex);
    OpenLogLocked();
    return g_logFile;
}

fs::path IniFile() { return BaseDir() / L"qvfr.ini"; }
fs::path BenchDir() { return BaseDir() / L"bench"; }

const Config& Cfg() { return g_cfg; }

namespace {

Config ReadConfig(const fs::path& ini) {
    Config c;
    const std::string path = IniString(ini, L"runtime", L"path", L"openvr");
    if (path == "openxr") {
        c.pathOpenXr = true;
    } else if (path != "openvr") {
        Log("Settings: unknown [runtime] path=\"%s\", using openvr", path.c_str());
    }
    c.openComposite = IniPathValue(ini, L"runtime", L"opencomposite");
    c.openxrLayer = IniInt(ini, L"runtime", L"openxr_layer", 1) != 0;
    c.wrap = IniInt(ini, L"proxy", L"wrap", 1) != 0;
    c.firstCalls = IniInt(ini, L"log", L"first_calls", c.firstCalls);
    c.statsIntervalS = IniInt(ini, L"log", L"stats_interval_s", c.statsIntervalS);
    c.warningBox = IniInt(ini, L"log", L"warning_box", 1) != 0;
    c.csvAlways = IniInt(ini, L"timing", L"csv_always", 0) != 0;
    c.gpuTiming = IniInt(ini, L"timing", L"gpu", 1) != 0;
    c.benchKeyMod = IniInt(ini, L"benchmark", L"modifier", c.benchKeyMod);
    c.benchKey = IniInt(ini, L"benchmark", L"key", c.benchKey);
    c.benchDelayS = IniFloat(ini, L"benchmark", L"delay_s", c.benchDelayS);
    c.benchDurationS = IniFloat(ini, L"benchmark", L"duration_s", c.benchDurationS);
    c.benchFreezePose = IniInt(ini, L"benchmark", L"freeze_pose", 1) != 0;
    c.benchBeep = IniInt(ini, L"benchmark", L"beep", 1) != 0;
    c.benchAutoStartS = IniFloat(ini, L"benchmark", L"auto_start_s", 0.0f);
    c.gazeRead = IniInt(ini, L"gaze", L"read", 1) != 0;
    std::string mode = IniString(ini, L"experiment", L"mode", L"off");
    if (mode == "focus") {
        c.expMode = ExpMode::Focus;
    } else if (mode == "outer") {
        c.expMode = ExpMode::Outer;
    } else if (mode != "off") {
        Log("Settings: unknown [experiment] mode=\"%s\", using off", mode.c_str());
    }
    c.expFocusSize = IniFloat(ini, L"experiment", L"focus_size", c.expFocusSize);
    if (!(c.expFocusSize >= 0.1f && c.expFocusSize <= 1.0f)) {
        Log("Settings: focus_size=%.3f outside 0.1..1, using 0.5", c.expFocusSize);
        c.expFocusSize = 0.5f;
    }
    c.expCenterX = IniFloat(ini, L"experiment", L"center_x", 0.0f);
    c.expCenterY = IniFloat(ini, L"experiment", L"center_y", 0.0f);
    c.expPad = IniInt(ini, L"experiment", L"pad", 1) != 0;
    c.traceCallers = IniInt(ini, L"trace", L"callers", 0) != 0;
    c.traceDepth = IniInt(ini, L"trace", L"depth", c.traceDepth);
    if (c.traceDepth < 1 || c.traceDepth > 48) {
        Log("Settings: [trace] depth=%d outside 1..48, using 12", c.traceDepth);
        c.traceDepth = 12;
    }
    if (c.traceCallers) Log("Settings: [trace] callers=1 depth=%d (unique caller stacks of key methods)", c.traceDepth);
    std::string quad = IniString(ini, L"quad", L"mode", L"off");
    if (quad == "probe") {
        c.quadMode = QuadMode::Probe;
    } else if (quad == "quad") {
        c.quadMode = QuadMode::Quad;
    } else if (quad != "off") {
        Log("Settings: unknown [quad] mode=\"%s\", using off", quad.c_str());
    }
    c.probeFrames = IniInt(ini, L"quad", L"probe_frames", c.probeFrames);
    if (c.probeFrames < 0 || c.probeFrames > 100) {
        Log("Settings: [quad] probe_frames=%d outside 0..100, using 3", c.probeFrames);
        c.probeFrames = 3;
    }
    c.probeEngine = IniInt(ini, L"quad", L"probe_engine", 1) != 0;
    c.probeD3D = IniInt(ini, L"quad", L"probe_d3d", 1) != 0;
    c.shotKeyMod = IniInt(ini, L"quad", L"shot_modifier", c.shotKeyMod);
    c.shotKey = IniInt(ini, L"quad", L"shot_key", c.shotKey);
    c.shotAutoS = IniFloat(ini, L"quad", L"shot_auto_s", 0.0f);
    if (c.quadMode == QuadMode::Quad) ReadQuadView(ini, c);
    c.followGaze = IniInt(ini, L"quad", L"follow_gaze", 1) != 0;
    c.gazeHoldS = IniFloat(ini, L"quad", L"gaze_hold_s", c.gazeHoldS);
    if (!(c.gazeHoldS >= 0.0f && c.gazeHoldS <= 10.0f)) {
        Log("Settings: [quad] gaze_hold_s=%.2f outside 0..10, using 0.6", c.gazeHoldS);
        c.gazeHoldS = 0.6f;
    }
    c.quadCompose = IniInt(ini, L"quad", L"compose", 1) != 0;
    c.quadMarginPx = IniInt(ini, L"quad", L"margin_px", c.quadMarginPx);
    if (c.quadMarginPx < 0 || c.quadMarginPx > 512) {
        Log("Settings: [quad] margin_px=%d outside 0..512, using 0", c.quadMarginPx);
        c.quadMarginPx = 0;
    }
    if (c.quadMode != QuadMode::Off)
        Log("Settings: [quad] mode=%s preset=%s peripheral_multiplier=%.3f focus_multiplier=%.3f focus_section=%.3f x %.3f "
            "fixed_section=%.3f x %.3f focus_offset=(%.3f, %.3f) fixed_offset=(%.3f, %.3f) "
            "smoothen_focus_view_edges=%.3f%s follow_gaze=%d gaze_hold_s=%.2f compose=%d margin_px=%d probe_frames=%d "
            "probe_engine=%d probe_d3d=%d shot_modifier=0x%X shot_key=0x%X shot_auto_s=%.1f",
            QuadModeName(c.quadMode), c.preset.c_str(), c.peripheralMultiplier, c.focusMultiplier, c.focusSectionX, c.focusSectionY,
            c.fixedSectionX, c.fixedSectionY, c.focusOffsetX, c.focusOffsetY, c.fixedOffsetX, c.fixedOffsetY,
            c.smoothen, c.legacyEdge ? " (overridden by the pre-M7 feather_px)" : "", c.followGaze, c.gazeHoldS,
            c.quadCompose, c.quadMarginPx, c.probeFrames, c.probeEngine, c.probeD3D, c.shotKeyMod, c.shotKey,
            c.shotAutoS);
    if (c.quadMode == QuadMode::Quad && c.expMode != ExpMode::Off) {
        Log("Settings: [experiment] mode=%s ignored: [quad] mode=quad sets the size and projection itself",
            ExpModeName(c.expMode));
        c.expMode = ExpMode::Off;
    }
    c.overlay = IniInt(ini, L"debug", L"overlay", 0) != 0;
    c.overlayKeyMod = IniInt(ini, L"debug", L"overlay_modifier", c.overlayKeyMod);
    c.overlayKey = IniInt(ini, L"debug", L"overlay_key", c.overlayKey);
    c.freezeKeyMod = IniInt(ini, L"debug", L"freeze_modifier", c.freezeKeyMod);
    c.freezeKey = IniInt(ini, L"debug", L"freeze_key", c.freezeKey);
    c.outerKeyMod = IniInt(ini, L"debug", L"outer_modifier", c.outerKeyMod);
    c.outerKey = IniInt(ini, L"debug", L"outer_key", c.outerKey);
    c.gazeKeyMod = IniInt(ini, L"debug", L"gaze_modifier", c.gazeKeyMod);
    c.gazeKey = IniInt(ini, L"debug", L"gaze_key", c.gazeKey);
    c.clones = IniInt(ini, L"debug", L"clones", 1) != 0;
    c.clonesKeyMod = IniInt(ini, L"debug", L"clones_modifier", c.clonesKeyMod);
    c.clonesKey = IniInt(ini, L"debug", L"clones_key", c.clonesKey);
    c.clonesViewportFix = IniInt(ini, L"debug", L"clones_viewport_fix", 1) != 0;
    c.clonesBindFix = IniInt(ini, L"debug", L"clones_bind_fix", 1) != 0;
    c.shotClones = IniInt(ini, L"debug", L"shot_clones", 0) != 0;
    if (c.quadMode == QuadMode::Quad)
        Log("Settings: [debug] overlay=%d overlay_modifier=0x%X overlay_key=0x%X freeze_modifier=0x%X freeze_key=0x%X "
            "outer_modifier=0x%X outer_key=0x%X gaze_modifier=0x%X gaze_key=0x%X clones=%d clones_modifier=0x%X "
            "clones_key=0x%X clones_viewport_fix=%d clones_bind_fix=%d shot_clones=%d",
            c.overlay, c.overlayKeyMod, c.overlayKey, c.freezeKeyMod, c.freezeKey, c.outerKeyMod, c.outerKey,
            c.gazeKeyMod, c.gazeKey, c.clones, c.clonesKeyMod, c.clonesKey, c.clonesViewportFix, c.clonesBindFix,
            c.shotClones);
    c.testFaultFrames = IniInt(ini, L"test", L"fault_frames", 0);
    if (c.testFaultFrames > 0) Log("Settings: [test] fault_frames=%d (deliberate fault for testing)", c.testFaultFrames);
    return c;
}


FILETIME g_iniTime{};
double g_iniCheckMs = 0;

void NoteIniTime() {
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (GetFileAttributesExW(IniFile().c_str(), GetFileExInfoStandard, &a)) g_iniTime = a.ftLastWriteTime;
}

}  // namespace

void LoadConfig() {
    std::error_code ec;
    fs::create_directories(BaseDir(), ec);
    fs::path ini = IniFile();
    if (!fs::exists(ini, ec)) {
        if (FILE* f = _wfopen(ini.c_str(), L"wb")) {
            fwrite(kDefaultIni, 1, sizeof(kDefaultIni) - 1, f);
            fclose(f);
            Log("Created default %s", Narrow(ini.wstring()).c_str());
        }
    }
    NoteIniTime();
    Config c = ReadConfig(ini);
    g_cfg = c;
    Log("Settings %s: [runtime] path=%s opencomposite=\"%s\" openxr_layer=%d", Narrow(ini.wstring()).c_str(),
        c.pathOpenXr ? "openxr" : "openvr", Narrow(c.openComposite).c_str(), c.openxrLayer);
    Log("Settings %s: wrap=%d first_calls=%d stats_interval_s=%d csv_always=%d gpu=%d | benchmark: modifier=0x%X key=0x%X "
        "delay_s=%.1f duration_s=%.1f freeze_pose=%d beep=%d auto_start_s=%.1f | gaze read=%d | experiment: mode=%s "
        "focus_size=%.3f center=(%.3f, %.3f) pad=%d",
        Narrow(ini.wstring()).c_str(), c.wrap, c.firstCalls, c.statsIntervalS, c.csvAlways, c.gpuTiming, c.benchKeyMod,
        c.benchKey,
        c.benchDelayS, c.benchDurationS, c.benchFreezePose, c.benchBeep, c.benchAutoStartS, c.gazeRead,
        ExpModeName(c.expMode), c.expFocusSize, c.expCenterX, c.expCenterY, c.expPad);
}

bool ReloadConfig() {
    const double now = NowMs();
    if (now - g_iniCheckMs < 1000) return false;
    g_iniCheckMs = now;
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(IniFile().c_str(), GetFileExInfoStandard, &a)) return false;
    if (CompareFileTime(&a.ftLastWriteTime, &g_iniTime) == 0) return false;
    g_iniTime = a.ftLastWriteTime;
    const Config n = ReadConfig(IniFile());
    Config& c = g_cfg;
    // Settings that need a restart: the size, projection and hooks are fixed for the VR session.
    std::string restart;
    auto need = [&](bool changed, const char* name) {
        if (changed) restart += restart.empty() ? name : std::string(", ") + name;
    };
    need(n.pathOpenXr != c.pathOpenXr || n.openComposite != c.openComposite || n.openxrLayer != c.openxrLayer,
         "[runtime]");
    need(n.wrap != c.wrap, "[proxy] wrap");
    need(n.firstCalls != c.firstCalls, "[log] first_calls");
    need(n.csvAlways != c.csvAlways, "[timing] csv_always");
    need(n.gpuTiming != c.gpuTiming, "[timing] gpu");
    need(n.benchAutoStartS != c.benchAutoStartS, "[benchmark] auto_start_s");
    need(n.expMode != c.expMode || n.expFocusSize != c.expFocusSize || n.expCenterX != c.expCenterX ||
             n.expCenterY != c.expCenterY || n.expPad != c.expPad,
         "[experiment]");
    need(n.traceCallers != c.traceCallers || n.traceDepth != c.traceDepth, "[trace]");
    need(n.quadMode != c.quadMode, "[quad] mode");
    need(n.preset != c.preset, "[quad] preset");
    need(n.peripheralMultiplier != c.peripheralMultiplier || n.focusMultiplier != c.focusMultiplier,
         "[quad] *_multiplier");
    need(n.focusSectionX != c.focusSectionX || n.focusSectionY != c.focusSectionY ||
             n.fixedSectionX != c.fixedSectionX || n.fixedSectionY != c.fixedSectionY,
         "[quad] *_section");
    need(n.probeFrames != c.probeFrames || n.probeEngine != c.probeEngine || n.probeD3D != c.probeD3D, "[quad] probe_*");
    c.statsIntervalS = n.statsIntervalS;
    c.warningBox = n.warningBox;
    c.benchKeyMod = n.benchKeyMod;
    c.benchKey = n.benchKey;
    c.benchDelayS = n.benchDelayS;
    c.benchDurationS = n.benchDurationS;
    c.benchFreezePose = n.benchFreezePose;
    c.benchBeep = n.benchBeep;
    c.gazeRead = n.gazeRead;
    c.focusOffsetX = n.focusOffsetX;
    c.focusOffsetY = n.focusOffsetY;
    c.fixedOffsetX = n.fixedOffsetX;
    c.fixedOffsetY = n.fixedOffsetY;
    c.smoothen = n.smoothen;
    c.quadCenterX = n.quadCenterX;
    c.quadCenterY = n.quadCenterY;
    c.legacyEdge = n.legacyEdge;
    c.quadFeatherPx = n.quadFeatherPx;
    c.followGaze = n.followGaze;
    c.gazeHoldS = n.gazeHoldS;
    c.quadCompose = n.quadCompose;
    c.quadMarginPx = n.quadMarginPx;
    c.shotKeyMod = n.shotKeyMod;
    c.shotKey = n.shotKey;
    c.shotAutoS = n.shotAutoS;
    c.overlay = n.overlay;
    c.overlayKeyMod = n.overlayKeyMod;
    c.overlayKey = n.overlayKey;
    c.freezeKeyMod = n.freezeKeyMod;
    c.freezeKey = n.freezeKey;
    c.outerKeyMod = n.outerKeyMod;
    c.outerKey = n.outerKey;
    c.gazeKeyMod = n.gazeKeyMod;
    c.gazeKey = n.gazeKey;
    c.clones = n.clones;
    c.clonesKeyMod = n.clonesKeyMod;
    c.clonesKey = n.clonesKey;
    c.clonesViewportFix = n.clonesViewportFix;
    c.clonesBindFix = n.clonesBindFix;
    c.shotClones = n.shotClones;
    Log("Settings reloaded (live): stats_interval_s=%d | benchmark modifier=0x%X key=0x%X delay_s=%.1f duration_s=%.1f "
        "freeze_pose=%d beep=%d | gaze read=%d | quad focus_offset=(%.3f, %.3f) fixed_offset=(%.3f, %.3f) "
        "smoothen_focus_view_edges=%.3f%s follow_gaze=%d gaze_hold_s=%.2f compose=%d margin_px=%d shot_modifier=0x%X "
        "shot_key=0x%X shot_auto_s=%.1f | debug overlay=%d clones=%d clones_viewport_fix=%d clones_bind_fix=%d "
        "shot_clones=%d keys 0x%X+0x%X, 0x%X+0x%X, 0x%X+0x%X, 0x%X+0x%X, 0x%X+0x%X%s%s",
        c.statsIntervalS, c.benchKeyMod, c.benchKey, c.benchDelayS, c.benchDurationS, c.benchFreezePose, c.benchBeep,
        c.gazeRead, c.focusOffsetX, c.focusOffsetY, c.fixedOffsetX, c.fixedOffsetY, c.smoothen,
        c.legacyEdge ? " (pre-M7 feather_px instead)" : "", c.followGaze, c.gazeHoldS, c.quadCompose, c.quadMarginPx,
        c.shotKeyMod, c.shotKey, c.shotAutoS, c.overlay, c.clones, c.clonesViewportFix, c.clonesBindFix, c.shotClones,
        c.overlayKeyMod, c.overlayKey, c.freezeKeyMod, c.freezeKey, c.outerKeyMod, c.outerKey, c.gazeKeyMod, c.gazeKey,
        c.clonesKeyMod, c.clonesKey, restart.empty() ? "" : "; CHANGED BUT NEEDS A GAME RESTART: ", restart.c_str());
    return true;
}

const char* ExpModeName(ExpMode m) {
    switch (m) {
        case ExpMode::Focus: return "focus";
        case ExpMode::Outer: return "outer";
        default: return "off";
    }
}

const char* QuadModeName(QuadMode m) {
    switch (m) {
        case QuadMode::Probe: return "probe";
        case QuadMode::Quad: return "quad";
        default: return "off";
    }
}

void ReportGuardFault(const char* where, unsigned long code) {
    bool was = g_diagDisabled.exchange(true);
    if (!was) {
        Log("ERROR: exception 0x%08lX in diagnostic code (%s). Diagnostics disabled for the rest of the session, "
            "the game keeps running (an active [experiment] keeps its size, projection and pad).", code, where);
    }
}

}  // namespace qvfr

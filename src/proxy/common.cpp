#include "common.h"

#include <shlobj.h>

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
    g_log = _wfopen((LogDir() / name).c_str(), L"wb");
}

const char kDefaultIni[] =
    "; ams2-qvfr: openvr_api.dll proxy settings. The file is created on the first game start.\n"
    "; Changes take effect after restarting the game.\n"
    "\n"
    "[proxy]\n"
    "; 1 = the proxy wraps IVRSystem, IVRCompositor and IVROverlay (logs, timings, benchmark).\n"
    "; 0 = the game gets the real interfaces and the proxy only logs startup (fallback).\n"
    "wrap=1\n"
    "\n"
    "[log]\n"
    "; How many first calls of key methods (projection, size, Submit) to log with values.\n"
    "first_calls=3\n"
    "; Interval in seconds for call counters and gaze state in the log (0 = only on game exit).\n"
    "stats_interval_s=60\n"
    "\n"
    "[timing]\n"
    "; 1 = frame timing CSV for the whole session (Documents\\ams2-qvfr\\bench\\<date>_session\\), 0 = only during a benchmark.\n"
    "csv_always=0\n"
    "\n"
    "[benchmark]\n"
    "; Hotkey: modifier + key (VK codes). 0xA3 = right Ctrl, 0x78 = F9. modifier=0 means the key alone.\n"
    "; Pressing it during the countdown or the recording aborts the measurement.\n"
    "modifier=0xA3\n"
    "key=0x78\n"
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
    "; 1 = read gaze through IVRSystem_026::GetEyeTrackedFoveationCenter every frame (log and CSV).\n"
    "read=1\n"
    "\n"
    "[experiment]\n"
    "; Size and projection test without engine hooks. The game reads both once per VR session, so restart the game\n"
    "; after a change. The size reported by SteamVR is treated as the 1:1 grid.\n"
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
    "; 1 = log where in the game the key OpenVR calls come from (size, projection, eye transforms, poses, Submit, overlays):\n"
    "; every unique call stack once as module+RVA, hit counts in the stats. For reverse engineering, normally 0.\n"
    "callers=0\n"
    "; Game frames per stack (1..48).\n"
    "depth=12\n"
    "\n"
    "[quad]\n"
    "; Quad views through extra eye groups in the game engine (hooks in AMS2AVX.exe, only for the known exe build).\n"
    ";   off   = no engine hooks\n"
    ";   probe = logging hooks only (task order, render context fields, D3D11 calls per eye); the image does not change\n"
    "mode=off\n"
    "; probe: how many frames to log in full after the first in-game frame (one more every stats_interval_s).\n"
    "probe_frames=3\n";

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

fs::path BaseDir() {
    if (g_base.empty()) {
        wchar_t env[MAX_PATH] = {};
        DWORD n = GetEnvironmentVariableW(L"QVFR_HOME", env, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            g_base = env;
        } else {
            PWSTR docs = nullptr;
            if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &docs))) {
                g_base = fs::path(docs) / L"ams2-qvfr";
                CoTaskMemFree(docs);
            } else {
                std::error_code ec;
                g_base = fs::temp_directory_path(ec) / L"ams2-qvfr";
            }
        }
    }
    return g_base;
}

fs::path LogDir() { return BaseDir() / L"logs"; }
fs::path BenchDir() { return BaseDir() / L"bench"; }

const Config& Cfg() { return g_cfg; }

void LoadConfig() {
    std::error_code ec;
    fs::create_directories(BaseDir(), ec);
    fs::path ini = BaseDir() / L"qvfr.ini";
    if (!fs::exists(ini, ec)) {
        if (FILE* f = _wfopen(ini.c_str(), L"wb")) {
            fwrite(kDefaultIni, 1, sizeof(kDefaultIni) - 1, f);
            fclose(f);
            Log("Created default %s", Narrow(ini.wstring()).c_str());
        }
    }
    Config c;
    c.wrap = IniInt(ini, L"proxy", L"wrap", 1) != 0;
    c.firstCalls = IniInt(ini, L"log", L"first_calls", c.firstCalls);
    c.statsIntervalS = IniInt(ini, L"log", L"stats_interval_s", c.statsIntervalS);
    c.csvAlways = IniInt(ini, L"timing", L"csv_always", 0) != 0;
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
    } else if (quad != "off") {
        Log("Settings: unknown [quad] mode=\"%s\", using off", quad.c_str());
    }
    c.probeFrames = IniInt(ini, L"quad", L"probe_frames", c.probeFrames);
    if (c.probeFrames < 0 || c.probeFrames > 100) {
        Log("Settings: [quad] probe_frames=%d outside 0..100, using 3", c.probeFrames);
        c.probeFrames = 3;
    }
    if (c.quadMode != QuadMode::Off) Log("Settings: [quad] mode=%s probe_frames=%d", QuadModeName(c.quadMode), c.probeFrames);
    c.testFaultFrames = IniInt(ini, L"test", L"fault_frames", 0);
    if (c.testFaultFrames > 0) Log("Settings: [test] fault_frames=%d (deliberate fault for testing)", c.testFaultFrames);
    g_cfg = c;
    Log("Settings %s: wrap=%d first_calls=%d stats_interval_s=%d csv_always=%d | benchmark: modifier=0x%X key=0x%X "
        "delay_s=%.1f duration_s=%.1f freeze_pose=%d beep=%d auto_start_s=%.1f | gaze read=%d | experiment: mode=%s "
        "focus_size=%.3f center=(%.3f, %.3f) pad=%d",
        Narrow(ini.wstring()).c_str(), c.wrap, c.firstCalls, c.statsIntervalS, c.csvAlways, c.benchKeyMod, c.benchKey,
        c.benchDelayS, c.benchDurationS, c.benchFreezePose, c.benchBeep, c.benchAutoStartS, c.gazeRead,
        ExpModeName(c.expMode), c.expFocusSize, c.expCenterX, c.expCenterY, c.expPad);
}

const char* ExpModeName(ExpMode m) {
    switch (m) {
        case ExpMode::Focus: return "focus";
        case ExpMode::Outer: return "outer";
        default: return "off";
    }
}

const char* QuadModeName(QuadMode m) { return m == QuadMode::Probe ? "probe" : "off"; }

void ReportGuardFault(const char* where, unsigned long code) {
    bool was = g_diagDisabled.exchange(true);
    if (!was) {
        Log("ERROR: exception 0x%08lX in diagnostic code (%s). Diagnostics disabled for the rest of the session, "
            "the game keeps running (an active [experiment] keeps its size, projection and pad).", code, where);
    }
}

}  // namespace qvfr

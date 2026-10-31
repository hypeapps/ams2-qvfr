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
    "read=1\n";

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
    g_cfg = c;
    Log("Settings %s: wrap=%d first_calls=%d stats_interval_s=%d csv_always=%d | benchmark: modifier=0x%X key=0x%X "
        "delay_s=%.1f duration_s=%.1f freeze_pose=%d beep=%d auto_start_s=%.1f | gaze read=%d",
        Narrow(ini.wstring()).c_str(), c.wrap, c.firstCalls, c.statsIntervalS, c.csvAlways, c.benchKeyMod, c.benchKey,
        c.benchDelayS, c.benchDurationS, c.benchFreezePose, c.benchBeep, c.benchAutoStartS, c.gazeRead);
}

void ReportGuardFault(const char* where, unsigned long code) {
    bool was = g_diagDisabled.exchange(true);
    if (!was) {
        Log("ERROR: exception 0x%08lX in diagnostic code (%s). Diagnostics disabled for the rest of the session, "
            "the game keeps running through the real OpenVR.", code, where);
    }
}

}  // namespace qvfr

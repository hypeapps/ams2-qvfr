// Shared proxy pieces: log, settings, paths, call counters and exception guard.
// This header does not include openvr.h, so both worlds (0.9.12 and 2.15.6) can use it.
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>

namespace qvfr {

namespace fs = std::filesystem;

// ---------------------------------------------------------------- time and log

double NowMs();                    // since proxy start, monotonic clock
std::string LocalTimeStamp();      // YYYYMMDD_HHMMSS
void Log(const char* fmt, ...);    // one line in the log file (with time), flushed after every line
void LogRaw(const char* text);     // without the time prefix
std::string Narrow(const std::wstring& w);
std::wstring Widen(const std::string& s);

// ---------------------------------------------------------------- settings (qvfr.ini)

// off: no change; focus: narrow FOV at 1:1 density; outer: full FOV in a texture of the focus size.
enum class ExpMode { Off, Focus, Outer };
const char* ExpModeName(ExpMode m);

// off: no engine hooks; probe: logging hooks in the engine and in D3D11 (M4.1), the image does not change;
// quad: extra eye groups in the engine render the focus views (M4.2+).
enum class QuadMode { Off, Probe, Quad };
const char* QuadModeName(QuadMode m);

struct Config {
    // [proxy]
    bool wrap = true;                 // 0 = the game gets the real interfaces, the proxy only logs startup
    // [log]
    int firstCalls = 3;               // how many first calls of key methods to log with values
    int statsIntervalS = 60;          // interval of call counters and gaze state in seconds (0 = only on exit)
    // [timing]
    bool csvAlways = false;           // frame timing CSV for the whole session, not only during a benchmark
    // [benchmark]
    int benchKeyMod = VK_RCONTROL;    // modifier key (VK code, 0 = none)
    int benchKey = VK_F9;             // start and abort key
    float benchDelayS = 5.0f;         // from key press to recording start (time to settle)
    float benchDurationS = 30.0f;     // recording length
    bool benchFreezePose = true;      // constant head pose for the game while recording
    bool benchBeep = true;            // beeps: countdown start, recording start, end
    float benchAutoStartS = 0.0f;     // > 0: the benchmark starts by itself this many seconds after the first Submit
    // [gaze]
    bool gazeRead = true;             // read gaze through IVRSystem_026 every frame
    // [experiment] (M2.5: the proxy alone changes the size and projection the game gets)
    ExpMode expMode = ExpMode::Off;
    float expFocusSize = 0.5f;        // focus window as a fraction of the eye's tangent span (both axes)
    float expCenterX = 0.0f;          // focus center as a view direction tangent (right = +), same for both eyes
    float expCenterY = 0.0f;          // up = +
    bool expPad = true;               // focus: copy the game texture 1:1 into a black full-FOV texture before Submit
    // [trace] (M3: where in the game the key OpenVR calls come from)
    bool traceCallers = false;        // log every unique call stack of the key methods once (module+RVA)
    int traceDepth = 12;              // game frames per stack (proxy frames are not counted)
    // [quad] (M4: focus views through extra eye groups in the engine)
    QuadMode quadMode = QuadMode::Off;
    int probeFrames = 3;              // probe: frames logged in full after the first in-game frame
    bool probeEngine = true;          // probe: hooks in the game engine (eye tasks, builder)
    bool probeD3D = true;             // probe: hooks in the D3D11 immediate context (draw and copy counters)
    int shotKeyMod = VK_RCONTROL;     // quad: shot of the four views (PNG + pixel differences), modifier (0 = none)
    int shotKey = VK_F10;             // quad: shot key (0 = no hotkey)
    float shotAutoS = 0.0f;           // quad: > 0 = one shot this many seconds after the first frame with focus groups
    // [test] (not in the default ini; used by qvfr_proxy_smoke --fault)
    int testFaultFrames = 0;          // > 0: deliberate access violation in the diagnostics at this proxy frame
};

const Config& Cfg();
void LoadConfig();                    // creates a default qvfr.ini if it does not exist

fs::path BaseDir();                   // %QVFR_HOME% or Documents\ams2-qvfr
fs::path LogDir();
fs::path BenchDir();

// ---------------------------------------------------------------- method call counters

void TraceCall(int iface, int method);  // called from the generated wrappers
void LogCallStats(bool totals);          // increments since the previous call, or totals

// ---------------------------------------------------------------- exception guard

// Proxy diagnostics (everything except forwarding calls) run inside Guarded. An SEH exception disables
// diagnostics for the rest of the session, and the game keeps talking to the real OpenVR.
extern std::atomic<bool> g_diagDisabled;
void ReportGuardFault(const char* where, unsigned long code);

template <class F>
void Guarded(const char* where, F&& f) {
    if (g_diagDisabled.load(std::memory_order_relaxed)) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        ReportGuardFault(where, code);
    }
}

}  // namespace qvfr

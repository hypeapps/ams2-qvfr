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
// quad: extra eye groups in the engine render the focus views (M4.2+); the game renders at the focus size (M4.3+).
enum class QuadMode { Off, Probe, Quad };
const char* QuadModeName(QuadMode m);

struct Config {
    // [runtime] (M8: what stands behind the proxy; read before the backend DLL is loaded, a change needs a restart)
    bool pathOpenXr = false;          // path=openvr: the Valve DLL; path=openxr: OpenComposite to the OpenXR runtime
    std::wstring openComposite;       // opencomposite=: OpenComposite DLL (empty = the default in x64\qvfr\opencomposite)
    bool openxrLayer = true;          // openxr_layer=: our OpenXR API layer for the gaze on the OpenXR path
    // [proxy]
    bool wrap = true;                 // 0 = the game gets the real interfaces, the proxy only logs startup
    // [log]
    int firstCalls = 3;               // how many first calls of key methods to log with values
    int statsIntervalS = 60;          // interval of call counters and gaze state in seconds (0 = only on exit)
    bool warningBox = true;           // serious findings (diag.h) also in a Windows message box
    // [timing]
    bool csvAlways = false;           // frame timing CSV for the whole session, not only during a benchmark
    bool gpuTiming = true;            // GPU timestamps per frame phase (frames.csv, benchmark summary, stats)
    // [benchmark]
    int benchKeyMod = VK_RCONTROL;    // modifier key (VK code, 0 = none)
    int benchKey = 0;                 // start and abort key (testers: VK_F9; 0 = no hotkey, as for players)
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
    int shotKey = 0;                  // quad: shot key (testers: VK_F10; 0 = no hotkey)
    float shotAutoS = 0.0f;           // quad: > 0 = one shot this many seconds after the first frame with focus groups
    // quad, M8.5: preset=quality|balanced|performance gives the defaults of peripheral_multiplier and the four
    // *_section keys; a key given in the ini wins. balanced = the mbucchia defaults below.
    std::string preset = "balanced";
    // quad, M7: the settings of the mbucchia Quad-Views-Foveated layer (names, meaning, defaults and ranges).
    // Densities multiply each dimension of the SteamVR size (the 1:1 grid); sections are fractions of the eye's
    // tangent span; offsets are NDC of the full eye projection, horizontal mirrored (+ = away from the nose).
    float peripheralMultiplier = 0.5f;  // outer density
    float focusMultiplier = 1.0f;       // focus density (> 1 = supersampling)
    float focusSectionX = 0.35f;        // focus window with DFR (follow_gaze=1 at the start of the VR session)
    float focusSectionY = 0.35f;
    float fixedSectionX = 0.5f;         // focus window with FFR (follow_gaze=0 at the start of the VR session)
    float fixedSectionY = 0.45f;
    float focusOffsetX = 0.0f;          // DFR: window center = gaze + offset
    float focusOffsetY = 0.0f;
    float fixedOffsetX = 0.0f;          // FFR (and DFR without a valid gaze): window center = straight ahead + offset
    float fixedOffsetY = 0.0f;
    float smoothen = 0.2f;              // smoothen_focus_view_edges: blend band per side, fraction of the shown focus
    bool followGaze = true;           // quad: the focus window follows the gaze every frame (DFR, M5); 0 = fixed (FFR)
    float gazeHoldS = 0.6f;           // quad: invalid gaze: the last valid one for this long, then the fixed center
    bool quadCompose = true;          // quad: the proxy composes outer and focus into one grid-size image per eye (M4.4)
    int quadMarginPx = 0;             // quad: hidden band at the focus edge (guard band), grid pixels; not in mbucchia
    // Pre-M7 keys, still read with a warning: focus_size (sets the sections and peripheral_multiplier when those are
    // not given), center_x/center_y (tangents added to the fixed center) and feather_px (the old edge curve).
    float quadCenterX = 0.0f;
    float quadCenterY = 0.0f;
    bool legacyEdge = false;          // feather_px given without smoothen_focus_view_edges
    int quadFeatherPx = 0;            // with legacyEdge: blend band inside the margin, grid pixels
    // [debug] (M4.5 tools; quad mode)
    bool overlay = false;             // debug overlay in the composed image at start (blend band edges, gaze point)
    int overlayKeyMod = VK_RCONTROL;  // overlay on/off hotkey
    int overlayKey = 0;               // testers: VK_F8 (debug hotkeys are off for players, M8.5)
    int freezeKeyMod = VK_RCONTROL;   // freeze/unfreeze the gaze point
    int freezeKey = 0;                // testers: VK_F7
    int outerKeyMod = VK_RCONTROL;    // quad <-> outer only: the focus groups do not run, SteamVR gets outer alone
    int outerKey = 0;                 // testers: VK_F11
    int gazeKeyMod = VK_RCONTROL;     // focus window: follow the gaze (DFR) <-> fixed (FFR)
    int gazeKey = VK_F6;
    // Clones of the engine's eye-size targets (M7.3, M7.4): 1 = on at start; the hotkey switches.
    bool clones = true;
    int clonesKeyMod = VK_RCONTROL;
    int clonesKey = 0;                // testers: VK_F5
    // Focus culling (M9.3): the focus groups draw a scene set culled for the focus window; the hotkey switches.
    bool focusCull = true;
    int cullKeyMod = VK_RCONTROL;
    int cullKey = 0;                  // testers: VK_F4
    // Crepuscular rays and lens flares of the focus views taken from outer (sun_fx.h); the hotkey switches.
    bool sunEffects = true;
    int sunKeyMod = VK_RCONTROL;
    int sunKey = 0;                   // testers: VK_F3
    // Own TAA history of the focus views (taa_views.h, dlaa-plan D2); the hotkey switches.
    bool taaViews = true;
    int taaViewsKeyMod = VK_RCONTROL;
    int taaViewsKey = 0;              // testers: VK_F2
    // Execute time of every task of the eye groups, one log line per group with the stats (M9 step 4 diagnostics).
    bool taskProfile = false;
    // A viewport the engine sets for an original while a smaller clone is bound is scaled to the clone (M7.4).
    bool clonesViewportFix = true;
    // In an eye group with clones every bind, clear and copy of a cloned original goes to its clone, and in the other
    // eye groups a clone goes back to its original (M7.4: engine passes that keep their own D3D11 views).
    bool clonesBindFix = true;
    // Each shot also saves every focus clone as focus L left it (M7.4 diagnostics; the read-back stalls the game for
    // about 3 s).
    bool shotClones = false;
    // NVIDIA Multi-View Rendering, gate B6 (mvr_vs.h): the game's vertex shaders patched and made through NVAPI, MVR
    // off. Not in the default ini (development); a change needs a restart.
    bool mvrVsPatch = false;
    // Gate B8 (mvr_pass.h): with mvr_vs_patch, outer's G-buffer pass of each eye draws focus too (MVR, view 1) and
    // the focus groups skip theirs. Not in the default ini; a change needs a restart. The hotkey switches (live).
    bool mvrGbuffer = false;
    int mvrKeyMod = VK_RCONTROL;
    int mvrKey = 0;                   // testers: VK_END
    // Live: view 1's matrix moves outer's TAA jitter to focus's own (else focus gets outer's, scaled; gate B7).
    bool mvrJitterFix = true;
    // Live hotkey (mvr_modifier + mvr_mode_key, read from mvr_late_key when absent; testers: VK_NEXT), diagnosis of
    // the B8 flicker: the focus groups run their own G-buffer pass and get outer's on top of it afterwards.
    int mvrModeKey = 0;
    // Live: the stash's depth made conservative for focus's later LESS_EQUAL passes (forward, velocity): the farthest of
    // its 3 x 3 neighbourhood (dilate), plus mvr_depth_slope pixels' worth of its local slope, plus this many float steps.
    int mvrDepthBias = 32;
    float mvrDepthSlope = 0.0f;
    bool mvrDepthDilate = true;
    // [dlaa] (prototype: NVIDIA DLSS in DLAA mode replaces the game's TAA resolve; needs TAA on in the game and a
    // proxy built with the DLSS SDK)
    bool dlaa = false;                // mode=dlaa (a change needs a restart)
    int dlaaPreset = 0;               // DLSS render preset hint for DLAA: 0 = DLSS default, 10 = J, 11 = K, ...
    float dlaaJitterScaleX = 0.5f;    // DLSS jitter in pixels = the engine's projection offset * scale
    float dlaaJitterScaleY = -0.5f;
    bool dlaaMvJittered = false;      // the engine's motion vectors include the jitter
    // DLSS exposure: auto dims the lights (dlaa-plan D4, tests 3-5); a fixed 0.002 keeps them (the game's TAA also
    // scales its color by 0.002).
    bool dlaaAutoExposure = false;    // DLSS auto exposure (0 = dlaaExposure)
    float dlaaExposure = 0.002f;      // with auto exposure off: fixed DLSS exposure (0 = DLSS default 1)
    float dlaaExposureScale = 1.0f;   // DLSS exposure scale (multiplies the auto or fixed exposure)
    float dlaaBiasMask = 0.0f;        // DLSS bias-current-color mask value at stencil bit 0x200 pixels (0 = no mask)
    bool dlaaSkipLocks = true;        // skip the engine's TAA locks pass while DLAA replaces the resolve
    int dlaaKeyMod = VK_RCONTROL;     // DLAA <-> the game's TAA (A/B comparison)
    int dlaaKey = 0;                  // testers: VK_F1 (F4 is focus culling in quad mode)
    int dlaaExpKeyMod = VK_RCONTROL;  // DLSS exposure steps (testers, dlaa.cpp kExpSteps)
    int dlaaExpKey = 0;               // testers: VK_F12
    // From the game's TAA instead of DLSS: 0 nothing, 1 displays and mirrors (stencil bit 0x800), 2 also the rest of the
    // player's car (bit 0x200). DLSS smears changing digits (dlaa-plan D4, tests 2-7).
    int dlaaCockpitTaa = 1;
    int dlaaPresetKeyMod = VK_RCONTROL;  // DLSS preset steps: ini, J, K, M, L (testers)
    int dlaaPresetKey = 0;               // testers: VK_HOME
    // [test] (not in the default ini; used by qvfr_proxy_smoke --fault)
    int testFaultFrames = 0;          // > 0: deliberate access violation in the diagnostics at this proxy frame
};

const Config& Cfg();
void LoadConfig();                    // creates a default qvfr.ini if it does not exist
// Render thread, every frame: at most once a second checks the ini date; a changed file updates the live settings
// (listed at the end of the default ini) and logs the ones that need a restart. True when it reloaded.
bool ReloadConfig();

fs::path ModuleDir();                 // folder of the proxy DLL (the game's x64\)
fs::path BaseDir();                   // %QVFR_HOME% (tests) or x64\qvfr; %TEMP%\ams2-qvfr when x64\qvfr is not writable
bool BaseDirIsFallback();             // BaseDir is the %TEMP% fallback
fs::path LogDir();
fs::path LogFile();                   // this session's log
fs::path IniFile();                   // qvfr.ini
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

#include "diag.h"

#include <windows.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "common.h"
#include "hooks.h"
#include "sysinfo.h"

namespace qvfr::diag {

namespace {

constexpr double kDetectedS = 20.0;  // the gaze needs a few seconds with the headset on

std::mutex g_m;
std::set<std::string> g_ids;
std::vector<std::pair<std::string, std::string>> g_facts;
std::vector<std::string> g_serious;  // texts of the serious findings, in order
size_t g_shown = 0;                  // how many of them a message box already had
int g_warnings = 0, g_notes = 0;

// The Detected block lists these keys first, in this order; any other key follows.
const char* const kOrder[] = {"Game", "Game settings", "Path", "Backend", "OpenXR active runtime", "OpenXR runtime",
                              "Headset", "Gaze", "Focus window", "Quad", "OpenXR layers", "OpenXR mod layer",
                              "OpenXR Toolkit", "Modules"};

// Render thread only (OnFrame), except the flag that OnShutdown also reads.
double g_firstFrameMs = -1;
uint64_t g_frames = 0, g_gazeFrames = 0, g_gazeOk = 0;
uint64_t g_src[4] = {};  // quad::SrcLive, SrcHeld, SrcFixed, SrcFrozen
const char* g_gazeSource = nullptr;
std::atomic<bool> g_detected{false};
std::string (*g_quadDescribe)() = nullptr;

std::string Percent(uint64_t part, uint64_t whole) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f%%", whole ? 100.0 * (double)part / (double)whole : 0.0);
    return buf;
}

void WriteDetected(const char* when) {
    if (g_detected.exchange(true)) return;
    const Config& c = Cfg();
    const bool quadOn = c.quadMode == QuadMode::Quad;
    char buf[512];
    if (g_frames == 0) {
        Set("Gaze", "no frames yet");
    } else if (g_gazeSource) {
        snprintf(buf, sizeof(buf), "%s, valid in %s of %llu frames", g_gazeSource, Percent(g_gazeOk, g_gazeFrames).c_str(),
                 (unsigned long long)g_gazeFrames);
        Set("Gaze", buf);
    } else {
        Set("Gaze", !c.gazeRead ? "not read ([gaze] read=0)"
                                : "no source (path=openvr: SteamVR has no IVRSystem_026; path=openxr: the OpenXR layer "
                                  "is not loaded)");
    }
    const bool noGaze = g_gazeFrames == 0 || g_gazeOk * 100 < g_gazeFrames;  // valid in under 1% of the frames
    if (quadOn && c.followGaze && g_frames > 0 && noGaze)
        Report(Level::Info, "no-gaze",
               "No valid gaze in the first %.0f s: a headset without eye tracking, eye tracking off or not calibrated, or "
               "the headset not worn yet. Without a gaze the focus window stays straight ahead (FFR).",
               kDetectedS);
    const uint64_t windows = g_src[0] + g_src[1] + g_src[2] + g_src[3];
    if (windows) {
        snprintf(buf, sizeof(buf), "follows the gaze in %s of the frames, held at the last gaze %s, fixed %s, frozen %s",
                 Percent(g_src[0], windows).c_str(), Percent(g_src[1], windows).c_str(), Percent(g_src[2], windows).c_str(),
                 Percent(g_src[3], windows).c_str());
        Set("Focus window", buf);
    }
    if (!quadOn)
        Set("Quad", std::string("off ([quad] mode=") + QuadModeName(c.quadMode) + ")");
    else if (g_quadDescribe) {
        std::string q = g_quadDescribe();
        if (q.rfind("Quad: ", 0) == 0) q.erase(0, 6);
        Set("Quad", q);
    }
    const std::string mods = sysinfo::KnownModules();
    Set("Modules", mods.empty() ? "none of the known ones" : mods);

    std::vector<std::pair<std::string, std::string>> facts;
    int warnings, notes;
    {
        std::lock_guard lock(g_m);
        for (const char* k : kOrder)
            for (const auto& f : g_facts)
                if (f.first == k) facts.push_back(f);
        for (const auto& f : g_facts) {
            bool listed = false;
            for (const char* k : kOrder) listed |= f.first == k;
            if (!listed) facts.push_back(f);
        }
        warnings = g_warnings;
        notes = g_notes;
    }
    Log("Detected (%s):", when);
    for (const auto& f : facts) Log("  %s: %s", f.first.c_str(), f.second.c_str());
    if (warnings || notes)
        Log("  Findings: %d warning(s), %d note(s); see the WARNING and Note lines", warnings, notes);
    else
        Log("  Findings: none");
    ShowPendingBox();
}

}  // namespace

void Report(Level level, const char* id, const char* fmt, ...) {
    char text[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    {
        std::lock_guard lock(g_m);
        if (!g_ids.insert(id).second) return;
        if (level == Level::Info) {
            ++g_notes;
        } else {
            ++g_warnings;
        }
        if (level == Level::Serious) g_serious.push_back(text);
    }
    Log("%s: %s", level == Level::Info ? "Note" : "WARNING", text);
}

void Set(const char* key, const std::string& value) {
    std::lock_guard lock(g_m);
    for (auto& f : g_facts) {
        if (f.first == key) {
            f.second = value;
            return;
        }
    }
    g_facts.emplace_back(key, value);
}

void ShowPendingBox() {
    std::vector<std::string> fresh;
    {
        std::lock_guard lock(g_m);
        if (g_shown >= g_serious.size()) return;
        fresh.assign(g_serious.begin() + (ptrdiff_t)g_shown, g_serious.end());
        g_shown = g_serious.size();
    }
    if (!Cfg().warningBox) {
        Log("Warning box: off ([log] warning_box=0), %zu serious finding(s) only in the log", fresh.size());
        return;
    }
    std::wstring text = L"ams2-qvfr found a problem with this game start:\n";
    for (const auto& s : fresh) text += L"\n\x2022 " + Widen(s) + L"\n";
    text += L"\nDetails: " + LogFile().wstring() +
            L"\nThe game keeps running. To stop these messages, set warning_box=0 in [log] of " + IniFile().wstring();
    Log("Warning box: shown with %zu finding(s)", fresh.size());
    // Its own thread: the game (and its VR frames) never waits for the player to click OK.
    std::thread([text] { MessageBoxW(nullptr, text.c_str(), L"ams2-qvfr", MB_OK | MB_ICONWARNING | MB_TOPMOST); }).detach();
}

void SetQuadDescribe(std::string (*describe)()) { g_quadDescribe = describe; }

void CheckStartup() {
    const Config& c = Cfg();
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::string name = Narrow(fs::path(exe).filename().wstring());
    if (c.quadMode == QuadMode::Quad) {
        hooks::Init();  // identifies the exe and its build and reports both
    } else {
        Set("Game", name + ", build not checked ([quad] mode=" + QuadModeName(c.quadMode) + ")");
    }
    const std::string aa = sysinfo::GameSetting("antialias"), taa = sysinfo::GameSetting("temporalantialiasenabled");
    if (aa.empty() && taa.empty()) {
        Set("Game settings", "graphicsconfigopenvrdx11.xml not found or without anti-aliasing settings");
    } else {
        Set("Game settings", "antialias=" + aa + " (MSAA), temporalantialiasenabled=" + taa + (taa == "1" ? " (TAA ON)" : ""));
        if (taa == "1" && c.quadMode == QuadMode::Quad)
            Report(Level::Serious, "taa",
                   "Temporal anti-aliasing (TAA) is on in the game. The mod is made for MSAA: with TAA the focus area can "
                   "smear or flicker. Turn TAA off in the game's graphics options and use MSAA.");
    }
    ShowPendingBox();
}

void NoteOpenVrSystem(vr::IVRSystem* real) {
    if (!real) return;
    auto prop = [&](vr::ETrackedDeviceProperty p) {
        char buf[256] = {};
        vr::TrackedPropertyError e = vr::TrackedProp_Success;
        real->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, p, buf, sizeof(buf), &e);
        return e == vr::TrackedProp_Success ? std::string(buf) : std::string("?");
    };
    Set("Headset", "\"" + prop(vr::Prop_ModelNumber_String) + "\" by \"" + prop(vr::Prop_ManufacturerName_String) +
                       "\" (SteamVR driver \"" + prop(vr::Prop_TrackingSystemName_String) + "\")");
}

void CheckModule(const std::wstring& fileNameLower) {
    if (fileNameLower.find(L"libmagic") != std::wstring::npos)
        Report(Level::Serious, "libmagic",
               "Pimax LibMagic (%s, Foveated rendering in Pimax Play) is loaded into the game. It does not work with AMS2 "
               "and can break the mod's image: turn off Foveated rendering in Pimax Play.",
               Narrow(fileNameLower).c_str());
}

void OnFrame(const char* gazeSource, bool gazeOk, int windowSrc) {
    if (g_detected.load(std::memory_order_relaxed)) return;
    const double now = NowMs();
    if (g_firstFrameMs < 0) g_firstFrameMs = now;
    ++g_frames;
    if (gazeSource) {
        g_gazeSource = gazeSource;
        ++g_gazeFrames;
        g_gazeOk += gazeOk;
    }
    if (windowSrc >= 0 && windowSrc < 4) ++g_src[windowSrc];
    if (now - g_firstFrameMs >= kDetectedS * 1000) {
        char when[64];
        snprintf(when, sizeof(when), "%.0f s after the first frame", kDetectedS);
        WriteDetected(when);
    }
}

void OnShutdown() {
    char when[96];
    if (g_firstFrameMs < 0)
        snprintf(when, sizeof(when), "at VR_Shutdown, no frames");
    else
        snprintf(when, sizeof(when), "at VR_Shutdown, %.0f s after the first frame", (NowMs() - g_firstFrameMs) / 1000);
    WriteDetected(when);
}

}  // namespace qvfr::diag

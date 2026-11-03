// qvfr_config --selftest <dir>: the ini, settings, log and file code without a window (run by the agent after a build).

#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "files.h"
#include "ini_file.h"
#include "log_reader.h"
#include "settings.h"

namespace fs = std::filesystem;
using namespace qvfr::config;

namespace {

int g_fail = 0;

void Expect(bool ok, const char* what) {
    printf("%s %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) ++g_fail;
}

bool Near(float a, float b) { return a > b - 1e-4f && a < b + 1e-4f; }

int CountComments(const std::string& t) {
    int n = 0;
    size_t p = 0;
    while ((p = t.find("\n;", p)) != std::string::npos) ++n, ++p;
    return n;
}

void Write(const fs::path& p, const std::string& text) {
    std::ofstream o(p, std::ios::binary);
    o << text;
}

}  // namespace

int RunSelfTest(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / L"x64" / L"qvfr" / L"logs", ec);
    const fs::path mod = dir / L"x64" / L"qvfr";
    printf("qvfr_config self-test in %ls\n", dir.c_str());

    IniFile ini;
    ini.LoadText(DefaultIniText());
    const int comments = CountComments(ini.Text());
    Settings s = Read(ini);
    Expect(s.quad && !s.openxr && s.preset == 1 && !s.Custom(), "default ini: quad, openvr, balanced, no overrides");
    Expect(Near(s.peripheral, 0.5f) && Near(s.focusX, 0.35f) && Near(s.fixedY, 0.45f) && Near(s.focusMultiplier, 1.0f),
          "default ini: balanced values");
    Expect(s.gazeKey == 0x75 && s.benchKey == 0 && s.outerKey == 0 && s.warningBox, "default ini: player hotkeys");

    Expect(ini.Set("quad", "peripheral_multiplier", Num(0.4f)), "set an empty key");
    s = Read(ini);
    Expect(s.givenPeripheral && Near(s.peripheral, 0.4f) && s.Custom(), "override wins over the preset");
    Expect(!ini.Set("quad", "peripheral_multiplier", "0.4"), "same value: no change");

    WritePreset(ini, 2);
    s = Read(ini);
    Expect(s.preset == 2 && !s.Custom() && Near(s.peripheral, 0.4f) && Near(s.focusX, 0.3f) && Near(s.fixedX, 0.4f),
          "preset performance clears the overrides");
    Expect(ini.GetOr("quad", "preset", "") == "performance", "preset written");
    Expect(CountComments(ini.Text()) == comments, "comments kept");

    IniFile before = ini;
    ini.Set("quad", "horizontal_focus_offset", Num(0.1f));
    Expect(ChangedRestartKeys(before, ini).empty(), "an offset needs no restart");
    ini.Set("quad", "focus_multiplier", Num(1.3f));
    const auto changed = ChangedRestartKeys(before, ini);
    Expect(changed.size() == 1 && changed[0] == "focus resolution", "focus resolution needs a restart");

    ini.Set("quad", "new_key", "1");
    ini.Set("newsection", "a", "b");
    std::string v;
    Expect(ini.Get("quad", "new_key", v) && v == "1" && ini.Get("newsection", "a", v) && v == "b", "keys and sections added");
    Expect(Num(0.35f) == "0.35" && Num(1.0f) == "1.0" && Num(-0.1f) == "-0.1" && Num(0.0f) == "0.0", "number format");

    const fs::path iniPath = mod / L"qvfr.ini";
    Expect(ini.Save(iniPath), "atomic save");
    IniFile back;
    Expect(back.Load(iniPath) && back.Text() == ini.Text(), "load what was saved");
    Expect(GetPrivateProfileIntW(L"quad", L"new_key", 0, iniPath.c_str()) == 1, "Windows profile API reads the saved file");

    Write(mod / L"logs" / L"qvfr_20261102_100000_AMS2AVX.log", "[ 1.0] old\n");
    Write(mod / L"logs" / L"qvfr_20261102_201721_AMS2AVX.log",
          "[       6.3] Settings C:\\x\\qvfr.ini: [runtime] path=openxr opencomposite=\"\" openxr_layer=1\n"
          "[     181.2] IVRSystem_026::GetRecommendedRenderTargetSize = 8192 x 6864\n"
          "[     595.4] IVRSystem::GetRecommendedRenderTargetSize = 3277 x 2746 (quad focus size; SteamVR 8192 x 6864)  [call 1]\n"
          "[     600.0] WARNING: The game runs as X.\n"
          "[   27527.9] Detected (20 s after the first frame):\n"
          "[   27527.9]   Game: AMS2AVX.exe, supported build\n"
          "[   27527.9]   Gaze: SteamVR, valid in 100.0% of 968 frames\n"
          "[   27527.9]   Findings: 1 warning(s), 0 note(s)\n"
          "[   27600.0] Note: something.\n");
    const LogInfo log = ReadNewestLog(mod / L"logs");
    Expect(log.found && log.file.filename() == L"qvfr_20261102_201721_AMS2AVX.log" && log.when == "2026-11-02 20:17:21", "newest log");
    Expect(log.gridW == 8192 && log.gridH == 6864 && log.openxr, "SteamVR resolution and path from the log");
    Expect(log.detectedWritten && log.detected.size() == 3 && log.detected[1].rfind("Gaze: ", 0) == 0, "Detected block");
    Expect(log.findings.size() == 2 && log.findings[0] == "WARNING: The game runs as X." && log.findings[1] == "Note: something.",
          "findings");

    // Files: a fake game folder. The spare proxy and the one in x64 differ, then Repair makes them the same.
    Write(dir / L"AMS2AVX.exe", "exe");
    fs::create_directories(mod / L"proxy", ec);
    Write(mod / L"proxy" / L"openvr_api.dll", "MZ...ams2-qvfr openvr_api.dll proxy, build B");
    Write(dir / L"x64" / L"openvr_api.dll", "MZ...ams2-qvfr openvr_api.dll proxy, build A");
    FileState f = CheckFiles(mod, false);
    Expect(f.gameFolder && f.spare && f.gameDll == GameDll::ProxyOther, "files: an older proxy in x64");
    Write(dir / L"x64" / L"openvr_api.dll", "MZ... OpenComposite ...");
    Expect(CheckFiles(mod, false).gameDll == GameDll::OpenComposite, "files: OpenComposite in x64");
    if (GameRunning()) {
        printf("  skip  Repair (the game is running)\n");
    } else {
        const std::string msg = RepairProxy(mod);
        f = CheckFiles(mod, false);
        Expect(f.gameDll == GameDll::Proxy, ("Repair: " + msg).c_str());
        size_t kept = 0;
        for (const auto& e : fs::directory_iterator(mod / L"backup", ec)) kept += e.path().extension() == L".dll";
        Expect(kept == 1, "Repair kept the OpenComposite DLL in backup");
    }
    printf("%s (%d failures)\n", g_fail ? "qvfr_config self-test FAILED" : "qvfr_config self-test OK", g_fail);
    return g_fail ? 1 : 0;
}

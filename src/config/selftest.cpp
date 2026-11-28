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
    Expect(s.quad && s.openxr && s.preset == 1 && !s.Custom(), "default ini: quad, openxr, balanced, no overrides");
    Expect(Near(s.peripheral, 0.5f) && Near(s.focusX, 0.35f) && Near(s.fixedY, 0.45f) && Near(s.focusMultiplier, 1.0f),
          "default ini: balanced values");
    Expect(s.gazeKey == 0x75 && s.benchKey == 0 && s.outerKey == 0 && s.warningBox, "default ini: player hotkeys");
    Expect(!s.dlss && !s.dlssUpscale && s.dlssPreset == "k" && s.dlssOuterPreset == "f" && s.dlssFocusPreset == "k" &&
               !s.dlaaAutoExposure && Near(s.dlaaExposure, 0.002f) && s.dlaaKey == 0 && s.dlaaExpKey == 0,
           "default ini: DLSS off, preset K, the periphery F, fixed exposure 0.002, no DLSS hotkeys");
    Expect(s.mvr, "default ini: NVIDIA Multi-View Rendering on");
    {
        IniFile d = ini;
        d.Set("mvr", "mode", "off");
        const auto ch = ChangedRestartKeys(ini, d);
        Expect(!Read(d).mvr && ch.size() == 1 && ch[0] == "NVIDIA Multi-View Rendering on/off",
               "MVR off needs a restart");
    }
    {
        IniFile d = ini;
        d.Set("dlaa", "mode", "dlaa");
        const auto ch = ChangedRestartKeys(ini, d);
        Expect(Read(d).dlss && !Read(d).dlssUpscale && ch.size() == 1 && ch[0] == "NVIDIA DLSS", "DLAA on needs a restart");
    }
    {
        // The DLSS options of the Rendering card, in quad views.
        IniFile d = ini;
        WriteDlss(d, true);
        Settings n = Read(d);
        Expect(d.GetOr("dlaa", "mode", "") == "dlaa" && DlssModeIndex(n.dlssOuterScale) == 0 && DlssModeIndex(n.dlssFocusScale) == 0,
               "DLSS on: DLAA in both views");
        WriteDlssScale(d, DlssView::Peripheral, kDlssModes[1].scale);
        n = Read(d);
        Expect(d.GetOr("dlaa", "mode", "") == "dlss" && DlssModeIndex(n.dlssOuterScale) == 1 && DlssModeIndex(n.dlssFocusScale) == 0,
               "peripheral Quality: mode=dlss, the focus stays DLAA");
        auto ch = ChangedRestartKeys(ini, d);
        Expect(ch.size() == 2 && ch[0] == "NVIDIA DLSS" && ch[1] == "DLSS mode of the periphery", "a DLSS mode needs a restart");
        WriteDlss(d, false);
        Expect(!Read(d).dlss && !Read(d).dlssUpscale, "DLSS off");
        WriteDlss(d, true);
        n = Read(d);
        Expect(n.dlssUpscale && DlssModeIndex(n.dlssOuterScale) == 1, "DLSS on again: the chosen modes are back");
        WriteDlssScale(d, DlssView::Peripheral, kDlssModes[0].scale);
        Expect(d.GetOr("dlaa", "mode", "") == "dlaa", "both views DLAA: mode=dlaa");
        WriteDlssPreset(d, DlssView::Focus, "m");
        n = Read(d);
        Expect(n.dlssFocusPreset == "m" && n.dlssOuterPreset == "f" && n.dlssPreset == "k", "a preset for the focus only");
        IniFile e = d;
        WriteDlssPreset(e, DlssView::Peripheral, "j");
        Expect(Read(e).dlssOuterPreset == "j" && ChangedRestartKeys(d, e).empty(), "a preset needs no restart");
        WriteDlssPreset(e, DlssView::Focus, "f");
        Expect(Read(e).dlssFocusPreset == "f" && e.GetOr("dlaa", "focus_preset", "") == "f", "preset F for the focus");
        // A game started with upscaling: its modes change live, also to DLAA in both views.
        WriteDlssScale(d, DlssView::Focus, kDlssModes[1].scale);
        e = d;
        WriteDlssScale(e, DlssView::Peripheral, kDlssModes[3].scale);
        Expect(ChangedRestartKeys(d, e).empty(), "started with upscaling: another mode needs no restart");
        WriteDlssScale(e, DlssView::Peripheral, kDlssModes[0].scale);
        WriteDlssScale(e, DlssView::Focus, kDlssModes[0].scale);
        Expect(e.GetOr("dlaa", "mode", "") == "dlaa" && ChangedRestartKeys(d, e).empty(),
               "started with upscaling: DLAA in both views needs no restart");
        WriteDlss(e, false);
        Expect(ChangedRestartKeys(d, e) == std::vector<std::string>{"NVIDIA DLSS"}, "started with upscaling: DLSS off needs a restart");
    }
    {
        // An ini made by hand: scales without their own keys, stale scales with mode=dlaa, a scale of no mode.
        IniFile d;
        d.LoadText("[quad]\nmode=quad\n[dlaa]\nmode=dlss\n");
        Settings n = Read(d);
        Expect(n.dlssUpscale && Near(n.dlssOuterScale, 0.67f) && Near(n.dlssFocusScale, 0.67f), "mode=dlss without scales: the proxy's 0.67");
        d.LoadText("[quad]\nmode=quad\n[dlaa]\nmode=dlss\nrender_scale=0.5\nfocus_render_scale=0.75\npreset=default\n");
        n = Read(d);
        Expect(DlssModeIndex(n.dlssOuterScale) == 3 && DlssModeIndex(n.dlssFocusScale) == -1 && n.dlssFocusPreset == "k",
               "render_scale as the fallback, a custom scale, preset default as K");
        d.LoadText("[quad]\nmode=quad\n[dlaa]\nmode=dlaa\nouter_render_scale=0.67\nfocus_render_scale=0.67\n");
        n = Read(d);
        Expect(n.dlss && !n.dlssUpscale && Near(n.dlssOuterScale, 1) && Near(n.dlssFocusScale, 1), "mode=dlaa shows DLAA whatever the scales");
        WriteDlssScale(d, DlssView::Focus, 0.5f);
        n = Read(d);
        Expect(n.dlssUpscale && Near(n.dlssFocusScale, 0.5f) && Near(n.dlssOuterScale, 1), "from mode=dlaa: the other view stays as shown");
        WriteDlss(d, false);
        d.Set("nr", "mode", "final");
        d.Set("dlaa", "mode", "dlaa");
        WriteDlss(d, false);
        WriteDlss(d, true);
        Expect(d.GetOr("dlaa", "mode", "") == "dlaa", "DLSS off from mode=dlaa: on again gives DLAA, not the old scales");
        d.LoadText("[quad]\nmode=off\n[dlaa]\nmode=dlss\nrender_scale=0.58\nouter_render_scale=1.0\nfocus_render_scale=1.0\n");
        n = Read(d);
        Expect(DlssModeIndex(n.dlssScale) == 2, "stereo: render_scale");
        d.Set("quad", "mode", "quad");
        SyncDlssMode(d);
        Expect(d.GetOr("dlaa", "mode", "") == "dlaa", "quad views on with nothing to upscale: mode=dlaa");
    }
    Expect(!s.nr && Near(s.nrIntensity, 1) && Near(s.nrStrength, 1) && s.nrStyle == 0 && s.nrPreset == 0 &&
               s.nrAutoMask && s.nrOuter && s.nrOuterStep == 2 && Near(s.nrOuterArea, 1) && !s.nrOuterSeparate,
           "default ini: DLSS 5 off, the proxy's defaults (outer on at 1/2)");
    {
        IniFile d = ini;
        WriteNr(d, true);
        Settings n = Read(d);
        const auto ch = ChangedRestartKeys(ini, d);
        Expect(n.nr && n.dlss && d.GetOr("nr", "mode", "") == "final" && ch.size() == 1 && ch[0] == "NVIDIA DLSS",
               "DLSS 5 on: [nr] mode=final and DLSS on (only DLSS needs a restart)");
        WriteNr(d, false);
        n = Read(d);
        Expect(!n.nr && n.dlss, "DLSS 5 off: DLSS stays on");
        WriteDlssScale(d, DlssView::Focus, kDlssModes[2].scale);
        WriteNr(d, true);
        Expect(d.GetOr("dlaa", "mode", "") == "dlss" && DlssModeIndex(Read(d).dlssFocusScale) == 2,
               "DLSS 5 on keeps the chosen DLSS mode");
        WriteDlss(d, false);
        n = Read(d);
        Expect(!n.dlss && !n.nr, "DLSS off turns DLSS 5 off too");
        d.Set("nr", "outer_resolution", "0.333");
        d.Set("nr", "outer_atlas", "separate");
        d.Set("nr", "intensity", "1.5");
        n = Read(d);
        Expect(n.nrOuterStep == 3 && n.nrOuterSeparate && Near(n.nrIntensity, 1.5f), "DLSS 5 options read back");
    }

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
    Write(mod / L"logs" / L"qvfr_sboys_driver.log", "[ 1.0] Note: No valid gaze in a saved copy.\n");
    Write(mod / L"logs" / L"qvfr_20261102_201721_AMS2AVX.log",
          "[       6.3] Settings C:\\x\\qvfr.ini: [runtime] path=openxr opencomposite=\"\" openxr_layer=1\n"
          "[     181.2] IVRSystem_026::GetRecommendedRenderTargetSize = 8192 x 6864\n"
          "[     595.4] IVRSystem::GetRecommendedRenderTargetSize = 3277 x 2746 (quad focus size; SteamVR 8192 x 6864)  [call 1]\n"
          "[     600.0] WARNING: The game runs as X.\n"
          "[   27527.9] Detected (20 s after the first frame):\n"
          "[   27527.9]   Game: AMS2AVX.exe, supported build\n"
          "[   27527.9]   Gaze: SteamVR, valid in 100.0% of 968 frames\n"
          "[   27527.9]   DLAA: on (waiting for the first TAA frame)\n"
          "[   27528.0]   Findings: 1 warning(s), 0 note(s)\n"
          "[   27600.0] Note: something.\n"
          "[   27700.0] Detected update: DLAA: on in quad views (DLSS Super Resolution: focus 67%, peripheral 50%)\n");
    const LogInfo log = ReadNewestLog(mod / L"logs");
    Expect(log.found && log.file.filename() == L"qvfr_20261102_201721_AMS2AVX.log" && log.when == "2026-11-02 20:17:21", "newest log");
    Expect(log.gridW == 8192 && log.gridH == 6864 && log.openxr, "SteamVR resolution and path from the log");
    Expect(log.detectedWritten && log.detected.size() == 4 && log.detected[1].rfind("Gaze: ", 0) == 0, "Detected block");
    Expect(log.detected[2] == "DLAA: on in quad views (DLSS Super Resolution: focus 67%, peripheral 50%)",
           "Detected update replaces its line");
    Expect(log.findings.size() == 2 && log.findings[0] == "WARNING: The game runs as X." && log.findings[1] == "Note: something.",
          "findings");

    // Files: a fake game folder. The spare proxy and the one in x64 differ, then Repair makes them the same.
    Write(dir / L"AMS2AVX.exe", "exe");
    fs::create_directories(mod / L"proxy", ec);
    Write(mod / L"proxy" / L"openvr_api.dll", "MZ...ams2-qvfr openvr_api.dll proxy, build B");
    Write(dir / L"x64" / L"openvr_api.dll", "MZ...ams2-qvfr openvr_api.dll proxy, build A");
    FileState f = CheckFiles(mod, false);
    Expect(f.gameFolder && f.spare && f.gameDll == GameDll::ProxyOther, "files: an older proxy in x64");
    auto dlssRow = [](const FileState& st) {
        for (const Check& c : st.checks)
            if (c.item == "x64\\qvfr\\nvngx_dlss.dll") return c.status;
        return Status::Ok;
    };
    Expect(!f.dlss && dlssRow(f) == Status::Info && dlssRow(CheckFiles(mod, false, true)) == Status::Error,
           "files: nvngx_dlss.dll missing is a note without DLAA, an error with it");
    Write(mod / L"nvngx_dlss.dll", "MZ dlss");
    f = CheckFiles(mod, false, true);
    Expect(f.dlss && dlssRow(f) == Status::Ok, "files: nvngx_dlss.dll present");
    auto rowOf = [](const FileState& st, const char* item) {
        for (const Check& c : st.checks)
            if (c.item == item) return (int)c.status;
        return -1;
    };
    Expect(rowOf(f, "x64\\qvfr\\nvngx_dlssnr.dll") == -1 &&
               rowOf(CheckFiles(mod, false, true, true), "x64\\qvfr\\nvngx_dlssnr.dll") == (int)Status::Error,
           "files: the DLSS 5 model is no row without DLSS 5, an error with it");
    Write(mod / L"nvngx_dlssnr.dll", "MZ nr");
    Write(mod / L"nvngx.dll_qvfr_nr.dll", "MZ fwd");
    f = CheckFiles(mod, false, true, true);
    Expect(f.nrModel && f.nrForwarder && rowOf(f, "x64\\qvfr\\nvngx_dlssnr.dll") == (int)Status::Ok &&
               rowOf(f, "x64\\qvfr\\nvngx.dll_qvfr_nr.dll") == (int)Status::Ok,
           "files: the DLSS 5 files present");
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

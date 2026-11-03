// qvfr_config: the settings GUI of ams2-qvfr (M8.6, docs/handoff-m86-gui.md). Lives in the game's x64\qvfr\ next to
// qvfr.ini and the logs. Pages: Settings (written to qvfr.ini at once; the proxy picks up the live keys within about a
// second), Resolution (outer and focus over the field of view, from the SteamVR resolution in the newest log), Status
// (the newest log's Detected block and findings), Files (check and repair) and Log (the end of the newest log).
//
// The window is RmlUi (HTML/CSS-like): ui/main.rml + ui/style.rcss and the fonts are resources of the exe; this file
// binds them to the settings code through one data model ("s").
//
//   qvfr_config.exe                   the mod folder is the exe's folder
//   qvfr_config.exe --home <folder>   another mod folder (tests)
//   qvfr_config.exe --tab <0..4>      open on that page (screenshots)
//   qvfr_config.exe --ui <folder>     load the .rml and .rcss files from that folder (styling without a rebuild)
//   qvfr_config.exe --selftest <dir>  tests of the ini, settings, log and file code in <dir> without a window
// QVFR_GUI_TRACE=<file> logs mouse events and their targets (for tests that click through window messages).

#include <windows.h>

#include <dwmapi.h>
#include <shellapi.h>

#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "RmlUi_Backend.h"
#include "files.h"
#include "ini_file.h"
#include "log_reader.h"
#include "presets.h"
#include "settings.h"

namespace fs = std::filesystem;
using namespace qvfr::config;
namespace presets = qvfr::presets;

int RunSelfTest(const fs::path& dir);  // selftest.cpp

namespace {

const char* kTitle = "AMS2 QVFR";  // also the window class name (the backend uses the title)

double Now() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string ReadFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string Fmt(const char* f, ...) {
    char b[256];
    va_list a;
    va_start(a, f);
    vsnprintf(b, sizeof(b), f, a);
    va_end(a);
    return b;
}

// ---------------------------------------------------------------- files: UI from resources, everything else from disk

class Files : public Rml::FileInterface {
public:
    fs::path uiDir;  // --ui: the UI files from disk

    Rml::FileHandle Open(const Rml::String& path) override {
        auto f = std::make_unique<Open_>();
        const fs::path p = fs::u8path(path);
        const std::string ext = p.extension().string();
        if (ext == ".rml" || ext == ".rcss") {
            if (!uiDir.empty()) {
                f->data = ReadFile(uiDir / p.filename());
            } else {
                std::wstring name = p.filename().wstring();  // main.rml -> MAIN_RML (qvfr_config.rc)
                for (wchar_t& c : name) c = c == L'.' ? L'_' : (wchar_t)towupper(c);
                HRSRC r = FindResourceW(nullptr, name.c_str(), MAKEINTRESOURCEW(10));  // RT_RCDATA
                HGLOBAL h = r ? LoadResource(nullptr, r) : nullptr;
                if (h) f->data.assign((const char*)LockResource(h), SizeofResource(nullptr, r));
            }
        } else {
            f->data = ReadFile(p);
        }
        if (f->data.empty()) return 0;
        return (Rml::FileHandle)f.release();
    }
    void Close(Rml::FileHandle file) override { delete (Open_*)file; }
    size_t Read(void* buffer, size_t size, Rml::FileHandle file) override {
        Open_* f = (Open_*)file;
        const size_t n = (std::min)(size, f->data.size() - f->pos);
        memcpy(buffer, f->data.data() + f->pos, n);
        f->pos += n;
        return n;
    }
    bool Seek(Rml::FileHandle file, long offset, int origin) override {
        Open_* f = (Open_*)file;
        const long long base = origin == SEEK_SET ? 0 : origin == SEEK_CUR ? (long long)f->pos : (long long)f->data.size();
        const long long to = base + offset;
        if (to < 0 || to > (long long)f->data.size()) return false;
        f->pos = (size_t)to;
        return true;
    }
    size_t Tell(Rml::FileHandle file) override { return ((Open_*)file)->pos; }

private:
    struct Open_ {
        std::string data;
        size_t pos = 0;
    };
};

// ---------------------------------------------------------------- application state

// Rows of the lists on the Status and Files pages, as the page binds them.
struct Finding {
    bool warn = false;
    Rml::String text;
};
struct KeyValue {
    Rml::String key, value;
};
struct FileRow {
    Rml::String cls, mark, item, detail;  // cls: ok / info / warn / err
};
struct LogLine {
    Rml::String cls, text;  // cls: warn (WARNING lines), note (Note: lines) or empty
};

constexpr size_t kLogLines = 400;  // the Log page shows the end of the log

struct App {
    fs::path home, iniPath, logDir;
    IniFile ini;
    unsigned long long iniTime = 0;
    bool dirty = false;
    double lastSave = 0, lastPoll = -10;
    std::string saveError;
    Settings s;
    // The game and the restart note: the ini as it was when the game was first seen running.
    bool game = false, haveStart = false;
    IniFile atStart;
    Settings start;       // Read(atStart): the values the running game started with
    std::string restart;  // labels of the changed restart keys while the game runs
    LogInfo log;
    unsigned long long logTime = 0;
    std::vector<Finding> findings;
    std::vector<KeyValue> detected;
    std::vector<LogLine> logLines;
    size_t logTotal = 0;     // lines in the whole log
    bool scrollLog = false;  // move the Log page to its end after the next layout
    bool noGaze = false;     // the last game start logged the proxy's no-gaze note (eye tracking on, no valid gaze)
    FileState files;
    std::vector<FileRow> fileRows;
    std::string filesMsg;
    std::string confirm;           // "repair" / "valve" while asking, else empty
    // Render resolution of the preview: taken from the newest log at the start and whenever a new log has another one;
    // the person can type another (preview only, not saved).
    int manualW = 4032, manualH = 3384;
    unsigned logGridW = 0, logGridH = 0;  // the newest log's value the fields were last set from
    // "Reset to the defaults" asks first; any change hides the question.
    bool askReset = false;
    // The preview shows the eye-tracking or the fixed size; it follows the setting until the person picks one there.
    bool previewDfr = true, lastFollow = true;
    int page = 0;
    Rml::DataModelHandle model;
};

App g;

void Dirty() {
    if (g.model) g.model.DirtyAllVariables();
}

void FollowSetting() {
    if (g.s.followGaze != g.lastFollow) g.previewDfr = g.lastFollow = g.s.followGaze;
}

void Changed() {
    g.s = Read(g.ini);
    FollowSetting();
    g.dirty = true;
    g.askReset = false;
    Dirty();
}

void Set(const char* sec, const char* key, const std::string& v) {
    if (g.ini.Set(sec, key, v)) Changed();
}

void Reload() {
    g.ini.Load(g.iniPath);
    g.iniTime = FileTime(g.iniPath);
    g.s = Read(g.ini);
    FollowSetting();
    Dirty();
}

void LoadLog() {
    g.log = ReadNewestLog(g.logDir);
    if (g.log.gridW && (g.log.gridW != g.logGridW || g.log.gridH != g.logGridH)) {
        g.manualW = (int)(g.logGridW = g.log.gridW);
        g.manualH = (int)(g.logGridH = g.log.gridH);
    }
    g.findings.clear();
    g.noGaze = false;
    for (const std::string& f : g.log.findings) {
        const bool warn = f.rfind("WARNING", 0) == 0;
        g.findings.push_back({warn, f.substr(warn ? 9 : 6)});
        if (f.rfind("Note: No valid gaze", 0) == 0) g.noGaze = true;
    }
    g.detected.clear();
    for (const std::string& l : g.log.detected) {
        const size_t c = l.find(": ");
        g.detected.push_back(c == std::string::npos ? KeyValue{"", l} : KeyValue{l.substr(0, c), l.substr(c + 2)});
    }
    // The end of the log for the Log page.
    std::vector<std::string> lines;
    if (g.log.found) {
        std::istringstream in(ReadFile(g.log.file));
        for (std::string l; std::getline(in, l);) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines.push_back(std::move(l));
        }
    }
    g.logTotal = lines.size();
    g.logLines.clear();
    for (size_t i = lines.size() > kLogLines ? lines.size() - kLogLines : 0; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        // "[  time] WARNING: ..." and "[  time] Note: ..." (not lines that only mention them).
        const size_t t = l.find("] ");
        const std::string body = t == std::string::npos ? l : l.substr(t + 2);
        const char* cls = body.rfind("WARNING", 0) == 0 ? "warn" : body.rfind("Note:", 0) == 0 ? "note" : "";
        g.logLines.push_back({cls, l});
    }
    g.scrollLog = true;
    Dirty();
}

void CheckNow() {
    g.files = CheckFiles(g.home, g.s.openxr);
    g.fileRows.clear();
    for (const Check& c : g.files.checks) {
        const char* cls = c.status == Status::Ok ? "ok" : c.status == Status::Info ? "info" : c.status == Status::Warning ? "warn" : "err";
        const char* mark = c.status == Status::Ok ? "OK" : c.status == Status::Info ? "--" : c.status == Status::Warning ? "!" : "X";
        g.fileRows.push_back({cls, mark, c.item, c.detail});
    }
    Dirty();
}

void Poll() {
    const double now = Now();
    if (g.dirty && now - g.lastSave > 0.15) {
        if (g.ini.Save(g.iniPath)) {
            g.dirty = false;
            g.iniTime = FileTime(g.iniPath);
            if (!g.saveError.empty()) Dirty();
            g.saveError.clear();
        } else {
            if (g.saveError.empty()) Dirty();
            g.saveError = "Could not save qvfr.ini (retrying)";
        }
        g.lastSave = now;
    }
    if (now - g.lastPoll < 1.0) return;
    g.lastPoll = now;
    const bool game = GameRunning();
    if (game && !g.game) {
        g.atStart = g.ini;  // the game reads the restart keys at its start; close enough when the GUI sees it start
        g.start = Read(g.atStart);
        g.haveStart = true;
    }
    if (!game) g.haveStart = false;
    std::string restart;
    if (g.haveStart)
        for (const std::string& l : ChangedRestartKeys(g.atStart, g.ini)) restart += (restart.empty() ? "" : ", ") + l;
    if (game != g.game || restart != g.restart) {
        g.game = game;
        g.restart = restart;
        Dirty();
    }
    if (!g.dirty && FileTime(g.iniPath) != g.iniTime) Reload();  // edited outside the GUI
    unsigned long long lt = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(g.logDir, ec))
        if (e.path().extension() == L".log") lt = (std::max)(lt, FileTime(e.path()));
    if (lt != g.logTime) {
        g.logTime = lt;
        LoadLog();
    }
}

// ---------------------------------------------------------------- resolution preview

struct Res {
    unsigned gw = 0, gh = 0;  // SteamVR resolution per eye (from the log, or typed in)
    bool dfr = true;
    float sx = 0, sy = 0, offX = 0, offY = 0, m = 0;
    unsigned ow = 0, oh = 0, fw = 0, fh = 0;
    double total = 0, full = 0;
};

Res Compute() {
    const Settings& s = g.s;
    Res r;
    r.gw = (unsigned)(std::max)(g.manualW, 16);
    r.gh = (unsigned)(std::max)(g.manualH, 16);
    r.dfr = g.previewDfr;
    r.sx = r.dfr ? s.focusX : s.fixedX;
    r.sy = r.dfr ? s.focusY : s.fixedY;
    r.offX = r.dfr ? s.focusOffsetX : s.fixedOffsetX;
    r.offY = r.dfr ? s.focusOffsetY : s.fixedOffsetY;
    r.m = s.smoothen;
    r.ow = (unsigned)std::lround(s.peripheral * r.gw);
    r.oh = (unsigned)std::lround(s.peripheral * r.gh);
    r.fw = (unsigned)std::lround(r.sx * s.focusMultiplier * r.gw);
    r.fh = (unsigned)std::lround(r.sy * s.focusMultiplier * r.gh);
    r.total = (double)r.ow * r.oh + (double)r.fw * r.fh;
    r.full = (double)r.gw * r.gh;
    return r;
}

// Grid lines whose spacing shows the pixel density (denser = sharper), as a decorator for data-style-decorator.
std::string Lines(float step, const char* color) {
    return Fmt("repeating-linear-gradient(90deg, %s 0dp, %s 1dp, #00000000 1dp, #00000000 %.1fdp), "
               "repeating-linear-gradient(180deg, %s 0dp, %s 1dp, #00000000 1dp, #00000000 %.1fdp)",
               color, color, step, color, color, step);
}

// ---------------------------------------------------------------- data model: the pages bind to these names

int Pct(float v) { return (int)std::lround(v * 100); }

float VarFloat(const Rml::Variant& v) {
    float f = 0;
    v.GetInto(f);
    return f;
}

void BindModel(Rml::Context* ctx) {
    Rml::DataModelConstructor c = ctx->CreateDataModel("s");
    if (auto h = c.RegisterStruct<Finding>()) {
        h.RegisterMember("warn", &Finding::warn);
        h.RegisterMember("text", &Finding::text);
    }
    if (auto h = c.RegisterStruct<KeyValue>()) {
        h.RegisterMember("key", &KeyValue::key);
        h.RegisterMember("value", &KeyValue::value);
    }
    if (auto h = c.RegisterStruct<FileRow>()) {
        h.RegisterMember("cls", &FileRow::cls);
        h.RegisterMember("mark", &FileRow::mark);
        h.RegisterMember("item", &FileRow::item);
        h.RegisterMember("detail", &FileRow::detail);
    }
    if (auto h = c.RegisterStruct<LogLine>()) {
        h.RegisterMember("cls", &LogLine::cls);
        h.RegisterMember("text", &LogLine::text);
    }
    c.RegisterArray<std::vector<Finding>>();
    c.RegisterArray<std::vector<KeyValue>>();
    c.RegisterArray<std::vector<FileRow>>();
    c.RegisterArray<std::vector<LogLine>>();

    // --- "Game restart needed" next to a setting: only while the game runs and when the setting differs from the
    // value the game started with (sizes compare as the effective values, so a preset change counts too).
    using Diff = bool (*)(const Settings&, const Settings&);
    auto rs = [&](const char* name, Diff diff) {
        c.BindFunc(name, [diff](Rml::Variant& v) { v = g.game && g.haveStart && diff(g.start, g.s); });
    };
    rs("rs_quad", [](const Settings& a, const Settings& b) { return a.quad != b.quad; });
    rs("rs_path", [](const Settings& a, const Settings& b) { return a.openxr != b.openxr; });
    rs("rs_preset", [](const Settings& a, const Settings& b) { return a.presetText != b.presetText || a.Custom() != b.Custom(); });
    rs("rs_focus_mult", [](const Settings& a, const Settings& b) { return Pct(a.focusMultiplier) != Pct(b.focusMultiplier); });
    rs("rs_peripheral", [](const Settings& a, const Settings& b) { return Pct(a.peripheral) != Pct(b.peripheral); });
    rs("rs_focus_x", [](const Settings& a, const Settings& b) { return Pct(a.focusX) != Pct(b.focusX); });
    rs("rs_focus_y", [](const Settings& a, const Settings& b) { return Pct(a.focusY) != Pct(b.focusY); });
    rs("rs_fixed_x", [](const Settings& a, const Settings& b) { return Pct(a.fixedX) != Pct(b.fixedX); });
    rs("rs_fixed_y", [](const Settings& a, const Settings& b) { return Pct(a.fixedY) != Pct(b.fixedY); });
    rs("rs_csv", [](const Settings& a, const Settings& b) { return a.csvAlways != b.csvAlways; });

    // --- Settings. A fraction as an integer percentage; the slider writes the fraction. The setters skip the value
    // already shown: the inputs report their value when the page binds them, which would otherwise write every preset
    // value as an override.
    auto pct = [&](const char* name, float Settings::* field, const char* sec, const char* key) {
        c.BindFunc(name, [field](Rml::Variant& v) { v = Pct(g.s.*field); }, [field, sec, key](const Rml::Variant& v) {
            const int p = (int)std::lround(VarFloat(v));
            if (p != Pct(g.s.*field)) Set(sec, key, Num(p / 100.0f));
        });
    };
    pct("focus_mult", &Settings::focusMultiplier, "quad", "focus_multiplier");
    pct("peripheral", &Settings::peripheral, "quad", "peripheral_multiplier");
    pct("focus_x", &Settings::focusX, "quad", "horizontal_focus_section");
    pct("focus_y", &Settings::focusY, "quad", "vertical_focus_section");
    pct("fixed_x", &Settings::fixedX, "quad", "horizontal_fixed_section");
    pct("fixed_y", &Settings::fixedY, "quad", "vertical_fixed_section");
    pct("smoothen", &Settings::smoothen, "quad", "smoothen_focus_view_edges");
    pct("offset_x", &Settings::focusOffsetX, "quad", "horizontal_focus_offset");
    pct("offset_y", &Settings::focusOffsetY, "quad", "vertical_focus_offset");
    pct("fixed_offset_x", &Settings::fixedOffsetX, "quad", "horizontal_fixed_offset");
    pct("fixed_offset_y", &Settings::fixedOffsetY, "quad", "vertical_fixed_offset");
    auto given = [&](const char* name, bool Settings::* field) { c.BindFunc(name, [field](Rml::Variant& v) { v = g.s.*field; }); };
    given("peripheral_given", &Settings::givenPeripheral);
    given("focus_x_given", &Settings::givenFocusX);
    given("focus_y_given", &Settings::givenFocusY);
    given("fixed_x_given", &Settings::givenFixedX);
    given("fixed_y_given", &Settings::givenFixedY);
    auto toggle = [&](const char* name, bool (*get)(), void (*set)(bool)) {
        c.BindFunc(name, [get](Rml::Variant& v) { v = get(); }, [get, set](const Rml::Variant& v) {
            if (v.Get<bool>() != get()) set(v.Get<bool>());
        });
    };
    toggle("quad", [] { return g.s.quad; }, [](bool on) { Set("quad", "mode", on ? "quad" : "off"); });
    toggle("follow", [] { return g.s.followGaze; }, [](bool on) { Set("quad", "follow_gaze", on ? "1" : "0"); });
    toggle("gaze_hotkey", [] { return g.s.gazeKey != 0; }, [](bool on) { Set("debug", "gaze_key", on ? "0x75" : "0"); });
    toggle("gaze_overlay", [] { return g.s.overlay; }, [](bool on) { Set("debug", "overlay", on ? "1" : "0"); });
    toggle("warning_box", [] { return g.s.warningBox; }, [](bool on) { Set("log", "warning_box", on ? "1" : "0"); });
    toggle("csv", [] { return g.s.csvAlways; }, [](bool on) { Set("timing", "csv_always", on ? "1" : "0"); });
    toggle("bench", [] { return g.s.benchKey != 0; }, [](bool on) { Set("benchmark", "key", on ? "0x78" : "0"); });
    toggle("debug_keys", [] { return g.s.overlayKey || g.s.freezeKey || g.s.outerKey || g.s.clonesKey || g.s.shotKey; },
           [](bool on) {
               g.ini.Set("debug", "clones_key", on ? "0x74" : "0");
               g.ini.Set("debug", "freeze_key", on ? "0x76" : "0");
               g.ini.Set("debug", "overlay_key", on ? "0x77" : "0");
               g.ini.Set("quad", "shot_key", on ? "0x79" : "0");
               g.ini.Set("debug", "outer_key", on ? "0x7A" : "0");
               Changed();
           });
    c.BindFunc("hold_tenths", [](Rml::Variant& v) { v = (int)std::lround(g.s.gazeHoldS * 10); },
               [](const Rml::Variant& v) {
                   const int t = (int)std::lround(VarFloat(v));
                   if (t != (int)std::lround(g.s.gazeHoldS * 10)) Set("quad", "gaze_hold_s", Num(t / 10.0f));
               });
    c.BindFunc("hold_text", [](Rml::Variant& v) { v = Fmt("%.1f s", std::round(g.s.gazeHoldS * 10) / 10.0f); });
    c.BindFunc("path", [](Rml::Variant& v) { v = Rml::String(g.s.openxr ? "openxr" : "openvr"); },
               [](const Rml::Variant& v) {
                   const Rml::String p = v.Get<Rml::String>();
                   if ((p == "openxr") != g.s.openxr) Set("runtime", "path", p);
               });
    c.BindFunc("preset", [](Rml::Variant& v) { v = g.s.preset; });
    c.BindFunc("preset_name", [](Rml::Variant& v) { v = Rml::String(g.s.presetText); });
    c.BindFunc("custom", [](Rml::Variant& v) { v = g.s.Custom(); });
    c.BindEventCallback("set_preset", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (args.empty()) return;
        const int i = args[0].Get<int>();
        if (i < 0 || i >= (int)std::size(presets::kPresets)) return;
        WritePreset(g.ini, i);
        Changed();
    });
    c.BindEventCallback("back_to_preset", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        ClearPresetOverrides(g.ini);
        Changed();
    });

    // --- Shared: page, game, restart note, save error.
    c.Bind("page", &g.page);
    c.BindFunc("game", [](Rml::Variant& v) { v = g.game; });
    c.BindFunc("restart", [](Rml::Variant& v) { v = Rml::String(g.restart); });
    c.BindFunc("save_error", [](Rml::Variant& v) { v = Rml::String(g.saveError); });
    c.BindEventCallback("open_url", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (args.empty()) return;
        const Rml::String which = args[0].Get<Rml::String>();
        const wchar_t* url = which == "github"    ? L"https://github.com/hypeapps/ams2-qvfr"
                             : which == "youtube" ? L"https://www.youtube.com/@hypevrtv"
                                                  : nullptr;
        if (url) ShellExecuteW(nullptr, L"open", url, nullptr, nullptr, SW_SHOWNORMAL);
    });

    // --- Reset to the defaults (asked first): the default qvfr.ini, keeping the VR API (part of the install, not a
    // preference).
    c.BindFunc("ask_reset", [](Rml::Variant& v) { v = g.askReset; });
    c.BindEventCallback("reset_defaults", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        g.askReset = true;
        Dirty();
    });
    c.BindEventCallback("cancel_reset", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        g.askReset = false;
        Dirty();
    });
    c.BindEventCallback("confirm_reset", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        if (!g.askReset) return;
        const std::string path = g.ini.GetOr("runtime", "path", "openxr");
        g.ini.LoadText(DefaultIniText());
        g.ini.Set("runtime", "path", path);
        Changed();
    });

    // --- Resolution.
    c.BindFunc("grid_known", [](Rml::Variant& v) { v = g.log.gridW != 0; });
    c.BindFunc("from_log", [](Rml::Variant& v) {
        v = g.log.gridW && (unsigned)g.manualW == g.log.gridW && (unsigned)g.manualH == g.log.gridH;
    });
    c.BindFunc("log_grid_text", [](Rml::Variant& v) { v = Fmt("%u x %u", g.log.gridW, g.log.gridH); });
    c.BindEventCallback("use_log_res", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        if (g.log.gridW) {
            g.manualW = (int)g.log.gridW;
            g.manualH = (int)g.log.gridH;
        }
        Dirty();
    });
    c.BindFunc("log_when", [](Rml::Variant& v) { v = Rml::String(g.log.when); });
    // The fields show these values one way and apply what was typed on Enter or when they lose the focus (applying on
    // every key would redraw the field while typing). Not a number in 16..16384: the field gets the old value back.
    c.BindFunc("manual_w", [](Rml::Variant& v) { v = g.manualW; });
    c.BindFunc("manual_h", [](Rml::Variant& v) { v = g.manualH; });
    auto apply = [](Rml::Event& e, const Rml::VariantList& args, bool onlyEnter) {
        if (onlyEnter && !e.GetParameter<bool>("linebreak", false)) return;
        auto* field = rmlui_dynamic_cast<Rml::ElementFormControl*>(e.GetCurrentElement());
        if (!field || args.empty()) return;
        int* target = args[0].Get<Rml::String>() == "h" ? &g.manualH : &g.manualW;
        const int n = std::atoi(field->GetValue().c_str());
        if (n >= 16 && n <= 16384) *target = n;
        field->SetValue(std::to_string(*target));
        Dirty();
    };
    c.BindEventCallback("res_apply", [apply](Rml::DataModelHandle, Rml::Event& e, const Rml::VariantList& a) { apply(e, a, false); });
    c.BindEventCallback("res_enter", [apply](Rml::DataModelHandle, Rml::Event& e, const Rml::VariantList& a) { apply(e, a, true); });
    c.BindFunc("pv_dfr", [](Rml::Variant& v) { v = g.previewDfr; });
    c.BindEventCallback("pv", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (!args.empty()) g.previewDfr = args[0].Get<bool>();
        Dirty();
    });
    c.BindFunc("canvas_h", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.0fdp", 400.0 * r.gh / r.gw); });
    // The pixel bar: outer and focus over a full render at the render resolution (the focus part capped at the bar).
    c.BindFunc("outer_mpx", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.1f", (double)r.ow * r.oh / 1e6); });
    c.BindFunc("focus_mpx", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.1f", (double)r.fw * r.fh / 1e6); });
    c.BindFunc("bar_outer", [](Rml::Variant& v) {
        const Res r = Compute();
        v = Fmt("%.2f%%", (std::min)(100.0, 100.0 * r.ow * r.oh / r.full));
    });
    c.BindFunc("bar_focus", [](Rml::Variant& v) {
        const Res r = Compute();
        const double o = (std::min)(100.0, 100.0 * r.ow * r.oh / r.full);
        v = Fmt("%.2f%%", (std::min)(100.0 - o, 100.0 * r.fw * r.fh / r.full));
    });
    c.BindFunc("fewer", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.1f", r.total > 0 ? r.full / r.total : 0.0); });
    c.BindFunc("saves", [](Rml::Variant& v) { const Res r = Compute(); v = r.total < r.full; });
    c.BindFunc("focus_pct", [](Rml::Variant& v) { v = Fmt("%.0f%%", g.s.focusMultiplier * 100); });
    c.BindFunc("focus_area", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.0f x %.0f%%", r.sx * 100, r.sy * 100); });
    // The fully sharp centre: the focus area less the blend on each side (none at a transition of 50%).
    c.BindFunc("sharp_area", [](Rml::Variant& v) {
        const Res r = Compute();
        v = Fmt("%.0f x %.0f%%", r.sx * (1 - 2 * r.m) * 100, r.sy * (1 - 2 * r.m) * 100);
    });
    c.BindFunc("sharp_none", [](Rml::Variant& v) { v = g.s.smoothen >= 0.495f; });
    c.BindFunc("shifted", [](Rml::Variant& v) { const Res r = Compute(); v = std::fabs(r.offX) > 0.004f || std::fabs(r.offY) > 0.004f; });
    c.BindFunc("focus_left", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.2f%%", 100 * (0.5f - r.offX / 2 - r.sx / 2)); });
    c.BindFunc("focus_top", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.2f%%", 100 * (0.5f - r.offY / 2 - r.sy / 2)); });
    c.BindFunc("focus_w", [](Rml::Variant& v) { v = Fmt("%.2f%%", 100 * Compute().sx); });
    c.BindFunc("focus_h", [](Rml::Variant& v) { v = Fmt("%.2f%%", 100 * Compute().sy); });
    c.BindFunc("sharp_inset", [](Rml::Variant& v) { v = Fmt("%.2f%%", 100 * g.s.smoothen); });
    c.BindFunc("sharp_size", [](Rml::Variant& v) { v = Fmt("%.2f%%", 100 * (std::max)(0.0f, 1 - 2 * g.s.smoothen)); });
    c.BindFunc("outer_lines", [](Rml::Variant& v) { v = Lines((std::max)(2.5f, 6 / g.s.peripheral), "#ffffff1f"); });
    c.BindFunc("focus_lines", [](Rml::Variant& v) { v = Lines((std::max)(2.0f, 6 / g.s.focusMultiplier), "#ffb3a847"); });
    c.BindFunc("mpx", [](Rml::Variant& v) { v = Fmt("%.1f", Compute().total / 1e6); });
    c.BindFunc("share", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%.0f", 100 * r.total / r.full); });
    c.BindFunc("full_mpx", [](Rml::Variant& v) { v = Fmt("%.1f", Compute().full / 1e6); });
    c.BindFunc("outer_size", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%u x %u", r.ow, r.oh); });
    c.BindFunc("outer_density", [](Rml::Variant& v) { v = Fmt("%.0f%%", g.s.peripheral * 100); });
    c.BindFunc("focus_size", [](Rml::Variant& v) { const Res r = Compute(); v = Fmt("%u x %u", r.fw, r.fh); });

    // --- Status.
    c.BindFunc("log_found", [](Rml::Variant& v) { v = g.log.found; });
    c.BindFunc("detected_written", [](Rml::Variant& v) { v = g.log.detectedWritten; });
    c.BindFunc("no_gaze", [](Rml::Variant& v) { v = g.noGaze; });
    c.Bind("findings", &g.findings);
    c.Bind("detected", &g.detected);
    c.BindEventCallback("open_log", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        ShellExecuteW(nullptr, L"open", g.log.file.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    });
    c.BindEventCallback("open_logs", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        ShellExecuteW(nullptr, L"explore", g.logDir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    });

    // --- Log.
    c.Bind("log_lines", &g.logLines);
    c.BindFunc("log_cut", [](Rml::Variant& v) { v = g.logTotal > g.logLines.size(); });
    c.BindFunc("log_total", [](Rml::Variant& v) { v = (int)g.logTotal; });
    c.BindFunc("log_shown", [](Rml::Variant& v) { v = (int)g.logLines.size(); });
    c.BindFunc("log_name", [](Rml::Variant& v) { v = g.log.file.filename().string(); });

    // --- Files. Repair and the Valve DLL ask first (an inline confirmation instead of a dialog).
    c.Bind("file_rows", &g.fileRows);
    c.BindFunc("files_msg", [](Rml::Variant& v) { v = Rml::String(g.filesMsg); });
    c.BindFunc("confirm", [](Rml::Variant& v) { v = Rml::String(g.confirm); });
    c.BindFunc("can_repair", [](Rml::Variant& v) {
        v = !g.game && g.files.gameFolder && g.files.spare && g.files.gameDll != GameDll::Proxy;
    });
    c.BindFunc("can_valve", [](Rml::Variant& v) { v = !g.game && g.files.gameFolder && g.files.gameDll != GameDll::Valve; });
    c.BindEventCallback("check_files", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        g.filesMsg.clear();
        CheckNow();
    });
    c.BindEventCallback("ask", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (args.empty() || g.game) return;
        const Rml::String what = args[0].Get<Rml::String>();
        const bool ok = what == "repair" ? g.files.gameFolder && g.files.spare && g.files.gameDll != GameDll::Proxy
                                         : g.files.gameFolder && g.files.gameDll != GameDll::Valve;
        if (ok) g.confirm = what;
        Dirty();
    });
    c.BindEventCallback("answer", [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (!args.empty() && args[0].Get<bool>()) g.filesMsg = g.confirm == "repair" ? RepairProxy(g.home) : RestoreValve(g.home);
        g.confirm.clear();
        CheckNow();
    });
    g.model = c.GetModelHandle();
}

// The embedded fonts (qvfr_config.rc): Inter for text (made for reading on screens), Barlow Condensed for the page
// title, the menu, buttons and values, Tabler Icons for icons. Each weight is registered under one family name so the
// page picks it by font-weight.
void LoadFonts() {
    struct Face {
        const wchar_t* resource;
        const char* family;
        int weight;
    } faces[] = {{L"FONT_BODY_400", "Inter", 400},            {L"FONT_BODY_500", "Inter", 500},
                 {L"FONT_BODY_600", "Inter", 600},            {L"FONT_HEAD_600", "Barlow Condensed", 600},
                 {L"FONT_HEAD_700", "Barlow Condensed", 700}, {L"FONT_HEAD_800", "Barlow Condensed", 800},
                 {L"FONT_ICONS", "Tabler Icons", 400}};
    for (const Face& f : faces) {
        HRSRC r = FindResourceW(nullptr, f.resource, MAKEINTRESOURCEW(10));  // RT_RCDATA; resources stay mapped
        HGLOBAL h = r ? LoadResource(nullptr, r) : nullptr;
        if (!h) continue;
        Rml::LoadFontFace({(const Rml::byte*)LockResource(h), (size_t)SizeofResource(nullptr, r)}, f.family,
                          Rml::Style::FontStyle::Normal, (Rml::Style::FontWeight)f.weight);
    }
    // The log in Consolas from Windows (not shipped: its license does not allow that); without it the page falls back
    // to Inter.
    wchar_t win[MAX_PATH];
    GetWindowsDirectoryW(win, MAX_PATH);
    static std::string consolas = ReadFile(fs::path(win) / L"Fonts" / L"consola.ttf");
    if (!consolas.empty())
        Rml::LoadFontFace({(const Rml::byte*)consolas.data(), consolas.size()}, "Log Mono", Rml::Style::FontStyle::Normal,
                          Rml::Style::FontWeight::Normal);
}

void StyleWindow() {
    HWND hwnd = FindWindowW(L"AMS2 QVFR", nullptr);
    if (!hwnd) return;
    const HINSTANCE inst = GetModuleHandleW(nullptr);
    SendMessageW(hwnd, WM_SETICON, ICON_BIG, (LPARAM)LoadIconW(inst, MAKEINTRESOURCEW(1)));
    SendMessageW(hwnd, WM_SETICON, ICON_SMALL,
                 (LPARAM)LoadImageW(inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0));
    const BOOL dark = TRUE;
    DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof(dark));
    const COLORREF caption = RGB(0x0c, 0x0a, 0x0a);
    DwmSetWindowAttribute(hwnd, 35 /* DWMWA_CAPTION_COLOR */, &caption, sizeof(caption));
    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    SetTimer(hwnd, 1, 1000, nullptr);  // wakes the power-save loop for the game, ini and log polling
}

// QVFR_GUI_TRACE=<file>: logs mousedown, click and change events with their targets.
struct Trace : Rml::EventListener {
    FILE* f = nullptr;
    void ProcessEvent(Rml::Event& e) override {
        Rml::Element* t = e.GetTargetElement();
        const Rml::Vector2f m = e.GetUnprojectedMouseScreenPos();
        fprintf(f, "%s at %.0f,%.0f on <%s class='%s'>\n", e.GetType().c_str(), m.x, m.y, t ? t->GetTagName().c_str() : "?",
                t ? t->GetClassNames().c_str() : "");
        fflush(f);
    }
};

fs::path ExeDir() {
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
}

}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    g.home = ExeDir();
    Files files;
    for (int i = 1; i < argc; ++i) {
        if (!wcscmp(argv[i], L"--home") && i + 1 < argc) g.home = argv[++i];
        else if (!wcscmp(argv[i], L"--tab") && i + 1 < argc) g.page = std::clamp(_wtoi(argv[++i]), 0, 4);
        else if (!wcscmp(argv[i], L"--ui") && i + 1 < argc) files.uiDir = argv[++i];
        else if (!wcscmp(argv[i], L"--selftest") && i + 1 < argc) {
            // Redirected output (a pipe) is already stdout; a console window needs attaching.
            const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
            if ((!out || GetFileType(out) == FILE_TYPE_UNKNOWN) && AttachConsole(ATTACH_PARENT_PROCESS)) {
                FILE* f = nullptr;
                freopen_s(&f, "CONOUT$", "w", stdout);
            }
            return RunSelfTest(argv[i + 1]);
        }
    }
    g.iniPath = g.home / L"qvfr.ini";
    g.logDir = g.home / L"logs";
    std::error_code ec;
    if (!fs::exists(g.iniPath, ec)) {
        IniFile d;
        d.LoadText(DefaultIniText());
        d.Save(g.iniPath);
    }
    Reload();
    g.previewDfr = g.lastFollow = g.s.followGaze;
    LoadLog();
    CheckNow();

    if (!Backend::Initialize(kTitle, 1160, 840, true)) {
        MessageBoxA(nullptr, "Direct3D 11 is not available.", kTitle, MB_ICONERROR);
        return 1;
    }
    StyleWindow();
    Rml::SetSystemInterface(Backend::GetSystemInterface());
    Rml::SetRenderInterface(Backend::GetRenderInterface());
    Rml::SetFileInterface(&files);
    Rml::Initialise();
    LoadFonts();
    Rml::Context* ctx = Rml::CreateContext("main", Rml::Vector2i(1160, 840));
    BindModel(ctx);
    Rml::ElementDocument* doc = ctx->LoadDocument("main.rml");
    if (!doc) {
        MessageBoxA(nullptr, "The window layout (main.rml) could not be loaded.", kTitle, MB_ICONERROR);
        return 1;
    }
    doc->Show();
    Trace trace;
    wchar_t tracePath[MAX_PATH];
    if (GetEnvironmentVariableW(L"QVFR_GUI_TRACE", tracePath, MAX_PATH) && (trace.f = _wfopen(tracePath, L"w")) != nullptr) {
        doc->AddEventListener(Rml::EventId::Mousedown, &trace, true);
        doc->AddEventListener(Rml::EventId::Click, &trace, true);
        doc->AddEventListener(Rml::EventId::Change, &trace, true);
    }

    // The pages share one scrolling #main; each keeps its own position.
    Rml::Element* main = doc->GetElementById("main");
    float pageScroll[5] = {};
    int shownPage = g.page;
    for (bool running = true, onLog = false; running;) {
        running = Backend::ProcessEvents(ctx, nullptr, true);
        Poll();
        if (g.page == 4 && !onLog) g.scrollLog = true;  // opening the Log page shows the end
        onLog = g.page == 4;
        const bool switched = g.page != shownPage && main;
        if (switched) pageScroll[shownPage] = main->GetScrollTop();  // still the old page's layout
        ctx->Update();
        if (switched) main->SetScrollTop(pageScroll[g.page]);
        shownPage = g.page;
        if (g.scrollLog && g.page == 4) {  // after the layout of the new lines: show the end of the log
            if (Rml::Element* box = doc->GetElementById("logbox")) box->SetScrollTop(box->GetScrollHeight());
            g.scrollLog = false;
        }
        Backend::BeginFrame();
        ctx->Render();
        Backend::PresentFrame();
    }
    if (g.dirty) g.ini.Save(g.iniPath);
    if (trace.f) fclose(trace.f);
    Rml::Shutdown();
    Backend::Shutdown();
    return 0;
}

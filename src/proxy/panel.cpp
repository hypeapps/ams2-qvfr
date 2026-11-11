// The tuning panel in the headset: see panel.h.
#include "panel.h"

#include <windows.h>

#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "frame.h"
#include "gpu.h"
#include "ini_file.h"
#include "nr.h"
#include "panel_input.h"
#include "panel_ui.h"
#include "wrappers.h"

namespace qvfr::panel {

namespace {

// The picture in px: the rows 900 wide, the measured values right of them (about 25 px per degree at the default
// place).
constexpr unsigned kW = 1320, kH = 1440;
constexpr char kOverlayKey[] = "ams2qvfr.tuning";
constexpr char kFooter[] = "Up/Down choose  \xC2\xB7  Left/Right change  \xC2\xB7  Enter switch  \xC2\xB7  Esc close";

using K = nr::Knob;
using R = panelui::Row;

enum class Special { None, NrOn, Save, Revert };

// One row: a heading, a value (left/right steps it), a toggle or a button. Values are the NR knobs' integers.
struct Item {
    R::Kind kind;
    const char* label;
    const char* hint;
    K knob = K::Count;
    Special special = Special::None;
    int min = 0, max = 1, step = 1;
    int fillMax = 0;                     // > 0: a bar of value / fillMax
    const char* const* names = nullptr;  // value names, from min
    const char* fmt = nullptr;           // printf of value / div
    float div = 1;
    bool outer = false;  // in effect only with NR on the outer views
};

const char* const kStyles[] = {"0  Default", "1", "2  Cinematic", "3"};
const char* const kPresets[] = {"Default", "#1", "#2", "#3"};
const char* const kOuterRes[] = {"Full", "1/2", "1/3"};
const char* const kEvaluation[] = {"With the focus", "Separate"};

// The DLSS 5 options in the order and with the names of DXL and nr-bridge, then the mod's own. The description is shown
// under the row in a smaller font. Not in the panel (the ini's value stays): UI correction, the edit map ([nr] debug)
// and skin structure (no skin in AMS2's cockpit).
const Item kItems[] = {
    {R::Head, "DLSS 5 Neural Rendering", nullptr},
    {R::Toggle, "Neural Rendering", "A/B switch, the same as right Ctrl + F8", K::Count, Special::NrOn},
    {R::Value, "Intensity", "How much of the model's change reaches the image", K::Intensity, Special::None, 0, 100,
     5, 100, nullptr, "%.2f", 100},
    {R::Value, "Style", "The model's look; Cinematic: softer, photographic light", K::Style, Special::None, 0, 3, 1, 0,
     kStyles},
    {R::Value, "Preset", "The model's variant; #1 costs about twice the GPU time", K::Preset, Special::None, 0, 3, 1,
     0, kPresets},
    {R::Value, "Strength", "Multiplies the model's change; above 1 stronger than the model", K::Strength,
     Special::None, 0, 300, 10, 300, nullptr, "%.1fx", 100},
    {R::Value, "Local tone", "Light and shadow in small areas (local contrast)", K::LocalTone, Special::None, 0, 200,
     10, 200, nullptr, "%.2f", 100},
    {R::Value, "Local structure", "Fine detail and texture of surfaces", K::LocalStructure, Special::None, 0, 200, 10,
     200, nullptr, "%.2f", 100},
    {R::Toggle, "Auto mask", "Leaves flat text and displays as they are", K::AutoMask},
    {R::Head, "DLSS 5 Foveation and performance", nullptr},
    {R::Toggle, "NR on the outer views", "The periphery too; hides the tone step at the focus edge, +0.5 ms",
     K::Outer},
    {R::Value, "Outer resolution", "Lower is cheaper; only the model's change is enlarged back", K::OuterRes,
     Special::None, 1, 3, 1, 0, kOuterRes, nullptr, 1, true},
    {R::Value, "Outer area", "Only this centre part of the periphery goes through the model", K::OuterArea,
     Special::None, 50, 100, 5, 100, nullptr, "%.0f%%", 1, true},
    {R::Value, "Outer evaluation", "Separate: the periphery does not change the focus", K::OuterSeparate,
     Special::None, 0, 1, 1, 0, kEvaluation, nullptr, 1, true},
    {R::Value, "Focus edge blend", "Width of the blend between the focus and the periphery", K::Smoothen,
     Special::None, 0, 50, 5, 50, nullptr, "%.2f", 100},  // the ini's range 0..0.5
    {R::Head, "Settings", nullptr},
    {R::Button, "Save to qvfr.ini", "Writes the options marked * to qvfr.ini", K::Count,
     Special::Save},
    {R::Button, "Restore to the ini", "Restore the changes not saved yet", K::Count, Special::Revert},
};
constexpr int kItemCount = (int)(sizeof(kItems) / sizeof(kItems[0]));

struct State {
    panelui::Ui ui;
    bool open = false;
    bool failed = false;   // no picture or no overlay: off for the session
    bool faulted = false;  // an exception: off for the session
    bool hotkeyDown = false;
    int sel = 1;
    uint32_t gen = 0;
    vr::VROverlayHandle_t handle = vr::k_ulOverlayHandleInvalid;
    bool created = false, shown = false;
    float placed[4] = {-1, -1, -1, -1};  // width, distance, down, tilt last set
    int held = 0;                        // the key held for repeats
    double next = 0;                     // its next repeat (ms)
    std::string status;                  // the Save button's result, shown until statusUntil (ms)
    double statusUntil = 0;
    // The measured values, once a second from the running totals (frame.h, nr.h).
    double statsAt = 0;                  // 0: take a new baseline
    frame::Totals frames;
    nr::Totals nr;
    std::vector<panelui::Stat> stats, details;
    std::string statsNote;
};
State& g = *new State;  // never destroyed: no D3D11 or RmlUi calls at the DLL's unload

template <class F>
void Protect(const char* where, F&& f) {
    if (g.faulted) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g.faulted = true;
        Log("ERROR: exception 0x%08lX in the tuning panel (%s); the panel is off for the session", code, where);
    }
}

bool Pressed(int mod, int key, bool& wasDown) {
    const bool down = key && (GetAsyncKeyState(key) & 0x8000) && (!mod || (GetAsyncKeyState(mod) & 0x8000));
    const bool edge = down && !wasDown;
    wasDown = down;
    return edge;
}

// ---------------------------------------------------------------- keys

// The keys are polled; panel_input.h takes them out of the game's DirectInput keyboard.
// A navigation key that fired this frame (a press, or a repeat while held), 0 none.
int NavKey() {
    static const int keys[] = {VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN, VK_ESCAPE};
    const double now = NowMs();
    for (int k : keys) {
        if (!(GetAsyncKeyState(k) & 0x8000)) {
            if (g.held == k) g.held = 0;
            continue;
        }
        if (g.held != k) {
            g.held = k;
            g.next = now + 400;
            return k;
        }
        if (k != VK_RETURN && k != VK_ESCAPE && now >= g.next) {
            g.next = now + 70;
            return k;
        }
    }
    return 0;
}

// ---------------------------------------------------------------- qvfr.ini

// The panel's options in qvfr.ini: its knob, section and key.
struct IniKey {
    K knob;
    const char* section;
    const char* key;
};
const IniKey kIniKeys[] = {
    {K::Intensity, "nr", "intensity"},
    {K::Style, "nr", "style"},
    {K::Preset, "nr", "preset"},
    {K::Strength, "nr", "strength"},
    {K::LocalTone, "nr", "local_tone"},
    {K::LocalStructure, "nr", "local_structure"},
    {K::AutoMask, "nr", "auto_mask"},
    {K::Outer, "nr", "outer"},
    {K::OuterRes, "nr", "outer_resolution"},
    {K::OuterArea, "nr", "outer_area"},
    {K::OuterSeparate, "nr", "outer_atlas"},
    {K::Smoothen, "quad", "smoothen_focus_view_edges"},
};

// A knob's value as the ini writes it (common.cpp reads it back to the same knob value).
std::string IniText(K k, int v) {
    char b[32];
    switch (k) {
        case K::Style:
        case K::Preset:
        case K::AutoMask:
        case K::Outer: snprintf(b, sizeof(b), "%d", v); break;
        case K::OuterRes: snprintf(b, sizeof(b), "%.3g", 1.0 / (double)(std::max)(v, 1)); break;  // 2 -> 0.5
        case K::OuterSeparate: return v ? "separate" : "shared";
        default: snprintf(b, sizeof(b), "%.2f", (double)v / 100.0); break;  // percent knobs
    }
    return b;
}

bool AnyToSave() {
    for (const IniKey& k : kIniKeys)
        if (nr::KnobSet(k.knob)) return true;
    return false;
}

void SetStatus(const std::string& s) {
    g.status = s;
    g.statusUntil = NowMs() + 4000;
}

// The changed options into qvfr.ini (in place, comments and order kept, atomic replace), the live settings read again
// at once, and their knobs back to the ini (which now has their values), so nothing jumps.
void Save() {
    std::vector<const IniKey*> keys;
    for (const IniKey& k : kIniKeys)
        if (nr::KnobSet(k.knob)) keys.push_back(&k);
    if (keys.empty()) {
        SetStatus("Nothing to save");
        return;
    }
    config::IniFile ini;
    const fs::path path = IniFile();
    if (!ini.Load(path)) {
        SetStatus("Cannot read qvfr.ini");
        Log("Panel: cannot read %s; nothing saved", Narrow(path.wstring()).c_str());
        return;
    }
    std::string list;
    for (const IniKey* k : keys) {
        const std::string v = IniText(k->knob, nr::GetKnob(k->knob));
        ini.Set(k->section, k->key, v);
        list += (list.empty() ? "" : ", ") + std::string("[") + k->section + "] " + k->key + "=" + v;
    }
    if (!ini.Save(path)) {
        SetStatus("Cannot write qvfr.ini");
        Log("Panel: cannot write %s; nothing saved", Narrow(path.wstring()).c_str());
        return;
    }
    ReloadConfig(true);
    for (const IniKey* k : keys) nr::ClearKnob(k->knob);
    SetStatus("Saved " + std::to_string(keys.size()));
    Log("Panel: saved to qvfr.ini: %s", list.c_str());
}

// ---------------------------------------------------------------- measured values

std::string Fmt(const char* f, double v) {
    char b[48];
    snprintf(b, sizeof(b), f, v);
    return b;
}

// Once a second: the frame rate, the GPU frame and NR's cost on the game's queue over the last second.
void UpdateStats() {
    const double now = NowMs();
    if (g.statsAt > 0 && now - g.statsAt < 1000) return;
    const frame::Totals f = frame::LiveTotals();
    const nr::Totals n = nr::LiveTotals();
    const bool base = g.statsAt <= 0;
    const frame::Totals pf = g.frames;
    const nr::Totals pn = g.nr;
    g.statsAt = now;
    g.frames = f;
    g.nr = n;
    std::vector<panelui::Stat> s(4), parts;
    s[0].label = "frame rate";
    s[1].label = "CPU frame (render thread)";
    s[2].label = "GPU frame";
    s[3].label = "NR on the GPU";
    for (auto& x : s) x.value = "...";
    g.statsNote = "Measuring";
    if (!base) {
        const uint64_t di = f.intervals - pf.intervals;
        if (di) {
            const double ms = (f.intervalMs - pf.intervalMs) / (double)di;
            s[0].value = ms > 0 ? Fmt("%.0f fps", 1000.0 / ms) : "-";
            s[0].label = Fmt("frame rate, %.1f ms", ms);
        }
        const uint64_t dc = f.cpuFrames - pf.cpuFrames;
        if (dc) s[1].value = Fmt("%.2f ms", (f.cpuMs - pf.cpuMs) / (double)dc);
        const uint64_t dg = f.gpuFrames - pf.gpuFrames;
        if (!gpu::Enabled())
            s[2] = {"off", "GPU frame ([timing] gpu=0)"};
        else if (dg)
            s[2].value = Fmt("%.2f ms", (f.gpuMs - pf.gpuMs) / (double)dg);
        const uint64_t d11 = n.d11Frames - pn.d11Frames, frames = n.frames - pn.frames;
        if (!d11 || !frames) {
            s[3].value = "-";
            g.statsNote = Cfg().nrMode != 2 ? "NR is off in qvfr.ini ([nr] mode=final)"
                          : !nr::Active()   ? "Neural Rendering is switched off"
                                            : "No images went through NR in the last second";
        } else {
            const double k = 1.0 / (double)d11;
            const double guides = (n.d11Guides - pn.d11Guides) * k, pack = (n.d11Pack - pn.d11Pack) * k,
                         handoff = (n.d11Handoff - pn.d11Handoff) * k, unpack = (n.d11Unpack - pn.d11Unpack) * k;
            const uint64_t samples = n.gpuSamples - pn.gpuSamples, calls = n.calls - pn.calls;
            const double model =
                samples ? (n.gpuMs - pn.gpuMs) / (double)samples * (double)calls / (double)frames : 0.0;
            s[3].value = Fmt("%.2f ms", guides + pack + handoff + unpack);
            parts = {{Fmt("%.2f ms", model), "Model (D3D12)"},
                     {Fmt("%.2f ms", (std::max)(handoff - model, 0.0)), "Queue switches"},
                     {Fmt("%.2f ms", guides), "Guides"},
                     {Fmt("%.2f ms", pack), "Pack"},
                     {Fmt("%.2f ms", unpack), "Unpack"}};
            if (n.atlasMp > 0) parts.push_back({Fmt("%.1f MP", n.atlasMp), "Atlas"});
            g.statsNote.clear();
        }
    }
    g.stats = std::move(s);
    g.details = std::move(parts);
}

int ValueOf(const Item& it) {
    if (it.special == Special::NrOn) return nr::Active() ? 1 : 0;
    if (it.knob == K::Count) return 0;
    const int v = nr::GetKnob(it.knob);
    return it.knob == K::SkinStructure && v < 0 ? it.min : v;  // the model's own: left of 0
}

std::string TextOf(const Item& it, int v) {
    if (it.kind == R::Toggle) return v ? "ON" : "OFF";
    if (it.special == Special::Save && NowMs() < g.statusUntil) return g.status;
    if (it.kind == R::Button) return "Enter";
    if (it.knob == K::SkinStructure && v < 0) return "Model's own";
    if (it.names) return it.names[std::clamp(v, it.min, it.max) - it.min];
    char b[32];
    snprintf(b, sizeof(b), it.fmt ? it.fmt : "%.0f", (double)v / (double)it.div);
    return b;
}

void Change(const Item& it, int dir) {
    if (it.special == Special::NrOn) {
        nr::SetActive(!nr::Active());
        Log("Panel: Neural Rendering %s", nr::Active() ? "on" : "off");
        return;
    }
    if (it.special == Special::Save) {
        if (!dir) Save();  // Enter only
        return;
    }
    if (it.special == Special::Revert) {
        if (dir) return;  // Enter only
        nr::ClearKnobs();
        g.statusUntil = 0;
        Log("Panel: every change back to the ini");
        return;
    }
    if (it.knob == K::Count) return;
    int v = ValueOf(it);
    if (it.kind == R::Toggle) {
        v = v ? 0 : 1;
    } else {
        if (!dir) return;
        v = std::clamp(v + dir * it.step, it.min, it.max);
    }
    nr::SetKnob(it.knob, it.knob == K::SkinStructure && v < 0 ? -1 : v);
    Log("Panel: %s = %s", it.label, TextOf(it, ValueOf(it)).c_str());
}

void Move(int dir) {
    for (int n = 0; n < kItemCount; ++n) {
        g.sel = (g.sel + dir + kItemCount) % kItemCount;
        if (kItems[g.sel].kind != R::Head) return;
    }
}

void Key(int vk) {
    switch (vk) {
        case VK_UP: Move(-1); break;
        case VK_DOWN: Move(1); break;
        case VK_LEFT: Change(kItems[g.sel], -1); break;
        case VK_RIGHT: Change(kItems[g.sel], 1); break;
        case VK_RETURN: Change(kItems[g.sel], 0); break;
        case VK_ESCAPE:
            g.open = false;
            Log("Panel: closed (Esc)");
            break;
        default: break;
    }
}

std::vector<R> Rows() {
    std::vector<R> rows;
    rows.reserve(kItemCount);
    const bool outer = nr::GetKnob(K::Outer) != 0;
    for (const Item& it : kItems) {
        R r;
        r.kind = it.kind;
        r.label = it.label;
        if (it.knob != K::Count && nr::KnobSet(it.knob)) r.label += "  *";  // changed in this session
        if (it.kind != R::Head) {
            const int v = ValueOf(it);
            r.value = TextOf(it, v);
            r.on = it.kind == R::Toggle && v != 0;
            if (it.fillMax > 0) r.fill = (float)(std::max)(v, 0) / (float)it.fillMax;
            r.dim = (it.outer && !outer) ||
                    (it.special == Special::Save && !AnyToSave() && NowMs() >= g.statusUntil);
        }
        if (it.hint) r.hint = it.hint;
        rows.push_back(std::move(r));
    }
    return rows;
}

// In the seated space like the game's HUD: ahead of and below the seated origin, tilted to face the eyes.
void Place(vr::IVROverlay* ov) {
    const Config& c = Cfg();
    const float want[4] = {c.panelWidthM, c.panelDistanceM, c.panelDownM, c.panelTiltDeg};
    if (std::equal(want, want + 4, g.placed)) return;
    std::copy(want, want + 4, g.placed);
    const float a = c.panelTiltDeg * 3.14159265f / 180.0f, ca = std::cos(a), sa = std::sin(a);
    vr::HmdMatrix34_t m{};
    m.m[0][0] = 1;
    m.m[1][1] = ca;
    m.m[1][2] = sa;
    m.m[1][3] = -c.panelDownM;
    m.m[2][1] = -sa;
    m.m[2][2] = ca;
    m.m[2][3] = -c.panelDistanceM;
    ov->SetOverlayWidthInMeters(g.handle, c.panelWidthM);
    ov->SetOverlayTransformAbsolute(g.handle, vr::TrackingUniverseSeated, &m);
    Log("Panel: overlay %.2f m wide, %.2f m ahead, %.2f m down, tilted %.0f deg (seated space)", c.panelWidthM,
        c.panelDistanceM, c.panelDownM, c.panelTiltDeg);
}

bool EnsureOverlay(vr::IVROverlay* ov) {
    if (g.created) return true;
    vr::EVROverlayError e = ov->CreateOverlay(kOverlayKey, "QVFR tuning", &g.handle);
    if (e == vr::VROverlayError_KeyInUse) e = ov->FindOverlay(kOverlayKey, &g.handle);
    if (e != vr::VROverlayError_None) {
        Log("Panel: CreateOverlay failed (error %d); the panel is off for the session", (int)e);
        g.failed = true;
        return false;
    }
    ov->SetOverlayAlpha(g.handle, 1.0f);
    g.created = true;
    for (float& p : g.placed) p = -1;
    Log("Panel: overlay %s (0x%llX)", kOverlayKey, (unsigned long long)g.handle);
    return true;
}

void Tick() {
    const Config& c = Cfg();
    panelinput::SetHotkey(c.panelKeyMod, c.panelKey);
    if (c.panelKey) panelinput::Start();
    if (c.panelKey && Pressed(c.panelKeyMod, c.panelKey, g.hotkeyDown) && !g.failed) {
        g.open = !g.open;
        g.statsAt = 0;       // new measurements from the opening on
        g.held = VK_ESCAPE;  // no key of the open panel fires from the hotkey's press
        Log("Panel: %s", g.open ? "open" : "closed");
    }
    if (g.open) Key(NavKey());
    panelinput::SetOpen(g.open && !g.failed);
    vr::IVROverlay* ov = wrap::RealOverlay();
    if (!g.open || g.failed) {
        if (g.shown && ov) ov->HideOverlay(g.handle);
        g.shown = false;
        return;
    }
    ID3D11DeviceContext* ctx = gpu::Context();
    if (!ctx || !ov) return;
    if (!g.ui.ready() || g.gen != gpu::Generation()) {
        g.ui.Release();
        ID3D11Device* dev = nullptr;
        ctx->GetDevice(&dev);
        const bool ok = dev && g.ui.Init(dev, kW, kH);
        if (dev) dev->Release();
        g.gen = gpu::Generation();
        if (!ok) {
            Log("Panel: no picture (%s); the panel is off for the session", g.ui.why().c_str());
            g.failed = true;
            return;
        }
    }
    const char* title = c.nrMode == 2 ? "DLSS 5 tuning" : "DLSS 5 tuning (NR off in qvfr.ini)";
    UpdateStats();
    g.ui.Set(Rows(), g.sel, title, kFooter, g.stats, "Measured, last second", g.details, "NR cost parts",
             g.statsNote);
    g.ui.Render(ctx);
    if (!EnsureOverlay(ov)) return;
    Place(ov);
    vr::Texture_t t{g.ui.Texture(), vr::API_DirectX, vr::ColorSpace_Gamma};
    ov->SetOverlayTexture(g.handle, &t);
    if (!g.shown) {
        ov->ShowOverlay(g.handle);
        g.shown = true;
    }
}

}  // namespace

void OnSubmit(vr::EVREye eye) {
    if (eye != vr::Eye_Right) return;
    Protect("frame", [] { Tick(); });
}

void Shutdown() {
    panelinput::SetOpen(false);
    Protect("shutdown", [] {
        vr::IVROverlay* ov = wrap::RealOverlay();
        if (g.created && ov) ov->DestroyOverlay(g.handle);
        g.created = g.shown = false;
        g.handle = vr::k_ulOverlayHandleInvalid;
        g.ui.Release();
        g.open = false;
    });
}

}  // namespace qvfr::panel

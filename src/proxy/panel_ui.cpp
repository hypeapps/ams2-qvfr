// The tuning panel's picture: see panel_ui.h.
#include "panel_ui.h"

#include <windows.h>

#include <d3d11_1.h>

#include <RmlUi/Core.h>

#include <algorithm>
#include <cstdio>

#include "RmlUi_Renderer_DX11.h"
#include "common.h"

namespace qvfr::panelui {

namespace {

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// A row as the document binds it.
struct RowView {
    Rml::String label, value, hint, fill;
    bool head = false, sel = false, on = false, toggle = false, button = false, dim = false, hasFill = false,
         showHint = false;
    bool operator==(const RowView& o) const {
        return label == o.label && value == o.value && hint == o.hint && fill == o.fill && head == o.head &&
               sel == o.sel && on == o.on && toggle == o.toggle && button == o.button && dim == o.dim &&
               hasFill == o.hasFill && showHint == o.showHint;
    }
};

struct StatView {
    Rml::String value, label;
    bool operator==(const StatView& o) const { return value == o.value && label == o.label; }
};

class System final : public Rml::SystemInterface {
public:
    double GetElapsedTime() override {
        LARGE_INTEGER f, t;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart / (double)f.QuadPart;
    }
    bool LogMessage(Rml::Log::Type type, const Rml::String& message) override {
        if (type <= Rml::Log::LT_WARNING) Log("Panel: RmlUi: %s", message.c_str());
        return true;
    }
};

HMODULE Self() {
    HMODULE h = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&Self), &h);
    return h;
}

// The look of the settings GUI (src/config/ui/style.rcss): dark graphite, one red accent, Barlow Condensed for the
// headings and values, Inter for the text. Sizes in px of the panel texture.
const char kRml[] = R"RML(<rml>
<head>
<title>QVFR panel</title>
<style>
div, body { display: block; }
body {
    width: 100%;
    height: 100%;
    background-color: #0d0c0e;
    color: #e9eaee;
    font-family: "Inter";
    font-weight: 400;
    font-size: 26px;
}
#top {
    display: flex;
    flex-direction: row;
    align-items: center;
    height: 84px;
    padding: 0 32px;
    background-color: #151418;
    border-bottom: 3px #e2231a;
}
.mark { width: 22px; height: 22px; border: 3px #ececf0; margin-right: 18px; }
.mark-in { width: 10px; height: 10px; margin: 2px 0 0 10px; background-color: #e2231a; }
.brand { font-family: "Barlow Condensed"; font-weight: 700; font-size: 38px; letter-spacing: 1px; margin-right: 18px; }
.title {
    font-family: "Barlow Condensed";
    font-weight: 600;
    font-size: 30px;
    letter-spacing: 1.5px;
    color: #9a9ea8;
    text-transform: uppercase;
}
#main { display: flex; flex-direction: row; }
#rows { width: 900px; padding: 8px 0 70px 0; }
#side {
    flex: 1;
    padding: 20px 32px 70px 32px;
    background-color: #111014;
    border-left: 2px #1d1c21;
}
.shead {
    font-family: "Barlow Condensed";
    font-weight: 600;
    font-size: 24px;
    letter-spacing: 2px;
    color: #e2231a;
    text-transform: uppercase;
}
.stat { margin-top: 18px; }
.sval { font-family: "Barlow Condensed"; font-weight: 700; font-size: 44px; color: #ffffff; }
.slab { font-size: 20px; color: #8d919b; }
.dhead { margin-top: 34px; }
.detail { display: flex; flex-direction: row; margin-top: 8px; font-size: 22px; color: #b4b8c1; }
.dlab { flex: 1; }
.dval { font-family: "Barlow Condensed"; font-weight: 600; font-size: 26px; color: #e9eaee; }
.snote { margin-top: 24px; font-size: 20px; color: #8d919b; }
.row { padding: 6px 32px 6px 28px; border-left: 4px #00000000; }
.row.head {
    padding: 20px 32px 4px 32px;
    font-family: "Barlow Condensed";
    font-weight: 600;
    font-size: 24px;
    letter-spacing: 2px;
    color: #e2231a;
    text-transform: uppercase;
}
.row.sel { background-color: #ffffff14; border-left: 4px #e2231a; }
.row.dim { color: #6b6e76; }
.line { display: flex; flex-direction: row; align-items: center; }
.label { flex: 1; }
.value {
    min-width: 170px;
    text-align: right;
    font-family: "Barlow Condensed";
    font-weight: 700;
    font-size: 32px;
    color: #ffffff;
}
.row.dim .value { color: #6b6e76; }
.value.toggle, .value.button {
    min-width: 0;
    padding: 0 14px;
    border-radius: 6px;
    background-color: #2a2a30;
    color: #9a9ea8;
    font-size: 26px;
}
.value.toggle.on { background-color: #e2231a; color: #ffffff; }
.bar { width: 190px; height: 6px; margin-right: 24px; background-color: #ffffff1a; }
.fill { height: 6px; background-color: #e2231a; }
.row.dim .fill { background-color: #6b6e76; }
.hint { margin-top: 2px; font-size: 20px; color: #8d919b; }
.row.sel .hint { color: #b4b8c1; }
#foot {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 30px;
    padding: 14px 32px;
    background-color: #151418;
    font-size: 20px;
    color: #9a9ea8;
}
</style>
</head>
<body data-model="panel">
<div id="top">
<div class="mark"><div class="mark-in"></div></div><div class="brand">QVFR</div><div class="title">{{title}}</div>
</div>
<div id="main">
<div id="rows">
<div data-for="r : rows" class="row" data-class-head="r.head" data-class-sel="r.sel" data-class-dim="r.dim">
<div class="line">
<div class="label">{{r.label}}</div>
<div class="bar" data-if="r.hasFill"><div class="fill" data-style-width="r.fill"></div></div>
<div class="value" data-if="!r.head" data-class-toggle="r.toggle" data-class-on="r.on"
     data-class-button="r.button">{{r.value}}</div>
</div>
<div class="hint" data-if="r.showHint">{{r.hint}}</div>
</div>
</div>
<div id="side" data-if="hasStats">
<div class="shead">{{statsHead}}</div>
<div class="stat" data-for="s : stats"><div class="sval">{{s.value}}</div><div class="slab">{{s.label}}</div></div>
<div class="shead dhead" data-if="hasDetails">{{detailsHead}}</div>
<div class="detail" data-for="d : details"><div class="dlab">{{d.label}}</div><div class="dval">{{d.value}}</div></div>
<div class="snote">{{statsNote}}</div>
</div>
</div>
<div id="foot">{{footer}}</div>
</body>
</rml>)RML";

}  // namespace

struct Ui::Impl {
    System sys;
    RenderInterface_DX11* render = nullptr;
    Rml::Context* ctx = nullptr;
    Rml::DataModelHandle model;
    std::vector<RowView> rows;
    std::vector<StatView> stats, details;
    Rml::String title, footer, statsHead, detailsHead, statsNote;
    bool hasStats = false, hasDetails = false;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3DDeviceContextState* state = nullptr;  // the panel's own D3D11 state, swapped in while it draws
    ID3D11DeviceContext1* ctx1 = nullptr;
    bool rml = false;  // Rml::Initialise done

    // Runs `f` with the panel's state on the immediate context; the game's state comes back after.
    template <class F>
    void InState(F&& f) {
        ID3DDeviceContextState* prev = nullptr;
        ctx1->SwapDeviceContextState(state, &prev);
        f();
        ctx1->SwapDeviceContextState(prev, nullptr);
        Rel(prev);
    }
};

bool Ui::Init(ID3D11Device* dev, unsigned w, unsigned h) {
    Release();
    auto fail = [&](const char* what, HRESULT hr = S_OK) {
        char b[192];
        snprintf(b, sizeof(b), "%s (0x%08lX)", what, (unsigned long)hr);
        why_ = b;
        Log("Panel: %s", b);
        Release();
        return false;
    };
    if (!dev || !w || !h) return fail("no device");
    // RmlUi's renderer compiles its shaders at run time; without the system's compiler there is no panel.
    if (!LoadLibraryW(L"d3dcompiler_47.dll")) return fail("no d3dcompiler_47.dll", HRESULT_FROM_WIN32(GetLastError()));
    impl_ = new Impl;
    Impl& m = *impl_;
    w_ = w;
    h_ = h;
    ID3D11Device1* dev1 = nullptr;
    HRESULT hr = dev->QueryInterface(__uuidof(ID3D11Device1), (void**)&dev1);
    if (FAILED(hr) || !dev1) return fail("no ID3D11Device1", hr);
    const D3D_FEATURE_LEVEL fl = dev->GetFeatureLevel();
    const UINT flags = (dev->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED)
                           ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED
                           : 0;
    hr = dev1->CreateDeviceContextState(flags, &fl, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device1), nullptr, &m.state);
    dev1->GetImmediateContext1(&m.ctx1);
    dev1->Release();
    if (FAILED(hr) || !m.state || !m.ctx1) return fail("CreateDeviceContextState", hr);

    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(hr = dev->CreateTexture2D(&d, nullptr, &tex_)) ||
        FAILED(hr = dev->CreateRenderTargetView(tex_, nullptr, &m.rtv)))
        return fail("panel texture", hr);

    const char* bad = nullptr;
    m.InState([&] {
        Rml::SetSystemInterface(&m.sys);
        m.render = new RenderInterface_DX11(dev);
        Rml::SetRenderInterface(m.render);
        if (!Rml::Initialise()) {
            bad = "Rml::Initialise";
            return;
        }
        m.rml = true;
        struct Face {
            const wchar_t* resource;
            const char* family;
            int weight;
        } faces[] = {{L"PANEL_FONT_BODY_400", "Inter", 400},
                     {L"PANEL_FONT_BODY_600", "Inter", 600},
                     {L"PANEL_FONT_HEAD_600", "Barlow Condensed", 600},
                     {L"PANEL_FONT_HEAD_700", "Barlow Condensed", 700}};
        const HMODULE self = Self();
        int fonts = 0;
        for (const Face& f : faces) {
            HRSRC r = FindResourceW(self, f.resource, MAKEINTRESOURCEW(10));  // RT_RCDATA; resources stay mapped
            HGLOBAL g = r ? LoadResource(self, r) : nullptr;
            if (g && Rml::LoadFontFace({(const Rml::byte*)LockResource(g), (size_t)SizeofResource(self, r)}, f.family,
                                       Rml::Style::FontStyle::Normal, (Rml::Style::FontWeight)f.weight))
                ++fonts;
        }
        if (fonts < 4) {
            bad = "the panel fonts (panel.rc) are missing";
            return;
        }
        m.ctx = Rml::CreateContext("qvfr_panel", Rml::Vector2i((int)w, (int)h));
        if (!m.ctx) {
            bad = "Rml::CreateContext";
            return;
        }
        Rml::DataModelConstructor mc = m.ctx->CreateDataModel("panel");
        if (auto s = mc.RegisterStruct<RowView>()) {
            s.RegisterMember("label", &RowView::label);
            s.RegisterMember("value", &RowView::value);
            s.RegisterMember("hint", &RowView::hint);
            s.RegisterMember("fill", &RowView::fill);
            s.RegisterMember("head", &RowView::head);
            s.RegisterMember("sel", &RowView::sel);
            s.RegisterMember("on", &RowView::on);
            s.RegisterMember("toggle", &RowView::toggle);
            s.RegisterMember("button", &RowView::button);
            s.RegisterMember("dim", &RowView::dim);
            s.RegisterMember("hasFill", &RowView::hasFill);
            s.RegisterMember("showHint", &RowView::showHint);
        }
        mc.RegisterArray<std::vector<RowView>>();
        if (auto s = mc.RegisterStruct<StatView>()) {
            s.RegisterMember("value", &StatView::value);
            s.RegisterMember("label", &StatView::label);
        }
        mc.RegisterArray<std::vector<StatView>>();
        mc.Bind("stats", &m.stats);
        mc.Bind("hasStats", &m.hasStats);
        mc.Bind("statsHead", &m.statsHead);
        mc.Bind("details", &m.details);
        mc.Bind("hasDetails", &m.hasDetails);
        mc.Bind("detailsHead", &m.detailsHead);
        mc.Bind("statsNote", &m.statsNote);
        mc.Bind("rows", &m.rows);
        mc.Bind("title", &m.title);
        mc.Bind("footer", &m.footer);
        m.model = mc.GetModelHandle();
        Rml::ElementDocument* doc = m.ctx->LoadDocumentFromMemory(kRml, "panel.rml");
        if (!doc) {
            bad = "the panel document";
            return;
        }
        doc->Show();
    });
    if (bad) return fail(bad);
    ready_ = true;
    dirty_ = true;
    Log("Panel: ready, %ux%u RGBA8 on the game's device (RmlUi %s)", w, h, Rml::GetVersion().c_str());
    return true;
}

void Ui::Set(const std::vector<Row>& rows, int selected, const std::string& title, const std::string& footer,
             const std::vector<Stat>& stats, const std::string& statsHead, const std::vector<Stat>& details,
             const std::string& detailsHead, const std::string& statsNote) {
    if (!impl_ || !ready_) return;
    Impl& m = *impl_;
    std::vector<RowView> v;
    v.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        RowView x;
        x.label = r.label;
        x.value = r.value;
        x.hint = r.hint;
        x.head = r.kind == Row::Head;
        x.toggle = r.kind == Row::Toggle;
        x.button = r.kind == Row::Button;
        x.on = r.on;
        x.dim = r.dim;
        x.sel = (int)i == selected;
        x.showHint = !r.hint.empty();
        x.hasFill = r.fill >= 0;
        x.fill = "0%";  // also without a bar: the style binding is read anyway
        if (x.hasFill) {
            char b[16];
            snprintf(b, sizeof(b), "%.1f%%", (double)(std::clamp)(r.fill, 0.0f, 1.0f) * 100.0);
            x.fill = b;
        }
        v.push_back(std::move(x));
    }
    if (v != m.rows) {
        m.rows = std::move(v);
        m.model.DirtyVariable("rows");
        dirty_ = true;
    }
    if (title != m.title) {
        m.title = title;
        m.model.DirtyVariable("title");
        dirty_ = true;
    }
    if (footer != m.footer) {
        m.footer = footer;
        m.model.DirtyVariable("footer");
        dirty_ = true;
    }
    std::vector<StatView> sv;
    for (const Stat& s : stats) sv.push_back({s.value, s.label});
    if (sv != m.stats) {
        m.stats = std::move(sv);
        m.model.DirtyVariable("stats");
        dirty_ = true;
    }
    if (m.hasStats != !stats.empty()) {
        m.hasStats = !stats.empty();
        m.model.DirtyVariable("hasStats");
        dirty_ = true;
    }
    std::vector<StatView> dv;
    for (const Stat& s : details) dv.push_back({s.value, s.label});
    if (dv != m.details) {
        m.details = std::move(dv);
        m.model.DirtyVariable("details");
        dirty_ = true;
    }
    if (m.hasDetails != !details.empty()) {
        m.hasDetails = !details.empty();
        m.model.DirtyVariable("hasDetails");
        dirty_ = true;
    }
    if (detailsHead != m.detailsHead) {
        m.detailsHead = detailsHead;
        m.model.DirtyVariable("detailsHead");
        dirty_ = true;
    }
    if (statsHead != m.statsHead) {
        m.statsHead = statsHead;
        m.model.DirtyVariable("statsHead");
        dirty_ = true;
    }
    if (statsNote != m.statsNote) {
        m.statsNote = statsNote;
        m.model.DirtyVariable("statsNote");
        dirty_ = true;
    }
}

bool Ui::Render(ID3D11DeviceContext* c, bool force) {
    if (!ready_ || !impl_ || !c || (!dirty_ && !force)) return false;
    Impl& m = *impl_;
    m.InState([&] {
        m.ctx->Update();
        m.render->SetViewport((int)w_, (int)h_);
        m.render->BeginFrame();
        m.ctx->Render();
        m.render->EndFrame(m.rtv);
    });
    dirty_ = false;
    return true;
}

void Ui::Release() {
    if (impl_) {
        Impl& m = *impl_;
        if (m.ctx1 && m.state) {
            m.InState([&] {
                if (m.ctx) Rml::RemoveContext("qvfr_panel");
                if (m.rml) Rml::Shutdown();
                delete m.render;  // after Shutdown, which releases RmlUi's textures through it
            });
        } else {
            delete m.render;
        }
        m.render = nullptr;
        Rel(m.rtv);
        Rel(m.state);
        Rel(m.ctx1);
        delete impl_;
        impl_ = nullptr;
    }
    Rel(tex_);
    ready_ = false;
    dirty_ = true;
}

}  // namespace qvfr::panelui

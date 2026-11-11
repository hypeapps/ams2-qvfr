// The tuning panel's picture (dlaa-plan D7.P): RmlUi (MIT, third_party/rmlui-6.3) lays out a list of rows and its
// D3D11 renderer draws it into a texture on the game's device. The game's whole D3D11 state is swapped out while the
// panel draws (ID3D11DeviceContext1::SwapDeviceContextState), so the game sees no change. Fonts come from the module's
// resources (panel.rc). No dependency on the rest of the proxy except Log (common.h), so qvfr_proxy_smoke --panel tests
// it directly. One Ui at a time (RmlUi is global); render thread only.
#pragma once

#include <d3d11.h>

#include <string>
#include <vector>

namespace qvfr::panelui {

struct Row {
    enum Kind { Head, Value, Toggle, Button };
    Kind kind = Value;
    std::string label;
    std::string value;   // the value as shown ("0.50", "1/2", "ON")
    std::string hint;    // a short description under the row, in a smaller font
    float fill = -1;     // 0..1: a bar under the value; < 0 none
    bool on = false;     // Toggle: switched on
    bool dim = false;    // not in effect now (e.g. outer options with the outer images off)
};

// A measured value in the column right of the rows ("90 fps", "frame rate").
struct Stat {
    std::string value, label;
    bool operator==(const Stat& o) const { return value == o.value && label == o.label; }
};

class Ui {
public:
    Ui() = default;
    Ui(const Ui&) = delete;
    Ui& operator=(const Ui&) = delete;
    ~Ui() { Release(); }

    // RmlUi, the fonts, the document and a w x h RGBA8 texture on `dev`. False: why().
    bool Init(ID3D11Device* dev, unsigned w, unsigned h);
    // The rows (selected: index into rows, -1 none), the title and the footer line, and in a column right of the
    // rows the measured values (stats: large; details: small label and value lines under them, e.g. the parts of a
    // cost) with a note; redraws at the next Render when something changed.
    void Set(const std::vector<Row>& rows, int selected, const std::string& title, const std::string& footer,
             const std::vector<Stat>& stats = {}, const std::string& statsHead = {},
             const std::vector<Stat>& details = {}, const std::string& detailsHead = {},
             const std::string& statsNote = {});
    // Draws into the texture when something changed (or `force`), on `c` (the device's immediate context). True: drawn.
    bool Render(ID3D11DeviceContext* c, bool force = false);
    ID3D11Texture2D* Texture() const { return tex_; }
    unsigned Width() const { return w_; }
    unsigned Height() const { return h_; }
    void Release();

    bool ready() const { return ready_; }
    const std::string& why() const { return why_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    ID3D11Texture2D* tex_ = nullptr;
    unsigned w_ = 0, h_ = 0;
    bool ready_ = false;
    bool dirty_ = true;
    std::string why_;
};

}  // namespace qvfr::panelui

#include "sr_probe.h"

#include <d3d11_1.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "clones.h"
#include "common.h"
#include "gpu.h"
#include "hooks.h"

namespace qvfr::srprobe {

namespace {

constexpr UINT kSlots = D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;
constexpr size_t kMaxRecords = 20000;  // constant buffer windows copied in the probed frame
// Draws copied per kind, phase, caller and shader pair: the scene's thousands of geometry draws used the whole limit in
// outer L's scene phase (run 2, 2026-11-12), so the passes after TAA and the focus groups went unread.
constexpr int kDrawsPerCombo = 2;
// Only the window a draw sees is copied (the engine binds windows of large buffers through *SetConstantBuffers1: a copy
// of each whole buffer ran the GPU out of memory and the game crashed, 2026-11-12), at most a constant buffer's 4096
// constants, and at most this much in all.
constexpr UINT kMaxWindow = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16;
constexpr uint64_t kMaxBytes = 128ull << 20;
constexpr size_t kMaxLines = 200;      // places in the report
constexpr size_t kMaxCallerLines = 60;

struct Rec {
    clones::SrGroup g;
    void* caller = nullptr;
    const char* draw = "";
    void* vs = nullptr;
    void* ps = nullptr;
    int stage = 0;  // 0 VS, 1 PS
    UINT slot = 0;
    UINT first = 0;  // where the copied window starts in the buffer, in 16-byte constants
    UINT bytes = 0;  // the copy's size (the window)
    ID3D11Buffer* staging = nullptr;
    UINT rtW = 0, rtH = 0;  // the bound target's size (first render target, else depth)
};

std::atomic<bool> g_armed{false};
bool g_installed = false;
bool g_fault = false;
bool g_keyDown = false;
uint64_t g_frame = 0, g_probedFrame = 0;
std::vector<Rec> g_recs;
std::multimap<UINT, ID3D11Buffer*> g_pool;  // free staging buffers by size
uint64_t g_draws = 0, g_srDraws = 0, g_dropped = 0, g_copyFailed = 0, g_bytes = 0;
bool g_stop = false;  // a staging buffer could not be made: no more copies in this probe frame
ID3D11DeviceContext1* g_ctx1 = nullptr;  // for the bound windows of VS/PSSetConstantBuffers1
// Per caller of the probed frame's draws in SR groups: draws per phase and kind, a sample of shaders and the target.
struct CallerInfo {
    uint64_t draws[2][2] = {};  // [kind][post]
    void* vs = nullptr;
    void* ps = nullptr;
    const char* draw = "";
    UINT rtW = 0, rtH = 0;
};
std::map<void*, CallerInfo> g_callers;
std::map<std::tuple<int, bool, void*, void*, void*>, int> g_combos;  // draws copied per kind, phase, caller, VS, PS
uint64_t g_skipped = 0;                                             // draws not copied (their combination had enough)
size_t g_lastPlaces = 0, g_lastSuspects = 0;  // of the last report
ID3D11DeviceContext* g_testCtx = nullptr;     // smoke test: the context probed instead of the game's


template <class F>
void Guard(const char* where, F&& f) {
    if (g_fault) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_fault = true;
        g_armed = false;
        Log("ERROR: exception 0x%08lX in the SR probe (%s); the probe is off for this session", code, where);
    }
}

std::string Where(const void* p) {
    if (!p) return "?";
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)p, &m) ||
        !m) {
        char b[32];
        snprintf(b, sizeof(b), "%p", p);
        return b;
    }
    wchar_t path[MAX_PATH] = L"";
    GetModuleFileNameW(m, path, MAX_PATH);
    const wchar_t* name = wcsrchr(path, L'\\');
    name = name ? name + 1 : path;
    char b[160];
    snprintf(b, sizeof(b), "%ls+0x%llX", name, (unsigned long long)((uintptr_t)p - (uintptr_t)m));
    return b;
}

ID3D11Buffer* TakeStaging(ID3D11Device* dev, UINT bytes) {
    if (auto it = g_pool.find(bytes); it != g_pool.end()) {
        ID3D11Buffer* b = it->second;
        g_pool.erase(it);
        return b;
    }
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = bytes;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* b = nullptr;
    return SUCCEEDED(dev->CreateBuffer(&d, nullptr, &b)) ? b : nullptr;
}

// A constant buffer cannot be copied in part, so it goes whole into a scratch buffer of its size (one per size, used
// again for every copy: the GPU runs the copies in order) and the window from there into the staging buffer.
std::map<UINT, ID3D11Buffer*> g_scratch;
uint64_t g_scratchBytes = 0;

ID3D11Buffer* ScratchFor(ID3D11Device* dev, UINT bytes) {
    if (auto it = g_scratch.find(bytes); it != g_scratch.end()) return it->second;
    if (g_scratchBytes + bytes > kMaxBytes) return nullptr;
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = bytes;
    d.Usage = D3D11_USAGE_DEFAULT;
    ID3D11Buffer* b = nullptr;
    if (FAILED(dev->CreateBuffer(&d, nullptr, &b)) || !b) return nullptr;
    g_scratch[bytes] = b;
    g_scratchBytes += bytes;
    return b;
}

void ReleaseScratch() {
    for (auto& [sz, b] : g_scratch) b->Release();
    g_scratch.clear();
    g_scratchBytes = 0;
}

void TargetSize(ID3D11DeviceContext* c, UINT& w, UINT& h) {
    w = h = 0;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(1, &rtv, &dsv);
    ID3D11View* v = rtv ? (ID3D11View*)rtv : (ID3D11View*)dsv;
    if (v) {
        ID3D11Resource* r = nullptr;
        v->GetResource(&r);
        ID3D11Texture2D* t = nullptr;
        if (r && SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
            D3D11_TEXTURE2D_DESC d{};
            t->GetDesc(&d);
            w = d.Width;
            h = d.Height;
            t->Release();
        }
        if (r) r->Release();
    }
    if (rtv) rtv->Release();
    if (dsv) dsv->Release();
}

void Capture(ID3D11DeviceContext* c, void* caller, const char* draw) {
    ++g_draws;
    clones::SrGroup g;
    if (!clones::SrGroupNow(g)) return;
    ++g_srDraws;
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    c->VSGetShader(&vs, nullptr, nullptr);
    c->PSGetShader(&ps, nullptr, nullptr);
    UINT rtW = 0, rtH = 0;
    TargetSize(c, rtW, rtH);
    CallerInfo& ci = g_callers[caller];
    ++ci.draws[g.kind & 1][g.post ? 1 : 0];
    if (!ci.ps) {
        ci.vs = vs;
        ci.ps = ps;
        ci.draw = draw;
        ci.rtW = rtW;
        ci.rtH = rtH;
    }
    const bool copy = ++g_combos[{g.kind, g.post, caller, (void*)vs, (void*)ps}] <= kDrawsPerCombo;
    if (!copy) ++g_skipped;
    for (int stage = 0; stage < 2 && copy; ++stage) {
        ID3D11Buffer* cb[kSlots] = {};
        UINT first[kSlots] = {}, num[kSlots] = {};
        if (g_ctx1) {
            if (stage == 0)
                g_ctx1->VSGetConstantBuffers1(0, kSlots, cb, first, num);
            else
                g_ctx1->PSGetConstantBuffers1(0, kSlots, cb, first, num);
        } else if (stage == 0) {
            c->VSGetConstantBuffers(0, kSlots, cb);
        } else {
            c->PSGetConstantBuffers(0, kSlots, cb);
        }
        for (UINT s = 0; s < kSlots; ++s) {
            if (!cb[s]) continue;
            D3D11_BUFFER_DESC d{};
            cb[s]->GetDesc(&d);
            // The window: [first, first + num) constants with *GetConstantBuffers1, else the buffer's start.
            const UINT off = g_ctx1 && num[s] ? first[s] * 16 : 0;
            const UINT want = g_ctx1 && num[s] ? num[s] * 16 : d.ByteWidth;
            const UINT len = off < d.ByteWidth ? std::min({want, d.ByteWidth - off, kMaxWindow}) & ~15u : 0;
            ID3D11Buffer* st = nullptr;
            ID3D11Buffer* scratch = nullptr;
            if (!len) {
                ++g_copyFailed;
            } else if (g_stop || g_recs.size() >= kMaxRecords || g_bytes + len > kMaxBytes) {
                ++g_dropped;
            } else if (!(scratch = ScratchFor(dev, d.ByteWidth)) || !(st = TakeStaging(dev, len))) {
                ++g_copyFailed;
                g_stop = true;
                Log("SR probe: no staging buffer of %u bytes (device %s); no more copies in this frame", len,
                    dev->GetDeviceRemovedReason() == S_OK ? "fine" : "REMOVED");
            }
            if (st) {
                {
                    clones::RawScope raw;  // our copy passes the clones' hooks as it is
                    c->CopyResource(scratch, cb[s]);
                    const D3D11_BOX box{off, 0, 0, off + len, 1, 1};
                    c->CopySubresourceRegion(st, 0, 0, 0, 0, scratch, 0, &box);
                }
                g_bytes += len;
                Rec r;
                r.g = g;
                r.caller = caller;
                r.draw = draw;
                r.vs = vs;
                r.ps = ps;
                r.stage = stage;
                r.slot = s;
                r.first = off / 16;
                r.bytes = len;
                r.staging = st;
                r.rtW = rtW;
                r.rtH = rtH;
                g_recs.push_back(r);
            }
            cb[s]->Release();
        }
    }
    if (vs) vs->Release();  // only compared and printed from here on
    if (ps) ps->Release();
    dev->Release();
}

// The sizes a float may be: R, O and G, each also at 1/2, 1/4, 1/8 (rounded down and up), as width or height, their
// reciprocal, half reciprocal (half a pixel) and double reciprocal.
struct Cand {
    float v;
    std::string label;
    char size;  // 'R', 'O', 'G' or 'B' (O and G the same size)
};

std::vector<Cand> Candidates(const clones::SrGroup& g) {
    std::vector<Cand> out;
    struct Sz {
        char name;
        uint32_t w, h;
    };
    const bool og = g.ow == g.gw && g.oh == g.gh;
    std::vector<Sz> sizes = {{'R', g.rw, g.rh}, {og ? 'B' : 'O', g.ow, g.oh}};
    if (!og) sizes.push_back({'G', g.gw, g.gh});
    for (const Sz& sz : sizes)
        for (int d : {1, 2, 4, 8})
            for (int up = 0; up < 2; ++up) {
                if (d == 1 && up) continue;
                const uint32_t w = up ? (sz.w + d - 1) / d : sz.w / d, h = up ? (sz.h + d - 1) / d : sz.h / d;
                if (!w || !h) continue;
                char frac[16] = "";
                if (d > 1) snprintf(frac, sizeof(frac), "/%d%s", d, up ? "^" : "");
                const char* nm = sz.name == 'B' ? "O=G" : sz.name == 'R' ? "R" : sz.name == 'O' ? "O" : "G";
                auto add = [&](float v, const char* form) {
                    out.push_back({v, std::string(nm) + frac + " " + form, sz.name});
                };
                add((float)w, "w");
                add((float)h, "h");
                add(1.0f / w, "1/w");
                add(1.0f / h, "1/h");
                add(0.5f / w, "0.5/w");
                add(0.5f / h, "0.5/h");
                add(2.0f / w, "2/w");
                add(2.0f / h, "2/h");
            }
    std::sort(out.begin(), out.end(), [](const Cand& a, const Cand& b) { return a.v < b.v; });
    return out;
}

// All sizes v may be: their labels joined with " | ", and whether one is the render size and one another size.
struct Matched {
    std::string label;
    bool r = false, other = false;
};
bool Match(const std::vector<Cand>& cs, float v, Matched& out) {
    out = {};
    if (!std::isfinite(v) || v <= 0) return false;
    const float tol = v * 2e-5f;
    auto it = std::lower_bound(cs.begin(), cs.end(), v - tol, [](const Cand& c, float x) { return c.v < x; });
    for (; it != cs.end() && it->v <= v + tol; ++it) {
        out.label += (out.label.empty() ? "" : " | ") + it->label;
        (it->size == 'R' ? out.r : out.other) = true;
    }
    return !out.label.empty();
}

struct Place {
    int kind;
    bool post;
    void* caller;
    int stage;
    UINT slot, index;
    std::string label;
    bool operator<(const Place& o) const {
        return std::tie(kind, post, caller, stage, slot, index, label) <
               std::tie(o.kind, o.post, o.caller, o.stage, o.slot, o.index, o.label);
    }
};
struct Found {
    uint64_t draws = 0;
    float value = 0;
    bool suspect = false;  // the wrong size for the phase only
    bool maybe = false;    // the wrong size or the right one (the same value)
    void* vs = nullptr;
    void* ps = nullptr;
    const char* draw = "";
    UINT rtW = 0, rtH = 0;
    unsigned groups = 0;  // bit per group number
};

void Report(ID3D11DeviceContext* c) {
    std::map<Place, Found> found;
    uint64_t mapped = 0, failed = 0;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, std::vector<Cand>> cache;
    for (Rec& r : g_recs) {
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(c->Map(r.staging, 0, D3D11_MAP_READ, 0, &m)) || !m.pData) {
            ++failed;
            continue;
        }
        ++mapped;
        auto key = std::make_tuple(r.g.rw, r.g.rh, r.g.ow, r.g.oh, r.g.gw, r.g.gh);
        auto ct = cache.find(key);
        if (ct == cache.end()) ct = cache.emplace(key, Candidates(r.g)).first;
        const float* f = (const float*)m.pData;
        const UINT n = r.bytes / 4;
        const UINT lo = 0, hi = n;  // the copy is the bound window
        for (UINT i = lo; i < hi; ++i) {
            Matched md;
            if (!Match(ct->second, f[i], md)) continue;
            Found& x = found[{r.g.kind, r.g.post, r.caller, r.stage, r.slot, i - lo, md.label}];
            ++x.draws;
            x.value = f[i];
            // After TAA the targets have the view size, before it the render size.
            const bool wrong = r.g.post ? md.r : md.other, right = r.g.post ? md.other : md.r;
            x.suspect = wrong && !right;
            x.maybe = wrong && right;
            if (!x.ps) {
                x.vs = r.vs;
                x.ps = r.ps;
                x.draw = r.draw;
                x.rtW = r.rtW;
                x.rtH = r.rtH;
            }
            if (r.g.group > 0 && r.g.group < 32) x.groups |= 1u << r.g.group;
        }
        c->Unmap(r.staging, 0);
    }
    for (Rec& r : g_recs) r.staging->Release();  // the memory back at once (no pool between probes)
    g_recs.clear();
    for (auto& [sz, b] : g_pool) b->Release();
    g_pool.clear();
    ReleaseScratch();
    std::vector<std::pair<Place, Found>> list(found.begin(), found.end());
    std::stable_sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
        const int ra = a.second.suspect ? 2 : a.second.maybe ? 1 : 0, rb = b.second.suspect ? 2 : b.second.maybe ? 1 : 0;
        if (ra != rb) return ra > rb;
        return a.second.draws > b.second.draws;
    });
    size_t suspects = 0;
    for (const auto& [p, x] : list) suspects += x.suspect ? 1 : 0;
    g_lastPlaces = list.size();
    g_lastSuspects = suspects;
    Log("SR probe (frame %llu): %llu draws, %llu in eye groups with DLSS SR's scene sets (%llu not copied: %d per kind, "
        "phase, caller and shaders); %llu constant buffer windows read, %.1f MB (%llu over the limits of %zu windows or %llu MB, %llu not copied, %llu not mapped); %zu places "
        "with a size (%zu SUSPECT: R after the TAA resolve, O or G before it; MAYBE: the value is both). Place = draw "
        "caller, stage, cb slot [float index in the bound window]:",
        (unsigned long long)g_probedFrame, (unsigned long long)g_draws, (unsigned long long)g_srDraws,
        (unsigned long long)g_skipped, kDrawsPerCombo, (unsigned long long)mapped, g_bytes / 1048576.0, (unsigned long long)g_dropped, kMaxRecords,
        (unsigned long long)(kMaxBytes >> 20), (unsigned long long)g_copyFailed, (unsigned long long)failed,
        list.size(), suspects);
    size_t shown = 0;
    for (const auto& [p, x] : list) {
        if (shown++ >= kMaxLines) break;
        std::string groups;
        for (int i = 1; i < 32; ++i)
            if (x.groups & (1u << i)) groups += (groups.empty() ? "" : ",") + std::to_string(i);
        char line[640];
        snprintf(line, sizeof(line),
                 "  %-7s %-5s %-6s %-28s %s cb%u[%u] = %.7g (%s) in %llu draws (%s; groups %s; VS %p PS %p; target "
                 "%ux%u)",
                 x.suspect ? "SUSPECT" : x.maybe ? "MAYBE" : "", p.kind ? "focus" : "outer", p.post ? "post" : "scene",
                 Where(p.caller).c_str(), p.stage ? "PS" : "VS", p.slot, p.index, x.value, p.label.c_str(),
                 (unsigned long long)x.draws, x.draw, groups.c_str(), x.vs, x.ps, x.rtW, x.rtH);
        LogRaw(line);
    }
    if (list.size() > kMaxLines) Log("SR probe: %zu more places not listed", list.size() - kMaxLines);
    // The callers after the TAA resolve (where the windscreen, particles, flares and the sun's effects draw).
    std::vector<std::pair<void*, CallerInfo>> cl(g_callers.begin(), g_callers.end());
    std::sort(cl.begin(), cl.end(), [](const auto& a, const auto& b) {
        const uint64_t pa = a.second.draws[0][1] + a.second.draws[1][1], pb = b.second.draws[0][1] + b.second.draws[1][1];
        if (pa != pb) return pa > pb;
        return a.second.draws[0][0] + a.second.draws[1][0] > b.second.draws[0][0] + b.second.draws[1][0];
    });
    Log("SR probe: draw callers in the SR groups (%zu; draws outer scene / outer post / focus scene / focus post, "
        "first draw's shaders and target):",
        cl.size());
    size_t n = 0;
    for (const auto& [caller, ci] : cl) {
        if (n++ >= kMaxCallerLines) break;
        char line[320];
        snprintf(line, sizeof(line), "  %-28s %5llu / %5llu / %5llu / %5llu  %s VS %p PS %p target %ux%u",
                 Where(caller).c_str(), (unsigned long long)ci.draws[0][0], (unsigned long long)ci.draws[0][1],
                 (unsigned long long)ci.draws[1][0], (unsigned long long)ci.draws[1][1], ci.draw, ci.vs, ci.ps,
                 ci.rtW, ci.rtH);
        LogRaw(line);
    }
    g_callers.clear();
    g_combos.clear();
}

}  // namespace

void Install(ID3D11DeviceContext* imm) {
    if (g_installed || !imm || !Cfg().srProbeKey) return;
    g_installed = true;
    Guard("Install", [&] {
        imm->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
        Log("SR probe: hotkey 0x%X+0x%X (Draw and DrawIndexed through sun effects, the instanced draws through the SR "
            "constants fix's hooks), constant buffer windows %s",
            Cfg().srProbeKeyMod, Cfg().srProbeKey, g_ctx1 ? "read (ID3D11DeviceContext1)" : "not known (whole buffers)");
    });
}

bool Armed() { return g_armed.load(std::memory_order_relaxed); }

void OnDraw(ID3D11DeviceContext* c, void* caller, const char* draw) {
    if (!g_armed.load(std::memory_order_relaxed) || c != (g_testCtx ? g_testCtx : gpu::Context())) return;
    Guard("draw", [&] { Capture(c, caller, draw); });
}

void OnFrameStart(ID3D11DeviceContext* imm) {
    if (!g_installed || g_fault) return;
    ++g_frame;
    Guard("frame", [&] {
        if (g_armed.exchange(false) && imm) Report(imm);
        const Config& c = Cfg();
        const bool down = c.srProbeKey && (GetAsyncKeyState(c.srProbeKey) & 0x8000) &&
                          (!c.srProbeKeyMod || (GetAsyncKeyState(c.srProbeKeyMod) & 0x8000));
        const bool pressed = down && !g_keyDown;
        g_keyDown = down;
        if (!pressed) return;
        g_draws = g_srDraws = g_dropped = g_copyFailed = g_bytes = g_skipped = 0;
        g_stop = false;
        g_callers.clear();
    g_combos.clear();
        g_probedFrame = g_frame;
        g_armed = true;
        Log("SR probe (hotkey): the constant buffers of frame %llu's draws in the eye groups are read (a short stall)",
            (unsigned long long)g_frame);
    });
}

void TestArm(ID3D11DeviceContext* c) {
    g_installed = true;
    g_testCtx = c;
    if (!g_ctx1 && c) c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
    g_draws = g_srDraws = g_dropped = g_copyFailed = g_bytes = g_skipped = 0;
        g_stop = false;
    g_callers.clear();
    g_combos.clear();
    g_armed = true;
}

size_t TestReport(ID3D11DeviceContext* c, size_t* suspects) {
    g_lastPlaces = g_lastSuspects = 0;
    if (g_armed.exchange(false)) Guard("test report", [&] { Report(c); });
    if (suspects) *suspects = g_lastSuspects;
    return g_lastPlaces;
}

void Shutdown() {
    g_armed = false;
    Guard("Shutdown", [&] {
        for (Rec& r : g_recs)
            if (r.staging) r.staging->Release();
        g_recs.clear();
        for (auto& [sz, b] : g_pool) b->Release();
        g_pool.clear();
        ReleaseScratch();
        g_callers.clear();
    g_combos.clear();
    });
}

}  // namespace qvfr::srprobe

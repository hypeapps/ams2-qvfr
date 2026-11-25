#include "fwd_diag.h"

#include <d3d11_1.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "dxbc.h"

namespace qvfr::fwddiag {
namespace {

constexpr int kDetailPasses = 3;            // per eye: the first passes with draws in a pause, every draw listed
constexpr double kDetailEveryMs = 300000.0;  // then one such pass per eye every 5 minutes
constexpr unsigned kDetailDraws = 160;       // draw lines per detailed pass
constexpr size_t kMaxKinds = 256;
constexpr size_t kMaxShaders = 65536;
constexpr size_t kMaxIdle = 16;  // distinct binds of a pause without draws, logged once each
constexpr int kStackFrames = 16;
constexpr size_t kMaxText = 3000;
const char* const kEye[2] = {"L", "R"};
// WKPDID_D3DDebugObjectName (d3dcommon.h), here so that no GUID library is needed.
constexpr GUID kDebugName = {0x429b8c22, 0x9188, 0x4b0c, {0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00}};

std::atomic<bool> g_dead{false};

void Fail(const char* where, unsigned long code) {
    if (g_dead.exchange(true)) return;
    Log("Forward diag: ERROR: exception 0x%08lX in %s; the forward diagnostics are off for this session", code, where);
}

template <class F>
void Protect(const char* where, F&& f) {
    if (g_dead.load(std::memory_order_relaxed)) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

// ---------------------------------------------------------------- shaders

struct Binding {
    uint32_t type = 0, point = 0, count = 0;  // D3D_SHADER_INPUT_TYPE, first register, registers
    std::string name;
};
struct Shader {
    char stage = 0;
    char sum[40] = {};
    uint32_t bytes = 0;
    std::string text;  // the reflection summary
    std::vector<Binding> binds;
    bool logged = false;
};
std::shared_mutex g_sm;
std::unordered_map<void*, Shader> g_shaders;
std::atomic<uint64_t> g_described{0}, g_skipped{0};

uint32_t Get32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

std::string Str(const std::vector<uint8_t>& d, uint32_t off) {
    if (off >= d.size()) return {};
    const char* s = reinterpret_cast<const char*>(d.data()) + off;
    return std::string(s, strnlen(s, d.size() - off));
}

char RegLetter(uint32_t type) {
    switch (type) {
        case 0: return 'b';  // constant buffer
        case 3: return 's';  // sampler
        case 4: case 6: case 8: case 9: case 10: case 11: return 'u';  // UAVs
        default: return 't';  // texture, tbuffer, structured and raw buffers
    }
}

void SumText(const void* code, size_t len, char out[40]) {
    uint32_t s[4] = {};
    if (len >= 20) memcpy(s, static_cast<const uint8_t*>(code) + 4, 16);
    snprintf(out, 40, "%08x-%08x-%08x-%08x", s[0], s[1], s[2], s[3]);
}

std::string SigText(std::vector<dxbc::Chunk>& chunks, uint32_t fourcc) {
    dxbc::Chunk* ch = dxbc::Find(chunks, fourcc);
    if (!ch) return "-";
    std::vector<dxbc::SigElement> el;
    if (!dxbc::ReadSignature(*ch, el, nullptr)) return "?";
    std::string s;
    for (const dxbc::SigElement& e : el) {
        if (!s.empty()) s += ' ';
        s += e.name + std::to_string(e.semanticIndex);
    }
    return s.empty() ? "-" : s;
}

void Describe(const void* code, size_t len, Shader& sh) {
    std::vector<dxbc::Chunk> chunks;
    if (!dxbc::Parse(code, len, chunks, nullptr)) {
        sh.text = "not a DXBC container";
        return;
    }
    std::string cb, tex, smp, uav;
    const dxbc::Chunk* rdef = dxbc::Find(chunks, dxbc::FourCC('R', 'D', 'E', 'F'));
    if (rdef && rdef->data.size() >= 28) {
        const std::vector<uint8_t>& d = rdef->data;
        const uint32_t bindCount = Get32(d.data() + 8), bindOffset = Get32(d.data() + 12), target = Get32(d.data() + 16);
        const uint32_t major = (target >> 8) & 0xFF, minor = target & 0xFF;
        const size_t stride = major == 5 && minor >= 1 ? 40 : 32;
        for (uint32_t i = 0; i < bindCount && bindOffset + stride * (i + 1) <= d.size(); ++i) {
            const uint8_t* b = d.data() + bindOffset + stride * i;
            Binding x;
            x.name = Str(d, Get32(b));
            x.type = Get32(b + 4);
            x.point = Get32(b + 20);
            x.count = Get32(b + 24);
            const char r = RegLetter(x.type);
            std::string& out = r == 'b' ? cb : r == 's' ? smp : r == 'u' ? uav : tex;
            char reg[40];
            if (x.count > 1)
                snprintf(reg, sizeof(reg), "@%c%u[%u]", r, x.point, x.count);
            else
                snprintf(reg, sizeof(reg), "@%c%u", r, x.point);
            if (!out.empty()) out += ' ';
            out += x.name + reg;
            sh.binds.push_back(std::move(x));
        }
    } else {
        sh.text = "no reflection; ";
    }
    auto list = [](const std::string& s) { return s.empty() ? std::string("-") : s; };
    sh.text += "cb: " + list(cb) + "; t: " + list(tex) + "; s: " + list(smp) + "; u: " + list(uav) +
               "; in: " + SigText(chunks, dxbc::FourCC('I', 'S', 'G', 'N')) +
               "; out: " + SigText(chunks, dxbc::FourCC('O', 'S', 'G', 'N'));
    if (sh.text.size() > kMaxText) sh.text.resize(kMaxText);
}

bool FindShader(void* p, Shader& out) {
    if (!p) return false;
    std::shared_lock lock(g_sm);
    const auto it = g_shaders.find(p);
    if (it == g_shaders.end()) return false;
    out = it->second;
    return true;
}

// True the first time for a shader (it goes to the log then).
bool FirstLog(void* p) {
    std::unique_lock lock(g_sm);
    const auto it = g_shaders.find(p);
    if (it == g_shaders.end() || it->second.logged) return false;
    it->second.logged = true;
    return true;
}

// ---------------------------------------------------------------- D3D11 state as text

std::string Where(void* a) {
    char buf[96];
    HMODULE mod = nullptr;
    wchar_t path[MAX_PATH];
    if (a && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                static_cast<LPCWSTR>(a), &mod) &&
        GetModuleFileNameW(mod, path, MAX_PATH)) {
        const wchar_t* name = wcsrchr(path, L'\\');
        snprintf(buf, sizeof(buf), "+0x%llX", (unsigned long long)((uintptr_t)a - (uintptr_t)mod));
        return Narrow(name ? name + 1 : path) + buf;
    }
    snprintf(buf, sizeof(buf), "%p", a);
    return buf;
}

struct Range {
    uintptr_t begin = 0, end = 0;
    std::string name;
};
Range ExeRange() {
    Range r;
    const auto* m = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
    if (!m) return r;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(m + dos->e_lfanew);
    r.begin = (uintptr_t)m;
    r.end = r.begin + nt->OptionalHeader.SizeOfImage;
    wchar_t path[MAX_PATH];
    if (GetModuleFileNameW(nullptr, path, MAX_PATH)) {
        const wchar_t* name = wcsrchr(path, L'\\');
        r.name = Narrow(name ? name + 1 : path);
    }
    return r;
}

// The game's frames of the call stack (return addresses in its exe), innermost first.
std::string ExeStack() {
    static const Range r = ExeRange();
    void* f[62];
    const USHORT n = RtlCaptureStackBackTrace(1, 61, f, nullptr);
    std::string s;
    int kept = 0;
    char buf[40];
    for (USHORT i = 0; i < n && kept < kStackFrames; ++i) {
        const uintptr_t a = (uintptr_t)f[i];
        if (a < r.begin || a >= r.end) continue;
        snprintf(buf, sizeof(buf), " +0x%llX", (unsigned long long)(a - r.begin));
        s += buf;
        ++kept;
    }
    return kept ? r.name + s : std::string("no frames in the game's exe");
}

std::string DebugName(ID3D11DeviceChild* o) {
    char name[128] = {};
    UINT size = sizeof(name) - 1;
    if (!o || FAILED(o->GetPrivateData(kDebugName, &size, name)) || !size) return {};
    name[std::min<UINT>(size, sizeof(name) - 1)] = 0;
    return std::string(" '") + name + "'";
}

std::string ResText(ID3D11Resource* r) {
    if (!r) return "-";
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dim);
    char b[200] = "?";
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ID3D11Texture2D* t = nullptr;
        if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
            D3D11_TEXTURE2D_DESC d{};
            t->GetDesc(&d);
            t->Release();
            snprintf(b, sizeof(b), "%ux%u f%d x%u arr %u mips %u", d.Width, d.Height, (int)d.Format,
                     d.SampleDesc.Count, d.ArraySize, d.MipLevels);
        }
    } else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE3D) {
        ID3D11Texture3D* t = nullptr;
        if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture3D), (void**)&t)) && t) {
            D3D11_TEXTURE3D_DESC d{};
            t->GetDesc(&d);
            t->Release();
            snprintf(b, sizeof(b), "3D %ux%ux%u f%d", d.Width, d.Height, d.Depth, (int)d.Format);
        }
    } else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE1D) {
        ID3D11Texture1D* t = nullptr;
        if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture1D), (void**)&t)) && t) {
            D3D11_TEXTURE1D_DESC d{};
            t->GetDesc(&d);
            t->Release();
            snprintf(b, sizeof(b), "1D %u f%d arr %u", d.Width, (int)d.Format, d.ArraySize);
        }
    } else if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ID3D11Buffer* t = nullptr;
        if (SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Buffer), (void**)&t)) && t) {
            D3D11_BUFFER_DESC d{};
            t->GetDesc(&d);
            t->Release();
            snprintf(b, sizeof(b), "buffer %u B stride %u", d.ByteWidth, d.StructureByteStride);
        }
    }
    return b + DebugName(r);
}

// A view: its resource and the view's format.
std::string ViewText(ID3D11View* v, int viewFormat) {
    if (!v) return "-";
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    std::string s = ResText(r);
    if (r) r->Release();
    char b[24];
    snprintf(b, sizeof(b), " view f%d", viewFormat);
    return s + b;
}
int RtvFormat(ID3D11RenderTargetView* v) {
    D3D11_RENDER_TARGET_VIEW_DESC d{};
    v->GetDesc(&d);
    return (int)d.Format;
}
int DsvFormat(ID3D11DepthStencilView* v) {
    D3D11_DEPTH_STENCIL_VIEW_DESC d{};
    v->GetDesc(&d);
    return (int)d.Format;
}
int SrvFormat(ID3D11ShaderResourceView* v) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    v->GetDesc(&d);
    return (int)d.Format;
}

std::string TargetsText(UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    std::string s;
    char b[16];
    for (UINT i = 0; i < n && rtv && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
        if (!rtv[i]) continue;
        snprintf(b, sizeof(b), "rt%u ", i);
        s += b + ViewText(rtv[i], RtvFormat(rtv[i])) + ", ";
    }
    s += "ds " + (dsv ? ViewText(dsv, DsvFormat(dsv)) : std::string("-"));
    return s;
}

// The name of a texture register in a shader's reflection.
std::string SlotName(const Shader* sh, UINT slot) {
    if (!sh) return {};
    for (const Binding& x : sh->binds) {
        const char r = RegLetter(x.type);
        if (r == 't' && slot >= x.point && slot < x.point + std::max(1u, x.count)) return x.name;
    }
    return {};
}

std::string SrvsText(ID3D11ShaderResourceView* const* srv, UINT n, const Shader* sh) {
    std::string s;
    char b[16];
    for (UINT i = 0; i < n; ++i) {
        if (!srv[i]) continue;
        const std::string name = SlotName(sh, i);
        snprintf(b, sizeof(b), "t%u", i);
        s += std::string(s.empty() ? "" : ", ") + b + (name.empty() ? "" : " " + name) + " " +
             ViewText(srv[i], SrvFormat(srv[i]));
    }
    return s.empty() ? "-" : s;
}

struct Call {
    const char* method = "?";
    unsigned count = 0, instances = 0;
    void* caller = nullptr;
};
Call g_call;  // render thread: the draw call being made

std::string DrawText(ID3D11DeviceContext* c, const Shader* vs, const Shader* ps) {
    char b[512];
    std::string s;
    snprintf(b, sizeof(b), "%s(%u", g_call.method, g_call.count);
    s += b;
    if (g_call.instances != 1) {
        snprintf(b, sizeof(b), " x %u instances", g_call.instances);
        s += b;
    }
    s += ") from " + Where(g_call.caller);
    s += "; VS " + std::string(vs ? vs->sum : "unknown") + ", PS " + std::string(ps ? ps->sum : "unknown");
    {
        ID3D11GeometryShader* gs = nullptr;
        ID3D11HullShader* hs = nullptr;
        ID3D11DomainShader* ds = nullptr;
        c->GSGetShader(&gs, nullptr, nullptr);
        c->HSGetShader(&hs, nullptr, nullptr);
        c->DSGetShader(&ds, nullptr, nullptr);
        if (gs || hs || ds) s += std::string(", ") + (gs ? "GS " : "") + (hs ? "HS " : "") + (ds ? "DS" : "");
        if (gs) gs->Release();
        if (hs) hs->Release();
        if (ds) ds->Release();
    }
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
    s += "; " + TargetsText(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
    for (ID3D11RenderTargetView* v : rtv)
        if (v) v->Release();
    if (dsv) dsv->Release();
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    D3D11_PRIMITIVE_TOPOLOGY topo = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    c->IAGetPrimitiveTopology(&topo);
    snprintf(b, sizeof(b), "; viewport %.0fx%.0f at (%.0f, %.0f) depth %.2f..%.2f; topology %d", vp.Width, vp.Height,
             vp.TopLeftX, vp.TopLeftY, vp.MinDepth, vp.MaxDepth, (int)topo);
    s += b;
    ID3D11DepthStencilState* dss = nullptr;
    UINT ref = 0;
    c->OMGetDepthStencilState(&dss, &ref);
    if (dss) {
        D3D11_DEPTH_STENCIL_DESC d{};
        dss->GetDesc(&d);
        dss->Release();
        snprintf(b, sizeof(b), "; depth %s write %d func %d, stencil %d (ref %u)", d.DepthEnable ? "on" : "off",
                 (int)d.DepthWriteMask, (int)d.DepthFunc, (int)d.StencilEnable, ref);
        s += b;
    } else {
        s += "; depth state default";
    }
    ID3D11BlendState* bs = nullptr;
    float factor[4] = {};
    UINT mask = 0;
    c->OMGetBlendState(&bs, factor, &mask);
    if (bs) {
        D3D11_BLEND_DESC d{};
        bs->GetDesc(&d);
        bs->Release();
        const D3D11_RENDER_TARGET_BLEND_DESC& r = d.RenderTarget[0];
        snprintf(b, sizeof(b), "; blend rt0 %s src %d dst %d op %d, alpha src %d dst %d op %d, write 0x%X",
                 r.BlendEnable ? "on" : "off", (int)r.SrcBlend, (int)r.DestBlend, (int)r.BlendOp,
                 (int)r.SrcBlendAlpha, (int)r.DestBlendAlpha, (int)r.BlendOpAlpha, (unsigned)r.RenderTargetWriteMask);
        s += b;
    } else {
        s += "; blend default";
    }
    ID3D11RasterizerState* rs = nullptr;
    c->RSGetState(&rs);
    if (rs) {
        D3D11_RASTERIZER_DESC d{};
        rs->GetDesc(&d);
        rs->Release();
        snprintf(b, sizeof(b), "; cull %d fill %d depth bias %d scissor %d", (int)d.CullMode, (int)d.FillMode,
                 d.DepthBias, (int)d.ScissorEnable);
        s += b;
    }
    ID3D11ShaderResourceView* srv[16] = {};
    c->VSGetShaderResources(0, 16, srv);
    s += "; VS textures " + SrvsText(srv, 16, vs);
    for (ID3D11ShaderResourceView*& v : srv)
        if (v) {
            v->Release();
            v = nullptr;
        }
    c->PSGetShaderResources(0, 16, srv);
    s += "; PS textures " + SrvsText(srv, 16, ps);
    for (ID3D11ShaderResourceView* v : srv)
        if (v) v->Release();
    if (s.size() > kMaxText) s.resize(kMaxText);
    return s;
}

// ---------------------------------------------------------------- passes and stats

struct PassState {
    bool on = false;
    int eye = 0;
    uint64_t frame = 0;
    bool live = false;
    unsigned w = 0, h = 0, fw = 0, fh = 0;
    unsigned binds = 0, draws = 0, lines = 0;
    bool decided = false, detail = false, bindLogged = false;
    std::string bind, stack;  // the latest bind of another size and the game's stack at it
};
PassState g_ps;

struct Stats {
    uint64_t passes[2] = {}, live[2] = {}, withDraws[2] = {}, draws[2] = {}, maxDraws[2] = {}, idle[2] = {},
             binds[2] = {}, toFocus[2] = {};
};
Stats g_st, g_tot;
double g_statsT0 = 0;
std::vector<std::string> g_kinds;  // kind index -> its text
std::map<std::string, size_t> g_kindIdx;
std::vector<uint64_t> g_kindN, g_kindTot;
uint64_t g_kindOver = 0;
std::vector<std::string> g_idleSeen;
int g_detailPasses[2] = {};
double g_lastDetail[2] = {-1e18, -1e18};
uint64_t g_pausedDraws = 0, g_detailLines = 0;
std::string g_lastDraw, g_lastPs;
std::mutex g_m;

int E() { return g_ps.eye == 1 ? 1 : 0; }

void ClosePassLocked() {
    if (!g_ps.on) return;
    const int e = E();
    if (g_ps.draws) {
        for (Stats* s : {&g_st, &g_tot}) {
            ++s->withDraws[e];
            s->draws[e] += g_ps.draws;
            s->maxDraws[e] = std::max<uint64_t>(s->maxDraws[e], g_ps.draws);
        }
        if (g_ps.detail)
            Log("Forward diag: end of the detailed pass of outer %s (frame %llu): %u draws while paused, %u binds of "
                "other sizes",
                kEye[e], (unsigned long long)g_ps.frame, g_ps.draws, g_ps.binds);
    } else if (g_ps.binds) {
        for (Stats* s : {&g_st, &g_tot}) ++s->idle[e];
        if (g_idleSeen.size() < kMaxIdle &&
            std::find(g_idleSeen.begin(), g_idleSeen.end(), g_ps.bind) == g_idleSeen.end()) {
            g_idleSeen.push_back(g_ps.bind);
            Log("Forward diag: outer %s's forward pass (frame %llu) binds a target of another size and draws nothing "
                "there (MVR goes on): %s; the game's stack at the bind: %s",
                kEye[e], (unsigned long long)g_ps.frame, g_ps.bind.c_str(), g_ps.stack.c_str());
        }
    }
    g_ps = PassState{};
}

void NewKindLocked(size_t k, const std::string& text, int e) {
    Log("Forward diag: kind %zu of draws while paused (first in outer %s's pass, frame %llu): %s", k + 1, kEye[e],
        (unsigned long long)g_ps.frame, text.c_str());
}

void LogShader(void* p, const Shader& sh) {
    if (!FirstLog(p)) return;
    Log("Forward diag: %s %s (%u bytes): %s", sh.stage == 'V' ? "VS" : "PS", sh.sum, sh.bytes, sh.text.c_str());
}

std::string TopKinds(const std::vector<uint64_t>& n) {
    std::vector<std::pair<uint64_t, size_t>> v;
    for (size_t i = 0; i < n.size(); ++i)
        if (n[i]) v.push_back({n[i], i});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string s;
    char b[48];
    for (size_t i = 0; i < v.size() && i < 12; ++i) {
        snprintf(b, sizeof(b), "%s#%zu %llu", i ? ", " : "", v[i].second + 1, (unsigned long long)v[i].first);
        s += b;
    }
    return s.empty() ? "-" : s;
}

}  // namespace

void OnShaderMade(void* shader, char stage, const void* code, size_t len) {
    if (!shader || !code || len < 32) return;
    Protect("shader", [&] {
        Shader sh;
        sh.stage = stage;
        sh.bytes = (uint32_t)len;
        SumText(code, len, sh.sum);
        Describe(code, len, sh);
        std::unique_lock lock(g_sm);
        if (g_shaders.size() >= kMaxShaders && !g_shaders.count(shader)) {
            g_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_shaders[shader] = std::move(sh);
        g_described.fetch_add(1, std::memory_order_relaxed);
    });
}

void NoteDrawCall(const char* method, unsigned count, unsigned instances, void* caller) {
    g_call = {method, count, instances, caller};
}

void OnPassStart(int eye, uint64_t frame, bool live, unsigned w, unsigned h, unsigned fw, unsigned fh) {
    Protect("pass start", [&] {
        std::lock_guard lock(g_m);
        static bool announced = false;
        if (!announced) {
            announced = true;
            Log("Forward diag: on; draws in outer's forward pass while no target at outer's size is bound are described "
                "here (shaders described so far %llu)",
                (unsigned long long)g_described.load());
        }
        ClosePassLocked();
        g_ps.on = true;
        g_ps.eye = eye;
        g_ps.frame = frame;
        g_ps.live = live;
        g_ps.w = w;
        g_ps.h = h;
        g_ps.fw = fw;
        g_ps.fh = fh;
        const int e = E();
        for (Stats* s : {&g_st, &g_tot}) {
            ++s->passes[e];
            s->live[e] += live;
        }
    });
}

void OnPauseBind(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    (void)c;
    Protect("pause bind", [&] {
        std::string bind = TargetsText(n, rtv, dsv);
        std::string stack = ExeStack();
        std::lock_guard lock(g_m);
        if (!g_ps.on) g_ps.on = true;  // a pass that began before these diagnostics knew it
        ++g_ps.binds;
        for (Stats* s : {&g_st, &g_tot}) ++s->binds[E()];
        g_ps.bind = std::move(bind);
        g_ps.stack = std::move(stack);
        g_ps.bindLogged = false;
        if (g_ps.detail && g_ps.lines < kDetailDraws) {
            Log("Forward diag:   bind while paused: %s; the game's stack at the bind: %s", g_ps.bind.c_str(),
                g_ps.stack.c_str());
            g_ps.bindLogged = true;
        }
    });
}

void OnPausedDraw(ID3D11DeviceContext* c) {
    Protect("paused draw", [&] {
        ID3D11VertexShader* vsp = nullptr;
        ID3D11PixelShader* psp = nullptr;
        c->VSGetShader(&vsp, nullptr, nullptr);
        c->PSGetShader(&psp, nullptr, nullptr);
        // The pointers stay keys only (the game keeps the shaders bound through the draw).
        if (vsp) vsp->Release();
        if (psp) psp->Release();
        Shader vs, ps;
        const bool haveVs = FindShader(vsp, vs), havePs = FindShader(psp, ps);
        ID3D11RenderTargetView* rt0 = nullptr;
        ID3D11DepthStencilView* ds = nullptr;
        c->OMGetRenderTargets(1, &rt0, &ds);
        ID3D11Resource* rt0r = nullptr;
        ID3D11Resource* dsr = nullptr;
        if (rt0) rt0->GetResource(&rt0r);
        if (ds) ds->GetResource(&dsr);
        char key[96];
        snprintf(key, sizeof(key), "%p %p %p %p", (void*)vsp, (void*)psp, (void*)rt0r, (void*)dsr);
        if (rt0r) rt0r->Release();
        if (dsr) dsr->Release();
        if (rt0) rt0->Release();
        if (ds) ds->Release();

        std::lock_guard lock(g_m);
        if (!g_ps.on) g_ps.on = true;
        const int e = E();
        ++g_ps.draws;
        ++g_pausedDraws;
        if (!g_ps.decided) {
            g_ps.decided = true;
            const double now = NowMs();
            g_ps.detail = g_detailPasses[e] < kDetailPasses || now - g_lastDetail[e] >= kDetailEveryMs;
            if (g_ps.detail) {
                ++g_detailPasses[e];
                g_lastDetail[e] = now;
                Log("Forward diag: outer %s's %s forward pass (frame %llu; outer %ux%u, focus %ux%u) draws while no "
                    "target at outer's size is bound, so MVR cannot draw it for the focus view and the focus group "
                    "draws its own forward pass. Detailed pass %d of this eye; every draw follows.",
                    kEye[e], g_ps.live ? "live" : "test", (unsigned long long)g_ps.frame, g_ps.w, g_ps.h, g_ps.fw,
                    g_ps.fh, g_detailPasses[e]);
            }
        }
        if (g_ps.detail && !g_ps.bindLogged) {
            Log("Forward diag:   bind: %s; the game's stack at the bind: %s", g_ps.bind.c_str(), g_ps.stack.c_str());
            g_ps.bindLogged = true;
        }
        size_t k = SIZE_MAX;
        const auto it = g_kindIdx.find(key);
        bool newKind = false;
        if (it != g_kindIdx.end()) {
            k = it->second;
        } else if (g_kinds.size() < kMaxKinds) {
            k = g_kinds.size();
            g_kindIdx[key] = k;
            g_kinds.push_back(key);
            g_kindN.push_back(0);
            g_kindTot.push_back(0);
            newKind = true;
        } else {
            ++g_kindOver;
        }
        if (k != SIZE_MAX) {
            ++g_kindN[k];
            ++g_kindTot[k];
        }
        const bool line = g_ps.detail && g_ps.lines < kDetailDraws;
        if (!newKind && !line) return;
        const std::string text = DrawText(c, haveVs ? &vs : nullptr, havePs ? &ps : nullptr);
        g_lastDraw = text;
        g_lastPs = havePs ? ps.text : std::string();
        if (newKind) {
            NewKindLocked(k, text, e);
            if (haveVs) LogShader(vsp, vs);
            if (havePs) LogShader(psp, ps);
        }
        if (line) {
            ++g_ps.lines;
            ++g_detailLines;
            Log("Forward diag:   draw %u (kind %zu): %s", g_ps.draws, k == SIZE_MAX ? 0 : k + 1, text.c_str());
            if (g_ps.lines == kDetailDraws) Log("Forward diag:   further draws of this pass only counted");
        }
    });
}

void OnPassEnd(const char* why) {
    Protect("pass end", [&] {
        std::lock_guard lock(g_m);
        if (!g_ps.on) return;
        const int e = E();
        if (why) {
            for (Stats* s : {&g_st, &g_tot}) ++s->toFocus[e];
            if (g_ps.detail)
                Log("Forward diag: outer %s's pass (frame %llu) went to the focus group: %s", kEye[e],
                    (unsigned long long)g_ps.frame, why);
        }
        ClosePassLocked();
    });
}

void LogStats(bool totals) {
    Protect("stats", [&] {
        std::lock_guard lock(g_m);
        const Stats& s = totals ? g_tot : g_st;
        const double now = NowMs();
        if (s.passes[0] + s.passes[1] + s.binds[0] + s.binds[1] == 0) {
            if (!totals) g_statsT0 = now;
            return;
        }
        Log("Forward diag%s (%.0f s): outer forward passes L %llu / R %llu (live L %llu / R %llu); with draws while "
            "paused L %llu / R %llu (draws L %llu / R %llu, at most %llu / %llu in a pass), to the focus group L %llu / "
            "R %llu; binds of other sizes L %llu / R %llu, pauses without draws L %llu / R %llu; draw kinds %zu%s (by "
            "draws: %s); shaders described %llu (not kept %llu)",
            totals ? " (session)" : "", totals ? now / 1000.0 : (now - g_statsT0) / 1000.0,
            (unsigned long long)s.passes[0], (unsigned long long)s.passes[1], (unsigned long long)s.live[0],
            (unsigned long long)s.live[1], (unsigned long long)s.withDraws[0], (unsigned long long)s.withDraws[1],
            (unsigned long long)s.draws[0], (unsigned long long)s.draws[1], (unsigned long long)s.maxDraws[0],
            (unsigned long long)s.maxDraws[1], (unsigned long long)s.toFocus[0], (unsigned long long)s.toFocus[1],
            (unsigned long long)s.binds[0], (unsigned long long)s.binds[1], (unsigned long long)s.idle[0],
            (unsigned long long)s.idle[1], g_kinds.size(), g_kindOver ? " (more not kept)" : "",
            TopKinds(totals ? g_kindTot : g_kindN).c_str(), (unsigned long long)g_described.load(),
            (unsigned long long)g_skipped.load());
        if (!totals) {
            g_st = Stats{};
            std::fill(g_kindN.begin(), g_kindN.end(), 0);
            g_statsT0 = now;
        }
    });
}

TestState TestGet() {
    std::lock_guard lock(g_m);
    TestState t;
    t.shaders = g_described.load();
    t.pausedDraws = g_pausedDraws;
    t.detailLines = g_detailLines;
    t.kinds = g_kinds.size();
    t.lastDraw = g_lastDraw;
    t.lastPs = g_lastPs;
    return t;
}

}  // namespace qvfr::fwddiag

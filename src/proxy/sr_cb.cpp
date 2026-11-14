#include "sr_cb.h"

#include <d3d11_1.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "clones.h"
#include "common.h"
#include "gpu.h"
#include "hooks.h"
#include "sr_fix_cs.h"  // kSrFixCS (fxc from shaders/sr.hlsl)
#include "sr_probe.h"

namespace qvfr::srcb {

namespace {

constexpr UINT kFirstSlot = 10, kSlotCount = 4;  // the engine's constant buffer banks with the view's sizes
constexpr UINT kMaxWindow = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16;
constexpr int kMaxPairs = 48;  // FixPairs in sr.hlsl
constexpr int kSlotDrawIndexedInstanced = 20, kSlotDrawInstanced = 21;  // ID3D11DeviceContext vtable (d3d11.h order)

struct FixPairs {
    uint32_t count = 0, unused = 0, pad[2] = {};
    float pairs[kMaxPairs][4] = {};
};

struct Fixed {
    ID3D11Buffer* work = nullptr;  // raw UAV buffer, the window fixed in place
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11Buffer* cb = nullptr;  // the fixed window as a constant buffer, bound in the game's place
    UINT bytes = 0;
};

bool g_installed = false;
bool g_fault = false;
std::atomic<bool> g_on{false};  // setting and hotkey
bool g_keyDown = false;
bool g_hotkeyOff = false;
ID3D11DeviceContext* g_imm = nullptr;
ID3D11DeviceContext1* g_ctx1 = nullptr;
ID3D11ComputeShader* g_cs = nullptr;
ID3D11Buffer* g_pairsCb = nullptr;
int g_group = -1;  // the SR group the cache is for (clones' group number)
uint64_t g_frameSeen = 0;
std::map<std::tuple<ID3D11Buffer*, UINT, UINT>, Fixed> g_cache;  // the game's buffer, window start and size
std::unordered_set<ID3D11Buffer*> g_ours;                         // our fixed constant buffers (bound by us)
std::multimap<UINT, Fixed> g_pool;                                // free fixed windows by size
uint64_t g_windows = 0, g_draws = 0, g_failed = 0;                // this stats window
uint64_t g_windowsTot = 0, g_drawsTot = 0, g_failedTot = 0;
uint64_t g_testWindows = 0;

using DiiFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DiFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
DiiFn o_dii = nullptr;
DiFn o_di = nullptr;

template <class F>
void Guard(const char* where, F&& f) {
    if (g_fault) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_fault = true;
        g_on = false;
        Log("ERROR: exception 0x%08lX in the SR constants fix (%s); off for this session", code, where);
    }
}

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

void Free(Fixed& f) {
    Rel(f.uav);
    Rel(f.work);
    Rel(f.cb);
}

// The windows of the last group go back to the pool (the GPU runs our later copies into them after its draws).
void DropCache() {
    for (auto& [k, f] : g_cache) g_pool.emplace(f.bytes, f);
    g_cache.clear();
}

bool EnsureShader(ID3D11Device* dev) {
    if (g_cs && g_pairsCb) return true;
    HRESULT hr = dev->CreateComputeShader(kSrFixCS, sizeof(kSrFixCS), nullptr, &g_cs);
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = sizeof(FixPairs);
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (SUCCEEDED(hr)) hr = dev->CreateBuffer(&d, nullptr, &g_pairsCb);
    if (FAILED(hr)) {
        Log("SR constants fix: shader or its constants not made (0x%08lX); off", (unsigned long)hr);
        g_fault = true;
        g_on = false;
        return false;
    }
    return true;
}

// The render-size pairs of this group and their view-size pairs.
void BuildPairs(ID3D11DeviceContext* c, const clones::SrGroup& g) {
    FixPairs fp;
    auto add = [&](float fx, float fy, float tx, float ty) {
        if ((fx == tx && fy == ty) || (int)fp.count >= kMaxPairs) return;
        for (uint32_t i = 0; i < fp.count; ++i)
            if (fp.pairs[i][0] == fx && fp.pairs[i][1] == fy) return;
        fp.pairs[fp.count][0] = fx;
        fp.pairs[fp.count][1] = fy;
        fp.pairs[fp.count][2] = tx;
        fp.pairs[fp.count][3] = ty;
        ++fp.count;
    };
    for (uint32_t d : {1u, 2u, 4u, 8u})
        for (int up = 0; up < 2; ++up) {
            if (d == 1 && up) continue;
            auto fr = [&](uint32_t s) { return std::max(1u, up ? (s + d - 1) / d : s / d); };
            const float rw = (float)fr(g.rw), rh = (float)fr(g.rh), ow = (float)fr(g.ow), oh = (float)fr(g.oh);
            add(rw, rh, ow, oh);
            for (float k : {1.0f, 0.5f, 2.0f}) add(k / rw, k / rh, k / ow, k / oh);
        }
    c->UpdateSubresource(g_pairsCb, 0, nullptr, &fp, 0, 0);
}

bool MakeFixed(ID3D11Device* dev, UINT bytes, Fixed& f) {
    if (auto it = g_pool.find(bytes); it != g_pool.end()) {
        f = it->second;
        g_pool.erase(it);
        return true;
    }
    f = {};
    f.bytes = bytes;
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = bytes;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    HRESULT hr = dev->CreateBuffer(&d, nullptr, &f.work);
    if (SUCCEEDED(hr)) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_TYPELESS;
        u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = bytes / 4;
        u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        hr = dev->CreateUnorderedAccessView(f.work, &u, &f.uav);
    }
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC cd{};
        cd.ByteWidth = bytes;
        cd.Usage = D3D11_USAGE_DEFAULT;
        cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        hr = dev->CreateBuffer(&cd, nullptr, &f.cb);
    }
    if (FAILED(hr)) {
        Free(f);
        return false;
    }
    g_ours.insert(f.cb);
    return true;
}

// The window [off, off + bytes) of `src` copied and fixed into f (our own compute state, the game's put back).
void RunFix(ID3D11DeviceContext* c, ID3D11Buffer* src, UINT off, Fixed& f) {
    clones::RawScope raw;  // our calls pass the clones' hooks as they are
    const D3D11_BOX box{off, 0, 0, off + f.bytes, 1, 1};
    c->CopySubresourceRegion(f.work, 0, 0, 0, 0, src, 0, &box);
    ID3D11ComputeShader* cs = nullptr;
    ID3D11ClassInstance* inst[256] = {};
    UINT ni = 256;
    c->CSGetShader(&cs, inst, &ni);
    ID3D11Buffer* cb0 = nullptr;
    c->CSGetConstantBuffers(0, 1, &cb0);
    ID3D11UnorderedAccessView* uav0 = nullptr;
    c->CSGetUnorderedAccessViews(0, 1, &uav0);
    c->CSSetShader(g_cs, nullptr, 0);
    c->CSSetConstantBuffers(0, 1, &g_pairsCb);
    c->CSSetUnorderedAccessViews(0, 1, &f.uav, nullptr);
    c->Dispatch((f.bytes / 8 + 63) / 64, 1, 1);
    c->CSSetUnorderedAccessViews(0, 1, &uav0, nullptr);
    c->CSSetConstantBuffers(0, 1, &cb0);
    c->CSSetShader(cs, inst, ni);
    Rel(uav0);
    Rel(cb0);
    Rel(cs);
    for (UINT i = 0; i < ni; ++i) Rel(inst[i]);
    c->CopyResource(f.cb, f.work);
}

void Fix(ID3D11DeviceContext* c) {
    clones::SrGroup g;
    if (!clones::SrGroupNow(g) || !g.post || !g.rw || !g.ow) return;
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return;
    if (!EnsureShader(dev)) {
        dev->Release();
        return;
    }
    if (g.group != g_group) {
        DropCache();
        g_group = g.group;
        BuildPairs(c, g);
    }
    bool any = false;
    for (int stage = 0; stage < 2; ++stage) {
        ID3D11Buffer* cb[kSlotCount] = {};
        UINT first[kSlotCount] = {}, num[kSlotCount] = {};
        if (stage == 0)
            g_ctx1->VSGetConstantBuffers1(kFirstSlot, kSlotCount, cb, first, num);
        else
            g_ctx1->PSGetConstantBuffers1(kFirstSlot, kSlotCount, cb, first, num);
        for (UINT s = 0; s < kSlotCount; ++s) {
            ID3D11Buffer* b = cb[s];
            if (!b) continue;
            if (!g_ours.count(b)) {
                D3D11_BUFFER_DESC d{};
                b->GetDesc(&d);
                const UINT off = num[s] ? first[s] * 16 : 0;
                const UINT want = num[s] ? num[s] * 16 : d.ByteWidth;
                const UINT len = off < d.ByteWidth ? std::min({want, d.ByteWidth - off, kMaxWindow}) & ~15u : 0;
                if (len) {
                    const auto key = std::make_tuple(b, off, len);
                    auto it = g_cache.find(key);
                    if (it == g_cache.end()) {
                        Fixed f;
                        if (MakeFixed(dev, len, f)) {
                            RunFix(c, b, off, f);
                            it = g_cache.emplace(key, f).first;
                            ++g_windows;
                            ++g_testWindows;
                        } else {
                            ++g_failed;
                        }
                    }
                    if (it != g_cache.end()) {
                        const UINT zero = 0, n = len / 16;
                        if (stage == 0)
                            g_ctx1->VSSetConstantBuffers1(kFirstSlot + s, 1, &it->second.cb, &zero, &n);
                        else
                            g_ctx1->PSSetConstantBuffers1(kFirstSlot + s, 1, &it->second.cb, &zero, &n);
                        any = true;
                    }
                }
            }
            b->Release();
        }
    }
    if (any) ++g_draws;
    dev->Release();
}

void STDMETHODCALLTYPE D_DrawIndexedInstanced(ID3D11DeviceContext* c, UINT n, UINT inst, UINT start, INT base,
                                              UINT startInst) {
    OnDraw(c);
    if (srprobe::Armed()) srprobe::OnDraw(c, _ReturnAddress(), "DrawIndexedInstanced");
    o_dii(c, n, inst, start, base, startInst);
}

void STDMETHODCALLTYPE D_DrawInstanced(ID3D11DeviceContext* c, UINT n, UINT inst, UINT start, UINT startInst) {
    OnDraw(c);
    if (srprobe::Armed()) srprobe::OnDraw(c, _ReturnAddress(), "DrawInstanced");
    o_di(c, n, inst, start, startInst);
}

}  // namespace

void Install(ID3D11DeviceContext* imm, bool hookDraws) {
    if (g_installed || !imm) return;
    g_installed = true;
    Guard("Install", [&] {
        g_imm = imm;
        imm->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
        if (!g_ctx1) {
            Log("SR constants fix: no ID3D11DeviceContext1; off (the passes after TAA keep the render size)");
            return;
        }
        void** vt = *(void***)imm;
        const bool a = !hookDraws || hooks::CreateAt(vt[kSlotDrawIndexedInstanced], (void*)D_DrawIndexedInstanced,
                                                     (void**)&o_dii, "DrawIndexedInstanced");
        const bool b =
            !hookDraws || hooks::CreateAt(vt[kSlotDrawInstanced], (void*)D_DrawInstanced, (void**)&o_di, "DrawInstanced");
        g_on = Cfg().srCbFix;
        Log("SR constants fix: %s; after the TAA resolve the windows of VS and PS constant buffer slots %u..%u get the view "
            "size in place of the render size; instanced draw hooks %s; hotkey 0x%X+0x%X",
            g_on.load() ? "on" : "off ([debug] sr_cb_fix=0)", kFirstSlot, kFirstSlot + kSlotCount - 1,
            !hookDraws ? "from mvr_pass" : a && b ? "installed" : "INCOMPLETE", Cfg().srProbeKeyMod, Cfg().srCbFixKey);
    });
}

void OnDraw(ID3D11DeviceContext* c) {
    if (!g_on.load(std::memory_order_relaxed) || !clones::SrPostNow() || c != g_imm || !g_ctx1) return;
    Guard("draw", [&] { Fix(c); });
}

void OnFrameStart() {
    if (!g_installed || g_fault) return;
    Guard("frame", [&] {
        const Config& c = Cfg();
        const bool down = c.srCbFixKey && (GetAsyncKeyState(c.srCbFixKey) & 0x8000) &&
                          (!c.srProbeKeyMod || (GetAsyncKeyState(c.srProbeKeyMod) & 0x8000));
        if (down && !g_keyDown) {
            g_hotkeyOff = !g_hotkeyOff;
            Log("SR constants fix: hotkey: %s", g_hotkeyOff ? "OFF (the passes after TAA keep the render size)" : "on");
        }
        g_keyDown = down;
        const bool on = c.srCbFix && !g_hotkeyOff && g_ctx1;
        if (on != g_on.load()) g_on = on;
        DropCache();  // a new frame: the game's windows hold new data
        g_group = -1;
        ++g_frameSeen;
    });
}

void LogStats(bool totals) {
    if (!g_installed) return;
    g_windowsTot += g_windows;
    g_drawsTot += g_draws;
    g_failedTot += g_failed;
    Log("SR constants fix%s: %s; windows fixed %llu, draws with a fixed window %llu, windows not fixed (no buffer) %llu",
        totals ? " (session total)" : "", g_on.load() ? "on" : "off",
        (unsigned long long)(totals ? g_windowsTot : g_windows), (unsigned long long)(totals ? g_drawsTot : g_draws),
        (unsigned long long)(totals ? g_failedTot : g_failed));
    g_windows = g_draws = g_failed = 0;
}

void Shutdown() {
    g_on = false;
    Guard("Shutdown", [&] {
        DropCache();
        for (auto& [sz, f] : g_pool) Free(f);
        g_pool.clear();
        g_ours.clear();
        Rel(g_cs);
        Rel(g_pairsCb);
        g_group = -1;
    });
}

void TestInit(ID3D11DeviceContext* c) {
    g_installed = true;
    g_imm = c;
    if (!g_ctx1) c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
    g_on = g_ctx1 != nullptr;
    g_group = -1;
    g_testWindows = 0;
}

uint64_t TestWindowsFixed() { return g_testWindows; }

}  // namespace qvfr::srcb

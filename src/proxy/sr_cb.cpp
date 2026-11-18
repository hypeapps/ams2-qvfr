#include "sr_cb.h"

#include <d3d11_1.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "clones.h"
#include "common.h"
#include "hooks.h"
#include "sr_fix_cs.h"  // kSrFixCS (fxc from shaders/sr.hlsl)
#include "sr_probe.h"

namespace qvfr::srcb {

namespace {

constexpr UINT kFirstSlot = 10, kSlotCount = 4;  // the engine's constant buffer banks with the view's sizes
constexpr UINT kMaxWindow = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16;
constexpr int kMaxPairs = 48;  // FixPairs in sr.hlsl
constexpr int kSlotMap = 14, kSlotDrawIndexedInstanced = 20, kSlotDrawInstanced = 21;  // ID3D11DeviceContext vtable
// The engine writes its frame's constants once into a 64 MB dynamic buffer and the passes after TAA bind windows of
// 512 to 4096 bytes lying close together (capture ams2_msaa_high2: 19 windows of an eye in 4 such blocks). One block is
// fixed once and serves every window inside it. D3D11.1 windows start and end on this grid.
constexpr UINT kBlockBytes = 32 * 1024, kBlockBack = 4 * 1024, kWindowGrid = 256;

struct FixPairs {
    uint32_t count = 0, unused = 0, pad[2] = {};
    float pairs[kMaxPairs][4] = {};
};

struct Fixed {
    ID3D11Buffer* work = nullptr;  // raw UAV buffer, the block fixed in place
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11Buffer* cb = nullptr;  // the fixed block as a constant buffer, bound in the game's place
    UINT bytes = 0;
};

using Sizes = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;  // render width, height, view width, height
using BlockKey = std::tuple<Sizes, ID3D11Buffer*, UINT>;           // the sizes, the game's buffer, the block's start

bool g_installed = false;
bool g_fault = false;
std::atomic<bool> g_on{false};  // setting and hotkey
bool g_keyDown = false;
bool g_hotkeyOff = false;
ID3D11DeviceContext* g_imm = nullptr;
ID3D11DeviceContext1* g_ctx1 = nullptr;
ID3D11ComputeShader* g_cs = nullptr;
ID3D11Buffer* g_pairsCb = nullptr;
bool g_shareBlocks = false;  // the Map hook is in, so a block can never outlive the data it was copied from
bool g_pairsSet = false;
Sizes g_pairsFor{};  // the sizes in g_pairsCb
std::map<BlockKey, Fixed> g_cache;         // this frame's fixed blocks
std::vector<ID3D11Buffer*> g_sources;      // the dynamic buffers with shared blocks in g_cache (see OnBufferWrite)
std::unordered_set<ID3D11Buffer*> g_ours;  // our fixed constant buffers (bound by us)
std::multimap<UINT, Fixed> g_pool;         // free fixed blocks by size
uint64_t g_blocks = 0, g_windows = 0, g_draws = 0, g_failed = 0, g_dropped = 0;  // this stats window
uint64_t g_blocksTot = 0, g_windowsTot = 0, g_drawsTot = 0, g_failedTot = 0, g_droppedTot = 0;
uint64_t g_testBlocks = 0;

using DiiFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DiFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using MapFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                          D3D11_MAPPED_SUBRESOURCE*);
DiiFn o_dii = nullptr;
DiFn o_di = nullptr;
MapFn o_map = nullptr;

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

// The blocks go back to the pool (the GPU runs our later copies into them after its draws).
void DropCache() {
    for (auto& [k, f] : g_cache) g_pool.emplace(f.bytes, f);
    g_cache.clear();
    g_sources.clear();
}

// The block of these sizes and buffer that holds [off, off + len) at a grid distance from its start, or null.
Fixed* FindBlock(const Sizes& sizes, ID3D11Buffer* b, UINT off, UINT len, UINT& start) {
    auto it = g_cache.upper_bound(BlockKey{sizes, b, off});
    for (int i = 0; i < 3 && it != g_cache.begin(); ++i) {
        --it;
        const auto& [ks, kb, ko] = it->first;
        if (ks != sizes || kb != b) return nullptr;
        if (off + len <= ko + it->second.bytes && (off - ko) % kWindowGrid == 0) {
            start = ko;
            return &it->second;
        }
    }
    return nullptr;
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

// The block [off, off + f.bytes) of `src` copied and fixed into f (our own compute state, the game's put back).
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
    // Blocks stay for the frame: the views with the same sizes (both eyes) share them.
    const Sizes sizes{g.rw, g.rh, g.ow, g.oh};
    if (!g_pairsSet || sizes != g_pairsFor) {
        BuildPairs(c, g);
        g_pairsFor = sizes;
        g_pairsSet = true;
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
                    UINT start = off;
                    Fixed* f = FindBlock(sizes, b, off, len, start);
                    if (!f) {
                        // A window of the engine's dynamic buffer takes the block around it; any other only itself.
                        const bool share = g_shareBlocks && d.Usage == D3D11_USAGE_DYNAMIC && num[s] &&
                                           off % kWindowGrid == 0 && len % kWindowGrid == 0;
                        UINT bytes = len;
                        start = off;
                        if (share) {
                            bytes = std::max(kBlockBytes, len);
                            start = off - std::min({kBlockBack, bytes - len, off});
                            bytes = std::min(bytes, d.ByteWidth - start) & ~15u;
                        }
                        Fixed made;
                        if (MakeFixed(dev, bytes, made)) {
                            RunFix(c, b, start, made);
                            f = &g_cache.emplace(BlockKey{sizes, b, start}, made).first->second;
                            if (share && std::find(g_sources.begin(), g_sources.end(), b) == g_sources.end())
                                g_sources.push_back(b);
                            ++g_blocks;
                            ++g_testBlocks;
                        } else {
                            ++g_failed;
                        }
                    }
                    if (f) {
                        const UINT at = (off - start) / 16, n = len / 16;
                        if (stage == 0)
                            g_ctx1->VSSetConstantBuffers1(kFirstSlot + s, 1, &f->cb, &at, &n);
                        else
                            g_ctx1->PSSetConstantBuffers1(kFirstSlot + s, 1, &f->cb, &at, &n);
                        ++g_windows;
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

HRESULT STDMETHODCALLTYPE D_Map(ID3D11DeviceContext* c, ID3D11Resource* r, UINT sub, D3D11_MAP type, UINT flags,
                                D3D11_MAPPED_SUBRESOURCE* out) {
    OnBufferWrite(c, r);
    return o_map(c, r, sub, type, flags, out);
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
        g_shareBlocks = hooks::CreateAt(vt[kSlotMap], (void*)D_Map, (void**)&o_map, "Map");
        g_on = Cfg().srCbFix;
        Log("SR constants fix: %s; after the TAA resolve the windows of VS and PS constant buffer slots %u..%u get the view "
            "size in place of the render size; %s; instanced draw hooks %s; hotkey 0x%X+0x%X",
            g_on.load() ? "on" : "off ([debug] sr_cb_fix=0)", kFirstSlot, kFirstSlot + kSlotCount - 1,
            g_shareBlocks ? "fixed in shared blocks" : "no Map hook, every window fixed on its own",
            !hookDraws ? "from mvr_pass" : a && b ? "installed" : "INCOMPLETE", Cfg().srProbeKeyMod, Cfg().srCbFixKey);
    });
}

void OnDraw(ID3D11DeviceContext* c) {
    if (!g_on.load(std::memory_order_relaxed) || !clones::SrPostNow() || c != g_imm || !g_ctx1) return;
    Guard("draw", [&] { Fix(c); });
}

void OnBufferWrite(ID3D11DeviceContext* c, ID3D11Resource* r) {
    if (c != g_imm || g_sources.empty()) return;
    Guard("map", [&] {
        auto src = std::find_if(g_sources.begin(), g_sources.end(),
                                [&](ID3D11Buffer* b) { return static_cast<ID3D11Resource*>(b) == r; });
        if (src == g_sources.end()) return;
        ID3D11Buffer* b = *src;
        g_sources.erase(src);
        for (auto it = g_cache.begin(); it != g_cache.end();) {
            if (std::get<1>(it->first) == b) {
                g_pool.emplace(it->second.bytes, it->second);
                it = g_cache.erase(it);
                ++g_dropped;
            } else {
                ++it;
            }
        }
    });
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
        DropCache();  // a new frame: the game's buffers hold new data
    });
}

void LogStats(bool totals) {
    if (!g_installed) return;
    g_blocksTot += g_blocks;
    g_windowsTot += g_windows;
    g_drawsTot += g_draws;
    g_failedTot += g_failed;
    g_droppedTot += g_dropped;
    Log("SR constants fix%s: %s; blocks fixed %llu, windows bound from them %llu, draws with a fixed window %llu, blocks "
        "dropped by a Map %llu, windows not fixed (no buffer) %llu",
        totals ? " (session total)" : "", g_on.load() ? "on" : "off",
        (unsigned long long)(totals ? g_blocksTot : g_blocks), (unsigned long long)(totals ? g_windowsTot : g_windows),
        (unsigned long long)(totals ? g_drawsTot : g_draws), (unsigned long long)(totals ? g_droppedTot : g_dropped),
        (unsigned long long)(totals ? g_failedTot : g_failed));
    g_blocks = g_windows = g_draws = g_failed = g_dropped = 0;
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
        g_pairsSet = false;
    });
}

void TestInit(ID3D11DeviceContext* c) {
    g_installed = true;
    g_imm = c;
    if (!g_ctx1) c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1);
    g_on = g_ctx1 != nullptr;
    g_shareBlocks = true;
    g_pairsSet = false;
    DropCache();
    g_testBlocks = 0;
}

uint64_t TestBlocksFixed() { return g_testBlocks; }

}  // namespace qvfr::srcb

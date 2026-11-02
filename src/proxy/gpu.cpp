#include "gpu.h"

#include <atomic>
#include <mutex>

#include "common.h"

namespace qvfr::gpu {

namespace {

// ---------------------------------------------------------------- device

std::mutex g_m;
ID3D11Device* g_dev = nullptr;                       // referenced
std::atomic<ID3D11DeviceContext*> g_ctx{nullptr};   // referenced
std::atomic<uint32_t> g_gen{0};
bool g_faultLogged = false;

template <class F>
bool Seh(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
        return true;
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_faultLogged) {
            g_faultLogged = true;
            Log("ERROR: exception 0x%08lX in the GPU module (%s); the device stays as it was", code, where);
        }
        return false;
    }
}

// ---------------------------------------------------------------- timestamps

// Marks: frame begin, frame end, then begin and end of every phase.
constexpr int kFrameBegin = 0, kFrameEnd = 1;
constexpr int kMarks = 2 + 2 * kPhases;
constexpr int kSlots = 6;  // frames in flight; at 72 Hz the results are ready after 2 to 3 frames
constexpr int kReady = 32;

struct Slot {
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[kMarks] = {};
    uint32_t issued = 0;  // bit per mark set in this frame
    uint64_t frame = 0;
    enum State { Free, Open, Pending } state = Free;
};

Slot g_slot[kSlots];
int g_cur = -1;
DWORD g_thread = 0;
bool g_createFailed = false;
bool g_firstLogged = false;
uint64_t g_dropped = 0, g_disjoint = 0;
Result g_ready[kReady];
int g_readyHead = 0, g_readyCount = 0;

const char* const kPhaseNames[kPhases] = {"outer_l", "outer_r", "focus_l", "focus_r", "compose_l", "compose_r"};

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

void ReleaseQueriesLocked() {
    for (Slot& s : g_slot) {
        SafeRelease(s.disjoint);
        for (auto*& q : s.ts) SafeRelease(q);
        s = Slot{};
    }
    g_cur = -1;
    g_readyHead = g_readyCount = 0;
    g_createFailed = false;
}

bool CreateQueriesLocked() {
    if (g_slot[0].disjoint) return true;
    if (g_createFailed || !g_dev) return false;
    D3D11_QUERY_DESC dj{D3D11_QUERY_TIMESTAMP_DISJOINT, 0}, ts{D3D11_QUERY_TIMESTAMP, 0};
    HRESULT hr = S_OK;
    for (Slot& s : g_slot) {
        if (SUCCEEDED(hr)) hr = g_dev->CreateQuery(&dj, &s.disjoint);
        for (auto*& q : s.ts)
            if (SUCCEEDED(hr)) hr = g_dev->CreateQuery(&ts, &q);
    }
    if (FAILED(hr)) {
        ReleaseQueriesLocked();
        g_createFailed = true;
        Log("GPU timing: cannot create the timestamp queries (0x%08lX); no GPU times", (unsigned long)hr);
        return false;
    }
    Log("GPU timing: %d frames x (1 disjoint + %d timestamp) queries", kSlots, kMarks);
    return true;
}

float Span(const uint64_t* t, uint32_t issued, int a, int b, double freq) {
    if (!(issued & (1u << a)) || !(issued & (1u << b)) || t[b] < t[a]) return -1;
    return (float)((t[b] - t[a]) * 1000.0 / freq);
}

// True when the slot is done with (read or discarded), false while the GPU has not finished it.
bool ReadLocked(ID3D11DeviceContext* ctx, Slot& s) {
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj{};
    if (ctx->GetData(s.disjoint, &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) return false;
    uint64_t t[kMarks] = {};
    for (int m = 0; m < kMarks; ++m) {
        if (!(s.issued & (1u << m))) continue;
        if (ctx->GetData(s.ts[m], &t[m], sizeof(t[m]), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) return false;
    }
    s.state = Slot::Free;
    if (dj.Disjoint || !dj.Frequency) {
        ++g_disjoint;
        return true;
    }
    Result r;
    r.frame = s.frame;
    r.frameMs = Span(t, s.issued, kFrameBegin, kFrameEnd, (double)dj.Frequency);
    for (int p = 0; p < kPhases; ++p) r.phaseMs[p] = Span(t, s.issued, 2 + 2 * p, 3 + 2 * p, (double)dj.Frequency);
    if (!g_firstLogged) {
        g_firstLogged = true;
        Log("GPU timing: first frame %llu read (timestamp frequency %llu Hz), frame %.3f ms",
            (unsigned long long)r.frame, (unsigned long long)dj.Frequency, r.frameMs);
    }
    if (g_readyCount == kReady) {  // nobody takes them: keep the newest
        g_readyHead = (g_readyHead + 1) % kReady;
        --g_readyCount;
    }
    g_ready[(g_readyHead + g_readyCount) % kReady] = r;
    ++g_readyCount;
    return true;
}

void MarkLocked(int m) {
    if (g_cur < 0 || GetCurrentThreadId() != g_thread) return;
    Slot& s = g_slot[g_cur];
    if (s.state != Slot::Open || (s.issued & (1u << m))) return;
    ID3D11DeviceContext* ctx = g_ctx.load();
    if (!ctx) return;
    ctx->End(s.ts[m]);
    s.issued |= 1u << m;
}

}  // namespace

ID3D11DeviceContext* Context() { return g_ctx.load(std::memory_order_acquire); }
uint32_t Generation() { return g_gen.load(); }

bool OnSubmitTexture(void* d3dTexture) {
    if (!d3dTexture) return false;
    bool changed = false;
    Seh("device check", [&] {
        ID3D11Texture2D* t = nullptr;
        if (FAILED(static_cast<IUnknown*>(d3dTexture)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) || !t)
            return;
        ID3D11Device* dev = nullptr;
        t->GetDevice(&dev);
        t->Release();
        if (!dev) return;
        std::lock_guard lock(g_m);
        if (dev == g_dev) {
            dev->Release();
            return;
        }
        ID3D11DeviceContext* ctx = nullptr;
        dev->GetImmediateContext(&ctx);
        if (!ctx) {
            dev->Release();
            return;
        }
        ReleaseQueriesLocked();
        ID3D11DeviceContext* old = g_ctx.exchange(ctx);
        if (old) old->Release();
        bool had = g_dev != nullptr;
        if (g_dev) g_dev->Release();
        g_dev = dev;  // the reference from GetDevice
        uint32_t gen = ++g_gen;
        changed = had;
        Log("GPU: %s D3D11 device %p, immediate context %p (generation %u)", had ? "NEW" : "game", (void*)dev,
            (void*)ctx, gen);
    });
    return changed;
}

const char* PhaseName(int p) { return p >= 0 && p < kPhases ? kPhaseNames[p] : "?"; }

bool Enabled() { return Cfg().gpuTiming && g_ctx.load() != nullptr; }

void FrameStart(uint64_t no) {
    if (!Cfg().gpuTiming) return;
    Guarded("gpu::FrameStart", [&] {
        std::lock_guard lock(g_m);
        ID3D11DeviceContext* ctx = g_ctx.load();
        if (!ctx || !CreateQueriesLocked()) return;
        g_thread = GetCurrentThreadId();
        if (g_cur >= 0 && g_slot[g_cur].state == Slot::Open) {
            ctx->End(g_slot[g_cur].disjoint);
            g_slot[g_cur].state = Slot::Pending;
        }
        // Oldest first; the GPU finishes frames in order, so the first one not ready ends the scan.
        for (int k = 1; k <= kSlots; ++k) {
            Slot& s = g_slot[(g_cur + k + kSlots) % kSlots];
            if (s.state == Slot::Pending && !ReadLocked(ctx, s)) break;
        }
        int next = (g_cur + 1 + kSlots) % kSlots;
        Slot& s = g_slot[next];
        if (s.state == Slot::Pending) {
            ++g_dropped;
            if (g_dropped == 1 || g_dropped % 1000 == 0)
                Log("GPU timing: frame %llu still not finished after %d frames, dropped (%llu so far)",
                    (unsigned long long)s.frame, kSlots, (unsigned long long)g_dropped);
        }
        s.state = Slot::Open;
        s.issued = 0;
        s.frame = no;
        g_cur = next;
        ctx->Begin(s.disjoint);
        MarkLocked(kFrameBegin);
    });
}

void FrameEnd() {
    if (!Cfg().gpuTiming) return;
    Guarded("gpu::FrameEnd", [&] {
        std::lock_guard lock(g_m);
        MarkLocked(kFrameEnd);
    });
}

void Begin(Phase p) {
    if (!Cfg().gpuTiming) return;
    Guarded("gpu::Begin", [&] {
        std::lock_guard lock(g_m);
        MarkLocked(2 + 2 * p);
    });
}

void End(Phase p) {
    if (!Cfg().gpuTiming) return;
    Guarded("gpu::End", [&] {
        std::lock_guard lock(g_m);
        MarkLocked(3 + 2 * p);
    });
}

int Take(Result* out, int max) {
    int n = 0;
    Guarded("gpu::Take", [&] {
        std::lock_guard lock(g_m);
        while (n < max && g_readyCount > 0) {
            out[n++] = g_ready[g_readyHead];
            g_readyHead = (g_readyHead + 1) % kReady;
            --g_readyCount;
        }
    });
    return n;
}

void Release() {
    Seh("Release", [&] {
        std::lock_guard lock(g_m);
        if (g_dropped || g_disjoint)
            Log("GPU timing: %llu frames dropped (not finished in time), %llu disjoint", (unsigned long long)g_dropped,
                (unsigned long long)g_disjoint);
        ReleaseQueriesLocked();
        ID3D11DeviceContext* old = g_ctx.exchange(nullptr);
        if (old) old->Release();
        SafeRelease(g_dev);
        ++g_gen;  // a later session starts from a new generation, so cached objects are dropped
        g_dropped = g_disjoint = 0;
        g_firstLogged = false;
    });
}

}  // namespace qvfr::gpu

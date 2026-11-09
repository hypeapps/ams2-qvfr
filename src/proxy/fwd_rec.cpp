#include "fwd_rec.h"

#include <atomic>
#include <cstdio>
#include <vector>

#include "common.h"
#include "hooks.h"

namespace qvfr::fwdrec {

namespace {

using hooks::Anchor;

// Engine layout (Ghidra, AMS2AVX.exe aa43bd68, docs/mvr_optimizations.md O2 step 3).
constexpr uintptr_t kEngineDevice = 0x2684ea0;            // DAT_142684ea0: the engine device
constexpr uintptr_t kVelPassKeyA = 0x2670870, kVelPassKeyB = 0x2670868;  // the velocity sub-passes' pass keys
constexpr size_t kCtxView = 0x27b04;                       // render context: the view being drawn
constexpr int kFocusView = 2;
constexpr size_t kPassCtx = 0x48;                          // pass state: recording context index
constexpr size_t kDevPassCount = 0xe290, kDevPassCur = 0xe294, kDevPassTable = 0xe2b8;  // 16 B entries, key at +0
constexpr uint32_t kMaxRecCtx = 64, kMaxPasses = 256;

using ForwardDrawFn = void(__fastcall*)(uint32_t recCtx, uint8_t* ctx, void* a3, void* a4);
using SubmitFn = uint64_t(__fastcall*)(uint8_t* dev, uint8_t* pass, uint32_t layer, void* obj, void* param);
ForwardDrawFn o_forwardDraw = nullptr;
SubmitFn o_submit = nullptr;
bool g_installed = false;

std::atomic<int> g_armed{-1};  // focus eye armed, -1 = none
uint64_t g_keyA = 0, g_keyB = 0;  // set before g_armed
std::atomic<bool> g_seen{false};
std::atomic<uint32_t> g_kept{0}, g_skipped{0}, g_drew{0};

thread_local int t_rec = -1;  // this thread records an armed focus forward pass
thread_local bool t_drew = false;
thread_local uint32_t t_kept = 0, t_skipped = 0, t_drewCount = 0;

void __fastcall D_ForwardDraw(uint32_t recCtx, uint8_t* ctx, void* a3, void* a4) {
    const int eye = g_armed.load(std::memory_order_acquire);
    int view = -1;
    if (eye >= 0 && ctx && hooks::TryRead(ctx + kCtxView, &view, 4) && view == kFocusView && t_rec < 0) {
        t_rec = eye;
        t_drew = false;
        t_kept = t_skipped = t_drewCount = 0;
        o_forwardDraw(recCtx, ctx, a3, a4);
        t_rec = -1;
        g_kept.fetch_add(t_kept, std::memory_order_relaxed);
        g_skipped.fetch_add(t_skipped, std::memory_order_relaxed);
        g_drew.fetch_add(t_drewCount, std::memory_order_relaxed);
        g_seen.store(true, std::memory_order_release);
        return;
    }
    o_forwardDraw(recCtx, ctx, a3, a4);
}

uint64_t __fastcall D_Submit(uint8_t* dev, uint8_t* pass, uint32_t layer, void* obj, void* param) {
    if (t_rec < 0 || !dev || !pass) return o_submit(dev, pass, layer, obj, param);
    const uint32_t rc = *(const uint32_t*)(pass + kPassCtx);
    if (rc < kMaxRecCtx) {
        const uint32_t n = *(const uint32_t*)(dev + kDevPassCount);
        const uint32_t idx = *(const uint32_t*)(dev + kDevPassCur + rc * 4);
        const uint8_t* table = *(uint8_t* const*)(dev + kDevPassTable);
        if (table && idx < n && n <= kMaxPasses) {
            const uint64_t key = *(const uint64_t*)(table + idx * 16);
            if (key == g_keyA || key == g_keyB) {
                if (t_drew) {
                    ++t_skipped;
                    return 0;  // the submit's own "nothing drawn" (its early outs return 0)
                }
                ++t_kept;
                const uint64_t r = o_submit(dev, pass, layer, obj, param);
                // Low byte: the entry went into the recording (the submit's early outs clear it, its success path
                // returns the object's submit result). The engine records into its own command buffer, so no D3D11
                // draw happens here; it comes when CForwardRender replays the buffer.
                if (r & 0xff) {
                    t_drew = true;
                    ++t_drewCount;
                }
                return r;
            }
        }
    }
    return o_submit(dev, pass, layer, obj, param);
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    if (!hooks::EngineAvailable()) return false;
    void* fwd = (void*)hooks::Address(Anchor::ForwardDraw);
    void* sub = (void*)hooks::Address(Anchor::DeviceSubmit);
    if (!fwd || !sub) {
        Log("Focus forward recording: anchors not found; its velocity sub-passes stay as they are");
        return false;
    }
    // The submit first: without the forward hook it never sees an armed thread.
    if (!hooks::CreateAt(sub, (void*)D_Submit, (void**)&o_submit, "DeviceSubmit") ||
        !hooks::CreateAt(fwd, (void*)D_ForwardDraw, (void**)&o_forwardDraw, "ForwardDraw")) {
        Log("Focus forward recording: hooks failed; its velocity sub-passes stay as they are");
        return false;
    }
    g_installed = true;
    Log("Focus forward recording: hooks on (forward AMS2AVX.exe+0x%llX, device submit +0x%llX)",
        (unsigned long long)((uintptr_t)fwd - hooks::ExeBase()), (unsigned long long)((uintptr_t)sub - hooks::ExeBase()));
    return true;
}

bool Installed() { return g_installed; }

namespace {

// SelfTest: a fake submit (layer 1 = the entry issues a draw) and a fake recording that runs a list of submits.
struct TestSubmit {
    uint32_t pass, layer;
};
uint8_t* g_tDev = nullptr;
uint8_t* g_tPass = nullptr;
const std::vector<TestSubmit>* g_tSeq = nullptr;
uint32_t g_tCalls = 0;

uint64_t __fastcall FakeSubmit(uint8_t*, uint8_t*, uint32_t layer, void*, void*) {
    ++g_tCalls;
    return layer ? 0x1234500 | layer : 0x1234500;  // low byte: the entry went into the recording
}

void __fastcall FakeForward(uint32_t, uint8_t*, void*, void*) {
    for (const TestSubmit& s : *g_tSeq) {
        *(uint32_t*)(g_tDev + kDevPassCur + 3 * 4) = s.pass;
        D_Submit(g_tDev, g_tPass, s.layer, nullptr, nullptr);
    }
}

}  // namespace

bool SelfTest(char* out, size_t n) {
    const ForwardDrawFn saveFwd = o_forwardDraw;
    const SubmitFn saveSub = o_submit;
    o_forwardDraw = &FakeForward;
    o_submit = &FakeSubmit;
    std::vector<uint8_t> dev(0xe300), pass(0x50), ctx(kCtxView + 4);
    const uint64_t table[3][2] = {{111, 0}, {222, 0}, {333, 0}};
    *(uint32_t*)(dev.data() + kDevPassCount) = 3;
    *(const void**)(dev.data() + kDevPassTable) = table;
    *(uint32_t*)(pass.data() + kPassCtx) = 3;
    g_tDev = dev.data();
    g_tPass = pass.data();
    // Pass 0 is not velocity, passes 1 and 2 are (keys 222, 333).
    const std::vector<TestSubmit> seq = {{0, 1}, {1, 0}, {1, 1}, {1, 1}, {2, 0}, {2, 1}, {0, 0}};
    g_tSeq = &seq;
    auto run = [&](int view, bool arm) {
        g_keyA = 222;
        g_keyB = 333;
        g_seen = false;
        g_kept = g_skipped = g_drew = 0;
        if (arm) g_armed = 0;
        *(int*)(ctx.data() + kCtxView) = view;
        g_tCalls = 0;
        D_ForwardDraw(0, ctx.data(), nullptr, nullptr);
        return Disarm();
    };
    const Result focus = run(kFocusView, true);
    const uint32_t focusCalls = g_tCalls;
    const Result other = run(1, true);
    const uint32_t otherCalls = g_tCalls;
    const Result off = run(kFocusView, false);
    const uint32_t offCalls = g_tCalls;
    o_forwardDraw = saveFwd;
    o_submit = saveSub;
    g_tSeq = nullptr;
    // Focus: 2 plain submits + 2 velocity ones up to the first draw; 3 velocity ones skipped.
    const bool ok = focus.armed && focus.seen && focus.kept == 2 && focus.skipped == 3 && focus.drew == 1 &&
                    focusCalls == 4 && other.armed && !other.seen && otherCalls == 7 && !off.armed && offCalls == 7 &&
                    t_rec < 0;
    snprintf(out, n,
             "Focus forward recording check %s: focus view: submits through %u of 7 (velocity kept %u, skipped %u, "
             "draws %u, seen %d); outer view while armed: through %u of 7 (seen %d); not armed: through %u of 7",
             ok ? "OK" : "FAILED", focusCalls, focus.kept, focus.skipped, focus.drew, (int)focus.seen, otherCalls,
             (int)other.seen, offCalls);
    return ok;
}

bool Arm(int eye, const char** why) {
    *why = nullptr;
    if (!g_installed) {
        *why = "hooks not installed";
        return false;
    }
    const uintptr_t base = hooks::ExeBase();
    uint8_t* dev = nullptr;
    uint64_t a = 0, b = 0;
    uint32_t n = 0;
    uint8_t* table = nullptr;
    if (!hooks::TryRead((void*)(base + kEngineDevice), &dev, 8) || !dev ||
        !hooks::TryRead((void*)(base + kVelPassKeyA), &a, 8) || !hooks::TryRead((void*)(base + kVelPassKeyB), &b, 8) ||
        !hooks::TryRead(dev + kDevPassCount, &n, 4) || !hooks::TryRead(dev + kDevPassTable, &table, 8)) {
        *why = "engine device or pass keys unreadable";
        return false;
    }
    if (!a || !b || a == b || !table || !n || n > kMaxPasses) {
        *why = "velocity pass keys or the device's pass table not as expected";
        return false;
    }
    bool foundA = false, foundB = false;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t k = 0;
        if (!hooks::TryRead(table + i * 16, &k, 8)) break;
        foundA |= k == a;
        foundB |= k == b;
    }
    if (!foundA || !foundB) {
        *why = "velocity pass keys not in the device's pass table";
        return false;
    }
    g_keyA = a;
    g_keyB = b;
    g_seen.store(false, std::memory_order_relaxed);
    g_kept.store(0, std::memory_order_relaxed);
    g_skipped.store(0, std::memory_order_relaxed);
    g_drew.store(0, std::memory_order_relaxed);
    g_armed.store(eye, std::memory_order_release);
    return true;
}

Result Disarm() {
    Result r;
    r.armed = g_armed.exchange(-1, std::memory_order_acq_rel) >= 0;
    if (!r.armed) return r;
    r.seen = g_seen.load(std::memory_order_acquire);
    r.kept = g_kept.load(std::memory_order_relaxed);
    r.skipped = g_skipped.load(std::memory_order_relaxed);
    r.drew = g_drew.load(std::memory_order_relaxed);
    return r;
}

}  // namespace qvfr::fwdrec

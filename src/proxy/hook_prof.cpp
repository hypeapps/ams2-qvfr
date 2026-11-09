// Hook CPU profile (hook_prof.h).
#include "hook_prof.h"

#include "common.h"

#include <string>

namespace qvfr::hookprof {

std::atomic<bool> g_on{false};
thread_local ThreadState t_state;

namespace {

constexpr int kGroups = 4;
constexpr int kOutside = 4;  // the render thread outside the eye groups
constexpr int kOther = 5;    // other threads
constexpr int kCalib = 6;    // the profile's own cost (SetOn)
constexpr int kBuckets = 7;

const char* const kBucketNames[] = {"outer L", "outer R", "focus L", "focus R", "render thread outside the eye groups",
                                    "other threads"};
const char* const kKindNames[kKinds] = {"draws",          "PSSetShader", "VSSetShader",      "VSSetConstantBuffers",
                                        "PSSetConstantBuffers",
                                        "PSSetShaderResources", "VS/GS/CSSetShaderResources", "OMSetRenderTargets*",
                                        "CSSetUnorderedAccessViews", "RSSetViewports", "copies", "clears",
                                        "task hook"};
const char* const kSubNames[kSubs] = {"ForwardDraw",  "GbufDraw",   "SetMultiViewMode", "mvrps::InfoOf",
                                      "mvrvs::IsPatched", "clones map", "taaviews::OnDraw", "DLAA draw check"};

struct Stat {
    std::atomic<uint64_t> calls{0}, cycles{0}, subs{0};
};
struct Bucket {
    Stat kind[kKinds];
    Stat sub[kSubs];
    std::atomic<uint64_t> counter[kCounters] = {};
};
Bucket g_b[kBuckets];
std::atomic<uint64_t> g_begins[kGroups] = {};

std::atomic<DWORD> g_renderTid{0};
std::atomic<int> g_group{-1};
thread_local bool t_calib = false;

// The profile's own cycles: per outermost scope (with its Orig) and per Sub inside one (taken out of the parent).
double g_ovTop = 0, g_ovSub = 0, g_ovSubIn = 0;
double g_last[kOther + 1] = {};  // the proxy's ms per frame per bucket in the latest log (LastTotals)
// TSC against QPC since the latest reset or log.
uint64_t g_tsc0 = 0;
int64_t g_qpc0 = 0;

int BucketOf() {
    if (t_calib) return kCalib;
    if (GetCurrentThreadId() != g_renderTid.load(std::memory_order_relaxed)) return kOther;
    const int g = g_group.load(std::memory_order_relaxed);
    return g >= 0 && g < kGroups ? g : kOutside;
}

void Add(std::atomic<uint64_t>& a, uint64_t v) { a.fetch_add(v, std::memory_order_relaxed); }

void ResetLocked() {
    for (Bucket& b : g_b) {
        for (Stat& s : b.kind) s.calls = s.cycles = s.subs = 0;
        for (Stat& s : b.sub) s.calls = s.cycles = s.subs = 0;
        for (auto& c : b.counter) c = 0;
    }
    for (auto& f : g_begins) f = 0;
    LARGE_INTEGER q{};
    QueryPerformanceCounter(&q);
    g_qpc0 = q.QuadPart;
    g_tsc0 = Tsc();
}

void Calibrate() {
    constexpr int kLoops = 20000;
    t_calib = true;
    Bucket& b = g_b[kCalib];
    for (int i = 0; i < kLoops; ++i) {
        Scope s(KDraw);
        Orig o;
    }
    const double top = (double)b.kind[KDraw].cycles.load() / kLoops;
    for (int i = 0; i < kLoops; ++i) {
        Scope s(KPsShader);
        {
            Orig o;
        }
        SubScope x(SPsInfo);
    }
    const double withSub = (double)b.kind[KPsShader].cycles.load() / kLoops;
    t_calib = false;
    g_ovTop = top;
    g_ovSub = withSub > top ? withSub - top : 0;
    g_ovSubIn = (double)b.sub[SPsInfo].cycles.load() / kLoops;  // an empty part's own span
}

}  // namespace

void Record(const Scope& s, uint64_t end) {
    Bucket& b = g_b[BucketOf()];
    Stat& k = b.kind[s.kind];
    Add(k.calls, 1);
    const uint64_t all = end - s.t0;
    Add(k.cycles, all > s.orig ? all - s.orig : 0);
    if (s.subs) Add(k.subs, s.subs);
}

void RecordSub(Sub s, uint64_t cycles) {
    Stat& k = g_b[BucketOf()].sub[s];
    Add(k.calls, 1);
    Add(k.cycles, cycles);
}

void Count(Counter c, uint32_t n) {
    if (On()) Add(g_b[BucketOf()].counter[c], n);
}

void SetGroup(int group) {
    if (group < 0) {  // an end: only on the thread whose group began
        if (GetCurrentThreadId() == g_renderTid.load(std::memory_order_relaxed)) g_group.store(-1, std::memory_order_relaxed);
        return;
    }
    g_renderTid.store(GetCurrentThreadId(), std::memory_order_relaxed);
    g_group.store(group, std::memory_order_relaxed);
    if (group >= 0 && group < kGroups && On()) Add(g_begins[group], 1);
}

void SetOn(bool on) {
    if (on == On()) return;
    if (!on) {
        g_on = false;
        Log("Hook profile: off");
        return;
    }
    g_on = true;  // the calibration scopes need it; the game's detours start counting too
    Calibrate();
    ResetLocked();
    LARGE_INTEGER f{};
    QueryPerformanceFrequency(&f);
    // A short TSC rate estimate for the first line (each log measures its own interval).
    LARGE_INTEGER q0{}, q1{};
    QueryPerformanceCounter(&q0);
    const uint64_t c0 = Tsc();
    do QueryPerformanceCounter(&q1);
    while (q1.QuadPart - q0.QuadPart < f.QuadPart / 200);  // 5 ms
    const double mhz = (double)(Tsc() - c0) / ((double)(q1.QuadPart - q0.QuadPart) / f.QuadPart) / 1e6;
    Log("Hook profile: on ([debug] hook_profile=1); the proxy's own time in its D3D11 detours and the task hook, "
        "without the original call; the profile's own cost %.0f cycles per detour, %.0f per measured part in its parent "
        "and %.0f inside the part (%.1f, %.1f and %.1f ns at a TSC of %.0f MHz), taken out of the times; it slows the "
        "frame (about 30 ns per call)",
        g_ovTop, g_ovSub, g_ovSubIn, g_ovTop * 1000 / mhz, g_ovSub * 1000 / mhz, g_ovSubIn * 1000 / mhz, mhz);
}

void LogStats(const double groupMs[4]) {
    for (double& v : g_last) v = 0;
    if (!On()) return;
    LARGE_INTEGER q{}, f{};
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    const uint64_t tsc = Tsc();
    const double secs = (double)(q.QuadPart - g_qpc0) / (double)f.QuadPart;
    const double perMs = secs > 0 ? (double)(tsc - g_tsc0) / (secs * 1000) : 0;  // TSC cycles per ms
    g_qpc0 = q.QuadPart;
    g_tsc0 = tsc;
    uint64_t begins[kGroups];
    for (int g = 0; g < kGroups; ++g) begins[g] = g_begins[g].exchange(0, std::memory_order_relaxed);
    if (perMs <= 0) return;
    for (int bi = 0; bi < kOther + 1; ++bi) {
        Bucket& b = g_b[bi];
        const uint64_t frames = bi < kGroups ? begins[bi] : begins[0];
        uint64_t calls[kKinds], cyc[kKinds], subs[kKinds], scalls[kSubs], scyc[kSubs], cnt[kCounters];
        for (int k = 0; k < kKinds; ++k) {
            calls[k] = b.kind[k].calls.exchange(0, std::memory_order_relaxed);
            cyc[k] = b.kind[k].cycles.exchange(0, std::memory_order_relaxed);
            subs[k] = b.kind[k].subs.exchange(0, std::memory_order_relaxed);
        }
        for (int k = 0; k < kSubs; ++k) {
            scalls[k] = b.sub[k].calls.exchange(0, std::memory_order_relaxed);
            scyc[k] = b.sub[k].cycles.exchange(0, std::memory_order_relaxed);
        }
        for (int k = 0; k < kCounters; ++k) cnt[k] = b.counter[k].exchange(0, std::memory_order_relaxed);
        if (!frames) continue;
        uint64_t any = 0;
        for (uint64_t c : calls) any += c;
        if (!any) continue;
        // Per frame: ms of the proxy's own time and calls; ns per call.
        auto ms = [&](double cycles) { return cycles / perMs / (double)frames; };
        double total = 0;
        uint64_t totalCalls = 0;
        std::string kinds;
        for (int k = 0; k < kKinds; ++k) {
            if (!calls[k]) continue;
            double c = (double)cyc[k] - (double)calls[k] * g_ovTop - (double)subs[k] * g_ovSub;
            if (c < 0) c = 0;
            total += c;
            totalCalls += calls[k];
            char item[128];
            snprintf(item, sizeof(item), "%s%s %.3f (%.0f calls, %.0f ns)", kinds.empty() ? "" : ", ", kKindNames[k],
                     ms(c), (double)calls[k] / frames, c / perMs * 1e6 / (double)calls[k]);
            kinds += item;
        }
        std::string parts;
        for (int k = 0; k < kSubs; ++k) {
            if (!scalls[k]) continue;
            double c = (double)scyc[k] - (double)scalls[k] * g_ovSubIn;
            if (c < 0) c = 0;
            char item[128];
            snprintf(item, sizeof(item), "%s%s %.3f (%.0f calls, %.0f ns)", parts.empty() ? "" : ", ", kSubNames[k],
                     ms(c), (double)scalls[k] / frames, c / perMs * 1e6 / (double)scalls[k]);
            parts += item;
        }
        g_last[bi] = ms(total);
        char share[64] = "";
        if (bi < kGroups && groupMs && groupMs[bi] > 0)
            snprintf(share, sizeof(share), "; group %.3f ms, proxy %.0f%% of it", groupMs[bi],
                     100 * ms(total) / groupMs[bi]);
        Log("Hook CPU per frame (%s, %llu frames%s): proxy %.3f ms over %.0f calls: %s; parts (inclusive): %s; per frame: "
            "PS slot 3 rebinds %.1f, t21-t26 binds %.1f (unchanged %.1f), PS Get* calls %.1f",
            kBucketNames[bi], (unsigned long long)frames, share, ms(total), (double)totalCalls / frames, kinds.c_str(),
            parts.empty() ? "none" : parts.c_str(), (double)cnt[CPsSlot3Rebind] / frames,
            (double)cnt[CTexBind] / frames, (double)cnt[CTexBindSame] / frames, (double)cnt[CPsGet] / frames);
    }
}

uint64_t TestCalls() {
    uint64_t n = 0;
    for (int bi = 0; bi < kCalib; ++bi)
        for (const Stat& s : g_b[bi].kind) n += s.calls.load(std::memory_order_relaxed);
    return n;
}

int TestDepth() { return t_state.depth; }

bool LastTotals(double ms[6]) {
    bool any = false;
    for (int i = 0; i < 6; ++i) any |= (ms[i] = g_last[i]) > 0;
    return any;
}

}  // namespace qvfr::hookprof

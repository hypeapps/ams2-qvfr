// Hook CPU profile (mvr_optimizations O1 step 0, [debug] hook_profile): the proxy's own time in its D3D11 detours and in
// the task hook, without the original call, per eye group and kind, measured with the TSC. Off by default: then every
// scope is one relaxed load.
//
// Scope: the whole detour (the outermost one on the thread counts; a detour called by the proxy inside it is the
// outer one's work). Orig: the original call inside a detour; its time is taken out, and detours the original calls
// (the task's Execute) count on their own. Sub: a part measured inclusive (nested in the scopes), with its calls.
#pragma once

#include <atomic>
#include <cstdint>

#include <intrin.h>

namespace qvfr::hookprof {

enum Kind : uint8_t {
    KDraw,       // Draw*, DrawIndexed* (sun_fx, mvr_pass, dlaa)
    KPsShader,   // PSSetShader (sun_fx)
    KVsShader,   // VSSetShader (mvr_pass)
    KVsCb,       // VSSetConstantBuffers(1) (mvr_pass)
    KPsCb,       // PSSetConstantBuffers(1) (mvr_pass)
    KPsSrv,      // PSSetShaderResources (clones)
    KOtherSrv,   // VS/GS/CSSetShaderResources (clones)
    KTargets,    // OMSetRenderTargets (quad), OMSetRenderTargetsAndUnorderedAccessViews (clones)
    KCsUav,      // CSSetUnorderedAccessViews (clones)
    KViewports,  // RSSetViewports (clones)
    KCopy,       // CopyResource, CopySubresourceRegion, ResolveSubresource, GenerateMips (clones)
    KClear,      // Clear*View (clones)
    KTask,       // RunTask (quad) without the task's Execute and the early focus run
    kKinds
};

enum Sub : uint8_t {
    SForwardDraw,   // mvr_pass ForwardDraw (per draw of a forward pass)
    SGbufDraw,      // mvr_pass GbufDraw (per draw of a G-buffer pass with two views)
    SSetViews,      // NvAPI_D3D_SetMultiViewMode
    SPsInfo,        // mvrps::InfoOf
    SVsPatched,     // mvrvs::IsPatched in VSSetShader
    SCloneMap,      // clones MapViews / MapRes past the live check (the mutex and the lookups)
    STaaDraw,       // taaviews::OnDraw
    SDlss,          // DLAA's check of a DrawIndexed for the TAA resolve (and its DLSS evaluation)
    kSubs
};

enum Counter : uint8_t {
    CPsSlot3Rebind,  // MVR forward or G-buffer draw: PS slot 3 rebound
    CTexBind,        // MVR forward draw: t21..t26 bound
    CTexBindSame,    //   of them with the same six views as the previous bind in this pass
    CPsGet,          // PSGetShader + PSGetShaderResources + PSGetConstantBuffers calls in MVR draws (O1 step 3: only
                     // those the hooks' state could not answer)
    kCounters
};

extern std::atomic<bool> g_on;
inline bool On() { return g_on.load(std::memory_order_relaxed); }

struct Scope;
struct ThreadState {
    int depth = 0;
    Scope* top = nullptr;
    unsigned aux = 0;  // Tsc's TSC_AUX output
};
extern thread_local ThreadState t_state;

// RDTSCP into the thread's state: the store keeps the reads in program order (the compiler reorders plain __rdtsc
// calls that do not depend on each other), and the CPU waits for the instructions before it.
inline uint64_t Tsc() { return __rdtscp(&t_state.aux); }

void Record(const Scope& s, uint64_t end);
void RecordSub(Sub s, uint64_t cycles);

struct Scope {
    uint64_t t0 = 0;
    uint64_t orig = 0;   // cycles in the original call
    uint32_t subs = 0;   // Sub scopes inside (their own overhead is taken out)
    Kind kind;
    bool active = false;
    bool top = false;
    explicit Scope(Kind k) : kind(k) {
        if (!On()) return;
        active = true;
        ThreadState& t = t_state;
        if (t.depth++ == 0) {
            top = true;
            t.top = this;
            t0 = Tsc();
        }
    }
    ~Scope() {
        if (!active) return;
        ThreadState& t = t_state;
        if (top) {
            Record(*this, Tsc());
            t.top = nullptr;
        }
        --t.depth;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

struct Orig {
    Scope* owner = nullptr;
    uint64_t t0 = 0;
    Orig() {
        ThreadState& t = t_state;
        if (!t.top || t.depth != 1) return;  // nested: the original call is part of the outer detour's work
        owner = t.top;
        t.top = nullptr;
        t.depth = 0;
        t0 = Tsc();
    }
    ~Orig() {
        if (!owner) return;
        const uint64_t t1 = Tsc();
        owner->orig += t1 - t0;
        ThreadState& t = t_state;
        t.top = owner;
        t.depth = 1;
    }
    Orig(const Orig&) = delete;
    Orig& operator=(const Orig&) = delete;
};

struct SubScope {
    uint64_t t0 = 0;
    Sub sub;
    bool active = false;
    explicit SubScope(Sub s) : sub(s) {
        if (!On()) return;
        active = true;
        if (Scope* top = t_state.top) ++top->subs;
        t0 = Tsc();
    }
    ~SubScope() {
        if (active) RecordSub(sub, Tsc() - t0);
    }
    SubScope(const SubScope&) = delete;
    SubScope& operator=(const SubScope&) = delete;
};

void Count(Counter c, uint32_t n = 1);

// Render thread: an eye group begins (0..3: outer L, outer R, focus L, focus R) or ends (-1).
void SetGroup(int group);
// [debug] hook_profile (live): on resets the counters and measures the profile's own cost per scope.
void SetOn(bool on);
// One line per bucket (eye groups, the render thread outside them, other threads), counters reset. groupMs: the eye
// groups' CPU per frame from quad's stats (0: unknown), for the share.
void LogStats(const double groupMs[4]);

// The proxy's ms per frame per bucket (outer L, outer R, focus L, focus R, outside, other threads) of the latest
// LogStats; false: the profile was off or measured nothing.
bool LastTotals(double ms[6]);

// qvfr_proxy_smoke --hook-profile: outermost scopes recorded since SetOn (all buckets), this thread's scope depth.
uint64_t TestCalls();
int TestDepth();

}  // namespace qvfr::hookprof

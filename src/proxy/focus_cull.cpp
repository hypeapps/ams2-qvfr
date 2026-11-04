#include "focus_cull.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "hooks.h"

namespace qvfr::focus_cull {

namespace {

using hooks::Anchor;

// Renderables set (gather result, 0x2f0 B, ctor 0xe21400). The gather stores its arguments here when it runs on a
// worker (0xe216f0), which CMainSceneGather always asks for, so they can be read back for the inline run.
constexpr size_t kSetEvent = 0x0;      // HANDLE, set when done
constexpr size_t kSetCount = 0x58;     // u32 renderables in the list (+0x60 begin, +0x68 cursor, 16 B entries)
constexpr size_t kSetWorld = 0x70;     // scene the gather walks
constexpr size_t kSetSpin = 0x78;      // shared "lists outside the tree taken" flag of the spin-off gathers (int*)
constexpr size_t kSetStatus = 0x84;    // 1 = running, 2 = done
constexpr size_t kSetSelf = 0xa8;      // the set itself (stored by the worker path only)
constexpr size_t kSetPass = 0xb0;      // pass type (4 = main view)
constexpr size_t kSetIncl = 0xd0;      // include flags (0 = none)
constexpr size_t kSetExcl = 0xd8;      // exclude flags (0 = none)
constexpr size_t kSetCamera = 0xf8;    // camera (distances, LOD)
constexpr size_t kSetFrustum = 0x100;  // copy of the frustum block it was asked for
constexpr size_t kSetArg6 = 0x2a4;     // int (2 for the main view)
constexpr size_t kSetArg7 = 0x2a8;     // pointer into the gather task
// Processor (0x210 B, ctor 0xe4d600; 0xe4d010 runs it, 0xe4c310 does the work).
constexpr size_t kProcEvent = 0x0;
constexpr size_t kProcSet = 0x88;      // renderables set of the last run
constexpr size_t kProcSelf = 0x90;
constexpr size_t kProcParams = 0xa0;   // copy of the view parameters (POD, 0x158 B, 0xe4d310)
constexpr size_t kParamsSize = 0x158;
constexpr size_t kParamFrustum = 0x20;  // const frustum block*
constexpr size_t kParamIncl = 0x28;     // const u64* include mask (null = none)
constexpr size_t kParamExcl = 0x30;     // const u64* exclude mask
constexpr size_t kProcStatus = 0x200;  // 1 = running, 2 = done
// Render context.
constexpr size_t kCtxSet = 0x670;       // mRenderablesInMainView
constexpr size_t kCtxProc = 0x690;      // mMainViewProcessor
constexpr size_t kCtxQueueCount = 0x69c;  // u32 entries of the draw queue the eye groups read (+0x6a0 begin, +0x6a8 end)
constexpr size_t kCtxView1Block = 0x190 + 1 * kBlockSize;

constexpr uint32_t kInline = 0xffffffff;  // gather and processor worker argument: run on this thread

using GatherFn = void*(__fastcall*)(void* set, int pass, const void* frustum, const uint64_t* incl,
                                     const uint64_t* excl, int arg6, void* arg7, void* camera, char occlusion,
                                     uint32_t worker, const char* name, int index, int count, void* const* children,
                                     void* arg15, void* helper, void* done);
using ProcessFn = void*(__fastcall*)(void* proc, void* set, const void* params, uint32_t worker);
using QueueCopyFn = void(__fastcall*)(void* unused, void* ctx);

// ---------------------------------------------------------------- plane math (double)

struct Plane {
    double n[3], d;
};

Plane ReadPlane(const float* f) { return {{f[0], f[1], f[2]}, f[3]}; }

double Dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

bool Normalize(Plane& p) {
    double len = std::sqrt(Dot3(p.n, p.n));
    if (len < 1e-12) return false;
    for (double& c : p.n) c /= len;
    p.d /= len;
    return true;
}

// m = a * b (row-major 4 x 4, row vectors: clip = p * view * proj).
void Mul(const float* a, const float* b, double* m) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += (double)a[i * 4 + k] * b[k * 4 + j];
            m[i * 4 + j] = s;
        }
}

// Clip-space edge plane: k * w + sign * x (col 0) or y (col 1) >= 0, from the columns of view * proj.
// Edges in Layout::side order: left, right, bottom, top.
Plane Edge(const double* m, int edge, double k) {
    const int col = edge < 2 ? 0 : 1;
    const double sign = (edge & 1) ? -1.0 : 1.0;
    Plane p;
    for (int i = 0; i < 3; ++i) p.n[i] = sign * m[i * 4 + col] + k * m[i * 4 + 3];
    p.d = sign * m[12 + col] + k * m[15];
    Normalize(p);
    return p;
}

// Point on three planes; false when they are (nearly) parallel.
bool Intersect(const Plane& a, const Plane& b, const Plane& c, double* out) {
    const double* n0 = a.n;
    const double* n1 = b.n;
    const double* n2 = c.n;
    double c12[3] = {n1[1] * n2[2] - n1[2] * n2[1], n1[2] * n2[0] - n1[0] * n2[2], n1[0] * n2[1] - n1[1] * n2[0]};
    double c20[3] = {n2[1] * n0[2] - n2[2] * n0[1], n2[2] * n0[0] - n2[0] * n0[2], n2[0] * n0[1] - n2[1] * n0[0]};
    double c01[3] = {n0[1] * n1[2] - n0[2] * n1[1], n0[2] * n1[0] - n0[0] * n1[2], n0[0] * n1[1] - n0[1] * n1[0]};
    double det = Dot3(n0, c12);
    if (std::fabs(det) < 1e-9) return false;
    for (int i = 0; i < 3; ++i) out[i] = -(a.d * c12[i] + b.d * c20[i] + c.d * c01[i]) / det;
    return true;
}

constexpr int kTriples = 20;
uint8_t g_triples[kTriples][3];
bool g_triplesReady = false;

void MakeTriples() {
    if (g_triplesReady) return;
    int t = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a + 1; b < 6; ++b)
            for (int c = b + 1; c < 6; ++c) {
                g_triples[t][0] = (uint8_t)a;
                g_triples[t][1] = (uint8_t)b;
                g_triples[t][2] = (uint8_t)c;
                ++t;
            }
    g_triplesReady = true;
}

uint32_t Bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

float FromBits(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

constexpr int kMaskBase = 0x18;   // float index of plane 0's mask; plane i at kMaskBase + 8 * i, its complement + 4
constexpr int kCornerBase = 0x48; // float index of corner 0; corner j at kCornerBase + 4 * j

// Masks and corners of planes pl into block f (lanes w of the masks stay as they are).
void WriteDerived(float* f, const Plane* pl, const Layout& l) {
    for (int i = 0; i < 6; ++i)
        for (int c = 0; c < 3; ++c) {
            uint32_t m = pl[i].n[c] >= 0 ? l.maskSet : l.maskClear;
            f[kMaskBase + 8 * i + c] = FromBits(m);
            f[kMaskBase + 8 * i + 4 + c] = FromBits(~m);
        }
    for (int j = 0; j < 8; ++j) {
        double p[3] = {};
        Intersect(pl[l.corner[j][0]], pl[l.corner[j][1]], pl[l.corner[j][2]], p);
        for (int c = 0; c < 3; ++c) f[kCornerBase + 4 * j + c] = (float)p[c];
        f[kCornerBase + 4 * j + 3] = 1.0f;
    }
}

GatherFn g_gather = nullptr;
ProcessFn g_process = nullptr;
QueueCopyFn g_queueCopy = nullptr;

void Resolve() {
    static bool done = false;
    if (done) return;
    done = true;
    g_gather = (GatherFn)hooks::Address(Anchor::SceneGather);
    g_process = (ProcessFn)hooks::Address(Anchor::SceneProcess);
    g_queueCopy = (QueueCopyFn)hooks::Address(Anchor::MainQueueCopy);
}

template <class T>
T& At(uint8_t* p, size_t off) {
    return *(T*)(p + off);
}

// The engine calls, under SEH (no C++ objects in this function). On a fault both objects are marked done and their
// events set, so the engine's own waits on them do not hang.
bool CallEngine(uint8_t* ctx, uint8_t* set, uint8_t* proc, const uint8_t* block, const uint64_t* incl,
                const uint64_t* excl, int arg6, void* arg7, void* camera, const uint8_t* params) {
    int* spin = At<int*>(set, kSetSpin);
    __try {
        // Without the spin-off flag the inline run also walks the lists outside the scene tree (dynamic objects),
        // which this frame's main gather already claimed.
        At<int*>(set, kSetSpin) = nullptr;
        void* r = g_gather(set, 4, block, incl, excl, arg6, arg7, camera, 0, kInline, "QvfrFocusQuery", 0, 0, nullptr,
                           nullptr, nullptr, nullptr);
        At<int*>(set, kSetSpin) = spin;
        if (!r || At<int>(set, kSetStatus) != 2) return false;
        g_process(proc, set, params, kInline);
        if (At<int>(proc, kProcStatus) != 2) return false;
        g_queueCopy(nullptr, ctx);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        At<int*>(set, kSetSpin) = spin;
        At<int>(set, kSetStatus) = 2;
        if (At<HANDLE>(set, kSetEvent)) SetEvent(At<HANDLE>(set, kSetEvent));
        At<int>(proc, kProcStatus) = 2;
        if (At<HANDLE>(proc, kProcEvent)) SetEvent(At<HANDLE>(proc, kProcEvent));
        return false;
    }
}

}  // namespace

bool Learn(const uint8_t* block, const float* view, const float* eyeProj, Layout& out, char* why, size_t whySize) {
    MakeTriples();
    const float* f = (const float*)block;
    Plane pl[6];
    for (int i = 0; i < 6; ++i) {
        pl[i] = ReadPlane(f + 4 * i);
        double len = std::sqrt(Dot3(pl[i].n, pl[i].n));
        if (len < 0.98 || len > 1.02) {
            snprintf(why, whySize, "block plane %d normal length %.4f (not unit)", i, len);
            return false;
        }
    }
    // Near and far: normals along the view's forward axis (world space: column 2 of the view rotation).
    double fwd[3] = {view[2], view[6], view[10]};
    double fl = std::sqrt(Dot3(fwd, fwd));
    if (fl < 0.5) {
        snprintf(why, whySize, "view matrix without a forward axis (|z| %.3f)", fl);
        return false;
    }
    for (double& c : fwd) c /= fl;
    int sides[6], nSides = 0, nNearFar = 0;
    for (int i = 0; i < 6; ++i) {
        if (std::fabs(Dot3(pl[i].n, fwd)) > 0.98)
            ++nNearFar;
        else
            sides[nSides++] = i;
    }
    if (nNearFar != 2 || nSides != 4) {
        snprintf(why, whySize, "%d block planes along the view axis, expected 2 (near and far)", nNearFar);
        return false;
    }
    // Eye position: view = p * R + t, so p = -t * R^T.
    double eye[3];
    for (int j = 0; j < 3; ++j) eye[j] = -(view[12] * view[j * 4] + view[13] * view[j * 4 + 1] + view[14] * view[j * 4 + 2]);
    double m[16];
    Mul(view, eyeProj, m);
    bool used[6] = {};
    for (int e = 0; e < 4; ++e) {
        Plane p = Edge(m, e, 1.0);
        int best = -1;
        double bestDot = -2;
        for (int s = 0; s < nSides; ++s) {
            double d = Dot3(p.n, pl[sides[s]].n);
            if (d > bestDot) {
                bestDot = d;
                best = sides[s];
            }
        }
        if (bestDot < 0.9 || used[best]) {
            snprintf(why, whySize, "eye edge %d matches no block side plane (best plane %d, dot %.3f%s)", e, best,
                     bestDot, best >= 0 && used[best] ? ", already taken" : "");
            return false;
        }
        used[best] = true;
        out.side[e] = best;
        double dist = Dot3(pl[best].n, eye) + pl[best].d;
        if (std::fabs(dist) > 0.05) {
            snprintf(why, whySize, "block side plane %d is %.3f m from the eye position of the view matrix", best, dist);
            return false;
        }
    }
    // Mask lanes: one bit pattern for components >= 0 and one for < 0, complements next to them.
    bool haveSet = false, haveClear = false;
    for (int i = 0; i < 6; ++i)
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(pl[i].n[c]) < 1e-6) continue;
            uint32_t v = Bits(f[kMaskBase + 8 * i + c]);
            uint32_t nv = Bits(f[kMaskBase + 8 * i + 4 + c]);
            if (nv != ~v) {
                snprintf(why, whySize, "mask of plane %d lane %d: 0x%08X, complement 0x%08X", i, c, v, nv);
                return false;
            }
            uint32_t& want = pl[i].n[c] >= 0 ? out.maskSet : out.maskClear;
            bool& have = pl[i].n[c] >= 0 ? haveSet : haveClear;
            if (!have) {
                have = true;
                want = v;
            } else if (want != v) {
                snprintf(why, whySize, "mask of plane %d lane %d: 0x%08X, other lanes of that sign 0x%08X", i, c, v, want);
                return false;
            }
        }
    if (!haveSet || !haveClear || out.maskSet == out.maskClear) {
        snprintf(why, whySize, "mask lanes do not depend on the normal's sign (0x%08X / 0x%08X)", out.maskSet,
                 out.maskClear);
        return false;
    }
    // Corners: the triple of planes each one is the intersection of.
    bool tripleUsed[kTriples] = {};
    for (int j = 0; j < 8; ++j) {
        const float* c = f + kCornerBase + 4 * j;
        if (c[3] != 1.0f) {
            snprintf(why, whySize, "corner %d has w %.4f, expected 1", j, c[3]);
            return false;
        }
        const double scale = std::max(1.0, std::sqrt((double)c[0] * c[0] + (double)c[1] * c[1] + (double)c[2] * c[2]));
        int best = -1;
        double bestErr = 1e30;
        for (int t = 0; t < kTriples; ++t) {
            double p[3];
            if (!Intersect(pl[g_triples[t][0]], pl[g_triples[t][1]], pl[g_triples[t][2]], p)) continue;
            double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
            double err = std::sqrt(dx * dx + dy * dy + dz * dz) / scale;
            if (err < bestErr) {
                bestErr = err;
                best = t;
            }
        }
        if (best < 0 || bestErr > 1e-3 || tripleUsed[best]) {
            snprintf(why, whySize, "corner %d (%.2f %.2f %.2f) is no intersection of three block planes (error %.2e)", j,
                     c[0], c[1], c[2], bestErr);
            return false;
        }
        tripleUsed[best] = true;
        memcpy(out.corner[j], g_triples[best], 3);
    }
    return true;
}

void Build(const uint8_t* eyeBlock, const Layout& l, const float* view, const float* focusProj, float margin,
           uint8_t* out) {
    memcpy(out, eyeBlock, kBlockSize);
    float* f = (float*)out;
    double m[16];
    Mul(view, focusProj, m);
    Plane pl[6];
    for (int i = 0; i < 6; ++i) pl[i] = ReadPlane(f + 4 * i);
    for (int e = 0; e < 4; ++e) pl[l.side[e]] = Edge(m, e, 1.0 + margin);
    for (int i = 0; i < 6; ++i) {
        f[4 * i + 0] = (float)pl[i].n[0];
        f[4 * i + 1] = (float)pl[i].n[1];
        f[4 * i + 2] = (float)pl[i].n[2];
        f[4 * i + 3] = (float)pl[i].d;
    }
    WriteDerived(f, pl, l);
}

bool Available() {
    Resolve();
    return g_gather && g_process && g_queueCopy;
}

bool Run(uint8_t* ctx, const uint8_t* block, Result& r, bool& fault, char* why, size_t whySize) {
    fault = false;
    if (!Available()) {
        snprintf(why, whySize, "engine functions not available");
        return false;
    }
    uint8_t* set = nullptr;
    uint8_t* proc = nullptr;
    if (!hooks::TryRead(ctx + kCtxSet, &set, sizeof(set)) || !hooks::TryRead(ctx + kCtxProc, &proc, sizeof(proc)) ||
        !set || !proc) {
        snprintf(why, whySize, "no renderables set or processor in the render context (%p, %p)", (void*)set,
                 (void*)proc);
        return false;
    }
    uint8_t s[0x2c0];
    uint8_t p[0x208];
    if (!hooks::TryRead(set, s, sizeof(s)) || !hooks::TryRead(proc, p, sizeof(p))) {
        snprintf(why, whySize, "renderables set %p or processor %p not readable", (void*)set, (void*)proc);
        return false;
    }
    if (At<int>(s, kSetStatus) == 0 && At<int>(p, kProcStatus) == 0) {
        r.notYet = true;  // the first frames of a graph (track loading) run the eye groups before any gather
        snprintf(why, whySize, "the engine has not culled the scene yet");
        return false;
    }
    if (At<int>(s, kSetStatus) != 2 || At<int>(p, kProcStatus) != 2) {
        snprintf(why, whySize, "gather or processor not done (status %d, %d)", At<int>(s, kSetStatus),
                 At<int>(p, kProcStatus));
        return false;
    }
    if (At<uint8_t*>(s, kSetSelf) != set || At<int>(s, kSetPass) != 4 || !At<void*>(s, kSetWorld) ||
        !At<void*>(s, kSetCamera)) {
        snprintf(why, whySize, "renderables set does not hold a worker gather of the main view (self %p, pass %d, "
                 "scene %p, camera %p)", At<void*>(s, kSetSelf), At<int>(s, kSetPass), At<void*>(s, kSetWorld),
                 At<void*>(s, kSetCamera));
        return false;
    }
    if (memcmp(s + kSetFrustum, ctx + kCtxView1Block, 0x60) != 0) {
        snprintf(why, whySize, "renderables set was not culled with view 1's frustum block in this frame");
        return false;
    }
    if (At<uint8_t*>(p, kProcSet) != set || At<uint8_t*>(p, kProcSelf) != proc) {
        snprintf(why, whySize, "processor %p did not run on the renderables set (%p, self %p)", (void*)proc,
                 At<void*>(p, kProcSet), At<void*>(p, kProcSelf));
        return false;
    }
    alignas(16) uint8_t params[kParamsSize];
    memcpy(params, p + kProcParams, kParamsSize);
    if (At<void*>(params, kParamIncl) || At<void*>(params, kParamExcl)) {
        snprintf(why, whySize, "processor view parameters hold mask pointers (%p, %p)", At<void*>(params, kParamIncl),
                 At<void*>(params, kParamExcl));
        return false;
    }
    At<const uint8_t*>(params, kParamFrustum) = block;
    static uint64_t incl, excl;  // read by the gather through the pointers
    incl = At<uint64_t>(s, kSetIncl);
    excl = At<uint64_t>(s, kSetExcl);
    r.setEye = At<uint32_t>(s, kSetCount);
    hooks::TryRead(ctx + kCtxQueueCount, &r.queueEye, 4);
    const double t0 = NowMs();
    bool ok = CallEngine(ctx, set, proc, block, incl ? &incl : nullptr, excl ? &excl : nullptr, At<int>(s, kSetArg6),
                         At<void*>(s, kSetArg7), At<void*>(s, kSetCamera), params);
    r.ms = NowMs() - t0;
    hooks::TryRead(set + kSetCount, &r.setFocus, 4);
    hooks::TryRead(ctx + kCtxQueueCount, &r.queueFocus, 4);
    if (!ok) {
        fault = true;
        snprintf(why, whySize, "engine call failed or faulted (set status %d, processor status %d)",
                 At<int>(set, kSetStatus), At<int>(proc, kProcStatus));
    }
    return ok;
}

}  // namespace qvfr::focus_cull

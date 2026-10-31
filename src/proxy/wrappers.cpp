#include "wrappers.h"

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "frame.h"
#include "generated/wrappers_0912.h"

namespace qvfr {

// ---------------------------------------------------------------- call counters

namespace {

constexpr int kMaxMethods = 64;
static_assert(vr::gen::kSystemMethodCount <= kMaxMethods && vr::gen::kCompositorMethodCount <= kMaxMethods &&
              vr::gen::kOverlayMethodCount <= kMaxMethods);

std::atomic<uint64_t> g_counts[vr::gen::kIfaceCount][kMaxMethods];
uint64_t g_lastCounts[vr::gen::kIfaceCount][kMaxMethods];
double g_lastStatsMs = 0;
std::mutex g_statsMutex;

struct IfaceInfo {
    const char* version;
    const char* const* names;
    int count;
};
const IfaceInfo kIfaces[vr::gen::kIfaceCount] = {
    {vr::gen::kSystemVersion, vr::gen::kSystemMethodNames, vr::gen::kSystemMethodCount},
    {vr::gen::kCompositorVersion, vr::gen::kCompositorMethodNames, vr::gen::kCompositorMethodCount},
    {vr::gen::kOverlayVersion, vr::gen::kOverlayMethodNames, vr::gen::kOverlayMethodCount},
};

}  // namespace

void TraceCall(int iface, int method) {
    uint64_t prev = g_counts[iface][method].fetch_add(1, std::memory_order_relaxed);
    if (prev == 0) {
        Log("First call of %s::%s (thread %lu)", kIfaces[iface].version, kIfaces[iface].names[method],
            GetCurrentThreadId());
    }
}

void LogOverlayStats();  // below, next to the IVROverlay wrapper

void LogCallStats(bool totals) {
    std::lock_guard lock(g_statsMutex);
    double now = NowMs();
    double sec = (now - g_lastStatsMs) / 1000.0;
    for (int i = 0; i < vr::gen::kIfaceCount; ++i) {
        std::string line;
        for (int m = 0; m < kIfaces[i].count; ++m) {
            uint64_t c = g_counts[i][m].load(std::memory_order_relaxed);
            uint64_t d = totals ? c : c - g_lastCounts[i][m];
            g_lastCounts[i][m] = c;
            if (!d) continue;
            char buf[128];
            if (totals)
                snprintf(buf, sizeof(buf), "%s %llu", kIfaces[i].names[m], (unsigned long long)d);
            else
                snprintf(buf, sizeof(buf), "%s %llu (%.1f/s)", kIfaces[i].names[m], (unsigned long long)d, d / sec);
            line += (line.empty() ? "" : ", ") + std::string(buf);
        }
        if (!line.empty()) Log("Calls %s%s: %s", kIfaces[i].version, totals ? " (session total)" : "", line.c_str());
    }
    g_lastStatsMs = now;
    LogOverlayStats();
}

namespace {

// ---------------------------------------------------------------- value logging

// The first first_calls calls with values, then only changes (at most 30 per key).
struct ValueLog {
    int calls = 0;
    int changes = 0;
    std::string last;
};
std::mutex g_valueMutex;

void LogValue(ValueLog& vl, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard lock(g_valueMutex);
    ++vl.calls;
    bool changed = vl.calls > 1 && vl.last != buf;
    if (vl.calls <= Cfg().firstCalls) {
        Log("%s  [call %d]", buf, vl.calls);
    } else if (changed) {
        if (++vl.changes <= 30)
            Log("%s  [CHANGED, call %d]", buf, vl.calls);
        else if (vl.changes == 31)
            Log("(further changes of this value are not logged)");
    }
    vl.last = buf;
}

const char* EyeName(vr::EVREye e) { return e == vr::Eye_Right ? "right" : "left"; }
int EyeIdx(vr::EVREye e) { return e == vr::Eye_Right ? 1 : 0; }

// ---------------------------------------------------------------- IVRSystem_009

class SystemWrap final : public vr::gen::SystemWrapBase {
public:
    using SystemWrapBase::SystemWrapBase;

    void GetRecommendedRenderTargetSize(uint32_t* w, uint32_t* h) override {
        SystemWrapBase::GetRecommendedRenderTargetSize(w, h);
        Guarded("GetRecommendedRenderTargetSize", [&] {
            if (!w || !h) return;
            frame::NoteRecommendedSize(*w, *h);
            LogValue(size_, "IVRSystem::GetRecommendedRenderTargetSize = %u x %u", *w, *h);
        });
    }

    vr::HmdMatrix44_t GetProjectionMatrix(vr::EVREye eye, float zn, float zf, vr::EGraphicsAPIConvention conv) override {
        vr::HmdMatrix44_t m = SystemWrapBase::GetProjectionMatrix(eye, zn, zf, conv);
        Guarded("GetProjectionMatrix", [&] {
            const auto& a = m.m;
            LogValue(projMat_[EyeIdx(eye)][conv == vr::API_OpenGL ? 1 : 0],
                     "IVRSystem::GetProjectionMatrix(%s, near %.4f, far %.1f, %s) = [%.5f %.5f %.5f %.5f | %.5f %.5f %.5f "
                     "%.5f | %.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f]",
                     EyeName(eye), zn, zf, conv == vr::API_OpenGL ? "OpenGL" : "DirectX", a[0][0], a[0][1], a[0][2],
                     a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0], a[2][1], a[2][2], a[2][3], a[3][0], a[3][1],
                     a[3][2], a[3][3]);
        });
        return m;
    }

    void GetProjectionRaw(vr::EVREye eye, float* l, float* r, float* t, float* b) override {
        SystemWrapBase::GetProjectionRaw(eye, l, r, t, b);
        Guarded("GetProjectionRaw", [&] {
            if (!l || !r || !t || !b) return;
            LogValue(projRaw_[EyeIdx(eye)], "IVRSystem::GetProjectionRaw(%s) = left %.5f right %.5f top %.5f bottom %.5f",
                     EyeName(eye), *l, *r, *t, *b);
        });
    }

    vr::HmdMatrix34_t GetEyeToHeadTransform(vr::EVREye eye) override {
        vr::HmdMatrix34_t m = SystemWrapBase::GetEyeToHeadTransform(eye);
        Guarded("GetEyeToHeadTransform", [&] {
            const auto& a = m.m;
            LogValue(eyeToHead_[EyeIdx(eye)],
                     "IVRSystem::GetEyeToHeadTransform(%s) = [%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f]",
                     EyeName(eye), a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0],
                     a[2][1], a[2][2], a[2][3]);
        });
        return m;
    }

    void GetDXGIOutputInfo(int32_t* adapter) override {
        SystemWrapBase::GetDXGIOutputInfo(adapter);
        Guarded("GetDXGIOutputInfo", [&] {
            if (adapter) LogValue(dxgi_, "IVRSystem::GetDXGIOutputInfo = adapter %d", *adapter);
        });
    }

    void GetDeviceToAbsoluteTrackingPose(vr::ETrackingUniverseOrigin origin, float predict, vr::TrackedDevicePose_t* poses,
                                         uint32_t count) override {
        SystemWrapBase::GetDeviceToAbsoluteTrackingPose(origin, predict, poses, count);
        Guarded("GetDeviceToAbsoluteTrackingPose", [&] {
            LogValue(devicePose_, "IVRSystem::GetDeviceToAbsoluteTrackingPose(origin %d, %u devices)", (int)origin, count);
            frame::FilterDevicePoses(origin, poses, count);
        });
    }

private:
    ValueLog size_, projMat_[2][2], projRaw_[2], eyeToHead_[2], dxgi_, devicePose_;
};

// ---------------------------------------------------------------- IVRCompositor_009

class CompositorWrap final : public vr::gen::CompositorWrapBase {
public:
    using CompositorWrapBase::CompositorWrapBase;

    vr::EVRCompositorError WaitGetPoses(vr::TrackedDevicePose_t* render, uint32_t nRender, vr::TrackedDevicePose_t* game,
                                        uint32_t nGame) override {
        frame::OnWaitGetPosesBegin();
        double t0 = NowMs();
        vr::EVRCompositorError err = CompositorWrapBase::WaitGetPoses(render, nRender, game, nGame);
        double t1 = NowMs();
        Guarded("WaitGetPoses (log)", [&] {
            LogValue(wgp_, "IVRCompositor::WaitGetPoses(render %u, game %u), thread %lu", nRender, nGame,
                     GetCurrentThreadId());
        });
        frame::OnWaitGetPosesEnd(real_, render, nRender, game, nGame, err, t1 - t0);
        return err;
    }

    vr::EVRCompositorError GetLastPoses(vr::TrackedDevicePose_t* render, uint32_t nRender, vr::TrackedDevicePose_t* game,
                                        uint32_t nGame) override {
        vr::EVRCompositorError err = CompositorWrapBase::GetLastPoses(render, nRender, game, nGame);
        frame::FilterLastPoses(render, nRender, game, nGame);
        return err;
    }

    vr::EVRCompositorError Submit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                  vr::EVRSubmitFlags flags) override {
        double t0 = NowMs();
        vr::EVRCompositorError err = CompositorWrapBase::Submit(eye, tex, bounds, flags);
        frame::OnSubmit(eye, tex, bounds, flags, err, NowMs() - t0);
        return err;
    }

    void SetTrackingSpace(vr::ETrackingUniverseOrigin origin) override {
        CompositorWrapBase::SetTrackingSpace(origin);
        frame::OnSetTrackingSpace(origin);
    }

private:
    ValueLog wgp_;
};

// ---------------------------------------------------------------- IVROverlay_007

class OverlayWrap final : public vr::gen::OverlayWrapBase {
public:
    using OverlayWrapBase::OverlayWrapBase;

    vr::EVROverlayError CreateOverlay(const char* key, const char* name, vr::VROverlayHandle_t* handle) override {
        vr::EVROverlayError err = OverlayWrapBase::CreateOverlay(key, name, handle);
        Guarded("CreateOverlay", [&] {
            Log("IVROverlay::CreateOverlay(\"%s\", \"%s\") = handle 0x%llX, error %d", key ? key : "", name ? name : "",
                handle ? (unsigned long long)*handle : 0ull, (int)err);
            if (handle && key) {
                std::lock_guard lock(m_);
                st_[*handle].key = key;
            }
        });
        return err;
    }

    vr::EVROverlayError DestroyOverlay(vr::VROverlayHandle_t h) override {
        Guarded("DestroyOverlay", [&] { Log("IVROverlay::DestroyOverlay(%s)", Name(h).c_str()); });
        return OverlayWrapBase::DestroyOverlay(h);
    }

    // Visibility, width and alpha: logged only on change (at most kLimit times per overlay and property).
    vr::EVROverlayError ShowOverlay(vr::VROverlayHandle_t h) override {
        vr::EVROverlayError err = OverlayWrapBase::ShowOverlay(h);
        Guarded("ShowOverlay", [&] { Visible(h, 1, err); });
        return err;
    }

    vr::EVROverlayError HideOverlay(vr::VROverlayHandle_t h) override {
        vr::EVROverlayError err = OverlayWrapBase::HideOverlay(h);
        Guarded("HideOverlay", [&] { Visible(h, 0, err); });
        return err;
    }

    vr::EVROverlayError SetOverlayWidthInMeters(vr::VROverlayHandle_t h, float w) override {
        vr::EVROverlayError err = OverlayWrapBase::SetOverlayWidthInMeters(h, w);
        Guarded("SetOverlayWidthInMeters", [&] {
            std::lock_guard lock(m_);
            State& s = st_[h];
            if (s.width != w && s.widthLogs++ < kLimit)
                Log("Overlay %s: width %.3f m", NameLocked(h).c_str(), w);
            s.width = w;
        });
        return err;
    }

    vr::EVROverlayError SetOverlayAlpha(vr::VROverlayHandle_t h, float a) override {
        vr::EVROverlayError err = OverlayWrapBase::SetOverlayAlpha(h, a);
        Guarded("SetOverlayAlpha", [&] {
            std::lock_guard lock(m_);
            State& s = st_[h];
            // During a fade the alpha changes every frame; only 0, 1 and the first intermediate values are logged.
            bool edge = a == 0.0f || a == 1.0f;
            if (s.alpha != a && (edge || s.alphaLogs < 5) && s.alphaLogs++ < kLimit)
                Log("Overlay %s: alpha %.3f", NameLocked(h).c_str(), a);
            s.alpha = a;
        });
        return err;
    }

    vr::EVROverlayError SetOverlayTransformAbsolute(vr::VROverlayHandle_t h, vr::ETrackingUniverseOrigin origin,
                                                    const vr::HmdMatrix34_t* m) override {
        vr::EVROverlayError err = OverlayWrapBase::SetOverlayTransformAbsolute(h, origin, m);
        Guarded("SetOverlayTransformAbsolute", [&] {
            std::lock_guard lock(m_);
            State& s = st_[h];
            if (!m) return;
            const auto& a = m->m;
            bool moved = s.lastPos[0] != a[0][3] || s.lastPos[1] != a[1][3] || s.lastPos[2] != a[2][3];
            if (moved) ++s.moves;
            ++s.transforms;
            if (++s.transformLogs <= 3)
                Log("Overlay %s: transform (origin %d) [%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]",
                    NameLocked(h).c_str(), (int)origin, a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2],
                    a[1][3], a[2][0], a[2][1], a[2][2], a[2][3]);
            s.lastPos[0] = a[0][3];
            s.lastPos[1] = a[1][3];
            s.lastPos[2] = a[2][3];
        });
        return err;
    }

    vr::EVROverlayError SetOverlayTexture(vr::VROverlayHandle_t h, const vr::Texture_t* tex) override {
        vr::EVROverlayError err = OverlayWrapBase::SetOverlayTexture(h, tex);
        Guarded("SetOverlayTexture", [&] {
            if (!tex) return;
            std::lock_guard lock(m_);
            State& s = st_[h];
            ++s.textures;
            if (std::find(s.tex.begin(), s.tex.end(), tex->handle) == s.tex.end() && s.tex.size() < 8) {
                s.tex.push_back(tex->handle);
                Log("Overlay %s: new texture %p (#%zu), type %d, color space %d, result %d; %s",
                    NameLocked(h).c_str(), tex->handle, s.tex.size(), (int)tex->eType, (int)tex->eColorSpace, (int)err,
                    frame::TextureInfo(tex->handle, tex->eType).c_str());
            }
        });
        return err;
    }

    // Every stats_interval_s and on exit: how many times each overlay got a texture and a transform.
    void LogStats() {
        std::lock_guard lock(m_);
        for (auto& [h, s] : st_) {
            if (!s.textures && !s.transforms) continue;
            Log("Overlay %s: %llu textures, %llu transforms (%llu with a position change), visible %d, width %.3f m, alpha %.3f",
                NameLocked(h).c_str(), (unsigned long long)s.textures, (unsigned long long)s.transforms,
                (unsigned long long)s.moves, s.visible, s.width, s.alpha);
            s.textures = s.transforms = s.moves = 0;
        }
    }

private:
    static constexpr int kLimit = 20;
    struct State {
        std::string key;
        std::vector<void*> tex;
        int visible = -1, visibleLogs = 0, widthLogs = 0, alphaLogs = 0;
        float width = -1, alpha = -1;
        float lastPos[3] = {};
        uint64_t textures = 0, transforms = 0, moves = 0;  // since the last stats
        int transformLogs = 0;
    };

    std::string NameLocked(vr::VROverlayHandle_t h) {
        char buf[96];
        auto it = st_.find(h);
        snprintf(buf, sizeof(buf), "%s (0x%llX)", it != st_.end() && !it->second.key.empty() ? it->second.key.c_str() : "?",
                 (unsigned long long)h);
        return buf;
    }
    std::string Name(vr::VROverlayHandle_t h) {
        std::lock_guard lock(m_);
        return NameLocked(h);
    }
    void Visible(vr::VROverlayHandle_t h, int v, vr::EVROverlayError err) {
        std::lock_guard lock(m_);
        State& s = st_[h];
        if (s.visible != v && s.visibleLogs++ < kLimit)
            Log("Overlay %s: %s (result %d)", NameLocked(h).c_str(), v ? "shown" : "hidden", (int)err);
        s.visible = v;
    }

    std::mutex m_;
    std::unordered_map<vr::VROverlayHandle_t, State> st_;
};

OverlayWrap* g_overlay = nullptr;  // for periodic stats

}  // namespace

void LogOverlayStats() {
    if (g_overlay) g_overlay->LogStats();
}

namespace {

// ---------------------------------------------------------------- factories

std::mutex g_wrapMutex;

// One map per wrapper type: after a VR restart a new object of another interface may reuse an old address.
// Wrappers live until the process exits. Returns nullptr when no wrapper could be created.
template <class W, class I>
W* GetWrapper(I* real) {
    static std::unordered_map<I*, W*> wrappers;
    try {
        std::lock_guard lock(g_wrapMutex);
        W*& w = wrappers[real];
        if (!w) {
            w = new W(real);
            Log("Wrapper %p for %p", (void*)w, (void*)real);
        }
        return w;
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

namespace wrap {

// Without a wrapper the game gets the real object: diagnostics stop, the game keeps working.
vr::IVRSystem* System(vr::IVRSystem* real) {
    if (!real) return nullptr;
    SystemWrap* w = GetWrapper<SystemWrap>(real);
    return w ? static_cast<vr::IVRSystem*>(w) : real;
}

vr::IVRCompositor* Compositor(vr::IVRCompositor* real) {
    if (!real) return nullptr;
    CompositorWrap* w = GetWrapper<CompositorWrap>(real);
    return w ? static_cast<vr::IVRCompositor*>(w) : real;
}

vr::IVROverlay* Overlay(vr::IVROverlay* real) {
    if (!real) return nullptr;
    OverlayWrap* w = GetWrapper<OverlayWrap>(real);
    if (!w) return real;
    g_overlay = w;
    return w;
}

void* Generic(const char* version, void* real) {
    if (!version || !real) return real;
    if (!strcmp(version, vr::IVRSystem_Version)) return System(static_cast<vr::IVRSystem*>(real));
    if (!strcmp(version, vr::IVRCompositor_Version)) return Compositor(static_cast<vr::IVRCompositor*>(real));
    if (!strcmp(version, vr::IVROverlay_Version)) return Overlay(static_cast<vr::IVROverlay*>(real));
    return real;
}

}  // namespace wrap

}  // namespace qvfr

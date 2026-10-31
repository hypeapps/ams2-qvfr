// OpenVR 2.15.6 header in the vrm namespace (macro), so it does not clash with 0.9.12 in other files.
// Its free functions are not used: the interfaces come from the original's VR_GetGenericInterface.
// Explicit path, because the proxy target has the 0.9.12 headers (also openvr.h) on its include path.
#undef VR_API_EXPORT
#define vr vrm
#include "../../third_party/openvr-2.15.6/headers/openvr.h"
#undef vr

#include "modern_vr.h"

#include <cmath>

#include "common.h"

namespace qvfr::modern {

namespace {

vrm::IVRSystem* g_sys = nullptr;          // IVRSystem_026
vrm::IVRCompositor* g_comp = nullptr;     // IVRCompositor_029
vrm::HmdMatrix44_t g_projL{};
float g_hz = 0;

void* Get(GetGenericFn fn, const char* version) {
    int err = 0;
    void* p = fn(version, &err);
    Log("VR_GetGenericInterface(\"%s\") = %p, error %d", version, p, err);
    return err == 0 ? p : nullptr;
}

constexpr double kRadToDeg = 57.29577951308232;

}  // namespace

void Init(GetGenericFn fn) {
    Guarded("modern::Init", [&] {
        g_sys = static_cast<vrm::IVRSystem*>(Get(fn, vrm::IVRSystem_Version));
        g_comp = static_cast<vrm::IVRCompositor*>(Get(fn, vrm::IVRCompositor_Version));
        if (g_sys) {
            Log("SteamVR: runtime version %s", g_sys->GetRuntimeVersion());
            char buf[256] = {};
            vrm::ETrackedPropertyError pe{};
            g_sys->GetStringTrackedDeviceProperty(vrm::k_unTrackedDeviceIndex_Hmd, vrm::Prop_ModelNumber_String, buf,
                                                  sizeof(buf), &pe);
            Log("HMD: Prop_ModelNumber_String = \"%s\"", buf);
            buf[0] = 0;
            g_sys->GetStringTrackedDeviceProperty(vrm::k_unTrackedDeviceIndex_Hmd, vrm::Prop_ControllerType_String, buf,
                                                  sizeof(buf), &pe);
            Log("HMD: Prop_ControllerType_String = \"%s\"", buf);
            bool eye = g_sys->GetBoolTrackedDeviceProperty(vrm::k_unTrackedDeviceIndex_Hmd,
                                                           vrm::Prop_SupportsXrEyeGazeInteraction_Bool, &pe);
            Log("HMD: Prop_SupportsXrEyeGazeInteraction_Bool = %d (error %d)", eye, (int)pe);
            float fps = g_sys->GetFloatTrackedDeviceProperty(vrm::k_unTrackedDeviceIndex_Hmd,
                                                            vrm::Prop_DisplayFrequency_Float, &pe);
            Log("HMD: Prop_DisplayFrequency_Float = %.2f Hz", fps);
            g_hz = fps;
            g_projL = g_sys->GetProjectionMatrix(vrm::Eye_Left, 0.1f, 100.0f);
            for (int e = 0; e < 2; ++e) {
                float l, r, t, b;
                g_sys->GetProjectionRaw((vrm::EVREye)e, &l, &r, &t, &b);
                Log("IVRSystem_026::GetProjectionRaw(%s) = left %.5f right %.5f top %.5f bottom %.5f",
                    e ? "right" : "left", l, r, t, b);
            }
            uint32_t w = 0, h = 0;
            g_sys->GetRecommendedRenderTargetSize(&w, &h);
            Log("IVRSystem_026::GetRecommendedRenderTargetSize = %u x %u", w, h);
        }
        if (g_comp) {
            Log("IVRCompositor_029: motion smoothing enabled %d, supported %d", g_comp->IsMotionSmoothingEnabled(),
                g_comp->IsMotionSmoothingSupported());
        }
    });
}

void Shutdown() {
    g_sys = nullptr;
    g_comp = nullptr;
}

bool HasSystem() { return g_sys != nullptr; }
bool HasCompositor() { return g_comp != nullptr; }
float DisplayHz() { return g_hz; }

bool ReadGaze(Gaze& out) {
    out = Gaze{};
    if (!g_sys) return false;
    vrm::HmdVector2_t l{}, r{};
    out.ok = g_sys->GetEyeTrackedFoveationCenter(&l, &r);
    out.ndcL[0] = l.v[0];
    out.ndcL[1] = l.v[1];
    out.ndcR[0] = r.v[0];
    out.ndcR[1] = r.v[1];
    if (out.ok && g_projL.m[0][0] != 0 && g_projL.m[1][1] != 0) {
        // NDC = m00 * tan + m02 * (-1) for direction (tan, ., -1), so tan = (ndc + m02) / m00.
        double tx = (l.v[0] + g_projL.m[0][2]) / g_projL.m[0][0];
        double ty = (l.v[1] + g_projL.m[1][2]) / g_projL.m[1][1];
        out.degX = (float)(std::atan(tx) * kRadToDeg);
        out.degY = (float)(std::atan(ty) * kRadToDeg);
    }
    return true;
}

uint32_t GetFrameTimings(FrameTiming* out, uint32_t n) {
    if (!g_comp || n == 0) return 0;
    vrm::Compositor_FrameTiming buf[32];
    if (n > 32) n = 32;
    buf[0].m_nSize = sizeof(vrm::Compositor_FrameTiming);
    uint32_t got = g_comp->GetFrameTimings(buf, n);
    for (uint32_t i = 0; i < got; ++i) {
        const auto& t = buf[i];
        FrameTiming& o = out[i];
        o.frameIndex = t.m_nFrameIndex;
        o.numFramePresents = t.m_nNumFramePresents;
        o.numMisPresented = t.m_nNumMisPresented;
        o.numDroppedFrames = t.m_nNumDroppedFrames;
        o.reprojectionFlags = t.m_nReprojectionFlags;
        o.systemTimeS = t.m_flSystemTimeInSeconds;
        o.preSubmitGpuMs = t.m_flPreSubmitGpuMs;
        o.postSubmitGpuMs = t.m_flPostSubmitGpuMs;
        o.totalRenderGpuMs = t.m_flTotalRenderGpuMs;
        o.compositorRenderGpuMs = t.m_flCompositorRenderGpuMs;
        o.compositorRenderCpuMs = t.m_flCompositorRenderCpuMs;
        o.compositorIdleCpuMs = t.m_flCompositorIdleCpuMs;
        o.clientFrameIntervalMs = t.m_flClientFrameIntervalMs;
        o.presentCallCpuMs = t.m_flPresentCallCpuMs;
        o.waitForPresentCpuMs = t.m_flWaitForPresentCpuMs;
        o.submitFrameMs = t.m_flSubmitFrameMs;
        o.waitGetPosesCalledMs = t.m_flWaitGetPosesCalledMs;
        o.newPosesReadyMs = t.m_flNewPosesReadyMs;
        o.newFrameReadyMs = t.m_flNewFrameReadyMs;
        o.compositorUpdateStartMs = t.m_flCompositorUpdateStartMs;
        o.compositorUpdateEndMs = t.m_flCompositorUpdateEndMs;
        o.compositorRenderStartMs = t.m_flCompositorRenderStartMs;
        o.numVSyncsReadyForUse = t.m_nNumVSyncsReadyForUse;
        o.numVSyncsToFirstView = t.m_nNumVSyncsToFirstView;
        o.transferLatencyMs = t.m_flTransferLatencyMs;
    }
    return got;
}

bool GetCumulative(Cumulative& out) {
    out = Cumulative{};
    if (!g_comp) return false;
    vrm::Compositor_CumulativeStats s{};
    g_comp->GetCumulativeStats(&s, sizeof(s));
    out.framePresents = s.m_nNumFramePresents;
    out.droppedFrames = s.m_nNumDroppedFrames;
    out.reprojectedFrames = s.m_nNumReprojectedFrames;
    out.frameSubmits = s.m_nNumFrameSubmits;
    out.timedOut = s.m_nNumTimedOut;
    out.loading = s.m_nNumLoading;
    out.sumAppGpuMs = s.m_flSumApplicationGPUTimeMS;
    out.sumAppCpuMs = s.m_flSumApplicationCPUTimeMS;
    out.sumCompositorGpuMs = s.m_flSumCompositorGPUTimeMS;
    out.sumCompositorCpuMs = s.m_flSumCompositorCPUTimeMS;
    return true;
}

}  // namespace qvfr::modern

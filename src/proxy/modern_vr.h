// Newer SteamVR interfaces (IVRSystem_026, IVRCompositor_029) obtained through VR_GetGenericInterface
// of the original openvr_api.dll 0.9.12. The implementation includes the 2.15.6 header in a separate namespace,
// so this header does not depend on any openvr.h.
#pragma once

#include <cstdint>

namespace qvfr::modern {

using GetGenericFn = void* (*)(const char* version, int* error);

void Init(GetGenericFn getGeneric);  // after a successful VR_Init; logs what is available
void Shutdown();
bool HasSystem();
bool HasCompositor();
float DisplayHz();               // HMD Prop_DisplayFrequency_Float (0 when unknown)

struct Gaze {
    bool ok = false;           // GetEyeTrackedFoveationCenter result
    float ndcL[2] = {};        // left eye NDC (x right, y up, relative to GetProjectionMatrix)
    float ndcR[2] = {};
    float degX = 0, degY = 0;  // gaze direction from the left eye in degrees (x right, y up)
};
bool ReadGaze(Gaze& out);      // false when the interface is not available

// One row of IVRCompositor_029::GetFrameTimings (fields as in Compositor_FrameTiming 2.15.6).
struct FrameTiming {
    uint32_t frameIndex;
    uint32_t numFramePresents, numMisPresented, numDroppedFrames, reprojectionFlags;
    double systemTimeS;
    float preSubmitGpuMs, postSubmitGpuMs, totalRenderGpuMs;
    float compositorRenderGpuMs, compositorRenderCpuMs, compositorIdleCpuMs;
    float clientFrameIntervalMs, presentCallCpuMs, waitForPresentCpuMs, submitFrameMs;
    float waitGetPosesCalledMs, newPosesReadyMs, newFrameReadyMs;
    float compositorUpdateStartMs, compositorUpdateEndMs, compositorRenderStartMs;
    uint32_t numVSyncsReadyForUse, numVSyncsToFirstView;
    float transferLatencyMs;
};
// The last n frames, oldest first; returns the number of filled rows.
uint32_t GetFrameTimings(FrameTiming* out, uint32_t n);

struct Cumulative {
    uint32_t framePresents, droppedFrames, reprojectedFrames, frameSubmits, timedOut, loading;
    double sumAppGpuMs, sumAppCpuMs, sumCompositorGpuMs, sumCompositorCpuMs;
};
bool GetCumulative(Cumulative& out);

}  // namespace qvfr::modern

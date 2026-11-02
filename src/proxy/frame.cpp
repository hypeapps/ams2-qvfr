#include <windows.h>

#include "frame.h"

#include <d3d11.h>
#include <dxgi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "experiment.h"
#include "gpu.h"
#include "modern_vr.h"
#include "quad.h"
#include "sysinfo.h"
#include "xr_layer.h"

namespace qvfr::frame {

namespace {

std::mutex g_mutex;  // WaitGetPoses and Submit may arrive from different threads

// ---------------------------------------------------------------- gaze from the OpenXR layer (path=openxr)

vr::IVRSystem* g_sys = nullptr;  // the backend's IVRSystem_009

// Per eye: what projects a head-relative direction to NDC of GetProjectionMatrix, the same NDC that
// GetEyeTrackedFoveationCenter gives on the OpenVR path.
struct EyeMap {
    bool ok = false;
    float m00 = 0, m02 = 0, m11 = 0, m12 = 0;
    float rot[3][3] = {};  // eye-to-head rotation (GetEyeToHeadTransform)
};
EyeMap g_eyeMap[2];
bool g_eyeMapTried = false;

void BuildEyeMaps() {
    g_eyeMapTried = true;
    if (!g_sys) return;
    for (int i = 0; i < 2; ++i) {
        const vr::EVREye ev = i ? vr::Eye_Right : vr::Eye_Left;
        const vr::HmdMatrix44_t m = g_sys->GetProjectionMatrix(ev, 0.1f, 100.0f, vr::API_DirectX);
        const vr::HmdMatrix34_t e = g_sys->GetEyeToHeadTransform(ev);
        EyeMap& em = g_eyeMap[i];
        em.m00 = m.m[0][0];
        em.m02 = m.m[0][2];
        em.m11 = m.m[1][1];
        em.m12 = m.m[1][2];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) em.rot[r][c] = e.m[r][c];
        em.ok = em.m00 != 0 && em.m11 != 0;
        Log("Gaze (OpenXR layer): %s eye projection m00 %.5f m02 %.5f m11 %.5f m12 %.5f, eye-to-head rotation rows "
            "(%.4f %.4f %.4f) (%.4f %.4f %.4f) (%.4f %.4f %.4f), offset (%.4f, %.4f, %.4f) m",
            i ? "right" : "left", em.m00, em.m02, em.m11, em.m12, e.m[0][0], e.m[0][1], e.m[0][2], e.m[1][0], e.m[1][1],
            e.m[1][2], e.m[2][0], e.m[2][1], e.m[2][2], e.m[0][3], e.m[1][3], e.m[2][3]);
    }
}

// The layer's gaze as the OpenVR path delivers it: NDC per eye (x right, y up) and the direction in degrees.
// The eye position offset is ignored (the gaze target is far away compared with half the IPD).
void XrGaze(modern::Gaze& g) {
    g = modern::Gaze{};
    xrlayer::Gaze x;
    if (!xrlayer::ReadGaze(x) || !x.ok) return;
    if (!g_eyeMapTried) BuildEyeMaps();
    float ndc[2][2] = {};
    for (int i = 0; i < 2; ++i) {
        const EyeMap& em = g_eyeMap[i];
        if (!em.ok) return;
        double d[3];
        for (int c = 0; c < 3; ++c) d[c] = em.rot[0][c] * x.dir[0] + em.rot[1][c] * x.dir[1] + em.rot[2][c] * x.dir[2];
        if (d[2] > -1e-3) return;  // not in front of the eye
        const double tx = d[0] / -d[2], ty = d[1] / -d[2];
        ndc[i][0] = (float)(em.m00 * tx - em.m02);
        ndc[i][1] = (float)(em.m11 * ty - em.m12);
    }
    g.ok = true;
    g.ndcL[0] = ndc[0][0];
    g.ndcL[1] = ndc[0][1];
    g.ndcR[0] = ndc[1][0];
    g.ndcR[1] = ndc[1][1];
    constexpr double kRadToDeg = 57.29577951308232;
    g.degX = (float)(std::atan2(x.dir[0], -x.dir[2]) * kRadToDeg);
    g.degY = (float)(std::atan2(x.dir[1], -x.dir[2]) * kRadToDeg);
}

// ---------------------------------------------------------------- proxy frame data

struct ProxyFrame {
    uint64_t no = 0;          // proxy frame number (also the GPU timing frame)
    uint32_t compIndex = 0;   // m_nFrameIndex of the newest compositor frame right after WaitGetPoses
    double wgpEnter = 0, wgpExit = 0, lastSubmit = 0;
    float wgpBlockMs = 0;
    float intervalMs = 0;     // since the previous entry into WaitGetPoses
    float submitMs = 0;       // total time spent in Submit
    int submits = 0;
    modern::Gaze gaze;
    quad::FrameWindow window;  // where the focus window of the frame came from (quad mode)
    bool frozen = false;
    bool valid = false;
};

constexpr uint32_t kRing = 128;
ProxyFrame g_ring[kRing];   // by compositor frame index (joined with GetFrameTimings rows)
ProxyFrame g_byNo[kRing];   // by proxy frame number (frames.csv)
ProxyFrame g_cur;
uint64_t g_proxyFrames = 0;
int g_badSubmitLogs = 0;
double g_prevWgpEnter = 0;

// ---------------------------------------------------------------- Submit

struct SubmitInfo {
    void* handle = nullptr;
    int type = -1, colorSpace = -1, flags = -1;
    float bounds[4] = {-1, -1, -1, -1};
};
SubmitInfo g_lastSubmit[2];
std::vector<void*> g_handles[2];
int g_submitChangeLogs = 0;
int g_submitErrors = 0;
bool g_sharedLogged = false;
bool g_deviceLogged = false;
void* g_frameHandle[2] = {};
std::string g_texDesc;        // the last texture description, for the benchmark summary
uint32_t g_recW = 0, g_recH = 0;    // size from SteamVR
uint32_t g_gameW = 0, g_gameH = 0;  // size returned to the game (differs in the experiment)
double g_firstSubmitMs = -1;

// ---------------------------------------------------------------- gaze and periodic stats

double g_lastStatsMs = 0;
uint64_t g_statsFrames = 0, g_gazeOk = 0, g_gazeSame = 0, g_gazeCalls = 0;
uint64_t g_submitsInInterval = 0;
modern::Gaze g_prevGaze;
bool g_gazeFirstLogged = false;
// Gaze stalls: a valid gaze that keeps exactly the same value for kStallMs or longer. The tracker lost the eye
// (blink, headset lifted), but SteamVR keeps reporting the last value as valid (M5 data, docs/gaze-test.md).
constexpr double kStallMs = 200;
constexpr double kPauseMs = 250;  // a longer frame interval is a pause of the game, not a stall
double g_valueSinceMs = -1;  // when the current valid value first came (-1 = no valid gaze)
double g_valueLastMs = -1;   // the last frame with it
uint64_t g_valueFrames = 0;  // frames with it
struct Stalls {
    uint64_t count = 0, frames = 0;
    double longestMs = 0;
} g_stalls, g_stallsTotal;

// ---------------------------------------------------------------- CSV and benchmark

enum class Bench { Idle, Countdown, Running };
Bench g_bench = Bench::Idle;
double g_benchT0 = 0;          // countdown or recording start
bool g_keyWasDown = false;
bool g_autoStarted = false;
uint32_t g_runFirstIndex = 0;
fs::path g_runDir;
FILE* g_runCsv = nullptr;
FILE* g_sessionCsv = nullptr;
bool g_sessionTried = false;
uint32_t g_lastWrittenIndex = 0;
double g_runSysT0 = -1;
double g_sessionSysT0 = -1;
modern::Cumulative g_cumStart{};

// frames.csv: one row per proxy frame with our own GPU timestamps; works without SteamVR frame timing.
struct GpuRow {
    uint64_t no = 0;
    bool ok = false;
    gpu::Result r;
};
GpuRow g_gpuRows[kRing];
uint64_t g_framesNext = 1;    // next proxy frame number to write
uint64_t g_runFirstNo = 0;    // first proxy frame of the benchmark
FILE* g_runFrames = nullptr;
FILE* g_sessionFrames = nullptr;
double g_runFramesT0 = -1, g_sessionFramesT0 = -1;
// Frames a GPU result may take before the row is written without it (gpu.cpp keeps 6 frames in flight).
constexpr uint64_t kGpuWait = 8;
// GPU averages for the periodic stats line.
struct GpuSums {
    uint64_t frames = 0, n[gpu::kPhases] = {};
    double frame = 0, phase[gpu::kPhases] = {};
} g_gpuSums;

bool g_frozen = false;
vr::TrackedDevicePose_t g_frozenPose{};
vr::ETrackingUniverseOrigin g_trackingSpace = vr::TrackingUniverseSeated;
bool g_trackingSpaceKnown = false;

struct Series {
    std::vector<float> v;
    void Add(float x) { v.push_back(x); }
};
struct RunStats {
    Series totalGpu, preGpu, postGpu, compGpu, compCpu, proxyInterval, proxyCpu, wgpBlock;
    Series gpuFrame, gpuPhase[gpu::kPhases], gpuCompose, gpuOther;  // our timestamps (frames.csv)
    uint64_t proxyRows = 0;
    double proxyFirstMs = -1, proxyLastMs = -1;
    uint64_t frames = 0, dropped = 0, mispresented = 0, reprojCpu = 0, reprojGpu = 0, reprojAny = 0, gazeOk = 0;
    double sysFirst = -1, sysLast = -1;
} g_run;

const char kCsvHeader[] =
    "frame,t_s,pre_submit_gpu_ms,post_submit_gpu_ms,total_gpu_ms,compositor_gpu_ms,compositor_cpu_ms,"
    "compositor_idle_cpu_ms,client_interval_ms,wait_get_poses_called_ms,new_poses_ready_ms,new_frame_ready_ms,"
    "submit_frame_ms,present_call_cpu_ms,wait_for_present_cpu_ms,presents,mispresented,dropped,reprojection_flags,"
    "vsyncs_ready,vsyncs_to_first_view,transfer_latency_ms,proxy_interval_ms,proxy_cpu_ms,proxy_wgp_block_ms,proxy_submit_ms,"
    "proxy_submits,gaze_ok,gaze_x_deg,gaze_y_deg,frozen\n";

const char kFramesHeader[] =
    "frame,t_s,interval_ms,cpu_ms,wgp_block_ms,submit_ms,submits,gpu_frame_ms,gpu_outer_l_ms,gpu_outer_r_ms,"
    "gpu_focus_l_ms,gpu_focus_r_ms,gpu_compose_ms,gpu_other_ms,gaze_ok,gaze_x_deg,gaze_y_deg,frozen,focus_src,"
    "focus_x0_l,focus_y0_l\n";

void Beeps(std::vector<std::pair<int, int>> tones) {
    if (!Cfg().benchBeep) return;
    std::thread([tones] {
        for (auto [f, d] : tones) {
            Beep(f, d);
            Sleep(60);
        }
    }).detach();
}

float BudgetMs() {
    float hz = modern::DisplayHz();
    return hz > 1 ? 1000.0f / hz : 1000.0f / 72.0f;
}

void WriteRow(FILE* f, const modern::FrameTiming& t, double t0, const ProxyFrame* p) {
    fprintf(f,
            "%u,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,0x%X,%u,%u,%.3f,",
            t.frameIndex, t0 >= 0 ? t.systemTimeS - t0 : 0.0, t.preSubmitGpuMs, t.postSubmitGpuMs, t.totalRenderGpuMs,
            t.compositorRenderGpuMs, t.compositorRenderCpuMs, t.compositorIdleCpuMs, t.clientFrameIntervalMs,
            t.waitGetPosesCalledMs, t.newPosesReadyMs, t.newFrameReadyMs, t.submitFrameMs, t.presentCallCpuMs,
            t.waitForPresentCpuMs, t.numFramePresents, t.numMisPresented, t.numDroppedFrames, t.reprojectionFlags,
            t.numVSyncsReadyForUse, t.numVSyncsToFirstView, t.transferLatencyMs);
    if (p) {
        float cpu = p->submits ? (float)(p->lastSubmit - p->wgpExit) : 0.0f;
        fprintf(f, "%.3f,%.3f,%.3f,%.3f,%d,%d,%.2f,%.2f,%d\n", p->intervalMs, cpu, p->wgpBlockMs, p->submitMs,
                p->submits, p->gaze.ok, p->gaze.degX, p->gaze.degY, g_frozen ? 1 : 0);
    } else {
        fprintf(f, ",,,,,,,,%d\n", g_frozen ? 1 : 0);
    }
}

const ProxyFrame* FindProxy(uint32_t compIndex) {
    const ProxyFrame& p = g_ring[compIndex % kRing];
    return p.valid && p.compIndex == compIndex ? &p : nullptr;
}

void Field(FILE* f, float ms) {
    if (ms >= 0)
        fprintf(f, "%.3f,", ms);
    else
        fputc(',', f);
}

// Compose = both eyes; other = the frame span minus the measured groups and the composition (shadows, reflections,
// postmain, the SteamVR copy of the left eye...).
float ComposeMs(const gpu::Result& r) {
    float a = r.phaseMs[gpu::ComposeL], b = r.phaseMs[gpu::ComposeR];
    return a < 0 && b < 0 ? -1 : std::max(a, 0.0f) + std::max(b, 0.0f);
}

float OtherMs(const gpu::Result& r) {
    if (r.frameMs < 0) return -1;
    float o = r.frameMs;
    for (int p = 0; p < gpu::kPhases; ++p) o -= std::max(r.phaseMs[p], 0.0f);
    return o;
}

void WriteFramesRow(FILE* f, const ProxyFrame& p, const gpu::Result* g, double& t0) {
    if (t0 < 0) t0 = p.wgpExit;
    float cpu = p.submits ? (float)(p.lastSubmit - p.wgpExit) : 0.0f;
    fprintf(f, "%llu,%.4f,%.3f,%.3f,%.3f,%.3f,%d,", (unsigned long long)p.no, (p.wgpExit - t0) / 1000.0, p.intervalMs,
            cpu, p.wgpBlockMs, p.submitMs, p.submits);
    if (g) {
        Field(f, g->frameMs);
        for (int k = gpu::OuterL; k <= gpu::FocusR; ++k) Field(f, g->phaseMs[k]);
        Field(f, ComposeMs(*g));
        Field(f, OtherMs(*g));
    } else {
        fputs(",,,,,,,", f);
    }
    fprintf(f, "%d,%.2f,%.2f,%d,", p.gaze.ok, p.gaze.degX, p.gaze.degY, p.frozen ? 1 : 0);
    if (p.window.src != quad::SrcNone)
        fprintf(f, "%s,%u,%u\n", quad::SourceName(p.window.src), p.window.x0, p.window.y0);
    else
        fputs(",,\n", f);
}

void AccumulateFrame(const ProxyFrame& p, const gpu::Result* g) {
    RunStats& r = g_run;
    ++r.proxyRows;
    if (r.proxyFirstMs < 0) r.proxyFirstMs = p.wgpExit;
    r.proxyLastMs = p.wgpExit;
    if (!g) return;
    if (g->frameMs >= 0) r.gpuFrame.Add(g->frameMs);
    for (int k = 0; k < gpu::kPhases; ++k)
        if (g->phaseMs[k] >= 0) r.gpuPhase[k].Add(g->phaseMs[k]);
    if (float c = ComposeMs(*g); c >= 0) r.gpuCompose.Add(c);
    if (float o = OtherMs(*g); o >= 0) r.gpuOther.Add(o);
}

// Rows of complete proxy frames (saved at the next WaitGetPoses) once their GPU result is in, or kGpuWait frames
// later without it. `current` is the frame that just started.
void ProcessFrames(uint64_t current) {
    gpu::Result res[16];
    int n;
    while ((n = gpu::Take(res, 16)) > 0) {
        for (int i = 0; i < n; ++i) {
            GpuRow& row = g_gpuRows[res[i].frame % kRing];
            row.no = res[i].frame;
            row.ok = true;
            row.r = res[i];
            const gpu::Result& g = res[i];
            if (g.frameMs >= 0) {
                ++g_gpuSums.frames;
                g_gpuSums.frame += g.frameMs;
            }
            for (int k = 0; k < gpu::kPhases; ++k) {
                if (g.phaseMs[k] < 0) continue;
                ++g_gpuSums.n[k];
                g_gpuSums.phase[k] += g.phaseMs[k];
            }
        }
    }
    if (current > g_framesNext + kRing / 2) g_framesNext = current - kRing / 2;  // after a long stall
    const bool waitGpu = gpu::Enabled();
    while (g_framesNext < current) {
        const uint64_t no = g_framesNext;
        const ProxyFrame& p = g_byNo[no % kRing];
        const GpuRow& row = g_gpuRows[no % kRing];
        const gpu::Result* g = row.ok && row.no == no ? &row.r : nullptr;
        if (!g && waitGpu && current - no <= kGpuWait) break;
        ++g_framesNext;
        if (!p.valid || p.no != no) continue;
        if (g_sessionFrames) WriteFramesRow(g_sessionFrames, p, g, g_sessionFramesT0);
        if (g_bench == Bench::Running && no >= g_runFirstNo) {
            if (g_runFrames) WriteFramesRow(g_runFrames, p, g, g_runFramesT0);
            AccumulateFrame(p, g);
        }
    }
}

void AccumulateRun(const modern::FrameTiming& t, const ProxyFrame* p) {
    RunStats& r = g_run;
    ++r.frames;
    r.totalGpu.Add(t.totalRenderGpuMs);
    r.preGpu.Add(t.preSubmitGpuMs);
    r.postGpu.Add(t.postSubmitGpuMs);
    r.compGpu.Add(t.compositorRenderGpuMs);
    r.compCpu.Add(t.compositorRenderCpuMs);
    r.dropped += t.numDroppedFrames;
    r.mispresented += t.numMisPresented;
    if (t.reprojectionFlags & 0x01) ++r.reprojCpu;
    if (t.reprojectionFlags & 0x02) ++r.reprojGpu;
    if (t.reprojectionFlags & 0x03) ++r.reprojAny;
    if (r.sysFirst < 0) r.sysFirst = t.systemTimeS;
    r.sysLast = t.systemTimeS;
    if (p) {
        if (p->intervalMs > 0) r.proxyInterval.Add(p->intervalMs);
        if (p->submits) r.proxyCpu.Add((float)(p->lastSubmit - p->wgpExit));
        r.wgpBlock.Add(p->wgpBlockMs);
        r.gazeOk += p->gaze.ok;
    }
}

std::string StatLine(const char* name, Series& s, float budget = 0) {
    char buf[256];
    if (s.v.empty()) {
        snprintf(buf, sizeof(buf), "%-26s no data", name);
        return buf;
    }
    std::vector<float> v = s.v;
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) { return v[std::min(v.size() - 1, (size_t)std::floor(q * (v.size() - 1) + 0.5))]; };
    double sum = 0;
    size_t over = 0;
    for (float x : v) {
        sum += x;
        if (budget > 0 && x > budget) ++over;
    }
    int n = snprintf(buf, sizeof(buf), "%-26s avg %6.2f  p50 %6.2f  p90 %6.2f  p99 %6.2f  max %6.2f ms", name,
                     sum / v.size(), pct(0.5), pct(0.9), pct(0.99), v.back());
    if (budget > 0)
        snprintf(buf + n, sizeof(buf) - n, "  > %.1f ms: %.1f%%", budget, 100.0 * over / v.size());
    return buf;
}

void StartCountdown(const char* why) {
    g_bench = Bench::Countdown;
    g_benchT0 = NowMs();
    Log("Benchmark: countdown %.1f s (%s). Sit still.", Cfg().benchDelayS, why);
    Beeps({{880, 120}});
}

void StartRun(uint32_t newestIndex) {
    g_runDir = BenchDir() / Widen(LocalTimeStamp() + experiment::RunSuffix() + quad::RunSuffix());
    std::error_code ec;
    fs::create_directories(g_runDir, ec);
    g_runCsv = _wfopen((g_runDir / L"timing.csv").c_str(), L"wb");
    if (g_runCsv) fputs(kCsvHeader, g_runCsv);
    g_runFrames = _wfopen((g_runDir / L"frames.csv").c_str(), L"wb");
    if (g_runFrames) fputs(kFramesHeader, g_runFrames);
    g_runFramesT0 = -1;
    g_run = RunStats{};
    experiment::MarkRunStart();
    g_runFirstIndex = newestIndex + 1;
    g_runFirstNo = g_proxyFrames + 1;
    g_runSysT0 = -1;
    modern::GetCumulative(g_cumStart);
    g_bench = Bench::Running;
    g_benchT0 = NowMs();
    if (Cfg().benchFreezePose) {
        g_frozenPose.bPoseIsValid = false;  // OnWaitGetPosesEnd captures the pose from the current frame
        g_frozen = true;
    }
    Log("Benchmark: recording for %.1f s to %s (constant pose: %s)", Cfg().benchDurationS,
        Narrow(g_runDir.wstring()).c_str(), Cfg().benchFreezePose ? "yes" : "no");
    // File copies and module listing run off the game's render thread, so the first recorded frames stay clean.
    std::thread([dir = g_runDir] {
        Guarded("benchmark start I/O", [&] {
            sysinfo::CopyGameConfig(dir);
            sysinfo::LogModules("benchmark start");
        });
    }).detach();
    Beeps({{1320, 90}, {1320, 90}});
}

void FinishRun(bool aborted) {
    g_frozen = false;
    g_frozenPose.bPoseIsValid = false;
    g_bench = Bench::Idle;
    if (g_runCsv) {
        fclose(g_runCsv);
        g_runCsv = nullptr;
    }
    if (g_runFrames) {
        fclose(g_runFrames);
        g_runFrames = nullptr;
    }
    modern::Cumulative end{};
    bool cum = modern::GetCumulative(end);
    RunStats& r = g_run;
    double dur = r.sysLast > r.sysFirst ? r.sysLast - r.sysFirst : 0;
    float budget = BudgetMs();

    std::vector<std::string> lines;
    char buf[512];
    snprintf(buf, sizeof(buf), "Benchmark %s: %s", aborted ? "ABORTED" : "finished", Narrow(g_runDir.wstring()).c_str());
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "Frames %llu in %.2f s (%.2f fps), budget %.2f ms (%.1f Hz)", (unsigned long long)r.frames,
             dur, dur > 0 ? (r.frames - 1) / dur : 0.0, budget, 1000.0 / budget);
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "Size recommended by SteamVR %u x %u, returned to the game %u x %u; Submit texture: %s", g_recW,
             g_recH, g_gameW, g_gameH, g_texDesc.c_str());
    lines.push_back(buf);
    lines.push_back(experiment::Describe());
    lines.push_back(quad::Describe());
    lines.push_back(StatLine("GPU total (total_gpu)", r.totalGpu, budget));
    lines.push_back(StatLine("Game GPU before Submit", r.preGpu, budget));
    lines.push_back(StatLine("Game GPU after Submit", r.postGpu));
    lines.push_back(StatLine("Compositor GPU", r.compGpu));
    lines.push_back(StatLine("Compositor CPU", r.compCpu));
    lines.push_back(StatLine("Frame interval (proxy)", r.proxyInterval, budget * 1.5f));
    lines.push_back(StatLine("Game CPU WGP->Submit (proxy)", r.proxyCpu, budget));
    lines.push_back(StatLine("Wait in WGP (proxy)", r.wgpBlock));
    double pdur = r.proxyLastMs > r.proxyFirstMs ? (r.proxyLastMs - r.proxyFirstMs) / 1000.0 : 0;
    snprintf(buf, sizeof(buf), "Proxy frames %llu in %.2f s (%.2f fps); GPU spans from proxy timestamps "
             "(frames.csv, GPU time between points in the game's commands, idle included):",
             (unsigned long long)r.proxyRows, pdur, pdur > 0 ? (r.proxyRows - 1) / pdur : 0.0);
    lines.push_back(buf);
    lines.push_back(StatLine("GPU frame (WGP->last Submit)", r.gpuFrame, budget));
    static const char* const phaseLabels[gpu::kPhases] = {"GPU outer L", "GPU outer R", "GPU focus L", "GPU focus R",
                                                          "GPU compose L", "GPU compose R"};
    for (int k = 0; k < gpu::kPhases; ++k)
        if (!r.gpuPhase[k].v.empty()) lines.push_back(StatLine(phaseLabels[k], r.gpuPhase[k]));
    if (!r.gpuCompose.v.empty()) lines.push_back(StatLine("GPU compose (both eyes)", r.gpuCompose));
    lines.push_back(StatLine("GPU other (frame - phases)", r.gpuOther));
    snprintf(buf, sizeof(buf), "Dropped %llu, mispresented %llu, reprojected frames %llu (CPU %llu, GPU %llu), gaze valid %.1f%%",
             (unsigned long long)r.dropped, (unsigned long long)r.mispresented, (unsigned long long)r.reprojAny,
             (unsigned long long)r.reprojCpu, (unsigned long long)r.reprojGpu, r.frames ? 100.0 * r.gazeOk / r.frames : 0.0);
    lines.push_back(buf);
    if (cum) {
        snprintf(buf, sizeof(buf), "SteamVR (cumulative stats, delta): presents %u, dropped %u, reprojected %u, Submit %u",
                 end.framePresents - g_cumStart.framePresents, end.droppedFrames - g_cumStart.droppedFrames,
                 end.reprojectedFrames - g_cumStart.reprojectedFrames, end.frameSubmits - g_cumStart.frameSubmits);
        lines.push_back(buf);
    }
    if (FILE* f = _wfopen((g_runDir / L"summary.txt").c_str(), L"wb")) {
        for (auto& l : lines) fprintf(f, "%s\n", l.c_str());
        fclose(f);
    }
    for (auto& l : lines) Log("%s", l.c_str());
    Beeps({{440, 450}});
}

void HandleHotkey(uint32_t newestIndex) {
    const Config& c = Cfg();
    bool down = c.benchKey && (GetAsyncKeyState(c.benchKey) & 0x8000) &&
                (!c.benchKeyMod || (GetAsyncKeyState(c.benchKeyMod) & 0x8000));
    bool pressed = down && !g_keyWasDown;
    g_keyWasDown = down;

    if (!g_autoStarted && c.benchAutoStartS > 0 && g_firstSubmitMs >= 0 &&
        NowMs() - g_firstSubmitMs >= c.benchAutoStartS * 1000.0 && g_bench == Bench::Idle) {
        g_autoStarted = true;
        StartCountdown("auto_start_s");
    }
    if (pressed) {
        if (g_bench == Bench::Idle) {
            StartCountdown("hotkey");
        } else if (g_bench == Bench::Countdown) {
            g_bench = Bench::Idle;
            Log("Benchmark: countdown cancelled by the hotkey");
            Beeps({{440, 120}, {440, 120}});
        } else {
            FinishRun(true);
        }
        return;
    }
    if (g_bench == Bench::Countdown && NowMs() - g_benchT0 >= c.benchDelayS * 1000.0) StartRun(newestIndex);
    if (g_bench == Bench::Running && NowMs() - g_benchT0 >= c.benchDurationS * 1000.0) FinishRun(false);
}

void OpenSessionCsv() {
    if (g_sessionTried || !Cfg().csvAlways) return;
    g_sessionTried = true;
    fs::path dir = BenchDir() / Widen(LocalTimeStamp() + "_session");
    std::error_code ec;
    fs::create_directories(dir, ec);
    g_sessionCsv = _wfopen((dir / L"timing.csv").c_str(), L"wb");
    if (g_sessionCsv) fputs(kCsvHeader, g_sessionCsv);
    g_sessionFrames = _wfopen((dir / L"frames.csv").c_str(), L"wb");
    if (g_sessionFrames) fputs(kFramesHeader, g_sessionFrames);
    g_sessionFramesT0 = -1;
    Log("Session CSV: %s (timing.csv, frames.csv)", Narrow(dir.wstring()).c_str());
}

void ProcessTimings(const modern::FrameTiming* rows, uint32_t n) {
    if (n == 0) return;
    uint32_t newest = rows[n - 1].frameIndex;
    if (newest < g_lastWrittenIndex) g_lastWrittenIndex = 0;  // the compositor restarted its frame counter
    // A frame row is complete only after the frame is displayed, so rows 3 frames old are taken.
    for (uint32_t i = 0; i < n; ++i) {
        const auto& t = rows[i];
        if (t.frameIndex + 3 > newest || t.frameIndex <= g_lastWrittenIndex) continue;
        g_lastWrittenIndex = t.frameIndex;
        const ProxyFrame* p = FindProxy(t.frameIndex);
        if (g_sessionCsv) {
            if (g_sessionSysT0 < 0) g_sessionSysT0 = t.systemTimeS;
            WriteRow(g_sessionCsv, t, g_sessionSysT0, p);
        }
        if (g_bench == Bench::Running && g_runCsv && t.frameIndex >= g_runFirstIndex) {
            if (g_runSysT0 < 0) g_runSysT0 = t.systemTimeS;
            WriteRow(g_runCsv, t, g_runSysT0, p);
            AccumulateRun(t, p);
        }
    }
}

// The current valid gaze value ends at `now`: a stall if it lasted kStallMs or longer.
void EndGazeValue(double now) {
    if (g_valueSinceMs < 0) return;
    const double ms = now - g_valueSinceMs;
    if (ms < kStallMs) return;
    for (Stalls* s : {&g_stalls, &g_stallsTotal}) {
        ++s->count;
        s->frames += g_valueFrames;
        s->longestMs = std::max(s->longestMs, ms);
    }
}

void PeriodicStats() {
    int interval = Cfg().statsIntervalS;
    double now = NowMs();
    if (g_lastStatsMs == 0) g_lastStatsMs = now;
    if (interval <= 0 || now - g_lastStatsMs < interval * 1000.0) return;
    double sec = (now - g_lastStatsMs) / 1000.0;
    g_lastStatsMs = now;
    Log("Stats %.0f s: %llu frames (%.1f/s), %llu Submit, %zu/%zu distinct textures (L/R)", sec,
        (unsigned long long)g_statsFrames, g_statsFrames / sec, (unsigned long long)g_submitsInInterval,
        g_handles[0].size(), g_handles[1].size());
    if (g_gazeCalls) {
        const double ongoing = g_valueSinceMs >= 0 ? now - g_valueSinceMs : 0;
        Log("Gaze: valid in %.1f%% of frames, same value as the previous frame in %.1f%%, last (%.1f°, %.1f°); stalls "
            "(same valid value for %.1f s or longer): %llu, %.1f%% of frames, longest %.1f s%s; session %llu, "
            "longest %.1f s",
            100.0 * g_gazeOk / g_gazeCalls, 100.0 * g_gazeSame / g_gazeCalls, g_prevGaze.degX, g_prevGaze.degY,
            kStallMs / 1000, (unsigned long long)g_stalls.count, 100.0 * g_stalls.frames / g_gazeCalls,
            g_stalls.longestMs / 1000, ongoing >= kStallMs ? " (one still going on)" : "",
            (unsigned long long)g_stallsTotal.count, g_stallsTotal.longestMs / 1000);
    }
    g_stalls = Stalls{};
    if (g_gpuSums.frames) {
        const GpuSums& s = g_gpuSums;
        auto avg = [&](int k) { return s.n[k] ? s.phase[k] / s.n[k] : -1.0; };
        Log("GPU (proxy timestamps, %llu frames): frame %.2f ms; outer L %.2f, outer R %.2f, focus L %.2f, focus R "
            "%.2f, compose L %.2f, compose R %.2f ms (-1 = did not run)",
            (unsigned long long)s.frames, s.frame / s.frames, avg(gpu::OuterL), avg(gpu::OuterR), avg(gpu::FocusL),
            avg(gpu::FocusR), avg(gpu::ComposeL), avg(gpu::ComposeR));
    }
    g_gpuSums = GpuSums{};
    LogCallStats(false);
    g_statsFrames = g_gazeOk = g_gazeSame = g_gazeCalls = g_submitsInInterval = 0;
}

const char* FormatName(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "R8G8B8A8_TYPELESS";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return "B8G8R8A8_TYPELESS";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R11G11B10_FLOAT: return "R11G11B10_FLOAT";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        default: return "other";
    }
}

void LogDevice(ID3D11Texture2D* tex) {
    ID3D11Device* dev = nullptr;
    tex->GetDevice(&dev);
    if (!dev) return;
    IDXGIDevice* dxgi = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi))) {
        IDXGIAdapter* ad = nullptr;
        if (SUCCEEDED(dxgi->GetAdapter(&ad))) {
            DXGI_ADAPTER_DESC d{};
            ad->GetDesc(&d);
            Log("D3D11: adapter %s (vendor 0x%04X, %llu MB), feature level 0x%X, device flags 0x%X",
                Narrow(d.Description).c_str(), d.VendorId, (unsigned long long)(d.DedicatedVideoMemory >> 20),
                dev->GetFeatureLevel(), dev->GetCreationFlags());
            ad->Release();
        }
        dxgi->Release();
    }
    dev->Release();
}

std::string DescribeTexture(void* handle, int type) {
    if (type != vr::API_DirectX || !handle) return "(not D3D)";
    std::string out = "(not an ID3D11Texture2D)";
    IUnknown* unk = static_cast<IUnknown*>(handle);
    ID3D11Texture2D* tex = nullptr;
    if (SUCCEEDED(unk->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex)) && tex) {
        D3D11_TEXTURE2D_DESC d{};
        tex->GetDesc(&d);
        char buf[256];
        snprintf(buf, sizeof(buf), "%u x %u, format %d (%s), samples %u (quality %u), mips %u, array %u, bind 0x%X, misc 0x%X",
                 d.Width, d.Height, (int)d.Format, FormatName(d.Format), d.SampleDesc.Count, d.SampleDesc.Quality,
                 d.MipLevels, d.ArraySize, d.BindFlags, d.MiscFlags);
        out = buf;
        if (!g_deviceLogged) {
            g_deviceLogged = true;
            LogDevice(tex);
        }
        tex->Release();
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- API

void NoteSystem(vr::IVRSystem* real) {
    std::lock_guard lock(g_mutex);
    g_sys = real;
    g_eyeMapTried = false;
}

void OnWaitGetPosesBegin() {
    Guarded("WaitGetPoses (before)", [&] {
        std::lock_guard lock(g_mutex);
        double now = NowMs();
        if (g_cur.wgpExit > 0) {
            if (g_cur.submits != 2 && g_badSubmitLogs < 5) {
                ++g_badSubmitLogs;
                Log("Frame %llu: %d Submit calls between WaitGetPoses (expected 2)",
                    (unsigned long long)g_proxyFrames, g_cur.submits);
            }
            g_cur.valid = true;
            g_ring[g_cur.compIndex % kRing] = g_cur;
            g_byNo[g_cur.no % kRing] = g_cur;
        }
        g_cur = ProxyFrame{};
        g_cur.wgpEnter = now;
        if (g_prevWgpEnter > 0) g_cur.intervalMs = (float)(now - g_prevWgpEnter);
        g_prevWgpEnter = now;
        g_frameHandle[0] = g_frameHandle[1] = nullptr;
    });
}

void OnWaitGetPosesEnd(vr::IVRCompositor* real, vr::TrackedDevicePose_t* render, uint32_t nRender,
                       vr::TrackedDevicePose_t* game, uint32_t nGame, vr::EVRCompositorError err, double blockMs) {
    Guarded("WaitGetPoses (after)", [&] {
        std::lock_guard lock(g_mutex);
        ++g_proxyFrames;
        ++g_statsFrames;
        g_cur.no = g_proxyFrames;
        g_cur.wgpExit = NowMs();
        gpu::FrameStart(g_proxyFrames);
        g_cur.wgpBlockMs = (float)blockMs;
        if (!g_trackingSpaceKnown) {
            g_trackingSpace = real->GetTrackingSpace();
            g_trackingSpaceKnown = true;
            Log("Compositor tracking space: %d (0 = seated, 1 = standing)", (int)g_trackingSpace);
        }
        if (err != vr::VRCompositorError_None && g_proxyFrames < 20) Log("WaitGetPoses: error %d", (int)err);

        modern::FrameTiming rows[16];
        uint32_t n = modern::GetFrameTimings(rows, 16);
        uint32_t newest = n ? rows[n - 1].frameIndex : 0;
        g_cur.compIndex = newest;
        OpenSessionCsv();
        ProcessTimings(rows, n);

        const bool xrGaze = !modern::HasSystem() && xrlayer::Loaded();
        if (Cfg().gazeRead && (modern::HasSystem() || xrGaze)) {
            modern::Gaze g;
            if (xrGaze) {
                XrGaze(g);
            } else {
                modern::ReadGaze(g);
            }
            g_cur.gaze = g;
            ++g_gazeCalls;
            g_gazeOk += g.ok;
            const bool same = g.ndcL[0] == g_prevGaze.ndcL[0] && g.ndcL[1] == g_prevGaze.ndcL[1];
            if (same) ++g_gazeSame;
            // A pause of the game (Alt+Tab, loading: one frame of seconds) ends the value at the previous frame, so it
            // never counts as a stall (M5.1 run: 7 pauses of 4-26 s).
            const bool pause = g_cur.intervalMs > kPauseMs;
            if (same && g.ok && g_prevGaze.ok && !pause) {
                ++g_valueFrames;
                g_valueLastMs = g_cur.wgpExit;
            } else {
                EndGazeValue(pause ? g_valueLastMs : g_cur.wgpExit);
                g_valueSinceMs = g_valueLastMs = g.ok ? g_cur.wgpExit : -1;
                g_valueFrames = g.ok ? 1 : 0;
            }
            // The OpenXR layer has no gaze until OpenComposite attaches its action sets, so its first valid one is logged.
            if (!g_gazeFirstLogged && (!xrGaze || g.ok)) {
                g_gazeFirstLogged = true;
                Log("Gaze in game (%s): valid %d at proxy frame %llu, NDC L (%.4f, %.4f) R (%.4f, %.4f), direction "
                    "(%.2f°, %.2f°)",
                    xrGaze ? "OpenXR layer, XR_EXT_eye_gaze_interaction" : "GetEyeTrackedFoveationCenter", g.ok,
                    (unsigned long long)g_proxyFrames, g.ndcL[0], g.ndcL[1], g.ndcR[0], g.ndcR[1], g.degX, g.degY);
            }
            g_prevGaze = g;
        }

        // Test of the kill-switch path (qvfr_proxy_smoke --fault N): a real access violation inside Guarded.
        if (Cfg().testFaultFrames > 0 && g_proxyFrames == (uint64_t)Cfg().testFaultFrames) {
            Log("Test: deliberate access violation at frame %llu", (unsigned long long)g_proxyFrames);
            *static_cast<volatile int*>(nullptr) = 0;
        }

        ProcessFrames(g_proxyFrames);
        HandleHotkey(newest);
        PeriodicStats();
        ReloadConfig();
        g_cur.frozen = g_frozen;
        // The gaze places this frame's focus window (DFR) and the overlay's gaze point; after the hotkeys, so a
        // benchmark that starts now already has the window at the center.
        g_cur.window = quad::SetGaze(g_cur.gaze.ok, g_cur.gaze.ndcL, g_cur.gaze.ndcR, g_bench != Bench::Idle);

        if (g_frozen) {
            if (!g_frozenPose.bPoseIsValid && render && nRender > 0 && render[0].bPoseIsValid) {
                g_frozenPose = render[0];
                g_frozenPose.vVelocity = {};
                g_frozenPose.vAngularVelocity = {};
                const auto& m = g_frozenPose.mDeviceToAbsoluteTracking.m;
                Log("Benchmark: HMD pose frozen, position (%.3f, %.3f, %.3f) m", m[0][3], m[1][3], m[2][3]);
            }
            if (g_frozenPose.bPoseIsValid) {
                if (render && nRender > 0) render[0] = g_frozenPose;
                if (game && nGame > 0) game[0] = g_frozenPose;
            }
        }
    });
}

void FilterLastPoses(vr::TrackedDevicePose_t* render, uint32_t nRender, vr::TrackedDevicePose_t* game, uint32_t nGame) {
    Guarded("GetLastPoses (filter)", [&] {
        std::lock_guard lock(g_mutex);
        if (!g_frozen || !g_frozenPose.bPoseIsValid) return;
        if (render && nRender > 0) render[0] = g_frozenPose;
        if (game && nGame > 0) game[0] = g_frozenPose;
    });
}

void FilterDevicePoses(vr::ETrackingUniverseOrigin origin, vr::TrackedDevicePose_t* poses, uint32_t n) {
    std::lock_guard lock(g_mutex);
    if (!g_frozen || !g_frozenPose.bPoseIsValid || !poses || n == 0 || origin != g_trackingSpace) return;
    poses[0] = g_frozenPose;
}

void OnSetTrackingSpace(vr::ETrackingUniverseOrigin origin) {
    Guarded("SetTrackingSpace", [&] {
        std::lock_guard lock(g_mutex);
        g_trackingSpace = origin;
        g_trackingSpaceKnown = true;
        Log("SetTrackingSpace(%d)", (int)origin);
    });
}

void OnSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds, vr::EVRSubmitFlags flags,
              vr::EVRCompositorError err, double durMs) {
    Guarded("Submit", [&] {
        std::lock_guard lock(g_mutex);
        double now = NowMs();
        if (g_firstSubmitMs < 0) {
            g_firstSubmitMs = now;
            Log("First Submit (thread %lu)", GetCurrentThreadId());
            sysinfo::LogModules("first Submit");
        }
        ++g_cur.submits;
        ++g_submitsInInterval;
        g_cur.lastSubmit = now;
        g_cur.submitMs += (float)durMs;
        if (err != vr::VRCompositorError_None && g_submitErrors < 10) {
            ++g_submitErrors;
            Log("Submit(%s): error %d", eye ? "right" : "left", (int)err);
        }
        int e = eye == vr::Eye_Right ? 1 : 0;
        SubmitInfo s;
        if (tex) {
            s.handle = tex->handle;
            s.type = tex->eType;
            s.colorSpace = tex->eColorSpace;
        }
        s.flags = flags;
        if (bounds) {
            s.bounds[0] = bounds->uMin;
            s.bounds[1] = bounds->vMin;
            s.bounds[2] = bounds->uMax;
            s.bounds[3] = bounds->vMax;
        }
        g_frameHandle[e] = s.handle;
        if (!g_sharedLogged && g_frameHandle[0] && g_frameHandle[0] == g_frameHandle[1]) {
            g_sharedLogged = true;
            Log("Submit: both eyes get the same texture %p (shared texture, eyes split by bounds)", s.handle);
        }

        auto& seen = g_handles[e];
        bool newHandle = s.handle && std::find(seen.begin(), seen.end(), s.handle) == seen.end();
        if (newHandle && seen.size() < 32) {
            seen.push_back(s.handle);
            std::string desc = DescribeTexture(s.handle, s.type);
            g_texDesc = desc;
            Log("Submit(%s): new texture %p (#%zu), type %d, color space %d, flags 0x%X, bounds %s u %.4f..%.4f "
                "v %.4f..%.4f, %.3f ms; %s",
                e ? "right" : "left", s.handle, seen.size(), s.type, s.colorSpace, s.flags, bounds ? "" : "(none)",
                s.bounds[0], s.bounds[2], s.bounds[1], s.bounds[3], durMs, desc.c_str());
        }
        SubmitInfo& last = g_lastSubmit[e];
        bool paramsChanged = last.type != s.type || last.colorSpace != s.colorSpace || last.flags != s.flags ||
                             memcmp(last.bounds, s.bounds, sizeof(s.bounds)) != 0;
        if (paramsChanged && last.type != -1 && !newHandle) {
            if (g_submitChangeLogs < 20) {
                ++g_submitChangeLogs;
                Log("Submit(%s): parameters changed: type %d, color space %d, flags 0x%X, bounds u %.4f..%.4f v %.4f..%.4f",
                    e ? "right" : "left", s.type, s.colorSpace, s.flags, s.bounds[0], s.bounds[2], s.bounds[1], s.bounds[3]);
            } else if (g_submitChangeLogs == 20) {
                ++g_submitChangeLogs;
                Log("Submit: further parameter changes are not logged");
            }
        }
        last = s;
    });
}

std::string TextureInfo(void* handle, int type) {
    std::string s = "(read error)";
    Guarded("TextureInfo", [&] { s = DescribeTexture(handle, type); });
    return s;
}

void NoteRecommendedSize(uint32_t realW, uint32_t realH, uint32_t returnedW, uint32_t returnedH) {
    g_recW = realW;
    g_recH = realH;
    g_gameW = returnedW;
    g_gameH = returnedH;
}

void OnShutdown() {
    Guarded("frame::OnShutdown", [&] {
        std::lock_guard lock(g_mutex);
        if (g_bench == Bench::Running) FinishRun(true);
        g_bench = Bench::Idle;
        if (g_sessionCsv) {
            fclose(g_sessionCsv);
            g_sessionCsv = nullptr;
        }
        if (g_sessionFrames) {
            fclose(g_sessionFrames);
            g_sessionFrames = nullptr;
        }
        Log("VR session end: %llu proxy frames", (unsigned long long)g_proxyFrames);
        // Per-session state, in case the game starts VR again in the same process.
        for (auto& f : g_ring) f = ProxyFrame{};
        for (auto& f : g_byNo) f = ProxyFrame{};
        for (auto& r : g_gpuRows) r = GpuRow{};
        g_framesNext = g_proxyFrames + 1;
        g_gpuSums = GpuSums{};
        g_cur = ProxyFrame{};
        g_prevWgpEnter = 0;
        g_lastWrittenIndex = 0;
        g_trackingSpaceKnown = false;
        g_sessionTried = false;
        g_sessionSysT0 = -1;
        g_frozen = false;
        g_frozenPose.bPoseIsValid = false;
        g_valueSinceMs = g_valueLastMs = -1;
        g_valueFrames = 0;
    });
    experiment::Shutdown();  // guards itself; runs even after a diagnostics fault
}

}  // namespace qvfr::frame

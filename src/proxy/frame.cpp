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
#include "modern_vr.h"
#include "sysinfo.h"

namespace qvfr::frame {

namespace {

std::mutex g_mutex;  // WaitGetPoses and Submit may arrive from different threads

// ---------------------------------------------------------------- proxy frame data

struct ProxyFrame {
    uint32_t compIndex = 0;   // m_nFrameIndex of the newest compositor frame right after WaitGetPoses
    double wgpEnter = 0, wgpExit = 0, lastSubmit = 0;
    float wgpBlockMs = 0;
    float intervalMs = 0;     // since the previous entry into WaitGetPoses
    float submitMs = 0;       // total time spent in Submit
    int submits = 0;
    modern::Gaze gaze;
    bool valid = false;
};

constexpr uint32_t kRing = 128;
ProxyFrame g_ring[kRing];
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
uint32_t g_recW = 0, g_recH = 0;
double g_firstSubmitMs = -1;

// ---------------------------------------------------------------- gaze and periodic stats

double g_lastStatsMs = 0;
uint64_t g_statsFrames = 0, g_gazeOk = 0, g_gazeSame = 0, g_gazeCalls = 0;
uint64_t g_submitsInInterval = 0;
modern::Gaze g_prevGaze;
bool g_gazeFirstLogged = false;

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
    uint64_t frames = 0, dropped = 0, mispresented = 0, reprojCpu = 0, reprojGpu = 0, reprojAny = 0, gazeOk = 0;
    double sysFirst = -1, sysLast = -1;
} g_run;

const char kCsvHeader[] =
    "frame,t_s,pre_submit_gpu_ms,post_submit_gpu_ms,total_gpu_ms,compositor_gpu_ms,compositor_cpu_ms,"
    "compositor_idle_cpu_ms,client_interval_ms,wait_get_poses_called_ms,new_poses_ready_ms,new_frame_ready_ms,"
    "submit_frame_ms,present_call_cpu_ms,wait_for_present_cpu_ms,presents,mispresented,dropped,reprojection_flags,"
    "vsyncs_ready,vsyncs_to_first_view,transfer_latency_ms,proxy_interval_ms,proxy_cpu_ms,proxy_wgp_block_ms,proxy_submit_ms,"
    "proxy_submits,gaze_ok,gaze_x_deg,gaze_y_deg,frozen\n";

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
    g_runDir = BenchDir() / Widen(LocalTimeStamp());
    std::error_code ec;
    fs::create_directories(g_runDir, ec);
    g_runCsv = _wfopen((g_runDir / L"timing.csv").c_str(), L"wb");
    if (g_runCsv) fputs(kCsvHeader, g_runCsv);
    g_run = RunStats{};
    g_runFirstIndex = newestIndex + 1;
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
    snprintf(buf, sizeof(buf), "Size recommended by SteamVR (returned to the game) %u x %u; Submit texture: %s", g_recW, g_recH,
             g_texDesc.c_str());
    lines.push_back(buf);
    lines.push_back(StatLine("GPU total (total_gpu)", r.totalGpu, budget));
    lines.push_back(StatLine("Game GPU before Submit", r.preGpu, budget));
    lines.push_back(StatLine("Game GPU after Submit", r.postGpu));
    lines.push_back(StatLine("Compositor GPU", r.compGpu));
    lines.push_back(StatLine("Compositor CPU", r.compCpu));
    lines.push_back(StatLine("Frame interval (proxy)", r.proxyInterval, budget * 1.5f));
    lines.push_back(StatLine("Game CPU WGP->Submit (proxy)", r.proxyCpu, budget));
    lines.push_back(StatLine("Wait in WGP (proxy)", r.wgpBlock));
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
    Log("Session CSV: %s", Narrow((dir / L"timing.csv").wstring()).c_str());
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
        Log("Gaze: valid in %.1f%% of frames, same value as the previous frame in %.1f%%, last (%.1f°, %.1f°)",
            100.0 * g_gazeOk / g_gazeCalls, 100.0 * g_gazeSame / g_gazeCalls, g_prevGaze.degX, g_prevGaze.degY);
    }
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
        g_cur.wgpExit = NowMs();
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

        if (Cfg().gazeRead && modern::HasSystem()) {
            modern::Gaze g;
            modern::ReadGaze(g);
            g_cur.gaze = g;
            ++g_gazeCalls;
            g_gazeOk += g.ok;
            if (g.ndcL[0] == g_prevGaze.ndcL[0] && g.ndcL[1] == g_prevGaze.ndcL[1]) ++g_gazeSame;
            if (!g_gazeFirstLogged) {
                g_gazeFirstLogged = true;
                Log("Gaze in game: GetEyeTrackedFoveationCenter = %d, NDC L (%.4f, %.4f) R (%.4f, %.4f), direction (%.2f°, %.2f°)",
                    g.ok, g.ndcL[0], g.ndcL[1], g.ndcR[0], g.ndcR[1], g.degX, g.degY);
            }
            g_prevGaze = g;
        }

        HandleHotkey(newest);
        PeriodicStats();

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

void NoteRecommendedSize(uint32_t w, uint32_t h) {
    g_recW = w;
    g_recH = h;
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
        Log("VR session end: %llu proxy frames", (unsigned long long)g_proxyFrames);
        // Per-session state, in case the game starts VR again in the same process.
        for (auto& f : g_ring) f = ProxyFrame{};
        g_cur = ProxyFrame{};
        g_prevWgpEnter = 0;
        g_lastWrittenIndex = 0;
        g_trackingSpaceKnown = false;
        g_sessionTried = false;
        g_sessionSysT0 = -1;
        g_frozen = false;
        g_frozenPose.bPoseIsValid = false;
    });
}

}  // namespace qvfr::frame

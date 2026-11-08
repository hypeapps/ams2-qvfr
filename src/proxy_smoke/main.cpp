// qvfr_proxy_smoke: tests the openvr_api.dll proxy without the game.
//
// Loads proxy\openvr_api.dll (next to the exe) with LoadLibrary and calls the same exports as AMS2.
// The proxy writes its logs to smoke_home\ next to the exe (QVFR_HOME variable), not to proxy\qvfr\.
//
//   qvfr_proxy_smoke.exe              Background application: VR_Init, projection queries, VR_Shutdown
//                                     (does not take the HMD from other applications; needs SteamVR running)
//   qvfr_proxy_smoke.exe --scene 12   Scene application: 12 s of D3D11 frames (grey background) with Submit,
//                                     the benchmark starts by itself (auto_start_s=2, delay_s=1, duration_s=5)
//   --experiment focus|outer          sets [experiment] mode (size and projection replaced by the proxy)
//   --center X Y                      with --experiment: focus center tangents (center_x, center_y)
//   --fault N                         [test] fault_frames=N: a deliberate exception in the proxy diagnostics at frame N
//   --trace                           [trace] callers=1; after VR_Shutdown checks the caller stacks in the proxy log
//                                     (they start in this exe, no proxy frames, each unique stack once, hits counted)
//   --probe                           [quad] mode=probe; the scene clears a "final" texture per eye and copies it into
//                                     the Submit texture, like AMS2; after VR_Shutdown checks the probe lines in the log
//                                     (engine hooks off on this exe, D3D11 hooks on, the copy into Submit found, clears
//                                     counted). No Draw: an empty Draw costs several ms in the NVIDIA driver.
//   --quad                            [quad] mode=quad; after VR_Shutdown checks that nothing was hooked or patched
//                                     (this exe is not the supported AMS2AVX.exe)
//   --dlaa                            [dlaa] mode=dlaa; after VR_Shutdown checks that DLAA stayed off on this exe
//                                     (no engine or draw hooks, no faults)
//   --shots                           without SteamVR and without the proxy DLL: the proxy's shot module on synthetic
//                                     textures (staging copies, PNG files, pixel differences)
//   --quad-compose                    without SteamVR and without the proxy DLL: the proxy's composition module on
//                                     synthetic textures (pixels by region, also with the window in grid corners,
//                                     pipeline state restored)
//   --focus-geometry                  without SteamVR: focus window placement for a gaze point (Dream Air frusta)
//   --focus-cull                      without SteamVR: focus culling's frustum block math on a synthetic engine block
//   --clones                          without SteamVR and without the proxy DLL: the proxy's clones module on fake engine
//                                     objects (learning, clones in the sets' sizes, swap of fields, sizes and bound
//                                     views, viewports, restore, leaks, destroy; focus or outer smaller)
//   --sun-fx                          without SteamVR and without the proxy DLL: the proxy's sun effects module (outer's
//                                     effect draws repeated into its accumulation, focus's replaced by the pass, at an
//                                     effect draw and at the anchor, module off) on a D3D11 device (sun_fx_smoke.cpp)
//   --mvr-vs                          without SteamVR and without the proxy DLL: the proxy's MVR vertex shader patch
//                                     (hook, NVAPI creation, image unchanged with MVR off, view 1 with MVR on, refusal)
//                                     on an NVIDIA D3D11 device (mvr_vs_smoke.cpp)
//   --mvr-ps                          without SteamVR and without the proxy DLL: the proxy's MVR pixel shader patch
//                                     (through the sun effects hook, kinds, image unchanged, refusal, layered GS count)
//                                     on an NVIDIA D3D11 device (mvr_ps_smoke.cpp)
//   --mvr-pass                        without SteamVR and without the proxy DLL: gate B7's test run (a fake task run
//                                     once more with MVR into two-slice copies, view 1 = the focus window in the corner
//                                     of slice 1 against the focus projection) on an NVIDIA D3D11 device
//                                     (mvr_pass_smoke.cpp)
//   --mvr-forward                     without SteamVR and without the proxy DLL: gate B9c's forward test (a fake
//                                     CPhase3Render with MVR, view 1 against focus's own forward pass; the patched VS
//                                     and PS, the copy-in of focus's clones) on an NVIDIA D3D11 device
//                                     (mvr_forward_smoke.cpp)
//   --mvr-motion                      without SteamVR and without the proxy DLL: the focus motion vectors from outer's
//                                     (B10) against a clip-space reference on a D3D11 device (mvr_motion_smoke.cpp)
//   --xr-layer                        without SteamVR: the proxy DLL as an OpenXR API layer between a fake loader and a
//                                     fake runtime (extension, gaze set in attach and sync, own sync, gaze, 2 sessions)
//   --openxr-path                     [runtime] path=openxr with the Valve DLL as the backend: loader environment,
//                                     layer manifest, backend path (needs no SteamVR running)
//   --openxr                          [runtime] path=openxr for the other modes (with --scene: the scene through
//                                     OpenComposite in proxy/qvfr/opencomposite/, the system's OpenXR runtime, our layer)
//   --openxr-nolayer                  the same without our OpenXR layer (openxr_layer=0), for A/B
// smoke_home\qvfr.ini is rewritten on every run, so options never leak from one run into the next.

#include <windows.h>
#include <d3d11.h>

#include <openvr.h>  // 0.9.12; functions are taken through GetProcAddress

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "clones.h"
#include "common.h"
#include "compose.h"

int RunSunFx();  // sun_fx_smoke.cpp
int RunMvrVs();  // mvr_vs_smoke.cpp
int RunMvrPs();  // mvr_ps_smoke.cpp
int RunMvrPass();  // mvr_pass_smoke.cpp
int RunMvrForward();  // mvr_forward_smoke.cpp
int RunTaaFix();  // taa_fix_smoke.cpp
int RunMvrMotion();  // mvr_motion_smoke.cpp
#include "focus_cull.h"
#include "focus_geometry.h"
#include "hooks.h"
#include "shots.h"
#include "xr_smoke.h"

namespace fs = std::filesystem;

namespace {

using InitFn = vr::IVRSystem*(VR_CALLTYPE*)(vr::EVRInitError*, vr::EVRApplicationType);
using ShutdownFn = void(VR_CALLTYPE*)();
using CompositorFn = vr::IVRCompositor*(VR_CALLTYPE*)();
using OverlayFn = vr::IVROverlay*(VR_CALLTYPE*)();
using ErrFn = const char*(VR_CALLTYPE*)(vr::EVRInitError);
using GenericFn = void*(VR_CALLTYPE*)(const char*, vr::EVRInitError*);

fs::path ExeDir() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
}

const char kSceneIni[] =
    "[proxy]\nwrap=1\n[log]\nwarning_box=0\nfirst_calls=2\nstats_interval_s=5\n[timing]\ncsv_always=1\n"
    "[benchmark]\nmodifier=0xA3\nkey=0x78\ndelay_s=1\nduration_s=5\nfreeze_pose=1\nbeep=0\nauto_start_s=2\n[gaze]\nread=1\n";
// warning_box=0: the smoke exe is never AMS2AVX.exe, so quad runs always have a serious finding (checked in the log).
const char kBackgroundIni[] = "[proxy]\nwrap=1\n[log]\nwarning_box=0\n";

int RunScene(vr::IVRSystem* sys, vr::IVRCompositor* comp, vr::IVROverlay* ov, double seconds, bool probe) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    // Like AMS2: textures of exactly the size the proxy returns, R8G8B8A8_TYPELESS with GENERATE_MIPS, typed RTV.
    uint32_t w = 0, h = 0;
    sys->GetRecommendedRenderTargetSize(&w, &h);
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    d.SampleDesc.Count = 1;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    d.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    ID3D11Texture2D* tex[2] = {};
    ID3D11Texture2D* final[2] = {};  // --probe: the eye image is drawn here and copied into the Submit texture
    ID3D11RenderTargetView* rtv[2] = {};
    for (int e = 0; e < 2; ++e) {
        ID3D11Texture2D*& target = probe ? final[e] : tex[e];
        if ((probe && FAILED(dev->CreateTexture2D(&d, nullptr, &tex[e]))) ||
            FAILED(dev->CreateTexture2D(&d, nullptr, &target)) || FAILED(dev->CreateRenderTargetView(target, &rv, &rtv[e]))) {
            printf("Cannot create the eye textures\n");
            return 1;
        }
    }
    // An overlay like "HUD Overlay" in AMS2: texture, width, alpha and transform every frame.
    vr::VROverlayHandle_t ovh = vr::k_ulOverlayHandleInvalid;
    if (ov) ov->CreateOverlay("qvfr.smoke", "Smoke HUD", &ovh);
    printf("Scene: %u x %u per eye for %.0f s, overlay 0x%llX\n", w, h, seconds, (unsigned long long)ovh);
    auto t0 = std::chrono::steady_clock::now();
    int frames = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        comp->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
        for (int e = 0; e < 2; ++e) {
            const float c[4] = {0.25f, 0.25f, 0.28f, 1.0f};
            ctx->ClearRenderTargetView(rtv[e], c);
            if (probe) ctx->CopyResource(tex[e], final[e]);
            vr::Texture_t t{tex[e], vr::API_DirectX, vr::ColorSpace_Gamma};
            comp->Submit((vr::EVREye)e, &t, nullptr, vr::Submit_Default);
        }
        if (ovh != vr::k_ulOverlayHandleInvalid) {
            vr::HmdMatrix34_t m = {{{1, 0, 0, 0}, {0, 1, 0, 1.0f}, {0, 0, 1, -1.5f - 0.0001f * frames}}};
            vr::Texture_t t{tex[frames & 1], vr::API_DirectX, vr::ColorSpace_Gamma};
            ov->SetOverlayTexture(ovh, &t);
            ov->SetOverlayWidthInMeters(ovh, 0.5f);
            ov->SetOverlayAlpha(ovh, frames < 20 ? frames / 20.0f : 1.0f);
            ov->SetOverlayTransformAbsolute(ovh, vr::TrackingUniverseStanding, &m);
            ov->ShowOverlay(ovh);
        }
        ctx->Flush();
        ++frames;
    }
    if (ovh != vr::k_ulOverlayHandleInvalid) ov->DestroyOverlay(ovh);
    printf("Frames: %d\n", frames);
    for (int e = 0; e < 2; ++e) {
        rtv[e]->Release();
        tex[e]->Release();
        if (final[e]) final[e]->Release();
    }
    ctx->Release();
    dev->Release();
    return 0;
}

// Newest proxy log in smoke_home\logs.
fs::path NewestLog(const fs::path& dir) {
    fs::path best;
    fs::file_time_type bestTime{};
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.path().extension() != L".log") continue;
        auto t = e.last_write_time(ec);
        if (best.empty() || t > bestTime) {
            best = e.path();
            bestTime = t;
        }
    }
    return best;
}

// Reads "Caller stack <method> #<n> (...): <frames>" and "Caller stacks <method> (session total): #1 <hits>, ..." lines.
int CheckTrace(const fs::path& logDir) {
    fs::path log = NewestLog(logDir);
    std::ifstream in(log);
    if (!in) {
        printf("Trace check FAILED: no proxy log in %ls\n", logDir.c_str());
        return 3;
    }
    const std::string kStack = "Caller stack ", kTotal = "Caller stacks ", kSelf = "qvfr_proxy_smoke.exe+";
    std::set<std::string> seen;
    int stacks = 0, fromSmoke = 0, withProxy = 0, duplicates = 0, totals = 0, counted = 0;
    std::string line;
    while (std::getline(in, line)) {
        size_t p = line.find(kStack);
        size_t colon = line.find("): ", p);
        if (p != std::string::npos && colon != std::string::npos) {
            std::string method = line.substr(p + kStack.size(), line.find(" #", p) - p - kStack.size());
            std::string frames = line.substr(colon + 3);
            ++stacks;
            if (frames.compare(0, kSelf.size(), kSelf) == 0) ++fromSmoke;
            if (frames.find("openvr_api.dll+") != std::string::npos) ++withProxy;
            if (!seen.insert(method + "|" + frames).second) ++duplicates;
        }
        p = line.find(kTotal);
        if (p != std::string::npos && line.find("(session total)", p) != std::string::npos) {
            ++totals;
            // A stack hit more than once was counted, not logged again.
            for (size_t h = line.find('#', p); h != std::string::npos; h = line.find('#', h + 1)) {
                size_t sp = line.find(' ', h);
                if (sp != std::string::npos && atoi(line.c_str() + sp + 1) > 1) ++counted;
            }
        }
    }
    bool ok = stacks > 0 && fromSmoke > 0 && !withProxy && !duplicates && totals > 0 && counted > 0;
    printf("Trace check %s: %d caller stacks (%d starting in qvfr_proxy_smoke.exe), %d with proxy frames, %d duplicates, "
           "%d session totals, %d stacks hit more than once (%ls)\n",
           ok ? "OK" : "FAILED", stacks, fromSmoke, withProxy, duplicates, totals, counted, log.filename().c_str());
    return ok ? 0 : 3;
}

// Probe lines: engine hooks off on this exe, D3D11 hooks installed, a copy into each eye's Submit texture,
// stats with immediate context clears (one ClearRenderTargetView per eye and frame in the scene).
int CheckProbe(const fs::path& logDir) {
    fs::path log = NewestLog(logDir);
    std::ifstream in(log);
    if (!in) {
        printf("Probe check FAILED: no proxy log in %ls\n", logDir.c_str());
        return 4;
    }
    bool engineOff = false, d3d = false, intoLeft = false, intoRight = false, total = false, faults = false;
    double clears = 0;
    std::string line;
    while (std::getline(in, line)) {
        engineOff |= line.find("Probe: engine hooks off") != std::string::npos;
        d3d |= line.find("Probe: D3D11 hooks installed") != std::string::npos;
        intoLeft |= line.find("into Submit texture of the left eye") != std::string::npos;
        intoRight |= line.find("into Submit texture of the right eye") != std::string::npos;
        faults |= line.find("exception") != std::string::npos && line.find("probe") != std::string::npos;
        size_t p = line.find("Probe stats (session total)");
        size_t q = line.find("clears ", p);
        if (p != std::string::npos && q != std::string::npos) {
            total = true;
            clears = atof(line.c_str() + q + strlen("clears "));
        }
    }
    bool ok = engineOff && d3d && intoLeft && intoRight && total && clears >= 1.5 && !faults;
    printf("Probe check %s: engine hooks off %d, D3D11 hooks %d, copy into Submit left %d right %d, session stats %d "
           "(%.2f immediate clears per frame, expected about 2), probe faults %d (%ls)\n",
           ok ? "OK" : "FAILED", engineOff, d3d, intoLeft, intoRight, total, clears, faults, log.filename().c_str());
    return ok ? 0 : 4;
}

// Quad mode on a foreign exe: the supported-build check fails, so no engine hook and no builder patch.
int CheckQuad(const fs::path& logDir) {
    fs::path log = NewestLog(logDir);
    std::ifstream in(log);
    if (!in) {
        printf("Quad check FAILED: no proxy log in %ls\n", logDir.c_str());
        return 5;
    }
    bool settings = false, notSupported = false, hooked = false, faults = false, size = false, resized = false;
    // M8.5: the default preset, the serious finding about the exe (log only with warning_box=0), the Detected block.
    bool preset = false, exeWarning = false, boxOff = false, detected = false, detectedGame = false;
    std::string line;
    while (std::getline(in, line)) {
        settings |= line.find("Settings: [quad] mode=quad") != std::string::npos;
        notSupported |= line.find("Quad: exe not supported") != std::string::npos;
        hooked |= line.find("Quad: engine hooks installed") != std::string::npos ||
                  line.find("builder patch applied") != std::string::npos;
        faults |= line.find("in quad mode") != std::string::npos;
        // Without the focus groups the game must keep the SteamVR size.
        size |= line.find("IVRSystem::GetRecommendedRenderTargetSize = ") != std::string::npos;
        resized |= line.find("quad focus size") != std::string::npos ||
                   line.find("size returned to the game") != std::string::npos;
        preset |= line.find("Settings: [quad] preset=balanced (peripheral_multiplier 0.50, focus section 0.35 x 0.35, "
                            "fixed section 0.50 x 0.45)") != std::string::npos;
        exeWarning |= line.find("WARNING: The game runs as qvfr_proxy_smoke.exe") != std::string::npos;
        boxOff |= line.find("Warning box: off ([log] warning_box=0), 1 serious finding(s)") != std::string::npos;
        detected |= line.find("] Detected (") != std::string::npos;
        detectedGame |= line.find("  Game: qvfr_proxy_smoke.exe (not AMS2AVX.exe): quad views off") != std::string::npos;
    }
    bool ok = settings && notSupported && !hooked && !faults && size && !resized && preset && exeWarning && boxOff &&
              detected && detectedGame;
    printf("Quad check %s: settings %d, exe not supported %d, hooks or patch %d, faults %d, size logged %d, size "
           "replaced %d, preset %d, exe warning %d, box off %d, Detected block %d with the game line %d (%ls)\n",
           ok ? "OK" : "FAILED", settings, notSupported, hooked, faults, size, resized, preset, exeWarning, boxOff,
           detected, detectedGame, log.filename().c_str());
    return ok ? 0 : 5;
}

int CheckDlaa(const fs::path& logDir) {
    fs::path log = NewestLog(logDir);
    std::ifstream in(log);
    if (!in) {
        printf("DLAA check FAILED: no proxy log in %ls\n", logDir.c_str());
        return 6;
    }
    bool settings = false, off = false, hooked = false, faults = false;
    std::string line;
    while (std::getline(in, line)) {
        settings |= line.find("Settings: [dlaa] mode=dlaa") != std::string::npos;
        // A proxy without the DLSS SDK says so instead of checking the exe.
        off |= line.find("DLAA: exe not supported") != std::string::npos ||
               line.find("DLAA: this proxy was built without the NVIDIA DLSS SDK") != std::string::npos;
        hooked |= line.find("DLAA: engine hooks installed") != std::string::npos ||
                  line.find("DLAA: D3D11 draw hooks") != std::string::npos;
        faults |= line.find("in DLAA") != std::string::npos;
    }
    bool ok = settings && off && !hooked && !faults;
    printf("DLAA check %s: settings %d, off on this exe %d, hooks %d, faults %d (%ls)\n", ok ? "OK" : "FAILED",
           settings, off, hooked, faults, log.filename().c_str());
    return ok ? 0 : 6;
}

// PNG width and height from the IHDR chunk.
bool PngSize(const fs::path& p, uint32_t& w, uint32_t& h) {
    std::ifstream f(p, std::ios::binary);
    unsigned char b[24] = {};
    if (!f.read((char*)b, sizeof(b)) || memcmp(b + 12, "IHDR", 4) != 0) return false;
    w = (uint32_t)b[16] << 24 | (uint32_t)b[17] << 16 | (uint32_t)b[18] << 8 | b[19];
    h = (uint32_t)b[20] << 24 | (uint32_t)b[21] << 16 | (uint32_t)b[22] << 8 | b[23];
    return true;
}

// The shot module on its own: four textures like the four views (outer_l == focus_l, focus_r differs from outer_r
// in a 16 x 16 block, one image in R10G10B10A2), a focus window (outer_l's 256 x 192 rectangle at (200, 150), each
// pixel repeated 2 x 2), then the log lines and the PNG files it writes.
int RunShots(const fs::path& home) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    const UINT w = 1024, h = 768;
    static const char* const names[] = {"outer_l", "outer_r", "focus_l", "focus_r", "tonemap", "focus_window",
                                        "hdr16",   "hdr11"};
    static const qvfr::shots::Pair pairs[] = {{2, 0, "focus L vs outer L"},
                                              {3, 1, "focus R vs outer R"},
                                              {0, 1, "outer L vs outer R"},
                                              {5, 0, "window vs outer L", 200, 150, 256, 192}};
    constexpr int kImages = 8;
    std::vector<uint32_t> px((size_t)w * h);
    ID3D11Texture2D* tex[kImages] = {};
    // HDR formats (M7.4 dump): RGBA16F (1, 0.5, 0) with one NaN pixel, R11G11B10 all 1.0; 64 x 64.
    {
        std::vector<uint16_t> h16(64 * 64 * 4);
        for (size_t p = 0; p < 64 * 64; ++p) {
            h16[p * 4 + 0] = p == 0 ? 0x7E00 : 0x3C00;
            h16[p * 4 + 1] = 0x3800;
            h16[p * 4 + 2] = 0;
            h16[p * 4 + 3] = 0x3C00;
        }
        std::vector<uint32_t> r11(64 * 64, 0x3C0u | (0x3C0u << 11) | (0x1E0u << 22));
        const DXGI_FORMAT fmts[2] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R11G11B10_FLOAT};
        const void* data[2] = {h16.data(), r11.data()};
        const UINT pitch[2] = {64 * 8, 64 * 4};
        for (int k = 0; k < 2; ++k) {
            D3D11_TEXTURE2D_DESC d{};
            d.Width = d.Height = 64;
            d.MipLevels = d.ArraySize = 1;
            d.Format = fmts[k];
            d.SampleDesc.Count = 1;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_SUBRESOURCE_DATA init{data[k], pitch[k], 0};
            if (FAILED(dev->CreateTexture2D(&d, &init, &tex[6 + k]))) {
                printf("Cannot create the HDR shot textures\n");
                return 1;
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        const UINT iw = i == 5 ? w / 2 : w, ih = i == 5 ? h / 2 : h;
        for (UINT y = 0; y < ih; ++y)
            for (UINT x = 0; x < iw; ++x) {
                UINT u = i == 5 ? 200 + x / 2 : x, v = i == 5 ? 150 + y / 2 : y;  // outer_l pixel shown here
                uint32_t r = u * 255 / w, g = v * 255 / h, b = (i == 1 || i == 3) ? 200 : 40;
                if (i == 3 && x >= 500 && x < 516 && y >= 300 && y < 316) r = (r + 128) % 256;
                px[(size_t)y * iw + x] = i == 4 ? (r << 2) | (g << 12) | (b << 22) | (3u << 30)
                                                : r | g << 8 | b << 16 | 0xFF000000u;
            }
        D3D11_TEXTURE2D_DESC d{};
        d.Width = iw;
        d.Height = ih;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = i == 4 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_TYPELESS;
        d.SampleDesc.Count = 1;
        d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA init{px.data(), iw * 4, 0};
        if (FAILED(dev->CreateTexture2D(&d, &init, &tex[i]))) {
            printf("Cannot create the shot textures\n");
            return 1;
        }
    }
    double t0 = qvfr::NowMs();
    bool begun = qvfr::shots::Begin(names, kImages, pairs, 4);
    bool again = qvfr::shots::Begin(names, kImages, pairs, 4);  // refused while the first shot is active
    int grabbed = 0;
    for (int i = 0; i < kImages; ++i) grabbed += qvfr::shots::Grab(ctx, tex[i], i);
    qvfr::shots::End(ctx);
    // The worker writes the files; Begin succeeds again once it is done.
    bool done = false;
    while (!done && qvfr::NowMs() - t0 < 30000) {
        Sleep(100);
        if (qvfr::shots::Begin(names, 0, nullptr, 0)) {
            qvfr::shots::Cancel();
            done = true;
        }
    }
    for (ID3D11Texture2D* t : tex) t->Release();
    ctx->Release();
    dev->Release();

    fs::path newest;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(home / L"shots", ec))
        if (e.is_directory() && (newest.empty() || e.path().filename() > newest.filename())) newest = e.path();
    int pngs = 0, halfSize = 0;
    for (int i = 0; i < kImages; ++i) {
        uint32_t pw = 0, ph = 0;
        if (PngSize(newest / (std::string(names[i]) + ".png"), pw, ph)) {
            ++pngs;
            halfSize += i >= 6 ? pw == 32 && ph == 32 : i == 5 ? pw == w / 4 && ph == h / 4 : pw == w / 2 && ph == h / 2;
        }
    }
    std::ifstream in(NewestLog(home / L"logs"));
    std::string line;
    bool same = false, differs = false, parallax = false, window = false, hdr16 = false, hdr11 = false;
    while (std::getline(in, line)) {
        hdr16 |= line.find("hdr16 (64 x 64 fmt 10): values 0..1, NaN or infinity in 1 pixels") != std::string::npos;
        hdr11 |= line.find("hdr11 (64 x 64 fmt 26): values 1..1, NaN or infinity in 0 pixels") != std::string::npos;
        window |= line.find("window vs outer L: mean abs difference 0.000 (0..255), max 0,") != std::string::npos &&
                  line.find("256.0 x 192.0 pixels at (200.0, 150.0)") != std::string::npos;
        same |= line.find("focus L vs outer L: mean abs difference 0.000 (0..255), max 0,") != std::string::npos;
        differs |= line.find("focus R vs outer R:") != std::string::npos &&
                   line.find("> 32: 0.033%") != std::string::npos;  // 256 of 786432 pixels
        parallax |= line.find("outer L vs outer R:") != std::string::npos &&
                    line.find("max 0,") == std::string::npos;
    }
    bool ok = begun && !again && grabbed == kImages && done && pngs == kImages && halfSize == kImages && same &&
              differs && parallax && window && hdr16 && hdr11;
    printf("Shots check %s: begin %d, second begin refused %d, grabbed %d, writer done %d (%.1f s), PNG %d (half size %d) in "
           "%ls, diffs: identical %d, block %d, different %d, focus window %d, HDR values RGBA16F %d R11G11B10 %d\n",
           ok ? "OK" : "FAILED", begun, !again, grabbed, done, (qvfr::NowMs() - t0) / 1000.0, pngs, halfSize,
           newest.c_str(), same, differs, parallax, window, hdr16, hdr11);
    return ok ? 0 : 6;
}

// The composition module on its own: a constant outer and a patterned focus (without shader resource binding, so the
// copy path runs) into a 1024 x 768 grid, both eyes, twice. Checks the pixels by region (focus 1:1 inside the blend
// band, outer in the margin and outside the window, between the two in the feather band) and that the pipeline
// state set before the draw is back afterwards.
int RunCompose() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    // With the debug layer (Graphics Tools) when installed: its warnings (binding hazards, state) fail the check.
    bool debugLayer = SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_DEBUG,
                                                  nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);
    // Pre-M7 edge curve first (margin, then feather from 0 to full focus); the mbucchia curve and a focus texture
    // larger than its window below.
    qvfr::compose::Layout l;
    l.gridW = 1024;
    l.gridH = 768;
    l.outerW = l.focusTexW = l.focusW = 512;
    l.outerH = l.focusTexH = l.focusH = 384;
    l.x0 = 300;
    l.y0 = 200;
    l.marginPx = 8;
    l.legacyEdge = true;
    l.featherPx = 32;
    const uint32_t outerPx = 0xFF967D32u;  // r 50, g 125, b 150
    const uint32_t flatPx = 0xFFC04020u;   // the supersampled focus: one color, so bilinear reads it exactly
    auto focusPx = [](uint32_t x, uint32_t y) {
        return ((x * 7) & 255) | ((y * 13) & 255) << 8 | (((x ^ y) * 5) & 255) << 16 | 0xFF000000u;
    };
    std::vector<uint32_t> px;
    auto make = [&](uint32_t w, uint32_t h, auto pixel, UINT bind, ID3D11Texture2D** t) {
        px.assign((size_t)w * h, 0);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) px[(size_t)y * w + x] = pixel(x, y);
        D3D11_TEXTURE2D_DESC d{};
        d.Width = w;
        d.Height = h;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        d.SampleDesc.Count = 1;
        d.BindFlags = bind;
        D3D11_SUBRESOURCE_DATA init{px.data(), w * 4, 0};
        return SUCCEEDED(dev->CreateTexture2D(&d, &init, t));
    };
    auto plain = [&](uint32_t, uint32_t) { return outerPx; };
    auto flat = [&](uint32_t, uint32_t) { return flatPx; };
    ID3D11Texture2D *outer = nullptr, *focus = nullptr, *focusBig = nullptr, *dummy = nullptr;
    ID3D11RenderTargetView* dummyRtv = nullptr;
    const UINT rtSrv = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    bool made = make(512, 384, plain, rtSrv, &outer) && make(512, 384, focusPx, D3D11_BIND_RENDER_TARGET, &focus) &&
                make(768, 576, flat, rtSrv, &focusBig) && make(512, 384, plain, D3D11_BIND_RENDER_TARGET, &dummy);
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    made = made && SUCCEEDED(dev->CreateRenderTargetView(dummy, &rv, &dummyRtv));
    if (!made) {
        printf("Cannot create the composition test textures\n");
        return 1;
    }
    // The "game" state: one render target, a viewport and a triangle strip.
    const D3D11_VIEWPORT gameVp = {1, 2, 300, 200, 0, 1};
    ctx->OMSetRenderTargets(1, &dummyRtv, nullptr);
    ctx->RSSetViewports(1, &gameVp);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ID3D11Texture2D* out[2] = {};
    for (int round = 0; round < 2; ++round)
        for (int e = 0; e < 2; ++e) out[e] = qvfr::compose::Draw(ctx, e, outer, focus, l);
    ID3D11RenderTargetView* rtvAfter = nullptr;
    ctx->OMGetRenderTargets(1, &rtvAfter, nullptr);
    D3D11_VIEWPORT vpAfter{};
    UINT nvp = 1;
    ctx->RSGetViewports(&nvp, &vpAfter);
    D3D11_PRIMITIVE_TOPOLOGY topoAfter{};
    ctx->IAGetPrimitiveTopology(&topoAfter);
    bool stateOk = rtvAfter == dummyRtv && nvp == 1 && !memcmp(&vpAfter, &gameVp, sizeof(gameVp)) &&
                   topoAfter == D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    if (rtvAfter) rtvAfter->Release();

    // Read back the left eye and check each pixel by its zone (outer, blend band, focus shown exactly); then the same
    // with the window in two corners of the grid (M5: the window follows the gaze up to the frustum edge), with the
    // mbucchia curve, and with a focus texture larger than its window (read bilinearly).
    uint64_t inner = 0, innerBad = 0, outside = 0, outsideBad = 0, band = 0, bandBad = 0;
    bool readOk = out[0] && out[1] && out[0] != out[1];
    D3D11_TEXTURE2D_DESC od{};
    if (readOk) out[0]->GetDesc(&od);
    // 0 = outer, 1 = blend band, 2 = focus; pixel center (qx, qy) in window pixels.
    auto zone = [](const qvfr::compose::Layout& w, float qx, float qy) {
        const float m = w.marginPx;
        if (w.legacyEdge) {
            const float d = std::min(std::min(qx, w.focusW - qx), std::min(qy, w.focusH - qy));
            return d >= m + w.featherPx ? 2 : d <= m ? 0 : 1;
        }
        const float ux = (qx - m) / (w.focusW - 2 * m), uy = (qy - m) / (w.focusH - 2 * m);
        if (ux <= 0 || ux >= 1 || uy <= 0 || uy >= 1) return 0;
        const float a = w.smoothen, eps = 1e-4f;
        if (a <= 0) return 2;
        return ux >= a + eps && ux <= 1 - a - eps && uy >= a + eps && uy <= 1 - a - eps ? 2 : 1;
    };
    auto checkPixels = [&](ID3D11Texture2D* tex, const qvfr::compose::Layout& w, bool bigFocus) {
        if (!tex) return false;
        D3D11_TEXTURE2D_DESC s = od;
        s.Usage = D3D11_USAGE_STAGING;
        s.BindFlags = 0;
        s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        D3D11_MAPPED_SUBRESOURCE m{};
        bool mapped = false;
        if (SUCCEEDED(dev->CreateTexture2D(&s, nullptr, &staging))) {
            ctx->CopyResource(staging, tex);
            if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
                mapped = true;
                auto near1 = [](uint32_t a, uint32_t b) {
                    for (int c = 0; c < 24; c += 8)
                        if (std::abs((int)((a >> c) & 255) - (int)((b >> c) & 255)) > 1) return false;
                    return true;
                };
                auto between = [](uint32_t v, uint32_t a, uint32_t b) {
                    for (int c = 0; c < 24; c += 8) {
                        int x = (v >> c) & 255, p = (a >> c) & 255, q = (b >> c) & 255;
                        if (x < std::min(p, q) - 1 || x > std::max(p, q) + 1) return false;
                    }
                    return true;
                };
                for (uint32_t y = 0; y < od.Height; ++y) {
                    const uint32_t* row = (const uint32_t*)((const uint8_t*)m.pData + (size_t)y * m.RowPitch);
                    for (uint32_t x = 0; x < od.Width; ++x) {
                        const uint32_t v = row[x];
                        const float qx = x + 0.5f - w.x0, qy = y + 0.5f - w.y0;
                        const int z = zone(w, qx, qy);
                        const uint32_t f = bigFocus ? flatPx : z ? focusPx(x - w.x0, y - w.y0) : 0;
                        if (z == 2) {
                            ++inner;
                            innerBad += bigFocus ? !near1(v, f) : v != f;
                        } else if (z == 0) {
                            ++outside;
                            outsideBad += !near1(v, outerPx);
                        } else {
                            ++band;
                            bandBad += !between(v, outerPx, f);
                        }
                    }
                }
                ctx->Unmap(staging, 0);
            }
            staging->Release();
        }
        return mapped;
    };
    int cases = 0;
    if (readOk) {
        readOk = checkPixels(out[0], l, false);
        ++cases;
        for (auto [x0, y0] : {std::pair<uint32_t, uint32_t>{0, 0}, {l.gridW - l.focusW, l.gridH - l.focusH}}) {
            qvfr::compose::Layout lc = l;
            lc.x0 = x0;
            lc.y0 = y0;
            readOk &= checkPixels(qvfr::compose::Draw(ctx, 0, outer, focus, lc), lc, false);
            ++cases;
        }
        // mbucchia's curve (smoothen_focus_view_edges 0.2, then a hard edge without a margin), then the focus texture
        // 1.5x its window.
        qvfr::compose::Layout lm = l;
        lm.legacyEdge = false;
        lm.featherPx = 0;
        lm.smoothen = 0.2f;
        readOk &= checkPixels(qvfr::compose::Draw(ctx, 0, outer, focus, lm), lm, false);
        qvfr::compose::Layout lh = lm;
        lh.smoothen = 0;
        lh.marginPx = 0;  // the default since M7.2
        lh.x0 = l.gridW - l.focusW;
        lh.y0 = 0;
        readOk &= checkPixels(qvfr::compose::Draw(ctx, 0, outer, focus, lh), lh, false);
        qvfr::compose::Layout lb = lm;
        lb.focusTexW = 768;
        lb.focusTexH = 576;
        readOk &= checkPixels(qvfr::compose::Draw(ctx, 0, outer, focusBig, lb), lb, true);
        // A focus texture of a size other than the layout's is refused.
        readOk &= qvfr::compose::Draw(ctx, 0, outer, focus, lb) == nullptr;
        cases += 4;
    }
    // Debug overlay: the gaze dot and the outlines of the margin and sharp edges, the rest unchanged.
    bool overlayOk = false;
    {
        qvfr::compose::Layout lo = l;
        lo.overlay = true;
        lo.gazeMode = 1;
        lo.gazeX = 100.5f;
        lo.gazeY = 100.5f;
        ID3D11Texture2D* o = qvfr::compose::Draw(ctx, 0, outer, focus, lo);
        D3D11_TEXTURE2D_DESC s = od;
        s.Usage = D3D11_USAGE_STAGING;
        s.BindFlags = 0;
        s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        D3D11_MAPPED_SUBRESOURCE m{};
        if (o && SUCCEEDED(dev->CreateTexture2D(&s, nullptr, &staging))) {
            ctx->CopyResource(staging, o);
            if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
                auto at = [&](uint32_t x, uint32_t y) {
                    return ((const uint32_t*)((const uint8_t*)m.pData + (size_t)y * m.RowPitch))[x];
                };
                auto ch = [](uint32_t v, int c) { return (int)((v >> (8 * c)) & 255); };
                const uint32_t dot = at(100, 100), gap = at(120, 100), edge = at(l.x0 + (uint32_t)l.marginPx, l.y0 + 100);
                const uint32_t sharp = at(l.x0 + (uint32_t)(l.marginPx + l.featherPx), l.y0 + 150);
                // White gaze dot, red margin edge (#ff5a4a as in the settings GUI), green sharp edge.
                overlayOk = ch(dot, 0) > 200 && ch(dot, 1) > 200 && ch(dot, 2) > 200 && gap == outerPx &&
                            ch(edge, 0) > 200 && ch(edge, 1) > 60 && ch(edge, 1) < 120 && ch(edge, 2) > 40 &&
                            ch(edge, 2) < 110 && ch(sharp, 0) < 60 && ch(sharp, 1) > 200 && ch(sharp, 2) < 160;
                printf("Overlay: gaze dot %08X, between dot and ring %08X, margin edge %08X, sharp edge %08X\n", dot,
                       gap, edge, sharp);
                ctx->Unmap(staging, 0);
            }
            staging->Release();
        }
    }
    qvfr::compose::Release();
    int warnings = 0;
    if (info) {
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> buf(size);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (FAILED(info->GetMessage(i, msg, &size)) || msg->Severity > D3D11_MESSAGE_SEVERITY_WARNING) continue;
            ++warnings;
            printf("D3D11 debug layer: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
        }
        info->Release();
    }
    for (IUnknown* u : {(IUnknown*)outer, (IUnknown*)focus, (IUnknown*)focusBig, (IUnknown*)dummy, (IUnknown*)dummyRtv})
        u->Release();
    ctx->Release();
    dev->Release();
    bool ok = readOk && stateOk && od.Width == l.gridW && od.Height == l.gridH &&
              od.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS && inner && outside && band && !innerBad && !outsideBad &&
              !bandBad && !warnings && overlayOk;
    printf("Compose check %s: outputs %p/%p (%u x %u fmt %d), state restored %d, %d layouts, pixels wrong: focus %llu of "
           "%llu, outer %llu of %llu, blend band %llu of %llu; overlay %s; debug layer %s, %d warnings\n",
           ok ? "OK" : "FAILED", (void*)out[0], (void*)out[1], od.Width, od.Height, (int)od.Format, stateOk, cases,
           (unsigned long long)innerBad, (unsigned long long)inner, (unsigned long long)outsideBad,
           (unsigned long long)outside, (unsigned long long)bandBad, (unsigned long long)band,
           overlayOk ? "ok" : "WRONG", debugLayer ? "on" : "not installed", warnings);
    return ok ? 0 : 7;
}

// Focus window placement for the gaze (M5) on the Dream Air frusta and grid from the game log (2026-11-02): the fixed
// window at the optical axis as in the log, NDC corners to the frustum tangents, the window clamped into the grid
// for a gaze at and beyond the frustum edge, and one gaze direction giving the same window tangents in both eyes.
int RunGeometry() {
    qvfr::focus::Geometry g;
    g.gridW = 6048;
    g.gridH = 5072;
    g.focusW = 3024;
    g.focusH = 2536;
    const float fr[2][4] = {{-1.37856f, 1.15212f, -1.05537f, 1.05537f}, {-1.15212f, 1.37856f, -1.05537f, 1.05537f}};
    for (int i = 0; i < 2; ++i) {
        qvfr::focus::Eye& e = g.eye[i];
        e.l = fr[i][0];
        e.r = fr[i][1];
        e.t = fr[i][2];
        e.b = fr[i][3];
    }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            ++bad;
            printf("Geometry: WRONG %s\n", what);
        }
    };
    auto same = [](float a, float b) { return std::fabs(a - b) <= 1e-5f; };
    // Window at the given center; its tangents must be its grid pixels and lie inside the frustum.
    auto place = [&](int i, float cx, float cy) {
        qvfr::focus::Eye e = g.eye[i];
        qvfr::focus::Place(g, e, cx, cy);
        const float px = (e.r - e.l) / g.gridW, py = (e.b - e.t) / g.gridH;
        expect(same(e.fr - e.fl, g.focusW * px) && same(e.fb - e.ft, g.focusH * py), "window size in tangents");
        expect(e.fl >= e.l - 1e-5f && e.fr <= e.r + 1e-5f && e.ft >= e.t - 1e-5f && e.fb <= e.b + 1e-5f,
               "window inside the frustum");
        expect(same(e.fl, e.l + e.x0 * px) && same(e.fb, e.b - e.y0 * py), "window edges on grid pixels");
        return e;
    };
    qvfr::focus::Eye a = place(0, 0, 0), b = place(1, 0, 0);
    expect(a.x0 == 1783 && a.y0 == 1268 && b.x0 == 1241 && b.y0 == 1268, "fixed window as in the game log");
    float cx, cy;
    qvfr::focus::NdcToTangent(g.eye[0], -1, -1, cx, cy);
    expect(same(cx, g.eye[0].l) && same(cy, g.eye[0].t), "NDC (-1, -1) = tangents (l, t)");
    qvfr::focus::NdcToTangent(g.eye[0], 1, 1, cx, cy);
    expect(same(cx, g.eye[0].r) && same(cy, g.eye[0].b), "NDC (1, 1) = tangents (r, b)");
    qvfr::focus::NdcToTangent(g.eye[0], 0, 0, cx, cy);
    a = place(0, cx, cy);
    expect(a.x0 == (g.gridW - g.focusW) / 2 && a.y0 == (g.gridH - g.focusH) / 2, "NDC (0, 0) = window in the grid center");
    qvfr::focus::NdcToTangent(g.eye[0], -1, 1, cx, cy);
    a = place(0, cx, cy);
    expect(a.x0 == 0 && a.y0 == 0, "gaze at the upper left frustum corner: window in the grid corner");
    qvfr::focus::NdcToTangent(g.eye[0], 1.5f, -2, cx, cy);
    a = place(0, cx, cy);
    expect(a.x0 == g.gridW - g.focusW && a.y0 == g.gridH - g.focusH, "gaze beyond the frustum: window clamped");
    // One direction (tangents 0.3, -0.2) seen through both eyes' projections, as GetEyeTrackedFoveationCenter gives it.
    float ndc[2][2];
    for (int i = 0; i < 2; ++i) {
        const qvfr::focus::Eye& e = g.eye[i];
        ndc[i][0] = (0.3f - e.l) / (e.r - e.l) * 2 - 1;
        ndc[i][1] = (-0.2f - e.t) / (e.b - e.t) * 2 - 1;
    }
    qvfr::focus::NdcToTangent(g.eye[0], ndc[0][0], ndc[0][1], cx, cy);
    a = place(0, cx, cy);
    qvfr::focus::NdcToTangent(g.eye[1], ndc[1][0], ndc[1][1], cx, cy);
    b = place(1, cx, cy);
    const float px = (g.eye[0].r - g.eye[0].l) / g.gridW;
    expect(std::fabs(a.fl - b.fl) <= px && std::fabs(a.fb - b.fb) <= px, "one gaze direction: same window in both eyes");
    printf("Geometry check %s: %d wrong; gaze (0.3, -0.2): left window (%u, %u) tangents l %.5f b %.5f, right (%u, %u) "
           "l %.5f b %.5f\n",
           bad ? "FAILED" : "OK", bad, a.x0, a.y0, a.fl, a.fb, b.x0, b.y0, b.fl, b.fb);
    return bad ? 8 : 0;
}

// Focus culling math (M9.3) on a synthetic engine frustum block: a rotated eye camera, a symmetric FOV block in a
// shuffled plane and corner order (the engine's order is learned in the game), the eye's asymmetric projection and a
// focus window. Learn must find the sides, Build must keep the window (and the margin) and drop the rest of the eye.
int RunFocusCull() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            ++bad;
            printf("Focus cull: WRONG %s\n", what);
        }
    };
    using V3 = std::array<double, 3>;
    auto norm = [](V3 v) {
        double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return V3{v[0] / l, v[1] / l, v[2] / l};
    };
    auto cross = [](V3 a, V3 b) { return V3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; };
    // Camera: forward, up and right in world space, position e. View (row vectors): rows of the rotation are the
    // world axes' view coordinates, so V[i][j] = axis_j[i]; row 3 = -e * R.
    const V3 fwd = norm({0.45, -0.18, 0.87});
    const V3 right = norm(cross({0.1, 1, 0.05}, fwd));
    const V3 up = cross(fwd, right);
    const V3 eye = {12.3, 1.2, -45.6};
    const V3 axis[3] = {right, up, fwd};
    float view[16] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) view[i * 4 + j] = (float)axis[j][i];
    for (int j = 0; j < 3; ++j) view[12 + j] = (float)-(eye[0] * axis[j][0] + eye[1] * axis[j][1] + eye[2] * axis[j][2]);
    view[15] = 1;
    const double nearZ = 0.015, farZ = 1000;
    // Projection for view tangents x/z in [l, r], y/z in [t, b] (the engine's eye data convention).
    auto proj = [&](double l, double r, double t, double b, float* m) {
        memset(m, 0, 64);
        m[0] = (float)(2 / (r - l));
        m[5] = (float)(2 / (b - t));
        m[8] = (float)(-(r + l) / (r - l));
        m[9] = (float)(-(b + t) / (b - t));
        m[10] = (float)(farZ / (farZ - nearZ));
        m[11] = 1;
        m[14] = (float)(-nearZ * farZ / (farZ - nearZ));
    };
    // World point from view tangents at depth z.
    auto at = [&](double tx, double ty, double z) {
        V3 p;
        for (int c = 0; c < 3; ++c) p[c] = eye[c] + tx * z * right[c] + ty * z * up[c] + z * fwd[c];
        return p;
    };
    struct P4 {
        double n[3], d;
    };
    // Inward plane through the eye and the edge x/z = tan (sign +1: inside is x/z >= tan) or y/z.
    auto side = [&](int ax, double tan, double sign) {
        const V3& a = axis[ax];
        V3 n;
        for (int c = 0; c < 3; ++c) n[c] = sign * (a[c] - tan * fwd[c]);
        n = norm(n);
        P4 p{{n[0], n[1], n[2]}, -(n[0] * eye[0] + n[1] * eye[1] + n[2] * eye[2])};
        return p;
    };
    const double hx = 1.45, hy = 1.12;  // the engine's symmetric FOV block, a superset of the eye
    P4 planes[6];
    // Shuffled order: right, left, near, far, bottom (y/z <= hy), top (y/z >= -hy).
    planes[0] = side(0, hx, -1);
    planes[1] = side(0, -hx, 1);
    {
        V3 pn = at(0, 0, nearZ), pf = at(0, 0, farZ);
        planes[2] = {{fwd[0], fwd[1], fwd[2]}, -(fwd[0] * pn[0] + fwd[1] * pn[1] + fwd[2] * pn[2])};
        planes[3] = {{-fwd[0], -fwd[1], -fwd[2]}, fwd[0] * pf[0] + fwd[1] * pf[1] + fwd[2] * pf[2]};
    }
    planes[4] = side(1, hy, -1);
    planes[5] = side(1, -hy, 1);
    alignas(16) uint8_t block[qvfr::focus_cull::kBlockSize] = {};
    float* f = (float*)block;
    for (int i = 0; i < 6; ++i) {
        for (int c = 0; c < 3; ++c) {
            f[4 * i + c] = (float)planes[i].n[c];
            uint32_t m = planes[i].n[c] >= 0 ? 0xFFFFFFFFu : 0u, nm = ~m;
            memcpy(&f[0x18 + 8 * i + c], &m, 4);
            memcpy(&f[0x18 + 8 * i + 4 + c], &nm, 4);
        }
        f[4 * i + 3] = (float)planes[i].d;
    }
    // Corners in a shuffled order: (x side, y side, depth) per corner.
    const int cx[8] = {1, 0, 0, 1, 1, 0, 1, 0}, cy[8] = {5, 5, 4, 4, 5, 4, 4, 5}, cz[8] = {3, 2, 3, 2, 2, 2, 3, 3};
    for (int j = 0; j < 8; ++j) {
        const double tx = cx[j] == 0 ? hx : -hx, ty = cy[j] == 4 ? hy : -hy, z = cz[j] == 2 ? nearZ : farZ;
        V3 p = at(tx, ty, z);
        for (int c = 0; c < 3; ++c) f[0x48 + 4 * j + c] = (float)p[c];
        f[0x48 + 4 * j + 3] = 1;
    }
    float eyeProj[16], focusProj[16];
    const double l = -1.37856, r = 1.15212, t = -1.05537, b = 1.05537;  // Dream Air left eye
    proj(l, r, t, b, eyeProj);
    const double fl = -0.42, fr = 0.21, ft = -0.33, fb = 0.27;  // focus window (view tangents)
    proj(fl, fr, ft, fb, focusProj);
    qvfr::focus_cull::Layout lay;
    char why[256] = "";
    const bool learned = qvfr::focus_cull::Learn(block, view, eyeProj, lay, why, sizeof(why));
    expect(learned, why);
    if (!learned) {
        printf("Focus cull check FAILED: %d wrong\n", bad);
        return 9;
    }
    // Edge order left, right, bottom, top: x/z >= l, x/z <= r, y/z >= t, y/z <= b.
    expect(lay.side[0] == 1 && lay.side[1] == 0 && lay.side[2] == 5 && lay.side[3] == 4, "side planes found");
    expect(lay.maskSet == 0xFFFFFFFFu && lay.maskClear == 0, "mask bits");
    alignas(16) uint8_t out[qvfr::focus_cull::kBlockSize];
    const float margin = 0.05f;
    qvfr::focus_cull::Build(block, lay, view, focusProj, margin, out);
    const float* o = (const float*)out;
    expect(!memcmp(o + 8, f + 8, 32), "near and far kept");
    auto inside = [&](const V3& p, int& plane) {
        for (int i = 0; i < 6; ++i)
            if (o[4 * i] * p[0] + o[4 * i + 1] * p[1] + o[4 * i + 2] * p[2] + o[4 * i + 3] < -1e-4) {
                plane = i;
                return false;
            }
        return true;
    };
    // Points by NDC of the focus window: inside up to the margin, outside beyond it.
    auto ndc = [&](double x, double y, double z) {
        const double tx = (fl + fr) / 2 + x * (fr - fl) / 2, ty = (ft + fb) / 2 + y * (fb - ft) / 2;
        return at(tx, ty, z);
    };
    int pl = -1;
    expect(inside(ndc(0, 0, 20), pl), "window center inside");
    for (double z : {0.5, 20.0, 900.0})
        for (int k = 0; k < 4; ++k) {
            const double sx = k == 0 ? -1 : k == 1 ? 1 : 0, sy = k == 2 ? -1 : k == 3 ? 1 : 0;
            expect(inside(ndc(sx * 1.03, sy * 1.03, z), pl), "point inside the margin kept");
            expect(!inside(ndc(sx * 1.1, sy * 1.1, z), pl) && pl == lay.side[k], "point beyond the margin dropped by its edge");
        }
    V3 rest = at(l * 0.9, b * 0.9, 20);  // in the eye, far from the window
    expect(!inside(rest, pl), "rest of the eye dropped");
    // The result is a well-formed block again: corners on their planes, masks by sign.
    qvfr::focus_cull::Layout again;
    expect(qvfr::focus_cull::Learn(out, view, focusProj, again, why, sizeof(why)), why);
    printf("Focus cull check %s: %d wrong; sides L/R/B/T in block planes %d %d %d %d\n", bad ? "FAILED" : "OK", bad,
           lay.side[0], lay.side[1], lay.side[2], lay.side[3]);
    return bad ? 9 : 0;
}

// A fake engine object: bytes with typed fields at the engine's offsets.
struct FakeObj {
    uint64_t q[0x108 / 8] = {};
    template <class T>
    T& at(size_t off) {
        return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(q) + off);
    }
};

// Clones (M7.3, M7.4) on fake engine objects with the engine's layout: texture objects (D3D11 texture at +0x58, views
// after it, u16 size at +0x68, name at +0xf0), a render target object (size at +0x24, view counts at +0x64/+0x66, view
// arrays at +0x78/+0x88, name at +0xb8) and an engine device (render target size per context at +0x9b18, dirty flags
// at +0xe448). Scenario 1, focus smaller than the game: learning sorts written from read-only textures; a focus group
// then swaps the fields (textures, views, sizes), the device sizes and the bound views, scales a viewport set for the
// original, a clear through the render target's array reaches the clone, everything goes back after the group, outer
// groups stay untouched, a write to an original is counted as a leak, and a destroyed object gets its originals back at
// once. Scenario 2, outer smaller: outer gets clones of what it writes, except a texture focus reads but does not write.
int RunClones() {
    namespace cl = qvfr::clones;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    qvfr::hooks::Init();
    cl::InstallD3DHooks(ctx);
    // Fake engine device: context k's render target size at +0x9b18 + k * 0xe60, its dirty flag at +0xe448 + k.
    std::vector<uint8_t> engineDev(0xe448 + 8);
    auto devSize = [&](int k) { return reinterpret_cast<float*>(engineDev.data() + 0x9b18 + k * 0xe60); };
    auto devDirty = [&](int k) -> uint8_t& { return engineDev[0xe448 + k]; };
    const float devInit[4][2] = {{64, 48}, {32, 24}, {100, 100}, {0, 0}};
    for (int k = 0; k < 4; ++k) {
        devSize(k)[0] = devInit[k][0];
        devSize(k)[1] = devInit[k][1];
    }
    // Postprocess object at device+0x17b0: the scene size as two ints at +0x100.
    FakeObj post;
    post.at<int32_t>(0x100) = 64;
    post.at<int32_t>(0x104) = 48;
    *reinterpret_cast<void**>(engineDev.data() + 0x17b0) = &post;
    cl::TestSetDevice(engineDev.data());
    cl::SetSizes(64, 48, 64, 48, 32, 24);
    auto tex = [&](UINT w, UINT h) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = w;
        d.Height = h;
        d.MipLevels = d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        ID3D11Texture2D* t = nullptr;
        dev->CreateTexture2D(&d, nullptr, &t);
        return t;
    };
    auto srv = [&](ID3D11Texture2D* t) {
        ID3D11ShaderResourceView* v = nullptr;
        dev->CreateShaderResourceView(t, nullptr, &v);
        return v;
    };
    auto rtv = [&](ID3D11Texture2D* t) {
        ID3D11RenderTargetView* v = nullptr;
        dev->CreateRenderTargetView(t, nullptr, &v);
        return v;
    };
    auto size = [](ID3D11Texture2D* t) {
        D3D11_TEXTURE2D_DESC d{};
        if (t) t->GetDesc(&d);
        return std::make_pair(d.Width, d.Height);
    };
    auto viewport = [&](float w, float h) {
        D3D11_VIEWPORT v{0, 0, w, h, 0, 1};
        ctx->RSSetViewports(1, &v);
    };
    auto boundViewport = [&]() {
        D3D11_VIEWPORT v{};
        UINT n = 1;
        ctx->RSGetViewports(&n, &v);
        return std::make_pair(v.Width, v.Height);
    };
    // A: eye size, written and read; B: 1/2, written by a copy; C: eye size, only read; D: not eye size, written.
    ID3D11Texture2D *texA = tex(64, 48), *texB = tex(32, 24), *texC = tex(64, 48), *texD = tex(100, 100),
                    *texE = tex(32, 24);
    ID3D11ShaderResourceView *srvA = srv(texA), *srvC = srv(texC);
    ID3D11RenderTargetView *rtvA = rtv(texA), *rtvA2 = rtv(texA), *rtvA3 = rtv(texA), *rtvD = rtv(texD);
    static const char* const names[] = {"Fake A", "Fake B", "Fake C", "Fake D", "Fake RT"};
    FakeObj A, B, C, D, R;
    A.at<void*>(0x58) = texA;
    A.at<void*>(0x70) = srvA;
    A.at<void*>(0x80) = rtvA;
    B.at<void*>(0x58) = texB;
    C.at<void*>(0x58) = texC;
    C.at<void*>(0x70) = srvC;
    D.at<void*>(0x58) = texD;
    FakeObj* objs[] = {&A, &B, &C, &D};
    ID3D11Texture2D* objTex[] = {texA, texB, texC, texD};
    for (int i = 0; i < 4; ++i) {
        objs[i]->at<const char*>(0xf0) = names[i];
        objs[i]->at<uint16_t>(0x68) = (uint16_t)size(objTex[i]).first;
        objs[i]->at<uint16_t>(0x6a) = (uint16_t)size(objTex[i]).second;
    }
    void* arr78[1] = {rtvA2};
    void* arr88[1] = {rtvA3};
    R.at<uint16_t>(0x24) = 64;
    R.at<uint16_t>(0x26) = 48;
    R.at<uint16_t>(0x64) = 0;
    R.at<uint16_t>(0x66) = 1;
    R.at<void**>(0x78) = arr78;
    R.at<void**>(0x88) = arr88;
    R.at<const char*>(0xb8) = names[4];
    for (FakeObj* o : objs) cl::TestRegister(false, o);
    cl::TestRegister(true, &R);
    int bad = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            ++bad;
            printf("Clones: WRONG %s\n", what);
        }
    };
    const float blue[4] = {0, 0, 1, 1}, red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1};
    // Learning: 16 outer and 16 focus groups (outer only clears A).
    for (int f = 0; f < 17; ++f) {
        cl::OnFrameStart(ctx, true);
        if (f == 16) break;
        cl::BeginGroup(ctx, cl::Outer);
        ctx->ClearRenderTargetView(rtvA, green);
        cl::EndGroup(ctx);
        cl::BeginGroup(ctx, cl::Focus);
        ctx->ClearRenderTargetView(A.at<ID3D11RenderTargetView*>(0x80), blue);
        ID3D11ShaderResourceView* s[2] = {srvA, srvC};
        ctx->PSSetShaderResources(0, 2, s);
        ctx->CopyResource(texB, texE);
        ctx->ClearRenderTargetView(rtvD, blue);
        cl::EndGroup(ctx);
    }
    cl::TestState st = cl::TestGet();
    expect(st.phase == 2, "phase after learning (ready)");
    expect(st.textures == 3 && st.renderTargets == 1, "listed: A, B, C and the render target (not D)");
    expect(st.clones[1] == 2 && st.clones[0] == 0, "focus clones for A and B only (C is read only), none for outer");
    expect(st.entries == 9, "entries: A +0x58 +0x68 +0x70 +0x80, B +0x58 +0x68, the render target's two arrays and size");
    // Outer at the game size: its groups keep the originals.
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Outer);
    expect(!cl::TestGet().swapped && A.at<void*>(0x58) == texA, "no swap in an outer group");
    cl::EndGroup(ctx);
    // A focus group with clones: fields, sizes and the bound views swapped, the viewport for the original scaled, a
    // clear through the array reaches the clone.
    ID3D11ShaderResourceView* s0[2] = {srvA, srvC};
    ctx->PSSetShaderResources(0, 2, s0);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    viewport(64, 48);
    cl::BeginGroup(ctx, cl::Focus);
    ID3D11Texture2D* cloneA = A.at<ID3D11Texture2D*>(0x58);
    ID3D11Texture2D* cloneB = B.at<ID3D11Texture2D*>(0x58);
    expect(cl::TestGet().swapped, "swapped inside the focus group");
    expect(cloneA != texA && A.at<void*>(0x70) != srvA && A.at<void*>(0x80) != rtvA, "A's texture and views swapped");
    expect(size(cloneA) == std::make_pair(32u, 24u) && size(cloneB) == std::make_pair(16u, 12u),
           "clones in focus's size: A 32 x 24, B (1/2) 16 x 12");
    expect(A.at<uint16_t>(0x68) == 32 && A.at<uint16_t>(0x6a) == 24 && B.at<uint16_t>(0x68) == 16 &&
               B.at<uint16_t>(0x6a) == 12,
           "texture size fields swapped");
    expect(R.at<uint16_t>(0x24) == 32 && R.at<uint16_t>(0x26) == 24, "render target size swapped");
    expect(C.at<void*>(0x58) == texC && C.at<void*>(0x70) == srvC && D.at<void*>(0x58) == texD &&
               C.at<uint16_t>(0x68) == 64,
           "C and D untouched");
    expect(arr78[0] != rtvA2 && arr88[0] != rtvA3, "render target arrays swapped");
    expect(devSize(0)[0] == 32 && devSize(0)[1] == 24 && devDirty(0) == 1 && devSize(1)[0] == 16 &&
               devSize(1)[1] == 12 && devDirty(1) == 1 && devSize(2)[0] == 100 && devDirty(2) == 0,
           "device render target sizes: game size and 1/2 swapped, others untouched");
    expect(post.at<int32_t>(0x100) == 32 && post.at<int32_t>(0x104) == 24, "postprocess scene size swapped");
    ID3D11ShaderResourceView* bound[2] = {};
    ctx->PSGetShaderResources(0, 2, bound);
    expect(bound[0] == A.at<ID3D11ShaderResourceView*>(0x70) && bound[1] == srvC, "bound SRV translated to the clone");
    for (auto* b : bound)
        if (b) b->Release();
    expect(boundViewport() == std::make_pair(64.0f, 48.0f), "viewport kept without a clone target bound");
    // The engine binds the render target's view (D3D11 unbinds A's clone SRV) and sets the original's viewport again.
    ctx->OMSetRenderTargets(1, (ID3D11RenderTargetView* const*)arr78, nullptr);
    cl::OnOMSetRenderTargets(ctx, 1, (ID3D11RenderTargetView* const*)arr78, nullptr, nullptr);
    viewport(64, 48);
    expect(boundViewport() == std::make_pair(32.0f, 24.0f), "viewport set for the original scaled to the clone");
    viewport(16, 12);
    expect(boundViewport() == std::make_pair(16.0f, 12.0f), "viewport that fits the clone kept");
    viewport(64, 48);
    ctx->ClearRenderTargetView((ID3D11RenderTargetView*)arr78[0], red);
    // A bind of targets of different sizes (the clone and D) is logged as a size mismatch.
    ID3D11RenderTargetView* mixedRtv[2] = {(ID3D11RenderTargetView*)arr78[0], rtvD};
    ctx->OMSetRenderTargetsAndUnorderedAccessViews(2, mixedRtv, nullptr, 0, D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr,
                                                   nullptr);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    cl::EndGroup(ctx);
    expect(!cl::TestGet().swapped && A.at<void*>(0x58) == texA && A.at<void*>(0x70) == srvA && A.at<void*>(0x80) == rtvA &&
               B.at<void*>(0x58) == texB && arr78[0] == rtvA2 && arr88[0] == rtvA3,
           "originals back after the group");
    expect(A.at<uint16_t>(0x68) == 64 && A.at<uint16_t>(0x6a) == 48 && R.at<uint16_t>(0x24) == 64 &&
               R.at<uint16_t>(0x26) == 48,
           "sizes back after the group");
    expect(devSize(0)[0] == 64 && devSize(0)[1] == 48 && devSize(1)[0] == 32 && devSize(1)[1] == 24,
           "device sizes back after the group");
    expect(post.at<int32_t>(0x100) == 64 && post.at<int32_t>(0x104) == 48,
           "postprocess scene size back after the group");
    expect(boundViewport() == std::make_pair(64.0f, 48.0f), "viewport as the engine asked after the group");
    st = cl::TestGet();
    expect(st.viewportsFixed == 3 && st.viewportMismatches == 3,
           "three viewports for the original scaled (at the bind and two RSSetViewports)");
    expect(cl::Original(cloneA) == texA && cl::Original(texA) == texA, "Original() of a clone");
    // Pixels: the original keeps the learning clear (blue), the clone got red.
    auto pixel = [&](ID3D11Texture2D* t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* s = nullptr;
        uint32_t px = 0;
        if (SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &s))) {
            ctx->CopyResource(s, t);
            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(ctx->Map(s, 0, D3D11_MAP_READ, 0, &m))) {
                px = *(const uint32_t*)m.pData;
                ctx->Unmap(s, 0);
            }
            s->Release();
        }
        return px;
    };
    const uint32_t pxA = pixel(texA), pxClone = pixel(cloneA);
    expect(pxA == 0xFFFF0000u && pxClone == 0xFF0000FFu, "original blue, clone red");
    expect(pixel(cloneB) == 0, "a clone nothing wrote yet is zero");
    // Bind fix: inside a focus group a write to an original through its field view or through a view the engine keeps
    // elsewhere (X) goes to the clone, no leak; the bound SRVs go back after the group.
    ID3D11RenderTargetView* rtvX = rtv(texA);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    ctx->PSSetShaderResources(0, 2, s0);
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Focus);
    ID3D11ShaderResourceView* cloneSrvA = A.at<ID3D11ShaderResourceView*>(0x70);
    ctx->ClearRenderTargetView(rtvA, red);
    ctx->ClearRenderTargetView(rtvX, green);
    ID3D11RenderTargetView* mapped[8] = {};
    ID3D11DepthStencilView* mappedDsv = nullptr;
    const bool mappedOk = cl::MapTargets(ctx, 1, &rtvX, nullptr, mapped, &mappedDsv, nullptr);
    ID3D11Resource* mappedRes = nullptr;
    if (mapped[0]) mapped[0]->GetResource(&mappedRes);
    expect(mappedOk && mappedRes == cloneA && !mappedDsv, "a bind of a view kept outside the fields goes to the clone");
    if (mappedRes) mappedRes->Release();
    cl::EndGroup(ctx);
    st = cl::TestGet();
    expect(st.leaks == 0 && st.cloneUses >= 1, "no leak, clone uses counted");
    expect(st.bindsToClone == 3 && st.lazyViews == 1, "bind fix: three to the clone, one view made on the clone");
    expect(pixel(texA) == 0xFFFF0000u && pixel(cloneA) == 0xFF00FF00u, "both clears reached the clone");
    ctx->PSGetShaderResources(0, 2, bound);
    expect(bound[0] == srvA && bound[1] == srvC, "bound SRV translated back after the focus group");
    for (auto* b : bound)
        if (b) b->Release();
    // In an outer group (no clones in) a clone view bound by a cached pointer goes back to the original.
    cl::BeginGroup(ctx, cl::Outer);
    ctx->PSSetShaderResources(0, 1, &cloneSrvA);
    ID3D11ShaderResourceView* back = nullptr;
    ctx->PSGetShaderResources(0, 1, &back);
    expect(back == srvA && cl::TestGet().bindsToOriginal == 1, "a clone bound in an outer group goes back to the original");
    if (back) back->Release();
    cl::EndGroup(ctx);
    ctx->PSGetShaderResources(0, 2, bound);
    expect(bound[0] == srvA && bound[1] == srvC, "bound SRV translated back");
    for (auto* b : bound)
        if (b) b->Release();
    // The original's target bound and its viewport set before the group: both go to the clone at the group start.
    ID3D11ShaderResourceView* none[2] = {};
    ctx->PSSetShaderResources(0, 2, none);
    ctx->OMSetRenderTargets(1, &rtvA, nullptr);
    viewport(64, 48);
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Focus);
    ID3D11RenderTargetView* boundRtv = nullptr;
    ctx->OMGetRenderTargets(1, &boundRtv, nullptr);
    expect(boundRtv && boundRtv == A.at<ID3D11RenderTargetView*>(0x80), "bound RTV translated to the clone");
    if (boundRtv) boundRtv->Release();
    expect(boundViewport() == std::make_pair(32.0f, 24.0f), "viewport of the original scaled at the group start");
    cl::EndGroup(ctx);
    expect(boundViewport() == std::make_pair(64.0f, 48.0f), "viewport back after the group");
    ctx->OMGetRenderTargets(1, &boundRtv, nullptr);
    expect(boundRtv == rtvA, "bound RTV translated back");
    if (boundRtv) boundRtv->Release();
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    // Focus R's twins (MVR B9d): A's focus clone gets a twin at the next frame start, with a twin of each focus view of
    // the clone; with eye 1 focus's swap, CloneOf and the bind fix give the twin (B stays shared), with eye 0 the clone.
    cl::WantTwin(texA);
    expect(!cl::HasTwin(texA), "no twin before the frame start");
    cl::OnFrameStart(ctx, true);
    st = cl::TestGet();
    expect(cl::HasTwin(texA) && !cl::HasTwin(texB) && st.twins == 1 && st.twinViews == 5,
           "A's twin with 5 views (SRV and RTV fields, the render target's two arrays, view X)");
    cl::SetFocusEye(1);
    ID3D11Texture2D* twinA = cl::CloneOf(texA, cl::Focus);
    expect(twinA && twinA != cloneA && size(twinA) == size(cloneA) && cl::CloneOf(texB, cl::Focus) == cloneB,
           "eye 1: CloneOf gives A's twin and B's shared clone");
    expect(cl::Original(twinA) == texA, "Original() of a twin");
    expect(twinA && pixel(twinA) == pixel(cloneA), "the twin starts with the clone's contents");
    cl::BeginGroup(ctx, cl::Focus);
    ID3D11ShaderResourceView* twinSrvA = A.at<ID3D11ShaderResourceView*>(0x70);
    expect(A.at<void*>(0x58) == twinA && twinSrvA != cloneSrvA && twinSrvA != srvA && B.at<void*>(0x58) == cloneB &&
               arr78[0] != rtvA2 && A.at<uint16_t>(0x68) == 32,
           "eye 1 swap: A's twin and its views in the fields, B's shared clone");
    ctx->ClearRenderTargetView((ID3D11RenderTargetView*)arr78[0], red);
    ID3D11RenderTargetView* mappedR[8] = {};
    ID3D11DepthStencilView* mappedDsvR = nullptr;
    cl::MapTargets(ctx, 1, &rtvX, nullptr, mappedR, &mappedDsvR, nullptr);
    ID3D11Resource* mappedResR = nullptr;
    if (mappedR[0]) mappedR[0]->GetResource(&mappedResR);
    expect(twinA && mappedResR == twinA, "eye 1: a view kept outside the fields goes to the twin");
    if (mappedResR) mappedResR->Release();
    cl::EndGroup(ctx);
    expect(A.at<void*>(0x58) == texA && A.at<void*>(0x70) == srvA && arr78[0] == rtvA2 && B.at<void*>(0x58) == texB,
           "originals back after the eye 1 group");
    expect(twinA && pixel(twinA) == 0xFF0000FFu && pixel(cloneA) == 0xFF00FF00u,
           "eye 1 wrote the twin (red), the clone kept eye 0's (green)");
    cl::SetFocusEye(0);
    expect(cl::CloneOf(texA, cl::Focus) == cloneA, "eye 0: CloneOf gives the shared clone");
    cl::BeginGroup(ctx, cl::Focus);
    expect(A.at<void*>(0x58) == cloneA && A.at<void*>(0x70) == cloneSrvA, "eye 0 swap: the shared clone");
    cl::EndGroup(ctx);
    cl::BeginGroup(ctx, cl::Outer);
    ctx->PSSetShaderResources(0, 1, &twinSrvA);
    back = nullptr;
    ctx->PSGetShaderResources(0, 1, &back);
    expect(back == srvA, "a twin's view bound in an outer group goes back to the original");
    if (back) back->Release();
    cl::EndGroup(ctx);
    ctx->PSSetShaderResources(0, 2, none);
    // An object destroyed inside a focus group gets its originals back at once; the list is watched again.
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Focus);
    cl::TestDestroy(false, &A);
    expect(A.at<void*>(0x58) == texA && A.at<void*>(0x70) == srvA && A.at<void*>(0x80) == rtvA &&
               A.at<uint16_t>(0x68) == 64,
           "A restored at destroy");
    cl::EndGroup(ctx);
    expect(B.at<void*>(0x58) == texB && arr78[0] == rtvA2 && R.at<uint16_t>(0x24) == 64,
           "the rest restored after the group");
    cl::OnFrameStart(ctx, true);
    st = cl::TestGet();
    expect(st.phase == 1 && st.textures == 2 && st.clones[1] == 0, "list watched again without A");
    // Off: no swap.
    cl::OnFrameStart(ctx, false);
    cl::BeginGroup(ctx, cl::Focus);
    expect(!cl::TestGet().swapped && B.at<void*>(0x58) == texB, "no swap while off");
    cl::EndGroup(ctx);
    const int bad1 = bad;
    const int translated1 = (int)cl::TestGet().translated;
    cl::Shutdown();

    // Scenario 2: outer smaller (32 x 24), focus at the game size. P: written by outer and focus; Q: written by outer,
    // read by focus (kept for outer); S: written by outer only.
    for (int k = 0; k < 4; ++k) {
        devSize(k)[0] = devInit[k][0];
        devSize(k)[1] = devInit[k][1];
        devDirty(k) = 0;
    }
    ID3D11Texture2D *texP = tex(64, 48), *texQ = tex(64, 48), *texS = tex(64, 48);
    ID3D11RenderTargetView *rtvP = rtv(texP), *rtvQ = rtv(texQ), *rtvS = rtv(texS);
    ID3D11ShaderResourceView* srvQ = srv(texQ);
    static const char* const names2[] = {"Fake P", "Fake Q", "Fake S"};
    FakeObj P, Q, S;
    FakeObj* objs2[] = {&P, &Q, &S};
    ID3D11Texture2D* objTex2[] = {texP, texQ, texS};
    void* objView2[] = {rtvP, rtvQ, rtvS};
    for (int i = 0; i < 3; ++i) {
        objs2[i]->at<void*>(0x58) = objTex2[i];
        objs2[i]->at<void*>(0x80) = objView2[i];
        objs2[i]->at<const char*>(0xf0) = names2[i];
        objs2[i]->at<uint16_t>(0x68) = 64;
        objs2[i]->at<uint16_t>(0x6a) = 48;
    }
    Q.at<void*>(0x70) = srvQ;
    cl::SetSizes(64, 48, 32, 24, 64, 48);
    for (FakeObj* o : objs2) cl::TestRegister(false, o);
    for (int f = 0; f < 17; ++f) {
        cl::OnFrameStart(ctx, true);
        if (f == 16) break;
        cl::BeginGroup(ctx, cl::Outer);
        for (ID3D11RenderTargetView* v : {rtvP, rtvQ, rtvS}) ctx->ClearRenderTargetView(v, green);
        cl::EndGroup(ctx);
        cl::BeginGroup(ctx, cl::Focus);
        ctx->ClearRenderTargetView(rtvP, blue);
        ctx->PSSetShaderResources(0, 1, &srvQ);
        cl::EndGroup(ctx);
    }
    st = cl::TestGet();
    expect(st.phase == 2 && st.textures == 3, "outer: ready, three textures listed");
    expect(st.clones[0] == 2 && st.clones[1] == 0, "outer clones for P and S (Q kept: focus reads it), none for focus");
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Outer);
    expect(cl::TestGet().swapped && size(P.at<ID3D11Texture2D*>(0x58)) == std::make_pair(32u, 24u) &&
               size(S.at<ID3D11Texture2D*>(0x58)) == std::make_pair(32u, 24u) && Q.at<void*>(0x58) == texQ,
           "outer group: P and S in 32 x 24 clones, Q the original");
    expect(devSize(0)[0] == 32 && devSize(0)[1] == 24, "outer group: device size swapped");
    cl::EndGroup(ctx);
    expect(P.at<void*>(0x58) == texP && S.at<void*>(0x58) == texS && devSize(0)[0] == 64, "outer: originals back");
    cl::BeginGroup(ctx, cl::Focus);
    expect(!cl::TestGet().swapped && P.at<void*>(0x58) == texP, "focus group at the game size: no swap");
    cl::EndGroup(ctx);
    cl::Shutdown();
    const int bad2 = bad;

    // Scenario 3 (MVR B17): a two-slice array in place of an engine texture. G: eye size, written by outer and focus,
    // its SRV at +0x70 and RTV at +0x80, a second RTV in a render target's array, a view X the engine keeps elsewhere.
    for (int k = 0; k < 4; ++k) {
        devSize(k)[0] = devInit[k][0];
        devSize(k)[1] = devInit[k][1];
        devDirty(k) = 0;
    }
    post.at<int32_t>(0x100) = 64;
    post.at<int32_t>(0x104) = 48;
    cl::SetSizes(64, 48, 64, 48, 32, 24);
    ID3D11Texture2D* texG = tex(64, 48);
    ID3D11ShaderResourceView* srvG = srv(texG);
    ID3D11RenderTargetView *rtvG = rtv(texG), *rtvG2 = rtv(texG), *rtvGX = rtv(texG);
    static const char* const names3[] = {"Fake G", "Fake RT G"};
    FakeObj G, RG;
    G.at<void*>(0x58) = texG;
    G.at<void*>(0x70) = srvG;
    G.at<void*>(0x80) = rtvG;
    G.at<const char*>(0xf0) = names3[0];
    G.at<uint16_t>(0x68) = 64;
    G.at<uint16_t>(0x6a) = 48;
    void* arrG[1] = {rtvG2};
    RG.at<uint16_t>(0x24) = 64;
    RG.at<uint16_t>(0x26) = 48;
    RG.at<uint16_t>(0x66) = 1;
    RG.at<void**>(0x78) = arrG;
    RG.at<const char*>(0xb8) = names3[1];
    cl::TestRegister(false, &G);
    cl::TestRegister(true, &RG);
    for (int f = 0; f < 17; ++f) {
        cl::OnFrameStart(ctx, true);
        if (f == 16) break;
        cl::BeginGroup(ctx, cl::Outer);
        ctx->ClearRenderTargetView(rtvG, green);
        cl::EndGroup(ctx);
        cl::BeginGroup(ctx, cl::Focus);
        ctx->ClearRenderTargetView(G.at<ID3D11RenderTargetView*>(0x80), blue);
        cl::EndGroup(ctx);
    }
    st = cl::TestGet();
    expect(st.phase == 2 && st.clones[1] == 1, "B17: ready with G's focus clone");
    ctx->ClearRenderTargetView(rtvG, red);  // the contents slice 0 starts with
    auto arraySize = [](ID3D11Texture2D* t) {
        D3D11_TEXTURE2D_DESC d{};
        if (t) t->GetDesc(&d);
        return d.ArraySize;
    };
    auto resOf = [](void* v) {
        ID3D11Resource* r = nullptr;
        static_cast<ID3D11View*>(v)->GetResource(&r);
        if (r) r->Release();
        return (void*)r;
    };
    cl::WantArray(texG);
    expect(!cl::IsArray(G.at<ID3D11Resource*>(0x58)), "B17: no array before the frame start");
    cl::OnFrameStart(ctx, true);
    ID3D11Texture2D* arrTex = G.at<ID3D11Texture2D*>(0x58);
    expect(arrTex != texG && cl::IsArray(arrTex) && arraySize(arrTex) == 2 && size(arrTex) == size(texG),
           "B17: G's texture replaced by a two-slice array of its size");
    ID3D11ShaderResourceView* srvArr = G.at<ID3D11ShaderResourceView*>(0x70);
    ID3D11RenderTargetView* rtvArr = G.at<ID3D11RenderTargetView*>(0x80);
    expect(srvArr != srvG && rtvArr != rtvG && arrG[0] != rtvG2 && resOf(srvArr) == arrTex && resOf(rtvArr) == arrTex &&
               resOf(arrG[0]) == arrTex,
           "B17: the field views and the render target's view are views of the array");
    D3D11_RENDER_TARGET_VIEW_DESC rvd{};
    rtvArr->GetDesc(&rvd);
    expect(rvd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D, "B17: the array's views keep the engine's TEXTURE2D");
    expect(pixel(arrTex) == 0xFF0000FFu, "B17: slice 0 starts with the texture's contents (red)");
    st = cl::TestGet();
    ID3D11Texture2D* cloneG = cl::CloneOf(arrTex, cl::Focus);
    expect(st.arrays == 1 && st.phase == 2 && st.clones[1] == 1 && cloneG && arraySize(cloneG) == 1 &&
               size(cloneG) == std::make_pair(32u, 24u),
           "B17: one array, focus's clone made again (single, 32 x 24)");
    // Outside the groups: a clear through view X goes to slice 0 of the array; the old texture keeps red.
    ctx->ClearRenderTargetView(rtvGX, green);
    uint32_t oldPx = 0;
    {
        cl::RawScope raw;
        oldPx = pixel(texG);
    }
    expect(pixel(arrTex) == 0xFF00FF00u && oldPx == 0xFF0000FFu, "B17: a view kept elsewhere clears the array");
    st = cl::TestGet();
    expect(st.arrayLazy == 1 && st.arrayFixes >= 1, "B17: one view made on the array for view X");
    // A bind of the engine's old SRV gets the array's.
    ctx->PSSetShaderResources(0, 1, &srvG);
    ID3D11ShaderResourceView* boundG = nullptr;
    ctx->PSGetShaderResources(0, 1, &boundG);
    expect(boundG == srvArr, "B17: the engine's old SRV bound as the array's");
    if (boundG) boundG->Release();
    ctx->PSSetShaderResources(0, 2, none);
    // CopyResource between the array and a single texture copies slice 0.
    ID3D11Texture2D* texS1 = tex(64, 48);
    const uint64_t copies0 = cl::TestGet().arrayCopies;
    ctx->CopyResource(texS1, arrTex);
    expect(pixel(texS1) == 0xFF00FF00u && cl::TestGet().arrayCopies > copies0, "B17: CopyResource array -> single");
    // A focus group swaps the clone in; after it the array's views are back.
    cl::OnFrameStart(ctx, true);
    cl::BeginGroup(ctx, cl::Focus);
    expect(cl::TestGet().swapped && G.at<void*>(0x58) == cloneG, "B17: focus group with G's clone");
    cl::EndGroup(ctx);
    expect(G.at<void*>(0x58) == arrTex && G.at<void*>(0x70) == srvArr && G.at<void*>(0x80) == rtvArr &&
               arrG[0] != rtvG2,
           "B17: the array and its views back after the focus group");
    // Destroyed objects get the engine's own texture and views back.
    cl::TestDestroy(true, &RG);
    expect(arrG[0] == rtvG2, "B17: the render target's view back at its destroy");
    cl::TestDestroy(false, &G);
    expect(G.at<void*>(0x58) == texG && G.at<void*>(0x70) == srvG && G.at<void*>(0x80) == rtvG,
           "B17: G's texture and views back at its destroy");
    cl::OnFrameStart(ctx, true);
    st = cl::TestGet();
    expect(st.arrays == 0 && !cl::IsArray(arrTex), "B17: no array left");
    cl::Shutdown();
    const int bad3 = bad;

    // Scenario 4 (MVR B12 step 5): a three-slice array whose slices 1 and 2 are focus L's and focus R's clone. H: eye
    // size, written by outer and focus, its RTV at +0x80, a second RTV in a render target's array.
    cl::SetSizes(64, 48, 64, 48, 32, 24);
    ID3D11Texture2D* texH = tex(64, 48);
    ID3D11RenderTargetView *rtvH = rtv(texH), *rtvH2 = rtv(texH);
    static const char* const names4[] = {"Fake H", "Fake RT H"};
    FakeObj H, RH;
    H.at<void*>(0x58) = texH;
    H.at<void*>(0x80) = rtvH;
    H.at<const char*>(0xf0) = names4[0];
    H.at<uint16_t>(0x68) = 64;
    H.at<uint16_t>(0x6a) = 48;
    void* arrH[1] = {rtvH2};
    RH.at<uint16_t>(0x24) = 64;
    RH.at<uint16_t>(0x26) = 48;
    RH.at<uint16_t>(0x66) = 1;
    RH.at<void**>(0x78) = arrH;
    RH.at<const char*>(0xb8) = names4[1];
    cl::TestRegister(false, &H);
    cl::TestRegister(true, &RH);
    for (int f = 0; f < 17; ++f) {
        cl::OnFrameStart(ctx, true);
        if (f == 16) break;
        cl::BeginGroup(ctx, cl::Outer);
        ctx->ClearRenderTargetView(rtvH, green);
        cl::EndGroup(ctx);
        cl::BeginGroup(ctx, cl::Focus);
        ctx->ClearRenderTargetView(H.at<ID3D11RenderTargetView*>(0x80), blue);
        cl::EndGroup(ctx);
    }
    expect(cl::TestGet().phase == 2 && cl::TestGet().clones[1] == 1, "slices: ready with H's focus clone");
    cl::WantArray(texH, 3);
    cl::OnFrameStart(ctx, true);
    ID3D11Texture2D* arr3 = H.at<ID3D11Texture2D*>(0x58);
    expect(arr3 != texH && cl::IsArray(arr3) && arraySize(arr3) == 3 && size(arr3) == size(texH),
           "slices: H's texture replaced by a three-slice array of its size");
    uint32_t sw = 0, shh = 0;
    expect(cl::CloneOf(arr3, cl::Focus) == arr3 && cl::FocusSlice(arr3, 0, &sw, &shh) == 1 && sw == 32 && shh == 24 &&
               cl::FocusSlice(arr3, 1) == 1,
           "slices: focus's clone is slice 1 of the array (32 x 24), focus R shares it before its twin");
    // Slice k's pixel (0, 0).
    auto slicePixel = [&](ID3D11Texture2D* t, UINT k) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        d.ArraySize = 1;
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* s = nullptr;
        uint32_t px = 0;
        if (SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &s))) {
            cl::RawScope raw;
            ctx->CopySubresourceRegion(s, 0, 0, 0, 0, t, D3D11CalcSubresource(0, k, d.MipLevels), nullptr);
            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(ctx->Map(s, 0, D3D11_MAP_READ, 0, &m))) {
                px = *(const uint32_t*)m.pData;
                ctx->Unmap(s, 0);
            }
            s->Release();
        }
        return px;
    };
    auto rtvSlice = [](void* v) {
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        if (v) static_cast<ID3D11RenderTargetView*>(v)->GetDesc(&d);
        return d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY && d.Texture2DArray.ArraySize == 1
                   ? (int)d.Texture2DArray.FirstArraySlice
                   : -1;
    };
    ctx->ClearRenderTargetView(H.at<ID3D11RenderTargetView*>(0x80), green);  // slice 0 (outside the groups)
    // The array's own view bound and a viewport of the original set before the group: slice 1 at focus's size in it.
    ID3D11RenderTargetView* hOwn = H.at<ID3D11RenderTargetView*>(0x80);
    ctx->OMSetRenderTargets(1, &hOwn, nullptr);
    viewport(64, 48);
    cl::BeginGroup(ctx, cl::Focus);
    ID3D11RenderTargetView* hBound = nullptr;
    ctx->OMGetRenderTargets(1, &hBound, nullptr);
    expect(rtvSlice(hBound) == 1 && boundViewport() == std::make_pair(32.0f, 24.0f),
           "slices: the bound view goes to slice 1, the original's viewport scaled to focus's size");
    if (hBound) hBound->Release();
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    void* hRtv = H.at<void*>(0x80);
    expect(cl::TestGet().swapped && H.at<void*>(0x58) == arr3 && resOf(hRtv) == arr3 && rtvSlice(hRtv) == 1 &&
               rtvSlice(arrH[0]) == 1 && H.at<uint16_t>(0x68) == 32 && RH.at<uint16_t>(0x24) == 32,
           "slices: focus L's fields hold the array, views of slice 1 and focus's size");
    ctx->ClearRenderTargetView((ID3D11RenderTargetView*)hRtv, blue);
    cl::EndGroup(ctx);
    expect(H.at<void*>(0x58) == arr3 && rtvSlice(H.at<void*>(0x80)) == -1 && H.at<uint16_t>(0x68) == 64,
           "slices: the array's own views back after the focus group");
    expect(slicePixel(arr3, 0) == 0xFF00FF00u && slicePixel(arr3, 1) == 0xFFFF0000u,
           "slices: focus L's clear in slice 1, slice 0 kept");
    // Focus R's twin: slice 2, no texture, nothing copied.
    cl::WantTwin(arr3);
    cl::OnFrameStart(ctx, true);
    expect(cl::HasTwin(arr3) && cl::FocusSlice(arr3, 1) == 2 && cl::FocusSlice(arr3, 0) == 1 &&
               cl::TestGet().twinViews == 2 && cl::TestGet().twins == 1,
           "slices: focus R's twin is slice 2 (two views)");
    cl::SetFocusEye(1);
    expect(cl::CloneOf(arr3, cl::Focus) == arr3, "slices: eye 1's clone is the array too");
    cl::BeginGroup(ctx, cl::Focus);
    expect(rtvSlice(H.at<void*>(0x80)) == 2 && rtvSlice(arrH[0]) == 2 && H.at<uint16_t>(0x68) == 32,
           "slices: focus R's fields hold views of slice 2");
    ctx->ClearRenderTargetView(H.at<ID3D11RenderTargetView*>(0x80), red);
    // A view the engine keeps elsewhere (the array's slice 0 RTV) goes to focus R's slice.
    ID3D11RenderTargetView* mappedH[8] = {};
    ID3D11DepthStencilView* mappedHd = nullptr;
    ID3D11RenderTargetView* keptH = rtv(arr3);
    const bool mappedHok = cl::MapTargets(ctx, 1, &keptH, nullptr, mappedH, &mappedHd, nullptr);
    expect(mappedHok && rtvSlice(mappedH[0]) == 2, "slices: a view kept outside the fields goes to slice 2 for focus R");
    cl::EndGroup(ctx);
    cl::SetFocusEye(0);
    expect(slicePixel(arr3, 0) == 0xFF00FF00u && slicePixel(arr3, 1) == 0xFFFF0000u &&
               slicePixel(arr3, 2) == 0xFF0000FFu,
           "slices: focus R's clear in slice 2, slices 0 and 1 kept");
    // An outer group binds the array's own views (slice 0) as they are.
    cl::BeginGroup(ctx, cl::Outer);
    ID3D11RenderTargetView* mappedO[8] = {};
    ID3D11DepthStencilView* mappedOd = nullptr;
    const bool mappedOk2 = cl::MapTargets(ctx, 1, &keptH, nullptr, mappedO, &mappedOd, nullptr);
    expect(!mappedOk2 || mappedO[0] == keptH, "slices: an outer group's bind of the array stays slice 0");
    cl::EndGroup(ctx);
    keptH->Release();
    cl::TestDestroy(true, &RH);
    cl::TestDestroy(false, &H);
    expect(H.at<void*>(0x58) == texH && H.at<void*>(0x80) == rtvH, "slices: H's texture and views back at its destroy");
    cl::OnFrameStart(ctx, true);
    expect(cl::TestGet().arrays == 0 && !cl::IsArray(arr3), "slices: no array left");
    cl::Shutdown();
    printf("Clones check %s: %d wrong (focus smaller %d, outer smaller %d, two-slice arrays %d, three-slice arrays %d); "
           "%d translations, original pixel 0x%08X, clone pixel 0x%08X\n",
           bad ? "FAILED" : "OK", bad, bad1, bad2 - bad1, bad3 - bad2, bad - bad3, translated1, pxA, pxClone);
    for (IUnknown* u : {(IUnknown*)texH, (IUnknown*)rtvH, (IUnknown*)rtvH2}) u->Release();
    for (IUnknown* u : {(IUnknown*)texG, (IUnknown*)srvG, (IUnknown*)rtvG, (IUnknown*)rtvG2, (IUnknown*)rtvGX,
                        (IUnknown*)texS1})
        u->Release();
    for (IUnknown* u : {(IUnknown*)srvA, (IUnknown*)srvC, (IUnknown*)rtvA, (IUnknown*)rtvA2, (IUnknown*)rtvA3,
                        (IUnknown*)rtvD, (IUnknown*)texA, (IUnknown*)texB, (IUnknown*)texC, (IUnknown*)texD,
                        (IUnknown*)texE, (IUnknown*)rtvX, (IUnknown*)rtvP, (IUnknown*)rtvQ, (IUnknown*)rtvS, (IUnknown*)srvQ,
                        (IUnknown*)texP, (IUnknown*)texQ, (IUnknown*)texS})
        u->Release();
    ctx->Release();
    dev->Release();
    return bad ? 9 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    double scene = 0;
    std::string experiment;
    std::string center;
    int fault = 0;
    bool trace = false;
    bool probe = false;
    bool quad = false;
    bool dlaa = false;
    bool shotsTest = false;
    bool composeTest = false;
    bool clonesTest = false;
    bool sunFxTest = false;
    bool mvrVsTest = false;
    bool mvrPsTest = false;
    bool mvrPassTest = false;
    bool mvrForwardTest = false;
    bool xrLayerTest = false;
    bool openxrPathTest = false;
    bool openxr = false;
    bool noLayer = false;
    auto hasValue = [&](int i) { return i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0; };
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scene")) scene = hasValue(i) ? atof(argv[++i]) : 12.0;
        else if (!strcmp(argv[i], "--experiment") && hasValue(i)) experiment = argv[++i];
        else if (!strcmp(argv[i], "--fault") && hasValue(i)) fault = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace")) trace = true;
        else if (!strcmp(argv[i], "--probe")) probe = true;
        else if (!strcmp(argv[i], "--quad")) quad = true;
        else if (!strcmp(argv[i], "--dlaa")) dlaa = true;
        else if (!strcmp(argv[i], "--shots")) shotsTest = true;
        else if (!strcmp(argv[i], "--quad-compose")) composeTest = true;
        else if (!strcmp(argv[i], "--focus-geometry")) return RunGeometry();
        else if (!strcmp(argv[i], "--focus-cull")) return RunFocusCull();
        else if (!strcmp(argv[i], "--clones")) clonesTest = true;
        else if (!strcmp(argv[i], "--sun-fx")) sunFxTest = true;
        else if (!strcmp(argv[i], "--taa-fix")) return RunTaaFix();
        else if (!strcmp(argv[i], "--mvr-motion")) return RunMvrMotion();
        else if (!strcmp(argv[i], "--mvr-vs")) mvrVsTest = true;
        else if (!strcmp(argv[i], "--mvr-ps")) mvrPsTest = true;
        else if (!strcmp(argv[i], "--mvr-pass")) mvrPassTest = true;
        else if (!strcmp(argv[i], "--mvr-forward")) mvrForwardTest = true;
        else if (!strcmp(argv[i], "--xr-layer")) xrLayerTest = true;
        else if (!strcmp(argv[i], "--openxr-path")) openxrPathTest = true;
        else if (!strcmp(argv[i], "--openxr")) openxr = true;
        else if (!strcmp(argv[i], "--openxr-nolayer")) openxr = noLayer = true;
        else if (!strcmp(argv[i], "--center") && i + 2 < argc) {
            center = std::string("center_x=") + argv[i + 1] + "\ncenter_y=" + argv[i + 2] + "\n";
            i += 2;
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }
    fs::path home = ExeDir() / L"smoke_home";
    std::error_code ec;
    fs::create_directories(home, ec);
    SetEnvironmentVariableW(L"QVFR_HOME", home.c_str());
    if (shotsTest) return RunShots(home);
    if (composeTest) return RunCompose();
    if (clonesTest) return RunClones();
    if (sunFxTest) return RunSunFx();
    if (mvrVsTest) return RunMvrVs();
    if (mvrPsTest) return RunMvrPs();
    if (mvrPassTest) return RunMvrPass();
    if (mvrForwardTest) return RunMvrForward();
    if (xrLayerTest) return RunXrLayer(ExeDir() / L"proxy" / L"openvr_api.dll", home);
    if (openxrPathTest) return RunOpenXrPath(ExeDir() / L"proxy" / L"openvr_api.dll", home);
    if (FILE* f = _wfopen((home / L"qvfr.ini").c_str(), L"wb")) {
        fputs(scene > 0 ? kSceneIni : kBackgroundIni, f);
        if (!experiment.empty()) fprintf(f, "[experiment]\nmode=%s\n%s", experiment.c_str(), center.c_str());
        if (fault > 0) fprintf(f, "[test]\nfault_frames=%d\n", fault);
        if (trace) fputs("[trace]\ncallers=1\ndepth=12\n", f);
        if (probe) fputs("[quad]\nmode=probe\nprobe_frames=2\n", f);
        if (quad) fputs("[quad]\nmode=quad\n", f);
        if (dlaa) fputs("[dlaa]\nmode=dlaa\n", f);
        if (openxr)
            fprintf(f, "[runtime]\npath=openxr\nopenxr_layer=%d\n", noLayer ? 0 : 1);
        else
            fputs("[runtime]\npath=openvr\n", f);  // openxr is the default
        fclose(f);
    }

    fs::path dll = ExeDir() / L"proxy" / L"openvr_api.dll";
    HMODULE m = LoadLibraryW(dll.c_str());
    if (!m) {
        printf("Cannot load %ls (error %lu)\n", dll.c_str(), GetLastError());
        return 1;
    }
    auto init = (InitFn)GetProcAddress(m, "VR_Init");
    auto shutdown = (ShutdownFn)GetProcAddress(m, "VR_Shutdown");
    auto compositor = (CompositorFn)GetProcAddress(m, "VRCompositor");
    auto overlay = (OverlayFn)GetProcAddress(m, "VROverlay");
    auto errDesc = (ErrFn)GetProcAddress(m, "VR_GetVRInitErrorAsEnglishDescription");
    auto generic = (GenericFn)GetProcAddress(m, "VR_GetGenericInterface");
    // Flat C exports are thunks to the backend (generated/thunks.asm), resolved on their first call.
    using FlatSizeFn = void(VR_CALLTYPE*)(intptr_t, uint32_t*, uint32_t*);
    using FlatProjFn = vr::HmdMatrix44_t(VR_CALLTYPE*)(intptr_t, vr::EVREye, float, float, vr::EGraphicsAPIConvention);
    auto flatSize = (FlatSizeFn)GetProcAddress(m, "VR_IVRSystem_GetRecommendedRenderTargetSize");
    auto flatProj = (FlatProjFn)GetProcAddress(m, "VR_IVRSystem_GetProjectionMatrix");
    printf("Exports: VR_Init %p, VR_Shutdown %p, VRCompositor %p, VROverlay %p, error description %p, generic %p, "
           "thunks VR_IVRSystem_GetRecommendedRenderTargetSize %p, VR_IVRSystem_GetProjectionMatrix %p\n",
           (void*)init, (void*)shutdown, (void*)compositor, (void*)overlay, (void*)errDesc, (void*)generic,
           (void*)flatSize, (void*)flatProj);
    if (!init || !shutdown || !compositor || !overlay || !errDesc || !generic || !flatSize || !flatProj) return 1;

    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem* sys = init(&err, scene > 0 ? vr::VRApplication_Scene : vr::VRApplication_Background);
    printf("VR_Init = %p, error %d: %s\n", (void*)sys, (int)err, errDesc(err));
    int rc = 0;
    if (sys) {
        uint32_t w = 0, h = 0;
        sys->GetRecommendedRenderTargetSize(&w, &h);
        float l, r, t, b;
        sys->GetProjectionRaw(vr::Eye_Left, &l, &r, &t, &b);
        sys->GetProjectionMatrix(vr::Eye_Left, 0.05f, 1000.0f, vr::API_DirectX);
        sys->GetEyeToHeadTransform(vr::Eye_Right);
        // A second call site hit several times: with --trace one more stack of this method, counted, not logged again.
        // The volatile counter keeps the compiler from unrolling the loop into separate call sites.
        for (volatile int i = 0; i < 4; ++i) sys->GetEyeToHeadTransform((vr::EVREye)(i & 1));
        printf("Size %u x %u, left eye: left %.4f right %.4f top %.4f bottom %.4f\n", w, h, l, r, t, b);
        vr::IVRCompositor* comp = compositor();
        vr::IVROverlay* ov = overlay();
        void* sys9 = generic(vr::IVRSystem_Version, &err);
        printf("VRCompositor %p, VROverlay %p, VR_GetGenericInterface(IVRSystem_009) %p (same as VR_Init: %s)\n",
               (void*)comp, (void*)ov, sys9, sys9 == sys ? "yes" : "no");
        // The thunks must hand every argument through: integer registers, xmm2/xmm3 floats, a stack argument and the
        // hidden return pointer of HmdMatrix44_t. OpenComposite has no flat exports, so there they return 0.
        uint32_t fw = 0, fh = 0;
        flatSize((intptr_t)sys, &fw, &fh);
        const vr::HmdMatrix44_t want = sys->GetProjectionMatrix(vr::Eye_Right, 0.05f, 1000.0f, vr::API_DirectX);
        const vr::HmdMatrix44_t got = flatProj((intptr_t)sys, vr::Eye_Right, 0.05f, 1000.0f, vr::API_DirectX);
        const bool sameProj = memcmp(&want, &got, sizeof(want)) == 0;
        printf("Thunks: GetRecommendedRenderTargetSize %u x %u (%s), GetProjectionMatrix(right) %s\n", fw, fh,
               fw == w && fh == h ? "same as the interface" : "DIFFERENT (0 x 0 = backend without the export)",
               sameProj ? "same as the interface" : "DIFFERENT");
        if (!openxr && (fw != w || fh != h || !sameProj)) rc = 1;
        if (scene > 0 && comp && rc == 0) rc = RunScene(sys, comp, ov, scene, probe);
        shutdown();
        if (trace && rc == 0) rc = CheckTrace(home / L"logs");
        if (probe && rc == 0) rc = CheckProbe(home / L"logs");
        if (quad && rc == 0) rc = CheckQuad(home / L"logs");
        if (dlaa && rc == 0) rc = CheckDlaa(home / L"logs");
    }
    printf("Proxy log: %ls\n", (home / L"logs").c_str());
    return rc;
}

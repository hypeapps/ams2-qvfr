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
//   --shots                           without SteamVR and without the proxy DLL: the proxy's shot module on synthetic
//                                     textures (staging copies, PNG files, pixel differences)
//   --quad-compose                    without SteamVR and without the proxy DLL: the proxy's composition module on
//                                     synthetic textures (pixels by region, also with the window in grid corners,
//                                     pipeline state restored)
//   --focus-geometry                  without SteamVR: focus window placement for a gaze point (Dream Air frusta)
//   --clones                          without SteamVR and without the proxy DLL: the proxy's clones module on fake engine
//                                     objects (learning, clones in the sets' sizes, swap of fields, sizes and bound
//                                     views, viewports, restore, leaks, destroy; focus or outer smaller)
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
    "[proxy]\nwrap=1\n[log]\nfirst_calls=2\nstats_interval_s=5\n[timing]\ncsv_always=1\n"
    "[benchmark]\nmodifier=0xA3\nkey=0x78\ndelay_s=1\nduration_s=5\nfreeze_pose=1\nbeep=0\nauto_start_s=2\n[gaze]\nread=1\n";
const char kBackgroundIni[] = "[proxy]\nwrap=1\n";

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
    }
    bool ok = settings && notSupported && !hooked && !faults && size && !resized;
    printf("Quad check %s: settings %d, exe not supported %d, hooks or patch %d, faults %d, size logged %d, size "
           "replaced %d (%ls)\n",
           ok ? "OK" : "FAILED", settings, notSupported, hooked, faults, size, resized, log.filename().c_str());
    return ok ? 0 : 5;
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
    // Debug overlay: the gaze dot and the outline of the margin edge, the rest unchanged.
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
                overlayOk = ch(dot, 0) > 200 && ch(dot, 1) < 120 && ch(dot, 2) < 120 && gap == outerPx &&
                            ch(edge, 0) > 200 && ch(edge, 1) > 200 && ch(edge, 2) < 60 && ch(sharp, 0) < 60 &&
                            ch(sharp, 1) > 200 && ch(sharp, 2) < 160;
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
    printf("Clones check %s: %d wrong (focus smaller %d, outer smaller %d); %d translations, original pixel 0x%08X, "
           "clone pixel 0x%08X\n",
           bad ? "FAILED" : "OK", bad, bad1, bad - bad1, translated1, pxA, pxClone);
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
    bool shotsTest = false;
    bool composeTest = false;
    bool clonesTest = false;
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
        else if (!strcmp(argv[i], "--shots")) shotsTest = true;
        else if (!strcmp(argv[i], "--quad-compose")) composeTest = true;
        else if (!strcmp(argv[i], "--focus-geometry")) return RunGeometry();
        else if (!strcmp(argv[i], "--clones")) clonesTest = true;
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
    if (xrLayerTest) return RunXrLayer(ExeDir() / L"proxy" / L"openvr_api.dll", home);
    if (openxrPathTest) return RunOpenXrPath(ExeDir() / L"proxy" / L"openvr_api.dll", home);
    if (FILE* f = _wfopen((home / L"qvfr.ini").c_str(), L"wb")) {
        fputs(scene > 0 ? kSceneIni : kBackgroundIni, f);
        if (!experiment.empty()) fprintf(f, "[experiment]\nmode=%s\n%s", experiment.c_str(), center.c_str());
        if (fault > 0) fprintf(f, "[test]\nfault_frames=%d\n", fault);
        if (trace) fputs("[trace]\ncallers=1\ndepth=12\n", f);
        if (probe) fputs("[quad]\nmode=probe\nprobe_frames=2\n", f);
        if (quad) fputs("[quad]\nmode=quad\n", f);
        if (openxr)
            fprintf(f, "[runtime]\npath=openxr\nopenxr_layer=%d\n", noLayer ? 0 : 1);
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
    }
    printf("Proxy log: %ls\n", (home / L"logs").c_str());
    return rc;
}

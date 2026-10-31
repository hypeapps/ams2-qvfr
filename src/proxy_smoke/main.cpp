// qvfr_proxy_smoke: tests the openvr_api.dll proxy without the game.
//
// Loads proxy\openvr_api.dll (next to the exe) with LoadLibrary and calls the same exports as AMS2.
// The proxy writes its logs to smoke_home\ next to the exe (QVFR_HOME variable), not to Documents.
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
// smoke_home\qvfr.ini is rewritten on every run, so options never leak from one run into the next.

#include <windows.h>
#include <d3d11.h>

#include <openvr.h>  // 0.9.12; functions are taken through GetProcAddress

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "shots.h"

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
    bool settings = false, notSupported = false, hooked = false, faults = false;
    std::string line;
    while (std::getline(in, line)) {
        settings |= line.find("Settings: [quad] mode=quad") != std::string::npos;
        notSupported |= line.find("Quad: exe not supported") != std::string::npos;
        hooked |= line.find("Quad: engine hooks installed") != std::string::npos ||
                  line.find("builder patch applied") != std::string::npos;
        faults |= line.find("in quad mode") != std::string::npos;
    }
    bool ok = settings && notSupported && !hooked && !faults;
    printf("Quad check %s: settings %d, exe not supported %d, hooks or patch %d, faults %d (%ls)\n", ok ? "OK" : "FAILED",
           settings, notSupported, hooked, faults, log.filename().c_str());
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
// in a 16 x 16 block, one image in R10G10B10A2), then the log lines and the PNG files it writes.
int RunShots(const fs::path& home) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    const UINT w = 1024, h = 768;
    static const char* const names[] = {"outer_l", "outer_r", "focus_l", "focus_r", "tonemap"};
    static const qvfr::shots::Pair pairs[] = {
        {2, 0, "focus L vs outer L"}, {3, 1, "focus R vs outer R"}, {0, 1, "outer L vs outer R"}};
    std::vector<uint32_t> px((size_t)w * h);
    ID3D11Texture2D* tex[5] = {};
    for (int i = 0; i < 5; ++i) {
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x) {
                uint32_t r = x * 255 / w, g = y * 255 / h, b = (i == 1 || i == 3) ? 200 : 40;
                if (i == 3 && x >= 500 && x < 516 && y >= 300 && y < 316) r = (r + 128) % 256;
                px[(size_t)y * w + x] = i == 4 ? (r << 2) | (g << 12) | (b << 22) | (3u << 30)
                                               : r | g << 8 | b << 16 | 0xFF000000u;
            }
        D3D11_TEXTURE2D_DESC d{};
        d.Width = w;
        d.Height = h;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = i == 4 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_TYPELESS;
        d.SampleDesc.Count = 1;
        d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA init{px.data(), w * 4, 0};
        if (FAILED(dev->CreateTexture2D(&d, &init, &tex[i]))) {
            printf("Cannot create the shot textures\n");
            return 1;
        }
    }
    double t0 = qvfr::NowMs();
    bool begun = qvfr::shots::Begin(names, 5, pairs, 3);
    bool again = qvfr::shots::Begin(names, 5, pairs, 3);  // refused while the first shot is active
    int grabbed = 0;
    for (int i = 0; i < 5; ++i) grabbed += qvfr::shots::Grab(ctx, tex[i], i);
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
    for (const char* n : names) {
        uint32_t pw = 0, ph = 0;
        if (PngSize(newest / (std::string(n) + ".png"), pw, ph)) {
            ++pngs;
            halfSize += pw == w / 2 && ph == h / 2;
        }
    }
    std::ifstream in(NewestLog(home / L"logs"));
    std::string line;
    bool same = false, differs = false, parallax = false;
    while (std::getline(in, line)) {
        same |= line.find("focus L vs outer L: mean abs difference 0.000 (0..255), max 0,") != std::string::npos;
        differs |= line.find("focus R vs outer R:") != std::string::npos &&
                   line.find("> 32: 0.033%") != std::string::npos;  // 256 of 786432 pixels
        parallax |= line.find("outer L vs outer R:") != std::string::npos &&
                    line.find("max 0,") == std::string::npos;
    }
    bool ok = begun && !again && grabbed == 5 && done && pngs == 5 && halfSize == 5 && same && differs && parallax;
    printf("Shots check %s: begin %d, second begin refused %d, grabbed %d, writer done %d (%.1f s), PNG %d (half size %d) in "
           "%ls, diffs: identical %d, block %d, different %d\n",
           ok ? "OK" : "FAILED", begun, !again, grabbed, done, (qvfr::NowMs() - t0) / 1000.0, pngs, halfSize,
           newest.c_str(), same, differs, parallax);
    return ok ? 0 : 6;
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
    auto hasValue = [&](int i) { return i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0; };
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scene")) scene = hasValue(i) ? atof(argv[++i]) : 12.0;
        else if (!strcmp(argv[i], "--experiment") && hasValue(i)) experiment = argv[++i];
        else if (!strcmp(argv[i], "--fault") && hasValue(i)) fault = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace")) trace = true;
        else if (!strcmp(argv[i], "--probe")) probe = true;
        else if (!strcmp(argv[i], "--quad")) quad = true;
        else if (!strcmp(argv[i], "--shots")) shotsTest = true;
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
    if (FILE* f = _wfopen((home / L"qvfr.ini").c_str(), L"wb")) {
        fputs(scene > 0 ? kSceneIni : kBackgroundIni, f);
        if (!experiment.empty()) fprintf(f, "[experiment]\nmode=%s\n%s", experiment.c_str(), center.c_str());
        if (fault > 0) fprintf(f, "[test]\nfault_frames=%d\n", fault);
        if (trace) fputs("[trace]\ncallers=1\ndepth=12\n", f);
        if (probe) fputs("[quad]\nmode=probe\nprobe_frames=2\n", f);
        if (quad) fputs("[quad]\nmode=quad\n", f);
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
    auto forwarded = GetProcAddress(m, "VR_IVRSystem_GetRecommendedRenderTargetSize");  // forwarded export
    printf("Exports: VR_Init %p, VR_Shutdown %p, VRCompositor %p, VROverlay %p, error description %p, generic %p, "
           "forwarded VR_IVRSystem_GetRecommendedRenderTargetSize %p\n",
           (void*)init, (void*)shutdown, (void*)compositor, (void*)overlay, (void*)errDesc, (void*)generic,
           (void*)forwarded);
    if (!init || !shutdown || !compositor || !overlay || !errDesc || !generic) return 1;

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
        // Forwarded exports work once the original is loaded (after the first call into the proxy).
        printf("Forwarded VR_IVRSystem_GetRecommendedRenderTargetSize after VR_Init: %p\n",
               (void*)GetProcAddress(m, "VR_IVRSystem_GetRecommendedRenderTargetSize"));
        if (scene > 0 && comp) rc = RunScene(sys, comp, ov, scene, probe);
        shutdown();
        if (trace && rc == 0) rc = CheckTrace(home / L"logs");
        if (probe && rc == 0) rc = CheckProbe(home / L"logs");
        if (quad && rc == 0) rc = CheckQuad(home / L"logs");
    }
    printf("Proxy log: %ls\n", (home / L"logs").c_str());
    return rc;
}

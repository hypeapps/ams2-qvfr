// qvfr_proxy_smoke: tests the openvr_api.dll proxy without the game.
//
// Loads proxy\openvr_api.dll (next to the exe) with LoadLibrary and calls the same exports as AMS2.
// The proxy writes its logs to smoke_home\ next to the exe (QVFR_HOME variable), not to Documents.
//
//   qvfr_proxy_smoke.exe              Background application: VR_Init, projection queries, VR_Shutdown
//                                     (does not take the HMD from other applications; needs SteamVR running)
//   qvfr_proxy_smoke.exe --scene 12   Scene application: 12 s of D3D11 frames (grey background) with Submit,
//                                     the benchmark starts by itself (auto_start_s=2, delay_s=1, duration_s=5)

#include <windows.h>
#include <d3d11.h>

#include <openvr.h>  // 0.9.12; functions are taken through GetProcAddress

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

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

int RunScene(vr::IVRSystem* sys, vr::IVRCompositor* comp, vr::IVROverlay* ov, double seconds) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    uint32_t w = 0, h = 0;
    sys->GetRecommendedRenderTargetSize(&w, &h);
    w /= 4;
    h /= 4;  // size does not matter for the test, smaller = faster
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    d.SampleDesc.Count = 1;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D* tex[2] = {};
    ID3D11RenderTargetView* rtv[2] = {};
    for (int e = 0; e < 2; ++e) {
        if (FAILED(dev->CreateTexture2D(&d, nullptr, &tex[e])) ||
            FAILED(dev->CreateRenderTargetView(tex[e], nullptr, &rtv[e]))) {
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
    }
    ctx->Release();
    dev->Release();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    double scene = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scene")) scene = i + 1 < argc ? atof(argv[++i]) : 12.0;
    }
    fs::path home = ExeDir() / L"smoke_home";
    std::error_code ec;
    fs::create_directories(home, ec);
    SetEnvironmentVariableW(L"QVFR_HOME", home.c_str());
    if (scene > 0) {
        if (FILE* f = _wfopen((home / L"qvfr.ini").c_str(), L"wb")) {
            fputs(kSceneIni, f);
            fclose(f);
        }
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
        printf("Size %u x %u, left eye: left %.4f right %.4f top %.4f bottom %.4f\n", w, h, l, r, t, b);
        vr::IVRCompositor* comp = compositor();
        vr::IVROverlay* ov = overlay();
        void* sys9 = generic(vr::IVRSystem_Version, &err);
        printf("VRCompositor %p, VROverlay %p, VR_GetGenericInterface(IVRSystem_009) %p (same as VR_Init: %s)\n",
               (void*)comp, (void*)ov, sys9, sys9 == sys ? "yes" : "no");
        // Forwarded exports work once the original is loaded (after the first call into the proxy).
        printf("Forwarded VR_IVRSystem_GetRecommendedRenderTargetSize after VR_Init: %p\n",
               (void*)GetProcAddress(m, "VR_IVRSystem_GetRecommendedRenderTargetSize"));
        if (scene > 0 && comp) rc = RunScene(sys, comp, ov, scene);
        shutdown();
    }
    printf("Proxy log: %ls\n", (home / L"logs").c_str());
    return rc;
}

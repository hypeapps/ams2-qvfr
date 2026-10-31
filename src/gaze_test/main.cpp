// qvfr_gaze_test: can the Dream Air gaze (sboys3 driver) be read through OpenVR,
// in which coordinate system and with what accuracy.
//
// Draws a head-locked angular grid in the headset, guides the gaze through checkpoints
// and writes several gaze sources to the log and CSV:
//   A    IVRSystem_026::GetEyeTrackedFoveationCenter (NDC per eye, no input manifest)
//   D    IVRSystem_026::GetEyeTrackedFoveationCenterForProjection (custom projection and the left eye matrix)
//   B    IVRInput_011::GetEyeTrackingDataForNextFrame (gaze ray, needs an action manifest)
//   C    %APPDATA%\CustomHeadset\diagnostic.json (angles from the driver, 4 Hz, reference)
//
// "legacy" mode initializes OpenVR with the old openvr_api.dll 0.9.12 (the same one AMS2 uses) and gets the newer
// interfaces through VR_GetGenericInterface. This answers whether a proxy in the game can get IVRSystem_026.
//
// Angular coordinates: tanX to the right, tanY up (direction (tanX, tanY, -1) in eye/head space).
// "Degrees" in the log are atan(tanX), atan(tanY).

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <openvr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------- log

FILE* g_log = nullptr;
FILE* g_csv = nullptr;

double NowSec() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Log(const char* fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g_log) {
        fprintf(g_log, "[%9.3f] %s\n", NowSec(), buf);
        fflush(g_log);
    }
    OutputDebugStringA(buf);
    OutputDebugStringA("\n");
}

std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

fs::path ExeDir() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
}

// ---------------------------------------------------------------- settings

const char kDefaultIni[] =
    "; qvfr_gaze_test: gaze test settings. The file is created on the first run.\n"
    "[gaze_test]\n"
    "; modern = openvr_api.dll 2.15.6, legacy = openvr_api_0912.dll (the same version AMS2 uses).\n"
    "; From the command line: --legacy or --modern.\n"
    "runtime=modern\n"
    "; Eye texture resolution scale relative to the SteamVR recommendation (does not affect accuracy).\n"
    "render_scale=0.5\n"
    "; IVRInput: 1 = test the action manifest path, 0 = skip.\n"
    "use_input=1\n"
    "; Device path in the bindings file (the sboys3 driver creates an /eyetracking component on the HMD).\n"
    "binding_path=/user/head/eyetracking\n"
    "; Controller type in the bindings file; auto = the HMD Prop_ControllerType_String.\n"
    "controller_type=auto\n"
    "; Time per point, and the moment from which samples are collected (seconds).\n"
    "dwell_seconds=3.5\n"
    "settle_seconds=1.2\n"
    "; Start the sequence this many seconds after the first frame (or press space in the program window).\n"
    "autostart_seconds=15\n"
    "; Half extent (in tangents) of the custom symmetric projection for ...ForProjection.\n"
    "focus_tan=0.5\n";

struct Config {
    bool legacy = false;
    float renderScale = 0.5f;
    bool useInput = true;
    std::string bindingPath = "/user/head/eyetracking";
    std::string controllerType = "auto";
    float dwell = 3.5f;
    float settle = 1.2f;
    float autostart = 15.0f;
    float focusTan = 0.5f;
};

std::wstring IniStr(const fs::path& ini, const wchar_t* key, const wchar_t* def) {
    wchar_t buf[512];
    GetPrivateProfileStringW(L"gaze_test", key, def, buf, 512, ini.c_str());
    return buf;
}

float IniFloat(const fs::path& ini, const wchar_t* key, float def) {
    std::wstring s = IniStr(ini, key, L"");
    return s.empty() ? def : (float)_wtof(s.c_str());
}

Config LoadConfig(const fs::path& dir) {
    fs::path ini = dir / L"gaze_test.ini";
    if (!fs::exists(ini)) {
        if (FILE* f = _wfopen(ini.c_str(), L"wb")) {
            fwrite(kDefaultIni, 1, sizeof(kDefaultIni) - 1, f);
            fclose(f);
        }
    }
    Config c;
    c.legacy = IniStr(ini, L"runtime", L"modern") == L"legacy";
    c.renderScale = std::clamp(IniFloat(ini, L"render_scale", c.renderScale), 0.1f, 2.0f);
    c.useInput = IniFloat(ini, L"use_input", 1.0f) != 0.0f;
    c.bindingPath = Narrow(IniStr(ini, L"binding_path", L"/user/head/eyetracking"));
    c.controllerType = Narrow(IniStr(ini, L"controller_type", L"auto"));
    c.dwell = std::max(1.0f, IniFloat(ini, L"dwell_seconds", c.dwell));
    c.settle = std::clamp(IniFloat(ini, L"settle_seconds", c.settle), 0.0f, c.dwell - 0.3f);
    c.autostart = std::max(0.0f, IniFloat(ini, L"autostart_seconds", c.autostart));
    c.focusTan = std::clamp(IniFloat(ini, L"focus_tan", c.focusTan), 0.05f, 3.0f);

    int argc = 0;
    if (wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc)) {
        for (int i = 1; i < argc; ++i) {
            if (wcscmp(argv[i], L"--legacy") == 0) c.legacy = true;
            if (wcscmp(argv[i], L"--modern") == 0) c.legacy = false;
        }
        LocalFree(argv);
    }
    return c;
}

// ---------------------------------------------------------------- OpenVR (loaded dynamically)

struct VrRuntime {
    HMODULE dll = nullptr;
    bool legacy = false;
    void*(VR_CALLTYPE* getGeneric)(const char*, vr::EVRInitError*) = nullptr;
    const char*(VR_CALLTYPE* errorDesc)(vr::EVRInitError) = nullptr;
    void(VR_CALLTYPE* shutdown)() = nullptr;

    const char* Desc(vr::EVRInitError e) const { return errorDesc ? errorDesc(e) : "?"; }

    template <class T>
    T* Get(const char* version) const {
        vr::EVRInitError e = vr::VRInitError_None;
        void* p = getGeneric(version, &e);
        Log("VR_GetGenericInterface(%s) -> %p, err=%d (%s)", version, p, (int)e, Desc(e));
        return e == vr::VRInitError_None ? static_cast<T*>(p) : nullptr;
    }
};

bool InitRuntime(VrRuntime& rt, bool legacy, const fs::path& dir) {
    rt.legacy = legacy;
    fs::path path = dir / (legacy ? L"openvr_api_0912.dll" : L"openvr_api.dll");
    rt.dll = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!rt.dll) {
        Log("ERROR: cannot load %s (GetLastError=%lu)", Narrow(path.wstring()).c_str(), GetLastError());
        return false;
    }
    Log("Loaded %s", Narrow(path.wstring()).c_str());
    rt.getGeneric = reinterpret_cast<decltype(rt.getGeneric)>(GetProcAddress(rt.dll, "VR_GetGenericInterface"));
    rt.errorDesc = reinterpret_cast<decltype(rt.errorDesc)>(GetProcAddress(rt.dll, "VR_GetVRInitErrorAsEnglishDescription"));

    vr::EVRInitError err = vr::VRInitError_None;
    if (legacy) {
        using InitFn = void*(VR_CALLTYPE*)(vr::EVRInitError*, int);
        auto init = reinterpret_cast<InitFn>(GetProcAddress(rt.dll, "VR_Init"));
        rt.shutdown = reinterpret_cast<decltype(rt.shutdown)>(GetProcAddress(rt.dll, "VR_Shutdown"));
        if (!init || !rt.getGeneric || !rt.shutdown) {
            Log("ERROR: missing exports VR_Init/VR_GetGenericInterface/VR_Shutdown");
            return false;
        }
        void* sys009 = init(&err, 1 /* VRApplication_Scene in 0.9.12 */);
        Log("VR_Init (0.9.12, Scene) -> IVRSystem_009=%p, err=%d (%s)", sys009, (int)err, rt.Desc(err));
    } else {
        using InitFn = uint32_t(VR_CALLTYPE*)(vr::EVRInitError*, vr::EVRApplicationType, const char*);
        auto init = reinterpret_cast<InitFn>(GetProcAddress(rt.dll, "VR_InitInternal2"));
        rt.shutdown = reinterpret_cast<decltype(rt.shutdown)>(GetProcAddress(rt.dll, "VR_ShutdownInternal"));
        if (!init || !rt.getGeneric || !rt.shutdown) {
            Log("ERROR: missing exports VR_InitInternal2/VR_GetGenericInterface/VR_ShutdownInternal");
            return false;
        }
        uint32_t token = init(&err, vr::VRApplication_Scene, nullptr);
        Log("VR_InitInternal2 (2.15.6, Scene) -> token=%u, err=%d (%s)", token, (int)err, rt.Desc(err));
    }
    if (err != vr::VRInitError_None) {
        rt.shutdown = nullptr;
        return false;
    }
    return true;
}

std::string DevStr(vr::IVRSystem* sys, vr::ETrackedDeviceProperty prop) {
    char buf[512] = {};
    vr::ETrackedPropertyError pe = vr::TrackedProp_Success;
    sys->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, prop, buf, sizeof(buf), &pe);
    if (pe != vr::TrackedProp_Success) return std::string("<error ") + std::to_string((int)pe) + ">";
    return buf;
}

// ---------------------------------------------------------------- geometry

struct Vec2 {
    float x = 0, y = 0;
};

struct EyeInfo {
    float raw[4] = {};  // GetProjectionRaw: left, right, top, bottom (top = -tan(up), y down)
    vr::HmdMatrix44_t proj = {};
};

// NDC from an OpenVR matrix (row-major, w = -Z) -> direction tangents.
Vec2 NdcToTan(const vr::HmdMatrix44_t& m, vr::HmdVector2_t ndc) {
    return {(ndc.v[0] + m.m[0][2]) / m.m[0][0], (ndc.v[1] + m.m[1][2]) / m.m[1][1]};
}

vr::HmdMatrix44_t SymmetricProjection(float halfTan) {
    const float n = 0.1f, f = 100.0f;
    vr::HmdMatrix44_t m = {};
    m.m[0][0] = 1.0f / halfTan;
    m.m[1][1] = 1.0f / halfTan;
    m.m[2][2] = -f / (f - n);
    m.m[2][3] = -f * n / (f - n);
    m.m[3][2] = -1.0f;
    return m;
}

// A world-space point to head space (inverse of the 3x4 pose).
void WorldToHead(const vr::HmdMatrix34_t& p, const float w[3], float out[3]) {
    float d[3] = {w[0] - p.m[0][3], w[1] - p.m[1][3], w[2] - p.m[2][3]};
    for (int i = 0; i < 3; ++i) out[i] = p.m[0][i] * d[0] + p.m[1][i] * d[1] + p.m[2][i] * d[2];
}

bool DirToTan(const float d[3], Vec2& out) {
    if (d[2] > -1e-6f) return false;
    out = {d[0] / -d[2], d[1] / -d[2]};
    return true;
}

constexpr float kRadToDeg = 57.29577951f;

double AngleBetweenTan(double ax, double ay, double bx, double by) {
    double la = std::sqrt(ax * ax + ay * ay + 1), lb = std::sqrt(bx * bx + by * by + 1);
    double c = (ax * bx + ay * by + 1) / (la * lb);
    return std::acos(std::clamp(c, -1.0, 1.0)) * kRadToDeg;
}

// ---------------------------------------------------------------- sboys3 driver diagnostic.json

struct DiagReader {
    fs::path path;
    std::string last;
    double lastRead = -1, lastChange = -1;
    bool ok = false, valid = false;
    double la[2] = {}, ra[2] = {};

    static bool FindNumber(const std::string& s, size_t from, const char* key, double& out) {
        size_t k = s.find(std::string("\"") + key + "\"", from);
        if (k == std::string::npos) return false;
        size_t c = s.find(':', k);
        if (c == std::string::npos) return false;
        out = strtod(s.c_str() + c + 1, nullptr);
        return true;
    }

    void Init() {
        PWSTR p = nullptr;
        if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &p))) {
            path = fs::path(p) / L"CustomHeadset" / L"diagnostic.json";
            CoTaskMemFree(p);
        }
        Log("diagnostic.json: %s (%s)", Narrow(path.wstring()).c_str(), fs::exists(path) ? "present" : "MISSING");
    }

    void Poll(double now) {
        if (now - lastRead < 0.05) return;
        lastRead = now;
        HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) return;
        char buf[8192];
        DWORD n = 0;
        BOOL r = ReadFile(h, buf, sizeof(buf) - 1, &n, nullptr);
        CloseHandle(h);
        if (!r || n == 0) return;
        std::string s(buf, n);
        if (s == last) return;
        size_t et = s.find("\"eyeTracking\"");
        size_t l = s.find("\"left\"", et), rr = s.find("\"right\"", et);
        if (et == std::string::npos || l == std::string::npos || rr == std::string::npos) return;  // write in progress
        double lx, ly, rx, ry;
        if (!FindNumber(s, l, "angleX", lx) || !FindNumber(s, l, "angleY", ly) || !FindNumber(s, rr, "angleX", rx) ||
            !FindNumber(s, rr, "angleY", ry))
            return;
        size_t v = s.find("\"valid\"", et);
        valid = v != std::string::npos && s.find("true", v) < s.find(',', v);
        la[0] = lx, la[1] = ly, ra[0] = rx, ra[1] = ry;
        ok = true;
        last = std::move(s);
        lastChange = now;
    }
};

// ---------------------------------------------------------------- IVRInput

struct InputPath {
    vr::IVRInput* input = nullptr;
    vr::VRActionSetHandle_t set = vr::k_ulInvalidActionSetHandle;
    vr::VRActionHandle_t action = vr::k_ulInvalidActionHandle;
    bool ok = false;

    void WriteText(const fs::path& p, const std::string& s) {
        if (FILE* f = _wfopen(p.c_str(), L"wb")) {
            fwrite(s.data(), 1, s.size(), f);
            fclose(f);
        }
    }

    void Init(vr::IVRInput* in, const fs::path& dir, const Config& cfg, const std::string& hmdControllerType) {
        input = in;
        if (!input) return;
        std::string type = cfg.controllerType == "auto" ? hmdControllerType : cfg.controllerType;
        fs::path idir = dir / L"input";
        std::error_code ec;
        fs::create_directories(idir, ec);
        std::string bindingFile = "bindings_" + type + ".json";
        WriteText(idir / L"actions.json",
                  "{\n"
                  "  \"default_bindings\": [ { \"controller_type\": \"" + type + "\", \"binding_url\": \"" + bindingFile + "\" } ],\n"
                  "  \"actions\": [ { \"name\": \"/actions/gaze/in/eyegaze\", \"type\": \"eyetracking\" } ],\n"
                  "  \"action_sets\": [ { \"name\": \"/actions/gaze\", \"usage\": \"single\" } ],\n"
                  "  \"localization\": [ { \"language_tag\": \"en_US\", \"/actions/gaze\": \"Gaze test\", \"/actions/gaze/in/eyegaze\": \"Eye gaze\" } ]\n"
                  "}\n");
        WriteText(idir / fs::path(bindingFile),
                  "{\n"
                  "  \"action_manifest_version\": 0,\n"
                  "  \"controller_type\": \"" + type + "\",\n"
                  "  \"name\": \"QVFR gaze test\",\n"
                  "  \"description\": \"Eye gaze binding for qvfr_gaze_test\",\n"
                  "  \"bindings\": {\n"
                  "    \"/actions/gaze\": {\n"
                  "      \"eyetracking\": [ { \"output\": \"/actions/gaze/in/eyegaze\", \"path\": \"" + cfg.bindingPath + "\" } ]\n"
                  "    }\n"
                  "  }\n"
                  "}\n");
        std::string manifest = Narrow((idir / L"actions.json").wstring());
        vr::EVRInputError e = input->SetActionManifestPath(manifest.c_str());
        Log("IVRInput::SetActionManifestPath(%s) -> %d (controller_type=%s, path=%s)", manifest.c_str(), (int)e,
            type.c_str(), cfg.bindingPath.c_str());
        vr::EVRInputError e1 = input->GetActionSetHandle("/actions/gaze", &set);
        vr::EVRInputError e2 = input->GetActionHandle("/actions/gaze/in/eyegaze", &action);
        Log("IVRInput: GetActionSetHandle -> %d (0x%llx), GetActionHandle -> %d (0x%llx)", (int)e1,
            (unsigned long long)set, (int)e2, (unsigned long long)action);
        ok = e == vr::VRInputError_None && e1 == vr::VRInputError_None && e2 == vr::VRInputError_None;
    }

    void LogOrigins() {
        if (!ok) return;
        vr::VRInputValueHandle_t origins[8] = {};
        vr::EVRInputError e = input->GetActionOrigins(set, action, origins, 8);
        int count = 0;
        for (auto o : origins) {
            if (o == vr::k_ulInvalidInputValueHandle) continue;
            ++count;
            char name[256] = {};
            input->GetOriginLocalizedName(o, name, sizeof(name), vr::VRInputString_All);
            vr::InputOriginInfo_t info = {};
            vr::EVRInputError ie = input->GetOriginTrackedDeviceInfo(o, &info, sizeof(info));
            char devPath[256] = {};
            // IVRInput has no GetPathFromHandle, so the device path is shown as a handle.
            snprintf(devPath, sizeof(devPath), "0x%llx", (unsigned long long)info.devicePath);
            Log("IVRInput: origin 0x%llx '%s' -> device %u, devicePath %s, component '%s' (err %d)",
                (unsigned long long)o, name, info.trackedDeviceIndex, devPath, info.rchRenderModelComponentName, (int)ie);
        }
        Log("IVRInput::GetActionOrigins -> %d, bound sources: %d", (int)e, count);
    }
};

// ---------------------------------------------------------------- render (D3D11)

const char kShader[] = R"(
cbuffer CB : register(b0)
{
    float4 g_raw;          // tan: left, right, top, bottom (top = -tan(up))
    float4 g_view;         // width, height, -, progress of the active target 0..1
    float4 g_active;       // active target: tanX, tanY, -, visible
    float4 g_mark[4];      // A, B, C, D: tanX, tanY, -, visible
    float4 g_targets[24];  // tanX, tanY, done, -
    float4 g_count;        // number of targets
};

struct VSOut { float4 pos : SV_Position; };

VSOut VSMain(uint id : SV_VertexID)
{
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float2 TanToPx(float2 t)
{
    float u = (t.x - g_raw.x) / (g_raw.y - g_raw.x);
    float v = (-t.y - g_raw.z) / (g_raw.w - g_raw.z);
    return float2(u * g_view.x, v * g_view.y);
}

float Ring(float d, float r, float w) { return saturate(1 - abs(d - r) / w); }
float Disc(float d, float r) { return saturate(r - d + 0.5); }

float4 PSMain(VSOut i) : SV_Target
{
    float2 px = i.pos.xy;
    float2 uv = px / g_view.xy;
    float tx = g_raw.x + uv.x * (g_raw.y - g_raw.x);
    float ty = -(g_raw.z + uv.y * (g_raw.w - g_raw.z));
    float2 deg = degrees(atan(float2(tx, ty)));
    float2 fw = max(fwidth(deg), 1e-4);

    float3 c = float3(0.05, 0.05, 0.06);
    float2 d5 = abs(frac(deg / 5 + 0.5) - 0.5) * 5 / fw;
    float2 d10 = abs(frac(deg / 10 + 0.5) - 0.5) * 10 / fw;
    float2 d0 = abs(deg) / fw;
    c = lerp(c, float3(0.15, 0.15, 0.17), saturate(1.0 - min(d5.x, d5.y)));
    c = lerp(c, float3(0.30, 0.30, 0.34), saturate(1.5 - min(d10.x, d10.y)));
    c = lerp(c, float3(0.50, 0.50, 0.56), saturate(2.0 - min(d0.x, d0.y)));

    float unit = g_view.y / 400;
    for (int k = 0; k < (int)g_count.x; ++k)
    {
        float d = length(px - TanToPx(g_targets[k].xy));
        float3 tc = g_targets[k].z > 0.5 ? float3(0.2, 0.55, 0.2) : float3(0.6, 0.6, 0.6);
        c = lerp(c, tc, Disc(d, 1.5 * unit));
    }
    if (g_active.w > 0.5)
    {
        float d = length(px - TanToPx(g_active.xy));
        float r = lerp(14, 4, saturate(g_view.w)) * unit;
        c = lerp(c, float3(1, 0.15, 0.1), Ring(d, r, 0.9 * unit));
        c = lerp(c, float3(1, 1, 1), Disc(d, 1.2 * unit));
    }
    if (g_mark[0].w > 0.5)  // A: green disc
    {
        float d = length(px - TanToPx(g_mark[0].xy));
        c = lerp(c, float3(0.1, 1.0, 0.2), 0.85 * Disc(d, 3.5 * unit));
    }
    if (g_mark[1].w > 0.5)  // B: blue ring
    {
        float d = length(px - TanToPx(g_mark[1].xy));
        c = lerp(c, float3(0.2, 0.7, 1.0), Ring(d, 5.5 * unit, 0.7 * unit));
    }
    if (g_mark[2].w > 0.5)  // C: yellow square
    {
        float2 q = abs(px - TanToPx(g_mark[2].xy));
        c = lerp(c, float3(1.0, 0.9, 0.1), saturate(1 - abs(max(q.x, q.y) - 8 * unit) / (0.7 * unit)));
    }
    if (g_mark[3].w > 0.5)  // D: purple cross
    {
        float2 q = abs(px - TanToPx(g_mark[3].xy));
        float cross = (min(q.x, q.y) < 0.6 * unit && max(q.x, q.y) < 5 * unit) ? 1 : 0;
        c = lerp(c, float3(1.0, 0.3, 1.0), cross);
    }
    return float4(c, 1);
}
)";

constexpr int kMaxTargets = 24;

struct alignas(16) Constants {
    float raw[4];
    float view[4];
    float active[4];
    float mark[4][4];
    float targets[kMaxTargets][4];
    float count[4];
};

struct Gfx {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<IDXGISwapChain1> swap;
    ComPtr<ID3D11RenderTargetView> backRtv;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11Buffer> cb;
    ComPtr<ID3D11Texture2D> eyeTex[2];
    ComPtr<ID3D11RenderTargetView> eyeRtv[2];
    uint32_t eyeW = 0, eyeH = 0, winW = 0, winH = 0;

    bool CompileShaders() {
        ComPtr<ID3DBlob> vsb, psb, err;
        HRESULT hr = D3DCompile(kShader, sizeof(kShader) - 1, "gaze.hlsl", nullptr, nullptr, "VSMain", "vs_5_0",
                                D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vsb, &err);
        if (FAILED(hr)) {
            Log("ERROR: VS compilation: %s", err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }
        hr = D3DCompile(kShader, sizeof(kShader) - 1, "gaze.hlsl", nullptr, nullptr, "PSMain", "ps_5_0",
                        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &psb, &err);
        if (FAILED(hr)) {
            Log("ERROR: PS compilation: %s", err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }
        dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
        D3D11_BUFFER_DESC bd = {sizeof(Constants), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        return SUCCEEDED(dev->CreateBuffer(&bd, nullptr, &cb)) && vs && ps;
    }

    bool Init(HWND hwnd, int adapterIndex, uint32_t ew, uint32_t eh, uint32_t ww, uint32_t wh) {
        eyeW = ew, eyeH = eh, winW = ww, winH = wh;
        ComPtr<IDXGIFactory2> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
        ComPtr<IDXGIAdapter1> adapter;
        if (adapterIndex >= 0) factory->EnumAdapters1((UINT)adapterIndex, &adapter);
        if (adapter) {
            DXGI_ADAPTER_DESC1 ad = {};
            adapter->GetDesc1(&ad);
            Log("Adapter %d (from GetDXGIOutputInfo): %s", adapterIndex, Narrow(ad.Description).c_str());
        } else {
            Log("Default adapter (GetDXGIOutputInfo returned %d)", adapterIndex);
        }
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
        HRESULT hr = D3D11CreateDevice(adapter.Get(), adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
                                       nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx);
        if (FAILED(hr)) {
            Log("ERROR: D3D11CreateDevice 0x%08lx", (unsigned long)hr);
            return false;
        }
        for (int e = 0; e < 2; ++e) {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = ew;
            td.Height = eh;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(dev->CreateTexture2D(&td, nullptr, &eyeTex[e]))) return false;
            dev->CreateRenderTargetView(eyeTex[e].Get(), nullptr, &eyeRtv[e]);
        }
        if (!hwnd) return CompileShaders();  // --preview mode: no window

        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = ww;
        sd.Height = wh;
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        hr = factory->CreateSwapChainForHwnd(dev.Get(), hwnd, &sd, nullptr, nullptr, &swap);
        if (FAILED(hr)) {
            Log("ERROR: CreateSwapChainForHwnd 0x%08lx", (unsigned long)hr);
            return false;
        }
        ComPtr<ID3D11Texture2D> back;
        swap->GetBuffer(0, IID_PPV_ARGS(&back));
        dev->CreateRenderTargetView(back.Get(), nullptr, &backRtv);
        return CompileShaders();
    }

    // Saves an eye texture to PNG (WIC); used in --preview mode.
    bool SavePng(int e, const fs::path& path) {
        D3D11_TEXTURE2D_DESC td = {};
        eyeTex[e]->GetDesc(&td);
        td.BindFlags = 0;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &staging))) return false;
        ctx->CopyResource(staging.Get(), eyeTex[e].Get());
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map))) return false;
        // RGBA -> BGRA, because the WIC PNG encoder natively takes 32bppBGRA.
        const UINT stride = td.Width * 4;
        std::vector<BYTE> px((size_t)stride * td.Height);
        for (UINT y = 0; y < td.Height; ++y) {
            const BYTE* src = (const BYTE*)map.pData + (size_t)y * map.RowPitch;
            BYTE* dst = px.data() + (size_t)y * stride;
            for (UINT x = 0; x < td.Width; ++x) {
                dst[x * 4 + 0] = src[x * 4 + 2];
                dst[x * 4 + 1] = src[x * 4 + 1];
                dst[x * 4 + 2] = src[x * 4 + 0];
                dst[x * 4 + 3] = src[x * 4 + 3];
            }
        }
        ctx->Unmap(staging.Get(), 0);

        ComPtr<IWICImagingFactory> wic;
        ComPtr<IWICStream> stream;
        ComPtr<IWICBitmapEncoder> enc;
        ComPtr<IWICBitmapFrameEncode> fr;
        WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
        HRESULT hr = S_OK;
        int step = 0;
        auto chk = [&](HRESULT r) { ++step; hr = r; return SUCCEEDED(r); };
        bool ok = chk(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))) &&
                  chk(wic->CreateStream(&stream)) && chk(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
                  chk(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) &&
                  chk(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache)) && chk(enc->CreateNewFrame(&fr, nullptr)) &&
                  chk(fr->Initialize(nullptr)) && chk(fr->SetSize(td.Width, td.Height)) && chk(fr->SetPixelFormat(&fmt)) &&
                  chk(fmt == GUID_WICPixelFormat32bppBGRA ? S_OK : E_FAIL) &&
                  chk(fr->WritePixels(td.Height, stride, (UINT)px.size(), px.data())) && chk(fr->Commit()) &&
                  chk(enc->Commit());
        if (!ok) Log("ERROR: PNG save, step %d, hr=0x%08lx", step, (unsigned long)hr);
        return ok;
    }

    void Draw(ID3D11RenderTargetView* rtv, uint32_t w, uint32_t h, const Constants& c) {
        ctx->UpdateSubresource(cb.Get(), 0, nullptr, &c, 0, 0);
        D3D11_VIEWPORT vp = {0, 0, (float)w, (float)h, 0, 1};
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        ctx->RSSetViewports(1, &vp);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(ps.Get(), nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, cb.GetAddressOf());
        ctx->Draw(3, 0);
    }
};

// ---------------------------------------------------------------- point sequence and statistics

enum Src { S_A_L, S_A_R, S_D, S_D_L, S_B, S_B_RAW, S_C_L, S_C_R, S_COUNT };

const char* const kSrcName[S_COUNT] = {
    "A_L   GetEyeTrackedFoveationCenter, left eye (NDC -> left eye matrix)",
    "A_R   GetEyeTrackedFoveationCenter, right eye (NDC -> right eye matrix)",
    "D     GetEyeTrackedFoveationCenterForProjection, custom symmetric projection",
    "D_L   GetEyeTrackedFoveationCenterForProjection, left eye matrix",
    "B     IVRInput GetEyeTrackingDataForNextFrame, converted to head space",
    "B_RAW IVRInput, raw data treated as head space",
    "C_L   diagnostic.json, left eye (tan of angles)",
    "C_R   diagnostic.json, right eye (tan of angles)",
};

struct Sample {
    bool ok[S_COUNT] = {};
    Vec2 t[S_COUNT];
};

struct Target {
    float xDeg, yDeg;
    Vec2 Tan() const { return {std::tan(xDeg / kRadToDeg), std::tan(yDeg / kRadToDeg)}; }
};

std::vector<Target> MakeTargets() {
    std::vector<Target> t = {{0, 0}};
    for (float y : {20.0f, 0.0f, -20.0f})
        for (float x : {-30.0f, -15.0f, 0.0f, 15.0f, 30.0f}) t.push_back({x, y});
    t.push_back({0, 0});
    return t;
}

double Median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void WriteSummary(int run, const std::vector<Target>& targets, const std::vector<std::vector<Sample>>& samples,
                  const int framesTotal[S_COUNT], const int framesOk[S_COUNT]) {
    Log("==================== SUMMARY of run %d ====================", run);
    Log("Errors in angular degrees. 'offset' = mean shift (x right, y up);");
    Log("'after offset' = error after removing the constant shift (precision); 'spread' = median sample deviation.");
    for (int s = 0; s < S_COUNT; ++s) {
        double pct = framesTotal[s] ? 100.0 * framesOk[s] / framesTotal[s] : 0.0;
        Log("--- %s", kSrcName[s]);
        Log("    valid frames: %d / %d (%.1f%%)", framesOk[s], framesTotal[s], pct);
        struct Row {
            int i;
            double mx, my, err, spread;
            size_t n;
        };
        std::vector<Row> rows;
        for (size_t i = 0; i < targets.size(); ++i) {
            std::vector<double> xs, ys;
            for (const Sample& sm : samples[i])
                if (sm.ok[s]) xs.push_back(sm.t[s].x), ys.push_back(sm.t[s].y);
            if (xs.size() < 5) continue;
            double mx = Median(xs), my = Median(ys);
            std::vector<double> dev;
            for (size_t k = 0; k < xs.size(); ++k) dev.push_back(AngleBetweenTan(xs[k], ys[k], mx, my));
            Vec2 tt = targets[i].Tan();
            rows.push_back({(int)i, mx, my, AngleBetweenTan(mx, my, tt.x, tt.y), Median(dev), xs.size()});
        }
        if (rows.empty()) {
            Log("    no data in the measurement windows");
            continue;
        }
        double offX = 0, offY = 0, sumErr = 0;
        for (const Row& r : rows) {
            offX += std::atan(r.mx) * kRadToDeg - targets[r.i].xDeg;
            offY += std::atan(r.my) * kRadToDeg - targets[r.i].yDeg;
            sumErr += r.err;
        }
        offX /= rows.size(), offY /= rows.size();
        double sumCorr = 0, sumFlipY = 0, sumFlipX = 0;
        for (const Row& r : rows) {
            Vec2 tt = targets[r.i].Tan();
            double cx = std::tan(std::atan(r.mx) - offX / kRadToDeg), cy = std::tan(std::atan(r.my) - offY / kRadToDeg);
            sumCorr += AngleBetweenTan(cx, cy, tt.x, tt.y);
            sumFlipY += AngleBetweenTan(r.mx, -r.my, tt.x, tt.y);
            sumFlipX += AngleBetweenTan(-r.mx, r.my, tt.x, tt.y);
            Log("    target %2d (%+5.1f, %+5.1f): median (%+6.2f, %+6.2f)  error %5.2f  spread %4.2f  n=%zu", r.i,
                targets[r.i].xDeg, targets[r.i].yDeg, std::atan(r.mx) * kRadToDeg, std::atan(r.my) * kRadToDeg, r.err,
                r.spread, r.n);
        }
        size_t n = rows.size();
        Log("    MEAN ERROR %.2f  | offset (%+.2f, %+.2f) | after offset %.2f | with Y flipped %.2f | with X flipped %.2f "
            "| targets %zu/%zu",
            sumErr / n, offX, offY, sumCorr / n, sumFlipY / n, sumFlipX / n, n, targets.size());
    }
    Log("==================================================================");
}

bool HasArg(const wchar_t* name) {
    int argc = 0;
    bool found = false;
    if (wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc)) {
        for (int i = 1; i < argc; ++i) found |= wcscmp(argv[i], name) == 0;
        LocalFree(argv);
    }
    return found;
}

// --preview: one left eye frame without OpenVR, with sample markers, to logs\preview_left.png.
int RunPreview(const fs::path& dir) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const uint32_t w = 1200, h = 1100;
    Gfx gfx;
    if (!gfx.Init(nullptr, -1, w, h, 0, 0)) return 3;
    const float raw[4] = {-1.45f, 0.95f, -1.10f, 1.05f};  // approximate asymmetric left eye frustum
    const std::vector<Target> targets = MakeTargets();
    const int cur = 4;
    Constants c = {};
    std::copy(std::begin(raw), std::end(raw), c.raw);
    c.view[0] = (float)w, c.view[1] = (float)h, c.view[3] = 0.4f;
    c.count[0] = (float)targets.size();
    for (size_t i = 0; i < targets.size(); ++i) {
        Vec2 t = targets[i].Tan();
        c.targets[i][0] = t.x, c.targets[i][1] = t.y, c.targets[i][2] = (int)i < cur ? 1.0f : 0.0f;
    }
    Vec2 t = targets[cur].Tan();
    c.active[0] = t.x, c.active[1] = t.y, c.active[3] = 1;
    const float off[4][2] = {{1.0f, -0.5f}, {-0.8f, 0.6f}, {1.5f, 1.2f}, {0.3f, -1.0f}};
    for (int m = 0; m < 4; ++m) {
        Target g = {targets[cur].xDeg + off[m][0], targets[cur].yDeg + off[m][1]};
        Vec2 gt = g.Tan();
        c.mark[m][0] = gt.x, c.mark[m][1] = gt.y, c.mark[m][3] = 1;
    }
    gfx.Draw(gfx.eyeRtv[0].Get(), w, h, c);
    bool ok = gfx.SavePng(0, dir / L"logs" / L"preview_left.png");
    Log("Preview: %s", ok ? "saved logs\\preview_left.png" : "PNG save ERROR");

    // Statistics check on synthetic data (noise +-0.3 degree). Expected: A_L offset (+1.00, -0.50),
    // after offset < 0.1; B error < 0.1; C_L large error, but "with Y flipped" < 0.1; other sources have no data.
    std::vector<std::vector<Sample>> ss(targets.size());
    int tot[S_COUNT] = {}, okc[S_COUNT] = {};
    uint32_t seed = 12345;
    auto noise = [&] {
        seed = seed * 1664525u + 1013904223u;
        return ((seed >> 8) / 16777216.0f - 0.5f) * 0.6f;  // +-0.3 degree
    };
    for (size_t i = 0; i < targets.size(); ++i) {
        for (int k = 0; k < 60; ++k) {
            Sample s;
            const Target& tg = targets[i];
            s.ok[S_A_L] = s.ok[S_B] = s.ok[S_C_L] = true;
            s.t[S_A_L] = Target{tg.xDeg + 1.0f + noise(), tg.yDeg - 0.5f + noise()}.Tan();
            s.t[S_B] = Target{tg.xDeg + noise(), tg.yDeg + noise()}.Tan();
            s.t[S_C_L] = Target{tg.xDeg + noise(), -tg.yDeg + noise()}.Tan();
            for (int src = 0; src < S_COUNT; ++src) ++tot[src], okc[src] += s.ok[src] ? 1 : 0;
            ss[i].push_back(s);
        }
    }
    WriteSummary(0, targets, ss, tot, okc);
    return ok ? 0 : 4;
}

// ---------------------------------------------------------------- window

bool g_running = true;
bool g_space = false;

LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_KEYDOWN:
            if (wp == VK_SPACE) g_space = true;
            if (wp == VK_ESCAPE) g_running = false;
            return 0;
        case WM_CLOSE:
        case WM_DESTROY:
            g_running = false;
            return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

HWND CreateMirrorWindow(HINSTANCE inst, uint32_t w, uint32_t h) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"QvfrGazeTest";
    RegisterClassW(&wc);
    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    RECT r = {0, 0, (LONG)w, (LONG)h};
    AdjustWindowRect(&r, style, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"QVFR gaze test", style, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left,
                              r.bottom - r.top, nullptr, nullptr, inst, nullptr);
    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

}  // namespace

// ---------------------------------------------------------------- main

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int) {
    const fs::path dir = ExeDir();
    Config cfg = LoadConfig(dir);

    std::error_code ec;
    fs::create_directories(dir / L"logs", ec);
    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t stamp[64];
    swprintf(stamp, 64, L"gaze_%04d%02d%02d_%02d%02d%02d_%s", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute,
             st.wSecond, cfg.legacy ? L"legacy" : L"modern");
    fs::path logPath = dir / L"logs" / (std::wstring(stamp) + L".log");
    fs::path csvPath = dir / L"logs" / (std::wstring(stamp) + L".csv");
    g_log = _wfopen(logPath.c_str(), L"wb");
    Log("qvfr_gaze_test, build " __DATE__ " " __TIME__ ", mode %s", cfg.legacy ? "legacy (openvr_api 0.9.12)" : "modern (openvr_api 2.15.6)");
    Log("Settings: render_scale=%.2f use_input=%d binding_path=%s controller_type=%s dwell=%.1f settle=%.1f "
        "autostart=%.1f focus_tan=%.2f",
        cfg.renderScale, (int)cfg.useInput, cfg.bindingPath.c_str(), cfg.controllerType.c_str(), cfg.dwell, cfg.settle,
        cfg.autostart, cfg.focusTan);
    if (HasArg(L"--preview")) return RunPreview(dir);
    g_csv = _wfopen(csvPath.c_str(), L"wb");
    if (!g_csv) {
        Log("Exit: cannot create %s", Narrow(csvPath.wstring()).c_str());
        return 5;
    }

    VrRuntime rt;
    if (!InitRuntime(rt, cfg.legacy, dir)) {
        Log("Exit: OpenVR did not start. Is SteamVR running and the HMD connected?");
        MessageBoxW(nullptr, L"OpenVR did not start. Details in logs\\*.log.", L"QVFR gaze test", MB_ICONERROR);
        return 1;
    }

    auto* sys = rt.Get<vr::IVRSystem>(vr::IVRSystem_Version);
    auto* comp = rt.Get<vr::IVRCompositor>(vr::IVRCompositor_Version);
    auto* settings = rt.Get<vr::IVRSettings>(vr::IVRSettings_Version);
    auto* input = cfg.useInput ? rt.Get<vr::IVRInput>(vr::IVRInput_Version) : nullptr;
    if (!sys) {
        Log("Exit: %s not available. Paths A and D are unavailable in this mode.", vr::IVRSystem_Version);
        rt.shutdown();
        return 2;
    }

    // --- system info
    Log("SteamVR runtime: %s", sys->GetRuntimeVersion());
    Log("HMD: %s / %s / %s, driver %s, controller_type=%s", DevStr(sys, vr::Prop_ManufacturerName_String).c_str(),
        DevStr(sys, vr::Prop_ModelNumber_String).c_str(), DevStr(sys, vr::Prop_TrackingSystemName_String).c_str(),
        DevStr(sys, vr::Prop_DriverVersion_String).c_str(), DevStr(sys, vr::Prop_ControllerType_String).c_str());
    {
        vr::ETrackedPropertyError pe = vr::TrackedProp_Success;
        bool eg = sys->GetBoolTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SupportsXrEyeGazeInteraction_Bool, &pe);
        float hz = sys->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float);
        Log("Prop_SupportsXrEyeGazeInteraction_Bool=%d (err %d), DisplayFrequency=%.1f Hz", (int)eg, (int)pe, hz);
    }
    if (settings) {
        const char* keys[] = {"allowEyeTracking", "eyeTrackingDominantEyeOnly", "motionSmoothing", "supersampleScale"};
        for (const char* k : keys) {
            vr::EVRSettingsError se = vr::VRSettingsError_None;
            if (strcmp(k, "supersampleScale") == 0) {
                float f = settings->GetFloat("steamvr", k, &se);
                Log("Setting steamvr/%s = %.3f (err %d)", k, f, (int)se);
            } else {
                bool b = settings->GetBool("steamvr", k, &se);
                Log("Setting steamvr/%s = %d (err %d)", k, (int)b, (int)se);
            }
        }
    }
    uint32_t recW = 0, recH = 0;
    sys->GetRecommendedRenderTargetSize(&recW, &recH);
    EyeInfo eye[2];
    for (int e = 0; e < 2; ++e) {
        auto ev = (vr::EVREye)e;
        sys->GetProjectionRaw(ev, &eye[e].raw[0], &eye[e].raw[1], &eye[e].raw[2], &eye[e].raw[3]);
        eye[e].proj = sys->GetProjectionMatrix(ev, 0.1f, 100.0f);
        vr::HmdMatrix34_t eh = sys->GetEyeToHeadTransform(ev);
        const float* r = eye[e].raw;
        const auto& m = eye[e].proj.m;
        Log("Eye %s: GetProjectionRaw L=%.5f R=%.5f T=%.5f B=%.5f (degrees: L %.2f R %.2f up %.2f down %.2f)",
            e ? "right" : "left", r[0], r[1], r[2], r[3], std::atan(r[0]) * kRadToDeg, std::atan(r[1]) * kRadToDeg,
            -std::atan(r[2]) * kRadToDeg, std::atan(r[3]) * kRadToDeg);
        Log("    GetProjectionMatrix: [%.5f %.5f %.5f %.5f] [%.5f %.5f %.5f %.5f] [%.5f %.5f %.5f %.5f] [%.5f %.5f %.5f %.5f]",
            m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3]);
        Log("    GetEyeToHeadTransform: [%.5f %.5f %.5f %.5f] [%.5f %.5f %.5f %.5f] [%.5f %.5f %.5f %.5f]", eh.m[0][0],
            eh.m[0][1], eh.m[0][2], eh.m[0][3], eh.m[1][0], eh.m[1][1], eh.m[1][2], eh.m[1][3], eh.m[2][0], eh.m[2][1],
            eh.m[2][2], eh.m[2][3]);
    }
    int32_t adapterIndex = -1;
    sys->GetDXGIOutputInfo(&adapterIndex);
    uint32_t eyeW = std::max(64u, (uint32_t)(recW * cfg.renderScale));
    uint32_t eyeH = std::max(64u, (uint32_t)(recH * cfg.renderScale));
    Log("GetRecommendedRenderTargetSize %ux%u, render %ux%u", recW, recH, eyeW, eyeH);

    // --- IVRInput (before the first PollNextEvent)
    InputPath ip;
    if (input) ip.Init(input, dir, cfg, DevStr(sys, vr::Prop_ControllerType_String));

    // --- window and D3D11
    uint32_t winH = 800, winW = (uint32_t)(800.0 * eyeW / eyeH);
    HWND hwnd = CreateMirrorWindow(inst, winW, winH);
    Gfx gfx;
    if (!gfx.Init(hwnd, adapterIndex, eyeW, eyeH, winW, winH)) {
        Log("Exit: D3D11 initialization error");
        rt.shutdown();
        return 3;
    }
    if (comp) comp->SetTrackingSpace(vr::TrackingUniverseSeated);

    DiagReader diag;
    diag.Init();

    const vr::HmdMatrix44_t focusProj = SymmetricProjection(cfg.focusTan);
    const std::vector<Target> targets = MakeTargets();
    std::vector<std::vector<Sample>> runSamples(targets.size());
    int framesTotal[S_COUNT] = {}, framesOk[S_COUNT] = {};

    enum class Phase { Waiting, Running, Free } phase = Phase::Waiting;
    int run = 0, cur = 0;
    double tFirstFrame = -1, tTarget = 0, tTitle = 0, tOrigins = 3.0;
    bool lastOk[S_COUNT] = {};
    int lastBErr = -999;
    int lastSubmitErr[2] = {0, 0};
    uint64_t frame = 0;

    fprintf(g_csv,
            "t,frame,run,phase,target,target_x_deg,target_y_deg,sampling,"
            "A_ok,A_lx,A_ly,A_rx,A_ry,D_ok,D_x,D_y,DL_ok,DL_x,DL_y,"
            "B_err,B_active,B_valid,B_tracked,B_ox,B_oy,B_oz,B_tx,B_ty,B_tz,B_tanx,B_tany,"
            "C_ok,C_valid,C_lax,C_lay,C_rax,C_ray,C_age,hmd_valid\n");

    Log("Loop start. Space = start/restart the sequence, Esc = quit.");
    while (g_running) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        vr::VREvent_t ev;
        while (sys->PollNextEvent(&ev, sizeof(ev))) {
            if (ev.eventType == vr::VREvent_Quit) {
                Log("VREvent_Quit");
                g_running = false;
            }
        }
        if (!g_running) break;

        // --- pose and compositor sync
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount] = {};
        if (comp) {
            vr::EVRCompositorError ce = comp->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
            if (ce != vr::VRCompositorError_None) Sleep(5);
        } else {
            Sleep(14);
            sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseSeated, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);
        }
        const double now = NowSec();
        if (tFirstFrame < 0) tFirstFrame = now;
        const vr::TrackedDevicePose_t& hmd = poses[vr::k_unTrackedDeviceIndex_Hmd];

        // --- gaze samples
        Sample smp;
        vr::HmdVector2_t ndcL = {}, ndcR = {}, ndcD = {}, ndcDL = {};
        bool aOk = sys->GetEyeTrackedFoveationCenter(&ndcL, &ndcR);
        bool dOk = sys->GetEyeTrackedFoveationCenterForProjection(&focusProj, &ndcD);
        bool dlOk = sys->GetEyeTrackedFoveationCenterForProjection(&eye[0].proj, &ndcDL);
        smp.ok[S_A_L] = smp.ok[S_A_R] = aOk;
        smp.t[S_A_L] = NdcToTan(eye[0].proj, ndcL);
        smp.t[S_A_R] = NdcToTan(eye[1].proj, ndcR);
        smp.ok[S_D] = dOk;
        smp.t[S_D] = NdcToTan(focusProj, ndcD);
        smp.ok[S_D_L] = dlOk;
        smp.t[S_D_L] = NdcToTan(eye[0].proj, ndcDL);

        vr::VREyeTrackingData_t et = {};
        int bErr = -1;
        if (ip.ok) {
            vr::VRActiveActionSet_t as = {};
            as.ulActionSet = ip.set;
            input->UpdateActionState(&as, sizeof(as), 1);
            bErr = (int)input->GetEyeTrackingDataForNextFrame(ip.action, vr::TrackingUniverseSeated, &et, sizeof(et));
            if (bErr != lastBErr) {
                Log("IVRInput::GetEyeTrackingDataForNextFrame -> %d (active %d valid %d tracked %d)", bErr, (int)et.bActive,
                    (int)et.bValid, (int)et.bTracked);
                lastBErr = bErr;
            }
            if (bErr == vr::VRInputError_None && et.bValid) {
                float oh[3], th[3];
                WorldToHead(hmd.mDeviceToAbsoluteTracking, et.vGazeOrigin.v, oh);
                WorldToHead(hmd.mDeviceToAbsoluteTracking, et.vGazeTarget.v, th);
                float d[3] = {th[0] - oh[0], th[1] - oh[1], th[2] - oh[2]};
                float dr[3] = {et.vGazeTarget.v[0] - et.vGazeOrigin.v[0], et.vGazeTarget.v[1] - et.vGazeOrigin.v[1],
                               et.vGazeTarget.v[2] - et.vGazeOrigin.v[2]};
                smp.ok[S_B] = hmd.bPoseIsValid && DirToTan(d, smp.t[S_B]);
                smp.ok[S_B_RAW] = DirToTan(dr, smp.t[S_B_RAW]);
            }
        }

        diag.Poll(now);
        if (diag.ok && diag.valid) {
            smp.ok[S_C_L] = smp.ok[S_C_R] = true;
            smp.t[S_C_L] = {(float)std::tan(diag.la[0]), (float)std::tan(diag.la[1])};
            smp.t[S_C_R] = {(float)std::tan(diag.ra[0]), (float)std::tan(diag.ra[1])};
        }

        for (int s = 0; s < S_COUNT; ++s) {
            if (smp.ok[s] != lastOk[s]) {
                Log("Source %.5s: %s", kSrcName[s], smp.ok[s] ? "data PRESENT" : "no data");
                lastOk[s] = smp.ok[s];
            }
        }
        if (now - tFirstFrame > tOrigins) {
            ip.LogOrigins();
            tOrigins += 30.0;
        }

        // --- point sequence
        bool start = g_space;
        g_space = false;
        int activeIdx = -1;
        float progress = 0;
        bool sampling = false;
        if (phase == Phase::Waiting) {
            activeIdx = 0;
            progress = cfg.autostart > 0 ? (float)((now - tFirstFrame) / cfg.autostart) : 1.0f;
            if (start || progress >= 1.0f) {
                phase = Phase::Running, cur = 0, tTarget = now, ++run;
                Log("Run %d: sequence start, %zu points", run, targets.size());
            }
        } else if (phase == Phase::Free && start) {
            phase = Phase::Running, cur = 0, tTarget = now, ++run;
            for (auto& v : runSamples) v.clear();
            std::fill(std::begin(framesTotal), std::end(framesTotal), 0);
            std::fill(std::begin(framesOk), std::end(framesOk), 0);
            Log("Run %d: sequence restart", run);
        }
        if (phase == Phase::Running) {
            double el = now - tTarget;
            if (el >= cfg.dwell) {
                if (++cur >= (int)targets.size()) {
                    phase = Phase::Free;
                    WriteSummary(run, targets, runSamples, framesTotal, framesOk);
                } else {
                    tTarget = now, el = 0;
                }
            }
            if (phase == Phase::Running) {
                activeIdx = cur;
                progress = (float)(el / cfg.dwell);
                sampling = el >= cfg.settle;
                for (int s = 0; s < S_COUNT; ++s) {
                    if (!sampling) continue;
                    ++framesTotal[s];
                    framesOk[s] += smp.ok[s] ? 1 : 0;
                }
                if (sampling) runSamples[cur].push_back(smp);
            }
        }

        // --- CSV
        {
            const Target* tg = activeIdx >= 0 ? &targets[activeIdx] : nullptr;
            fprintf(g_csv,
                    "%.4f,%llu,%d,%d,%d,%.1f,%.1f,%d,"
                    "%d,%.5f,%.5f,%.5f,%.5f,%d,%.5f,%.5f,%d,%.5f,%.5f,"
                    "%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
                    "%d,%d,%.5f,%.5f,%.5f,%.5f,%.3f,%d\n",
                    now, (unsigned long long)frame, run, (int)phase, activeIdx, tg ? tg->xDeg : 0.0f, tg ? tg->yDeg : 0.0f,
                    (int)sampling, (int)aOk, ndcL.v[0], ndcL.v[1], ndcR.v[0], ndcR.v[1], (int)dOk, ndcD.v[0], ndcD.v[1],
                    (int)dlOk, ndcDL.v[0], ndcDL.v[1], bErr, (int)et.bActive, (int)et.bValid, (int)et.bTracked,
                    et.vGazeOrigin.v[0], et.vGazeOrigin.v[1], et.vGazeOrigin.v[2], et.vGazeTarget.v[0],
                    et.vGazeTarget.v[1], et.vGazeTarget.v[2], smp.ok[S_B] ? smp.t[S_B].x : 0.0f,
                    smp.ok[S_B] ? smp.t[S_B].y : 0.0f, (int)diag.ok, (int)diag.valid, diag.la[0], diag.la[1], diag.ra[0],
                    diag.ra[1], diag.ok ? now - diag.lastChange : -1.0, (int)hmd.bPoseIsValid);
        }

        // --- render
        Constants c = {};
        c.view[3] = progress;
        if (activeIdx >= 0) {
            Vec2 t = targets[activeIdx].Tan();
            c.active[0] = t.x, c.active[1] = t.y, c.active[3] = 1;
        }
        c.count[0] = (float)std::min<size_t>(targets.size(), kMaxTargets);
        for (size_t i = 0; i < targets.size() && i < (size_t)kMaxTargets; ++i) {
            Vec2 t = targets[i].Tan();
            bool done = phase == Phase::Free || (phase == Phase::Running && (int)i < cur);
            c.targets[i][0] = t.x, c.targets[i][1] = t.y, c.targets[i][2] = done ? 1.0f : 0.0f;
        }
        auto setMark = [&](int slot, Src s) {
            c.mark[slot][0] = smp.t[s].x, c.mark[slot][1] = smp.t[s].y, c.mark[slot][3] = smp.ok[s] ? 1.0f : 0.0f;
        };
        for (int e = 0; e < 2; ++e) {
            std::copy(std::begin(eye[e].raw), std::end(eye[e].raw), c.raw);
            setMark(0, e ? S_A_R : S_A_L);
            setMark(1, S_B);
            setMark(2, e ? S_C_R : S_C_L);
            setMark(3, S_D);
            c.view[0] = (float)eyeW, c.view[1] = (float)eyeH;
            gfx.Draw(gfx.eyeRtv[e].Get(), eyeW, eyeH, c);
            if (comp) {
                vr::Texture_t tex = {gfx.eyeTex[e].Get(), vr::TextureType_DirectX, vr::ColorSpace_Gamma};
                int se = (int)comp->Submit((vr::EVREye)e, &tex, nullptr, vr::Submit_Default);
                if (se != lastSubmitErr[e]) {
                    Log("IVRCompositor::Submit(eye %d) -> %d", e, se);
                    lastSubmitErr[e] = se;
                }
            }
            if (e == 0) {  // desktop window: left eye view
                c.view[0] = (float)winW, c.view[1] = (float)winH;
                gfx.Draw(gfx.backRtv.Get(), winW, winH, c);
            }
        }
        gfx.swap->Present(0, 0);

        if (now - tTitle > 0.5) {
            tTitle = now;
            wchar_t title[256];
            const wchar_t* ph = phase == Phase::Waiting ? L"waiting (space = start)" : phase == Phase::Running ? L"measuring" : L"free (space = repeat)";
            swprintf(title, 256, L"QVFR gaze test [%s] | %s %d/%zu | A:%s B:%s C:%s D:%s", cfg.legacy ? L"legacy" : L"modern",
                     ph, activeIdx + 1, targets.size(), smp.ok[S_A_L] ? L"ok" : L"-", smp.ok[S_B] ? L"ok" : L"-",
                     smp.ok[S_C_L] ? L"ok" : L"-", smp.ok[S_D] ? L"ok" : L"-");
            SetWindowTextW(hwnd, title);
        }
        ++frame;
    }

    if (phase == Phase::Running && run > 0) {
        Log("Aborted during run %d; partial summary:", run);
        WriteSummary(run, targets, runSamples, framesTotal, framesOk);
    }
    Log("Exit, frames: %llu", (unsigned long long)frame);
    if (rt.shutdown) rt.shutdown();
    if (g_csv) fclose(g_csv);
    if (g_log) fclose(g_log);
    return 0;
}

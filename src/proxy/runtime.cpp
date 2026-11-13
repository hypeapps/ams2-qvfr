#include "runtime.h"

#include "common.h"
#include "diag.h"
#include "xr_layer.h"

namespace qvfr::runtime {

namespace {

bool g_openxr = false;
std::wstring g_backend;

std::wstring GetEnv(const wchar_t* name) {
    wchar_t buf[4096];
    DWORD n = GetEnvironmentVariableW(name, buf, 4096);
    return n > 0 && n < 4096 ? std::wstring(buf, n) : std::wstring();
}

// Puts our value first and keeps what the player already had (';' separates paths and layer names on Windows).
void PrependEnv(const wchar_t* name, const std::wstring& value) {
    const std::wstring old = GetEnv(name);
    const std::wstring v = old.empty() ? value : value + L";" + old;
    SetEnvironmentVariableW(name, v.c_str());
    Log("Runtime: %s=%s%s", Narrow(name).c_str(), Narrow(v).c_str(), old.empty() ? "" : " (the previous value kept after ours)");
}

bool Elevated() {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) return false;
    TOKEN_ELEVATION e{};
    DWORD n = 0;
    BOOL ok = GetTokenInformation(tok, TokenElevation, &e, sizeof(e), &n);
    CloseHandle(tok);
    return ok && e.TokenIsElevated;
}

std::wstring Lower(std::wstring s) {
    for (auto& ch : s) ch = (wchar_t)towlower(ch);
    return s;
}

DWORD RegDword(HKEY key, const wchar_t* name, DWORD def) {
    DWORD v = 0, len = sizeof(v);
    return RegGetValueW(key, nullptr, name, RRF_RT_REG_DWORD, nullptr, &v, &len) == ERROR_SUCCESS ? v : def;
}

std::string g_layersOn, g_layersOff;  // "folder\file.json" of the enabled and disabled implicit layers

// OpenXR Toolkit keeps its settings per application in HKCU\SOFTWARE\OpenXR_Toolkit\<application name>; under
// OpenComposite the name is "OpenComposite_<exe name>". The mod never changes them (decision M8.4).
void CheckToolkit() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::wstring app = L"OpenComposite_" + fs::path(exe).stem().wstring();
    HKEY key = nullptr;
    DWORD bypass = 0, eye = 0, vrs = 0, scaling = 0;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, (L"SOFTWARE\\OpenXR_Toolkit\\" + app).c_str(), 0, KEY_READ, &key) == ERROR_SUCCESS) {
        bypass = RegDword(key, L"bypass", 0);
        eye = RegDword(key, L"eye_tracking", 0);
        vrs = RegDword(key, L"vrs", 0);
        scaling = RegDword(key, L"scaling_type", 0);
        RegCloseKey(key);
    }
    char buf[256];
    if (bypass) {
        snprintf(buf, sizeof(buf), "installed, off for %s (bypass=1)", Narrow(app).c_str());
        diag::Set("OpenXR Toolkit", buf);
        return;
    }
    snprintf(buf, sizeof(buf), "ON for %s (eye_tracking=%lu, vrs=%lu, scaling_type=%lu)", Narrow(app).c_str(), eye, vrs,
             scaling);
    diag::Set("OpenXR Toolkit", buf);
    if (eye) {
        diag::Report(diag::Level::Warning, "toolkit",
                     "OpenXR Toolkit is on for AMS2 with its Eye tracking on. The Toolkit then attaches its own eye "
                     "tracking actions, and OpenComposite stops the game at start with the error "
                     "XR_ERROR_ACTIONSETS_ALREADY_ATTACHED. Turn off Eye tracking in the Toolkit for AMS2, or turn the "
                     "Toolkit off for AMS2.");
        return;
    }
    std::string extra;
    if (vrs) extra += " Its Foveated rendering is on (vrs=" + std::to_string(vrs) + ").";
    if (scaling) extra += " Its upscaling is on (scaling_type=" + std::to_string(scaling) + ").";
    diag::Report(diag::Level::Warning, "toolkit",
                 "OpenXR Toolkit is on for AMS2. Keep its Eye tracking, Foveated rendering and upscaling off for AMS2 "
                 "(the mod does the foveated rendering).%s If the game stops at start with the OpenComposite error "
                 "XR_ERROR_ACTIONSETS_ALREADY_ATTACHED (some Toolkit builds attach eye tracking actions anyway), turn the "
                 "Toolkit off for AMS2.",
                 extra.c_str());
}

void LogImplicitLayers(HKEY root, const char* rootName) {
    HKEY key = nullptr;
    if (RegOpenKeyExW(root, L"SOFTWARE\\Khronos\\OpenXR\\1\\ApiLayers\\Implicit", 0, KEY_READ, &key) != ERROR_SUCCESS) {
        Log("Runtime: OpenXR implicit layers in %s: none", rootName);
        return;
    }
    for (DWORD i = 0;; ++i) {
        wchar_t name[1024];
        DWORD nameLen = 1024, type = 0, data = 0, dataLen = sizeof(data);
        LONG r = RegEnumValueW(key, i, name, &nameLen, nullptr, &type, (BYTE*)&data, &dataLen);
        if (r == ERROR_NO_MORE_ITEMS) break;
        if (r != ERROR_SUCCESS) continue;
        const bool on = type == REG_DWORD && data == 0;
        Log("Runtime: OpenXR implicit layer (%s) %s: %s", rootName, Narrow(name).c_str(),
            type != REG_DWORD ? "unknown value type" : on ? "enabled" : "disabled");
        const fs::path manifest = name;
        const std::string shortName =
            Narrow((manifest.parent_path().filename() / manifest.filename()).wstring());
        std::string& list = on ? g_layersOn : g_layersOff;
        list += (list.empty() ? "" : ", ") + shortName;
        if (!on) continue;
        const std::wstring lower = Lower(name);
        if (lower.find(L"xr_apilayer_mbucchia_toolkit") != std::wstring::npos) CheckToolkit();
        if (lower.find(L"quad-views-foveated") != std::wstring::npos)
            diag::Report(diag::Level::Info, "mbucchia-quad-views",
                         "The Quad-Views-Foveated OpenXR layer (mbucchia) is enabled. It has no effect on AMS2: the mod "
                         "never asks the runtime for quad views. Quad views for AMS2 are set in %s.",
                         Narrow(IniFile().wstring()).c_str());
    }
    RegCloseKey(key);
}

void LogActiveRuntime() {
    wchar_t buf[1024] = {};
    DWORD len = sizeof(buf);
    LONG r = RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Khronos\\OpenXR\\1", L"ActiveRuntime", RRF_RT_REG_SZ, nullptr, buf,
                          &len);
    const std::wstring env = GetEnv(L"XR_RUNTIME_JSON");
    const std::string active = r == ERROR_SUCCESS ? Narrow(buf) : "(none)";
    Log("Runtime: OpenXR active runtime in the system: %s%s%s", active.c_str(),
        env.empty() ? "" : "; XR_RUNTIME_JSON set outside the mod: ", Narrow(env).c_str());
    diag::Set("OpenXR active runtime", env.empty() ? active : Narrow(env) + " (XR_RUNTIME_JSON, set outside the mod)");
    const std::wstring used = Lower(env.empty() ? std::wstring(r == ERROR_SUCCESS ? buf : L"") : env);
    if (used.find(L"pimax") != std::wstring::npos)
        diag::Report(diag::Level::Info, "pimax-quad-views",
                     "The active OpenXR runtime is Pimax's. Its quad views (Pimax OpenXR Quadviews in Pimax Play) have no "
                     "effect on AMS2: the mod never asks the runtime for quad views. Quad views for AMS2 are set in %s.",
                     Narrow(IniFile().wstring()).c_str());
}

}  // namespace

// The release layout (docs/architecture.md): x64\openvr_api.dll = the proxy, x64\qvfr\openvr\ = the Valve DLL,
// x64\qvfr\opencomposite\ = OpenComposite, x64\qvfr\XR_APILAYER_AMS2QVFR_eye_gaze.json = the layer manifest (shipped,
// library_path "..\openvr_api.dll" relative to it). The mod never chooses the OpenXR runtime.
std::wstring Prepare(HMODULE self) {
    wchar_t selfPath[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(self, selfPath, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    const fs::path dir = fs::path(selfPath).parent_path();
    const fs::path qvfr = dir / L"qvfr";
    const Config& c = Cfg();
    g_openxr = c.pathOpenXr;
    if (c.fovCrop.Active()) {
        diag::Set("FOV crop", "requested; OpenXR layer has not applied it yet");
        if (!g_openxr || !c.openxrLayer)
            diag::Report(diag::Level::Warning, "crop-path",
                         "FOV cropping requires path=openxr and openxr_layer=1. The requested crop is not active.");
    }
    if (!g_openxr) {
        g_backend = (qvfr / L"openvr" / L"openvr_api.dll").wstring();
        return g_backend;
    }
    fs::path oc = c.openComposite.empty() ? qvfr / L"opencomposite" / L"openvr_api.dll" : fs::path(c.openComposite);
    if (oc.is_relative()) oc = dir / oc;
    g_backend = oc.lexically_normal().wstring();
    Log("Runtime: path=openxr, backend OpenComposite %s", Narrow(g_backend).c_str());
    if (!c.openxrLayer) {
        Log("Runtime: openxr_layer=0: no gaze on the OpenXR path (FFR only)");
        diag::Set("OpenXR mod layer", "off (openxr_layer=0): no eye tracking on path=openxr");
        return g_backend;
    }
    const fs::path manifest = qvfr / (Widen(xrlayer::kLayerName) + L".json");
    std::error_code ec;
    if (!fs::exists(manifest, ec)) {
        Log("Runtime: ERROR: the OpenXR layer manifest %s is missing (reinstall the mod); no gaze on the OpenXR path",
            Narrow(manifest.wstring()).c_str());
        diag::Report(diag::Level::Serious, "layer-manifest",
                     "The mod's file %s is missing, so there is no eye tracking on path=openxr (the focus area stays "
                     "straight ahead). Reinstall the mod.",
                     Narrow(manifest.wstring()).c_str());
        return g_backend;
    }
    if (Elevated())
        diag::Report(diag::Level::Warning, "elevated",
                     "The game runs as administrator. The OpenXR loader then ignores the mod's environment, so the mod's "
                     "OpenXR layer does not load and there is no eye tracking on path=openxr. Start the game normally.");
    PrependEnv(L"XR_API_LAYER_PATH", qvfr.wstring());
    PrependEnv(L"XR_ENABLE_API_LAYERS", Widen(xrlayer::kLayerName));
    diag::Set("OpenXR mod layer", "set up for the OpenXR loader, not loaded yet");
    return g_backend;
}

void LogStartup() {
    diag::Set("Path", g_openxr ? "openxr (OpenComposite to the system's OpenXR runtime)"
                               : "openvr (the Valve openvr_api.dll to SteamVR)");
    if (!g_openxr) return;
    LogActiveRuntime();
    LogImplicitLayers(HKEY_LOCAL_MACHINE, "HKLM");
    LogImplicitLayers(HKEY_CURRENT_USER, "HKCU");
    diag::Set("OpenXR layers", "enabled: " + (g_layersOn.empty() ? std::string("none") : g_layersOn) +
                                   "; disabled: " + (g_layersOff.empty() ? std::string("none") : g_layersOff));
    Log("Runtime: the loader puts implicit layers closest to the game and XR_ENABLE_API_LAYERS layers (ours) after "
        "them, closest to the runtime");
}

bool OpenXrPath() { return g_openxr; }

}  // namespace qvfr::runtime

#include "runtime.h"

#include "common.h"
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
        Log("Runtime: OpenXR implicit layer (%s) %s: %s", rootName, Narrow(name).c_str(),
            type != REG_DWORD ? "unknown value type" : data == 0 ? "enabled" : "disabled");
    }
    RegCloseKey(key);
}

void LogActiveRuntime() {
    wchar_t buf[1024] = {};
    DWORD len = sizeof(buf);
    LONG r = RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Khronos\\OpenXR\\1", L"ActiveRuntime", RRF_RT_REG_SZ, nullptr, buf,
                          &len);
    const std::wstring env = GetEnv(L"XR_RUNTIME_JSON");
    Log("Runtime: OpenXR active runtime in the system: %s%s%s", r == ERROR_SUCCESS ? Narrow(buf).c_str() : "(none)",
        env.empty() ? "" : "; XR_RUNTIME_JSON set outside the mod: ", Narrow(env).c_str());
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
        return g_backend;
    }
    const fs::path manifest = qvfr / (Widen(xrlayer::kLayerName) + L".json");
    std::error_code ec;
    if (!fs::exists(manifest, ec)) {
        Log("Runtime: ERROR: the OpenXR layer manifest %s is missing (reinstall the mod); no gaze on the OpenXR path",
            Narrow(manifest.wstring()).c_str());
        return g_backend;
    }
    if (Elevated())
        Log("Runtime: WARNING: the game runs elevated (as administrator); the OpenXR loader then ignores the "
            "environment, so the layer does not load and there is no gaze");
    PrependEnv(L"XR_API_LAYER_PATH", qvfr.wstring());
    PrependEnv(L"XR_ENABLE_API_LAYERS", Widen(xrlayer::kLayerName));
    return g_backend;
}

void LogStartup() {
    if (!g_openxr) return;
    LogActiveRuntime();
    LogImplicitLayers(HKEY_LOCAL_MACHINE, "HKLM");
    LogImplicitLayers(HKEY_CURRENT_USER, "HKCU");
    Log("Runtime: the loader puts implicit layers closest to the game and XR_ENABLE_API_LAYERS layers (ours) after "
        "them, closest to the runtime");
}

bool OpenXrPath() { return g_openxr; }

}  // namespace qvfr::runtime

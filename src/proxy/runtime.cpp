#include "runtime.h"

#include <shlobj.h>

#include <cstdio>

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

// SteamVR's OpenXR runtime manifest from %LOCALAPPDATA%\openvr\openvrpaths.vrpath (the first "runtime" entry).
std::wstring SteamVrRuntimeJson() {
    PWSTR local = nullptr;
    if (FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local))) return {};
    fs::path p = fs::path(local) / L"openvr" / L"openvrpaths.vrpath";
    CoTaskMemFree(local);
    std::string text;
    if (FILE* f = _wfopen(p.c_str(), L"rb")) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
    }
    size_t k = text.find("\"runtime\"");
    if (k == std::string::npos) return {};
    size_t b = text.find('[', k), q = b == std::string::npos ? b : text.find('"', b);
    if (q == std::string::npos) return {};
    std::string dir;
    for (size_t i = q + 1; i < text.size() && text[i] != '"'; ++i) {
        if (text[i] == '\\' && i + 1 < text.size()) ++i;  // \\ and \/ in JSON
        dir += text[i];
    }
    if (dir.empty()) return {};
    return (fs::path(Widen(dir)) / L"steamxr_win64.json").wstring();
}

std::string JsonEscape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '\\' || ch == '"') o += '\\';
        o += ch;
    }
    return o;
}

// The layer's manifest, written at every start so library_path is always this proxy DLL.
bool WriteManifest(const fs::path& dir, const fs::path& self) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path file = dir / (Widen(xrlayer::kLayerName) + L".json");
    FILE* f = _wfopen(file.c_str(), L"wb");
    if (!f) {
        Log("Runtime: ERROR: cannot write the OpenXR layer manifest %s", Narrow(file.wstring()).c_str());
        return false;
    }
    fprintf(f,
            "{\n"
            "  \"file_format_version\": \"1.0.0\",\n"
            "  \"api_layer\": {\n"
            "    \"name\": \"%s\",\n"
            "    \"library_path\": \"%s\",\n"
            "    \"api_version\": \"1.0\",\n"
            "    \"implementation_version\": \"1\",\n"
            "    \"description\": \"ams2-qvfr eye gaze for the openvr_api.dll proxy (enabled only in AMS2 by the proxy)\"\n"
            "  }\n"
            "}\n",
            xrlayer::kLayerName, JsonEscape(Narrow(self.wstring())).c_str());
    fclose(f);
    Log("Runtime: OpenXR layer manifest %s (library_path = the proxy)", Narrow(file.wstring()).c_str());
    return true;
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
    Log("Runtime: OpenXR active runtime in the system: %s", r == ERROR_SUCCESS ? Narrow(buf).c_str() : "(none)");
}

}  // namespace

std::wstring Prepare(HMODULE self) {
    wchar_t selfPath[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(self, selfPath, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    const fs::path dir = fs::path(selfPath).parent_path();
    const Config& c = Cfg();
    g_openxr = c.pathOpenXr;
    if (!g_openxr) {
        g_backend = (dir / L"openvr_api_orig.dll").wstring();
        return g_backend;
    }
    fs::path oc = c.openComposite.empty() ? dir / L"qvfr" / L"opencomposite" / L"openvr_api.dll" : fs::path(c.openComposite);
    if (oc.is_relative()) oc = dir / oc;
    g_backend = oc.lexically_normal().wstring();
    Log("Runtime: path=openxr, backend OpenComposite %s", Narrow(g_backend).c_str());
    if (Elevated())
        Log("Runtime: WARNING: the game runs elevated (as administrator); the OpenXR loader then ignores the "
            "environment, so neither the layer nor openxr_runtime work");

    if (!c.openxrRuntime.empty()) {
        std::wstring json = c.openxrRuntime == L"steamvr" ? SteamVrRuntimeJson() : c.openxrRuntime;
        std::error_code ec;
        if (json.empty() || !fs::exists(json, ec)) {
            Log("Runtime: ERROR: openxr_runtime=%s: runtime manifest %s not found; the system's runtime is used",
                Narrow(c.openxrRuntime).c_str(), json.empty() ? "(SteamVR not found)" : Narrow(json).c_str());
        } else {
            const std::wstring old = GetEnv(L"XR_RUNTIME_JSON");
            SetEnvironmentVariableW(L"XR_RUNTIME_JSON", json.c_str());
            Log("Runtime: XR_RUNTIME_JSON=%s (only for the game%s)", Narrow(json).c_str(),
                old.empty() ? "" : (", replaces " + Narrow(old)).c_str());
        }
    }
    if (c.openxrLayer) {
        const fs::path md = BaseDir() / L"openxr";
        if (WriteManifest(md, selfPath)) {
            PrependEnv(L"XR_API_LAYER_PATH", md.wstring());
            PrependEnv(L"XR_ENABLE_API_LAYERS", Widen(xrlayer::kLayerName));
        }
    } else {
        Log("Runtime: openxr_layer=0: no gaze on the OpenXR path (FFR only)");
    }
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

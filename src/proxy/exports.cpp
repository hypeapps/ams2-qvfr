// openvr_api.dll proxy exports. AMS2 imports VR_Init, VR_Shutdown, VRCompositor, VROverlay
// and VR_GetVRInitErrorAsEnglishDescription; the other main functions are handled here too. The remaining exports
// (flat C API, Unity) are forwarded to the original by generated/forwards.inc.
//
// Original: openvr_api_orig.dll next to the proxy (the official Valve 0.9.12 DLL, the same one the game ships).

#include <windows.h>

#include <openvr.h>  // 0.9.12 with VR_API_EXPORT (defined in CMake), so the declarations are dllexport

#include <bcrypt.h>

#include <cstring>
#include <mutex>
#include <set>
#include <string>

#include "common.h"
#include "experiment.h"
#include "frame.h"
#include "modern_vr.h"
#include "probe.h"
#include "quad.h"
#include "sysinfo.h"
#include "wrappers.h"

#include "generated/forwards.inc"

namespace {

using namespace qvfr;

HMODULE g_self = nullptr;
HMODULE g_orig = nullptr;
std::once_flag g_once;

constexpr const char* kOrigMd5 = "864be5a92276ce2b902b6f3a0f850703";  // official openvr_api.dll 0.9.12 (win64)

struct Orig {
    vr::IVRSystem*(VR_CALLTYPE* init)(vr::EVRInitError*, vr::EVRApplicationType) = nullptr;
    void(VR_CALLTYPE* shutdown)() = nullptr;
    bool(VR_CALLTYPE* isHmdPresent)() = nullptr;
    bool(VR_CALLTYPE* isRuntimeInstalled)() = nullptr;
    void*(VR_CALLTYPE* getGeneric)(const char*, vr::EVRInitError*) = nullptr;
    const char*(VR_CALLTYPE* errSymbol)(vr::EVRInitError) = nullptr;
    const char*(VR_CALLTYPE* errEnglish)(vr::EVRInitError) = nullptr;
    const char*(VR_CALLTYPE* hmdError)(int) = nullptr;
    vr::IVRSystem*(VR_CALLTYPE* system)() = nullptr;
    vr::IVRCompositor*(VR_CALLTYPE* compositor)() = nullptr;
    vr::IVROverlay*(VR_CALLTYPE* overlay)() = nullptr;
} g_fn;

const char kNoOrig[] = "ams2-qvfr: openvr_api_orig.dll not found next to the proxy";

wchar_t g_origPath[MAX_PATH + 32] = {};
DWORD g_loadError = 0;
const char* g_missing[16] = {};
int g_missingCount = 0;

template <class T>
void Resolve(T& fn, const char* name) {
    fn = reinterpret_cast<T>(GetProcAddress(g_orig, name));
    if (!fn && g_missingCount < 16) g_missing[g_missingCount++] = name;
}

// Loads the real DLL with plain Win32 calls that cannot throw, before any diagnostics: a failing log,
// ini or hash must never keep the game from starting VR.
void LoadOriginal() noexcept {
    wchar_t self[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(g_self, self, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        g_loadError = n ? ERROR_INSUFFICIENT_BUFFER : GetLastError();
        return;
    }
    wchar_t* slash = wcsrchr(self, L'\\');
    size_t dirLen = slash ? (size_t)(slash - self + 1) : 0;
    wmemcpy(g_origPath, self, dirLen);
    wcscpy_s(g_origPath + dirLen, _countof(g_origPath) - dirLen, L"openvr_api_orig.dll");
    g_orig = LoadLibraryExW(g_origPath, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_orig || g_orig == g_self) {
        g_loadError = g_orig ? ERROR_INVALID_HANDLE : GetLastError();
        g_orig = nullptr;
        return;
    }
    Resolve(g_fn.init, "VR_Init");
    Resolve(g_fn.shutdown, "VR_Shutdown");
    Resolve(g_fn.isHmdPresent, "VR_IsHmdPresent");
    Resolve(g_fn.isRuntimeInstalled, "VR_IsRuntimeInstalled");
    Resolve(g_fn.getGeneric, "VR_GetGenericInterface");
    Resolve(g_fn.errSymbol, "VR_GetVRInitErrorAsSymbol");
    Resolve(g_fn.errEnglish, "VR_GetVRInitErrorAsEnglishDescription");
    Resolve(g_fn.hmdError, "VR_GetStringForHmdError");
    Resolve(g_fn.system, "VRSystem");
    Resolve(g_fn.compositor, "VRCompositor");
    Resolve(g_fn.overlay, "VROverlay");
}

void LogStartup() {
    wchar_t self[MAX_PATH] = {};
    GetModuleFileNameW(g_self, self, MAX_PATH);
    Log("ams2-qvfr openvr_api.dll proxy, build " __DATE__ " " __TIME__);
    Log("Proxy: %s", Narrow(self).c_str());
    sysinfo::LogProcessInfo();

    fs::path orig = g_origPath;
    std::error_code ec;
    auto size = fs::file_size(orig, ec);
    std::string md5 = sysinfo::HashFile(orig, BCRYPT_MD5_ALGORITHM);
    Log("Original: %s (%llu B, MD5 %s%s)", Narrow(orig.wstring()).c_str(), ec ? 0ull : (unsigned long long)size,
        md5.c_str(), md5 == kOrigMd5 ? ", official 0.9.12" : ", NOT THE OFFICIAL 0.9.12");
    if (!g_orig) {
        Log("ERROR: cannot load the original (error %lu). The game will not start VR.", g_loadError);
        return;
    }
    for (int i = 0; i < g_missingCount; ++i) Log("ERROR: the original has no export %s", g_missing[i]);
    sysinfo::LogGameConfig();
    sysinfo::LogModules("proxy start");
}

bool Ready() {
    std::call_once(g_once, [] {
        LoadOriginal();
        Guarded("LoadConfig", LoadConfig);
        Guarded("LogStartup", LogStartup);
    });
    return g_orig != nullptr;
}

// After a diagnostics fault the game normally gets the real interfaces. Not while the experiment is engaged:
// the game already renders with a replaced size and projection, and only the wrappers keep Submit consistent.
bool Wrap() { return Cfg().wrap && (!g_diagDisabled.load() || experiment::Engaged() || quad::Engaged()); }

void LogFirst(const char* what) {
    Guarded("LogFirst", [&] {
        static std::mutex m;
        static std::set<std::string> seen;
        std::lock_guard lock(m);
        if (seen.insert(what).second) Log("First call of export %s (thread %lu)", what, GetCurrentThreadId());
    });
}

}  // namespace

namespace vr {

IVRSystem* VR_CALLTYPE VR_Init(EVRInitError* peError, EVRApplicationType eApplicationType) {
    if (!Ready() || !g_fn.init) {
        if (peError) *peError = VRInitError_Init_InstallationNotFound;
        return nullptr;
    }
    Log("VR_Init(application type %d)", (int)eApplicationType);
    EVRInitError err = VRInitError_None;
    IVRSystem* real = g_fn.init(&err, eApplicationType);
    if (peError) *peError = err;
    Log("VR_Init = %p, error %d (%s)", (void*)real, (int)err, g_fn.errEnglish ? g_fn.errEnglish(err) : "?");
    if (!real) return nullptr;
    if (!Cfg().wrap) {
        Log("wrap=0: the game gets the real interfaces");
        return real;
    }
    if (g_fn.getGeneric) modern::Init(reinterpret_cast<modern::GetGenericFn>(g_fn.getGeneric));
    Guarded("LogModules", [] { sysinfo::LogModules("after VR_Init"); });
    probe::Install();  // [quad] mode=probe; guards itself
    quad::Install();   // [quad] mode=quad; guards itself
    return Wrap() ? wrap::System(real) : real;
}

void VR_CALLTYPE VR_Shutdown() {
    Log("VR_Shutdown");
    if (!Ready()) return;
    frame::OnShutdown();
    Guarded("LogCallStats", [] { LogCallStats(true); });
    probe::Shutdown();
    quad::Shutdown();
    modern::Shutdown();
    if (g_fn.shutdown) g_fn.shutdown();
}

bool VR_CALLTYPE VR_IsHmdPresent() {
    LogFirst("VR_IsHmdPresent");
    return Ready() && g_fn.isHmdPresent && g_fn.isHmdPresent();
}

bool VR_CALLTYPE VR_IsRuntimeInstalled() {
    LogFirst("VR_IsRuntimeInstalled");
    return Ready() && g_fn.isRuntimeInstalled && g_fn.isRuntimeInstalled();
}

const char* VR_CALLTYPE VR_GetVRInitErrorAsSymbol(EVRInitError error) {
    return Ready() && g_fn.errSymbol ? g_fn.errSymbol(error) : kNoOrig;
}

const char* VR_CALLTYPE VR_GetVRInitErrorAsEnglishDescription(EVRInitError error) {
    LogFirst("VR_GetVRInitErrorAsEnglishDescription");
    return Ready() && g_fn.errEnglish ? g_fn.errEnglish(error) : kNoOrig;
}

void* VR_CALLTYPE VR_GetGenericInterface(const char* pchInterfaceVersion, EVRInitError* peError) {
    if (!Ready() || !g_fn.getGeneric) {
        if (peError) *peError = VRInitError_Init_InterfaceNotFound;
        return nullptr;
    }
    void* real = g_fn.getGeneric(pchInterfaceVersion, peError);
    Guarded("VR_GetGenericInterface (log)", [&] {
        static std::mutex m;
        static std::set<std::string> seen;
        std::lock_guard lock(m);
        if (pchInterfaceVersion && seen.insert(pchInterfaceVersion).second)
            Log("Game: VR_GetGenericInterface(\"%s\") = %p, error %d", pchInterfaceVersion, real, peError ? (int)*peError : -1);
    });
    return Wrap() ? wrap::Generic(pchInterfaceVersion, real) : real;
}

IVRSystem* VR_CALLTYPE VRSystem() {
    LogFirst("VRSystem");
    if (!Ready() || !g_fn.system) return nullptr;
    IVRSystem* real = g_fn.system();
    return Wrap() ? wrap::System(real) : real;
}

IVRCompositor* VR_CALLTYPE VRCompositor() {
    LogFirst("VRCompositor");
    if (!Ready() || !g_fn.compositor) return nullptr;
    IVRCompositor* real = g_fn.compositor();
    return Wrap() ? wrap::Compositor(real) : real;
}

IVROverlay* VR_CALLTYPE VROverlay() {
    LogFirst("VROverlay");
    if (!Ready() || !g_fn.overlay) return nullptr;
    IVROverlay* real = g_fn.overlay();
    return Wrap() ? wrap::Overlay(real) : real;
}

}  // namespace vr

// Not declared in the 0.9.12 header, but the original exports it.
extern "C" __declspec(dllexport) const char* VR_CALLTYPE VR_GetStringForHmdError(int error) {
    return Ready() && g_fn.hmdError ? g_fn.hmdError(error) : kNoOrig;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        // No DisableThreadLibraryCalls: the static CRT relies on thread notifications.
        g_self = inst;
    }
    return TRUE;
}

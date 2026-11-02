// openvr_api.dll proxy exports. AMS2 imports VR_Init, VR_Shutdown, VRCompositor, VROverlay
// and VR_GetVRInitErrorAsEnglishDescription; the other main functions are handled here too. The remaining exports
// (flat C API, Unity) are forwarded to the original by generated/forwards.inc.
//
// Backend ([runtime] path, runtime.cpp): openvr = x64\qvfr\openvr\openvr_api.dll (the official Valve 0.9.12 DLL, the
// same one the game ships); openxr = OpenComposite. Every other export is a thunk (generated/thunks.asm) that jumps to
// the backend's export of the same name; its address is looked up on the first call (qvfr_ResolveThunk).

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
#include "gpu.h"
#include "modern_vr.h"
#include "probe.h"
#include "quad.h"
#include "runtime.h"
#include "sysinfo.h"
#include "wrappers.h"

namespace {

// Stand-ins for an export the backend does not have (OpenComposite has 27 of the 253), so its thunk does not jump to
// nothing. The x64 convention lets a caller of any signature survive them (the caller cleans its own stack); a struct
// returned by value comes through a hidden pointer in rcx that the caller expects back in rax.
uint64_t MissingZero() { return 0; }
template <size_t N>
void* MissingStruct(void* ret) {
    if (ret) memset(ret, 0, N);
    return ret;
}

}  // namespace

#include "generated/exports.inc"  // kThunkNames, kThunkMissing, /EXPORT directives for generated/thunks.asm

extern "C" void* qvfr_thunk_table[];

namespace {

using namespace qvfr;

HMODULE g_self = nullptr;
HMODULE g_orig = nullptr;
std::once_flag g_once;

constexpr const char* kOrigMd5 = "864be5a92276ce2b902b6f3a0f850703";  // official openvr_api.dll 0.9.12 (win64)
// OpenComposite builds seen on the user's machine (only named in the log).
constexpr const char* kOpenCompositeMd5 = "1a89f94450ab870d74b5c20da2e52d85";  // 2025-10-17, the user's copy

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

const char kNoOrig[] = "ams2-qvfr: the backend DLL (x64\\qvfr\\openvr or OpenComposite) could not be loaded";

wchar_t g_origPath[MAX_PATH + 32] = {};
DWORD g_loadError = 0;
const char* g_missing[16] = {};
int g_missingCount = 0;

template <class T>
void Resolve(T& fn, const char* name) {
    fn = reinterpret_cast<T>(GetProcAddress(g_orig, name));
    if (!fn && g_missingCount < 16) g_missing[g_missingCount++] = name;
}

// Loads the real DLL with plain Win32 calls that cannot throw. `backend` comes from runtime::Prepare; when it is empty
// (the settings or Prepare failed) the default x64\qvfr\openvr\openvr_api.dll is used, so a failing log, ini or
// hash never keeps the game from starting VR.
void LoadOriginal(const wchar_t* backend) noexcept {
    if (backend && backend[0] && wcslen(backend) < _countof(g_origPath)) {
        wcscpy_s(g_origPath, backend);
    } else {
        wchar_t self[MAX_PATH] = {};
        DWORD n = GetModuleFileNameW(g_self, self, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) {
            g_loadError = n ? ERROR_INSUFFICIENT_BUFFER : GetLastError();
            return;
        }
        wchar_t* slash = wcsrchr(self, L'\\');
        size_t dirLen = slash ? (size_t)(slash - self + 1) : 0;
        wmemcpy(g_origPath, self, dirLen);
        wcscpy_s(g_origPath + dirLen, _countof(g_origPath) - dirLen, L"qvfr\\openvr\\openvr_api.dll");
    }
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

void LogHeader() {
    wchar_t self[MAX_PATH] = {};
    GetModuleFileNameW(g_self, self, MAX_PATH);
    Log("ams2-qvfr openvr_api.dll proxy, build " __DATE__ " " __TIME__);
    Log("Proxy: %s", Narrow(self).c_str());
    Log("Mod folder (qvfr.ini, logs, CSV, shots): %s%s", Narrow(BaseDir().wstring()).c_str(),
        BaseDirIsFallback() ? " (x64\\qvfr is not writable, so the temporary folder is used)" : "");
}

void LogStartup() {
    sysinfo::LogProcessInfo();
    runtime::LogStartup();

    fs::path orig = g_origPath;
    std::error_code ec;
    auto size = fs::file_size(orig, ec);
    std::string md5 = sysinfo::HashFile(orig, BCRYPT_MD5_ALGORITHM);
    const char* what = md5 == kOrigMd5            ? ", official Valve 0.9.12"
                       : md5 == kOpenCompositeMd5 ? ", OpenComposite 2025-10-17 (the user's copy)"
                       : runtime::OpenXrPath()    ? ", OpenComposite of unknown build"
                                                  : ", NOT THE OFFICIAL 0.9.12";
    Log("Backend: %s (%llu B, MD5 %s%s)", Narrow(orig.wstring()).c_str(), ec ? 0ull : (unsigned long long)size,
        md5.c_str(), what);
    if (!g_orig) {
        Log("ERROR: cannot load the backend (error %lu). The game will not start VR.", g_loadError);
        return;
    }
    for (int i = 0; i < g_missingCount; ++i) Log("ERROR: the backend has no export %s", g_missing[i]);
    sysinfo::LogGameConfig();
    sysinfo::LogModules("proxy start");
}

bool Ready() {
    std::call_once(g_once, [] {
        // The settings come first: [runtime] picks the backend and prepares the OpenXR loader's environment before
        // OpenComposite is loaded. A fault there leaves `backend` empty and LoadOriginal takes the default.
        std::wstring backend;
        Guarded("LogHeader", LogHeader);
        Guarded("LoadConfig", LoadConfig);
        Guarded("runtime::Prepare", [&] { backend = runtime::Prepare(g_self); });
        LoadOriginal(backend.c_str());
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
    // OpenComposite answers an interface it does not implement with a message box ("Missing interface") and an abort,
    // not with an error code, so on the OpenXR path the proxy never asks it for the newer SteamVR interfaces.
    if (g_fn.getGeneric && !runtime::OpenXrPath())
        modern::Init(reinterpret_cast<modern::GetGenericFn>(g_fn.getGeneric));
    frame::NoteSystem(real);  // eye frusta for the gaze from the OpenXR layer
    Guarded("VR_Init (backend log)", [] {
        if (!modern::HasSystem())
            Log("%s: gaze %s, no SteamVR frame timing (GPU timestamps only)",
                runtime::OpenXrPath() ? "path=openxr: IVRSystem_026 and IVRCompositor_029 not requested (OpenComposite "
                                        "aborts on interfaces it does not have)"
                                      : modern::HasCompositor() ? "Backend has no IVRSystem_026"
                                                                : "Backend has no IVRSystem_026 and no IVRCompositor_029",
                runtime::OpenXrPath() ? "through the OpenXR layer" : "unavailable");
    });
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
    gpu::Release();  // after everything that drew with the device
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

// ---------------------------------------------------------------- export thunks

// First call of thunk i (from qvfr_lazy_common, any thread): loads the backend if needed, puts the backend's export
// into the thunk's slot and returns it; the next calls jump there directly.
extern "C" void* qvfr_ResolveThunk(unsigned i) {
    void* p = nullptr;
    if (i < kThunkCount && Ready()) p = reinterpret_cast<void*>(GetProcAddress(g_orig, kThunkNames[i]));
    Guarded("export thunk (log)", [&] {
        static std::atomic<int> logs{0};
        if (logs++ < 32)
            Log("Export %s: first call, %s", i < kThunkCount ? kThunkNames[i] : "?",
                p ? "goes to the backend" : "THE BACKEND HAS NO SUCH EXPORT, returns zeros");
    });
    if (!p) p = i < kThunkCount ? kThunkMissing[i] : reinterpret_cast<void*>(&MissingZero);
    if (i < kThunkCount) InterlockedExchangePointer(&qvfr_thunk_table[i], p);
    return p;
}

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

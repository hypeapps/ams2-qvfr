// nvngx.dll_qvfr_nr.dll: the Neural Rendering model (nvngx_dlssnr.dll) checks the module that calls its entry points
// and refuses any whose file path does not contain "nvngx.dll". This DLL is named to pass that check and only forwards
// (NR prototype NR-P1, docs: dlaa-plan D7). Each result goes through a volatile, so the compiler cannot turn a forward
// into a tail call (the caller's return address would then be the proxy's).
//
// Ported from nr-bridge's src/nr_forwarder.cpp (https://github.com/lowpoplabs/nr-bridge).
// MIT License, Copyright (c) 2026 LowPopLabs. Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
// following conditions: The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <windows.h>

#include <d3d12.h>

namespace {

using InitFn = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
using CreateFn = int(__cdecl*)(ID3D12GraphicsCommandList*, int, void*, void**);
using EvaluateFn = int(__cdecl*)(ID3D12GraphicsCommandList*, const void*, void*, void*);
using ReleaseFn = int(__cdecl*)(void*);
using ShutdownFn = int(__cdecl*)(ID3D12Device*);

HMODULE g_mod = nullptr;
InitFn g_init = nullptr;
CreateFn g_create = nullptr;
EvaluateFn g_eval = nullptr;
ReleaseFn g_release = nullptr;
ShutdownFn g_shutdown = nullptr;

}  // namespace

extern "C" {

// 1: the model is loaded and has the D3D12 entry points; 0: entry points missing; < 0: -GetLastError() of the load.
__declspec(dllexport) int __cdecl qnr_load(const wchar_t* path) {
    if (g_mod) return g_create ? 1 : 0;
    g_mod = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_mod) return -(int)GetLastError();
    g_init = (InitFn)GetProcAddress(g_mod, "NVSDK_NGX_D3D12_Init_Ext");
    g_create = (CreateFn)GetProcAddress(g_mod, "NVSDK_NGX_D3D12_CreateFeature");
    g_eval = (EvaluateFn)GetProcAddress(g_mod, "NVSDK_NGX_D3D12_EvaluateFeature");
    g_release = (ReleaseFn)GetProcAddress(g_mod, "NVSDK_NGX_D3D12_ReleaseFeature");
    g_shutdown = (ShutdownFn)GetProcAddress(g_mod, "NVSDK_NGX_D3D12_Shutdown1");
    return (g_init && g_create && g_eval) ? 1 : 0;
}

__declspec(dllexport) int __cdecl qnr_init(unsigned long long appId, const wchar_t* dataPath, ID3D12Device* dev, int sdk,
                                           const void* commonInfo) {
    if (!g_init) return 0;
    volatile int r = g_init(appId, dataPath, dev, sdk, commonInfo);
    return r;
}

__declspec(dllexport) int __cdecl qnr_create(ID3D12GraphicsCommandList* cl, int feature, void* params, void** handle) {
    if (!g_create) return 0;
    volatile int r = g_create(cl, feature, params, handle);
    return r;
}

__declspec(dllexport) int __cdecl qnr_evaluate(ID3D12GraphicsCommandList* cl, void* handle, void* params) {
    if (!g_eval) return 0;
    volatile int r = g_eval(cl, handle, params, nullptr);
    return r;
}

__declspec(dllexport) int __cdecl qnr_release(void* handle) {
    if (!g_release) return 0;
    volatile int r = g_release(handle);
    return r;
}

__declspec(dllexport) int __cdecl qnr_shutdown(ID3D12Device* dev) {
    if (!g_shutdown) return 0;
    volatile int r = g_shutdown(dev);
    return r;
}

}  // extern "C"

BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }

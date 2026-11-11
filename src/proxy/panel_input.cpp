// The tuning panel's keys away from the game: see panel_input.h.
#include "panel_input.h"

#include <windows.h>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

#include <atomic>
#include <mutex>

#include "common.h"
#include "hooks.h"

namespace qvfr::panelinput {

namespace {

std::atomic<bool> g_open{false};
std::atomic<int> g_modVk{0}, g_keyDik{0};
std::once_flag g_once;
bool g_started = false;

constexpr int kNav[] = {DIK_UP, DIK_DOWN, DIK_LEFT, DIK_RIGHT, DIK_RETURN, DIK_NUMPADENTER, DIK_ESCAPE};

bool ModDown() {
    const int m = g_modVk.load();
    return !m || (GetAsyncKeyState(m) & 0x8000);
}

// The panel wants this key now (at its press).
bool Wanted(int dik) {
    if (dik == g_keyDik.load() && ModDown()) return true;
    if (!g_open.load()) return false;
    for (int k : kNav)
        if (k == dik) return true;
    return false;
}

bool Candidate(int dik) {
    if (dik == g_keyDik.load()) return true;
    for (int k : kNav)
        if (k == dik) return true;
    return false;
}

// The keys the game lost, separately for the state and the buffered events (a game may read both). The game reads
// its keyboard on one thread.
bool g_stateMasked[256] = {};
bool g_dataMasked[256] = {};

// Keyboard devices among the devices that call GetDeviceData (wheels and the mouse share the implementation).
struct Known {
    void* dev;
    bool keyboard;
};
Known g_known[32] = {};
int g_knownN = 0;
std::mutex g_knownM;

using GetCapsFn = HRESULT(STDMETHODCALLTYPE*)(void*, DIDEVCAPS*);

bool IsKeyboard(void* dev) {
    std::lock_guard lock(g_knownM);
    for (int i = 0; i < g_knownN; ++i)
        if (g_known[i].dev == dev) return g_known[i].keyboard;
    DIDEVCAPS caps{};
    caps.dwSize = sizeof(caps);
    auto getCaps = reinterpret_cast<GetCapsFn>((*reinterpret_cast<void***>(dev))[3]);  // GetCapabilities
    const bool kb = SUCCEEDED(getCaps(dev, &caps)) && GET_DIDEVICE_TYPE(caps.dwDevType) == DI8DEVTYPE_KEYBOARD;
    if (g_knownN < 32) g_known[g_knownN++] = {dev, kb};
    return kb;
}

void FilterState(DWORD cb, void* data) {
    if (cb != 256 || !data) return;  // only the keyboard's state is 256 bytes
    auto* s = static_cast<BYTE*>(data);
    __try {
        for (int k = 0; k < 256; ++k) {
            if (!Candidate(k) && !g_stateMasked[k]) continue;
            if (!(s[k] & 0x80)) {
                g_stateMasked[k] = false;
                continue;
            }
            if (!g_stateMasked[k] && Wanted(k)) g_stateMasked[k] = true;
            if (g_stateMasked[k]) s[k] = 0;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

void FilterData(void* dev, DWORD cb, DIDEVICEOBJECTDATA* rg, DWORD* inOut, DWORD flags) {
    if (!rg || !inOut || !*inOut || cb < 8 || !IsKeyboard(dev)) return;
    const bool peek = (flags & DIGDD_PEEK) != 0;
    auto* base = reinterpret_cast<BYTE*>(rg);
    DWORD kept = 0;
    __try {
        for (DWORD i = 0; i < *inOut; ++i) {
            BYTE* e = base + (size_t)i * cb;
            const DWORD ofs = *reinterpret_cast<const DWORD*>(e);      // dwOfs: the DIK code
            const DWORD val = *reinterpret_cast<const DWORD*>(e + 4);  // dwData: 0x80 pressed
            bool drop = false;
            if (ofs < 256) {
                if (val & 0x80) {
                    drop = g_dataMasked[ofs] || Wanted((int)ofs);
                    if (drop && !peek) g_dataMasked[ofs] = true;
                } else {
                    drop = g_dataMasked[ofs];
                    if (drop && !peek) g_dataMasked[ofs] = false;
                }
            }
            if (drop) continue;
            if (kept != i) memmove(base + (size_t)kept * cb, e, cb);
            ++kept;
        }
        *inOut = kept;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

using StateFn = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, void*);
using DataFn = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, DIDEVICEOBJECTDATA*, DWORD*, DWORD);
StateFn g_stateW = nullptr, g_stateA = nullptr;
DataFn g_dataW = nullptr, g_dataA = nullptr;

HRESULT STDMETHODCALLTYPE StateW(void* dev, DWORD cb, void* data) {
    const HRESULT hr = g_stateW(dev, cb, data);
    if (SUCCEEDED(hr)) FilterState(cb, data);
    return hr;
}
HRESULT STDMETHODCALLTYPE StateA(void* dev, DWORD cb, void* data) {
    const HRESULT hr = g_stateA(dev, cb, data);
    if (SUCCEEDED(hr)) FilterState(cb, data);
    return hr;
}
HRESULT STDMETHODCALLTYPE DataW(void* dev, DWORD cb, DIDEVICEOBJECTDATA* rg, DWORD* n, DWORD flags) {
    const HRESULT hr = g_dataW(dev, cb, rg, n, flags);
    if (SUCCEEDED(hr)) FilterData(dev, cb, rg, n, flags);
    return hr;
}
HRESULT STDMETHODCALLTYPE DataA(void* dev, DWORD cb, DIDEVICEOBJECTDATA* rg, DWORD* n, DWORD flags) {
    const HRESULT hr = g_dataA(dev, cb, rg, n, flags);
    if (SUCCEEDED(hr)) FilterData(dev, cb, rg, n, flags);
    return hr;
}

// A keyboard device of our own, only for its methods' addresses (the game's devices use the same code).
bool Methods(bool wide, void*& state, void*& data) {
    using CreateFn = HRESULT(WINAPI*)(HINSTANCE, DWORD, REFIID, LPVOID*, LPUNKNOWN);
    HMODULE m = LoadLibraryW(L"dinput8.dll");
    auto create = m ? reinterpret_cast<CreateFn>(GetProcAddress(m, "DirectInput8Create")) : nullptr;
    if (!create) return false;
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&Methods), &self);
    bool ok = false;
    if (wide) {
        IDirectInput8W* di = nullptr;
        if (SUCCEEDED(create(self, DIRECTINPUT_VERSION, IID_IDirectInput8W, (void**)&di, nullptr)) && di) {
            IDirectInputDevice8W* dev = nullptr;
            if (SUCCEEDED(di->CreateDevice(GUID_SysKeyboard, &dev, nullptr)) && dev) {
                void** vt = *reinterpret_cast<void***>(dev);
                state = vt[9];  // GetDeviceState
                data = vt[10];  // GetDeviceData
                ok = true;
                dev->Release();
            }
            di->Release();
        }
    } else {
        IDirectInput8A* di = nullptr;
        if (SUCCEEDED(create(self, DIRECTINPUT_VERSION, IID_IDirectInput8A, (void**)&di, nullptr)) && di) {
            IDirectInputDevice8A* dev = nullptr;
            if (SUCCEEDED(di->CreateDevice(GUID_SysKeyboard, &dev, nullptr)) && dev) {
                void** vt = *reinterpret_cast<void***>(dev);
                state = vt[9];
                data = vt[10];
                ok = true;
                dev->Release();
            }
            di->Release();
        }
    }
    return ok;
}

}  // namespace

bool Start() {
    std::call_once(g_once, [] {
        if (!hooks::Init()) {
            Log("Panel: no MinHook; the game also gets the panel's keys");
            return;
        }
        void *sw = nullptr, *dw = nullptr, *sa = nullptr, *da = nullptr;
        const bool w = Methods(true, sw, dw), a = Methods(false, sa, da);
        if (!w && !a) {
            Log("Panel: no DirectInput 8 keyboard; the game also gets the panel's keys");
            return;
        }
        int n = 0;
        if (w) {
            n += hooks::CreateAt(sw, (void*)&StateW, (void**)&g_stateW, "IDirectInputDevice8W::GetDeviceState");
            n += hooks::CreateAt(dw, (void*)&DataW, (void**)&g_dataW, "IDirectInputDevice8W::GetDeviceData");
        }
        if (a && sa != sw)
            n += hooks::CreateAt(sa, (void*)&StateA, (void**)&g_stateA, "IDirectInputDevice8A::GetDeviceState");
        if (a && da != dw)
            n += hooks::CreateAt(da, (void*)&DataA, (void**)&g_dataA, "IDirectInputDevice8A::GetDeviceData");
        g_started = n > 0;
        Log("Panel: %d DirectInput keyboard methods hooked%s; the panel's keys go only to the panel", n,
            a && sa == sw ? " (the A and W devices share them)" : "");
    });
    return g_started;
}

void SetOpen(bool open) { g_open = open; }

void SetHotkey(int modVk, int keyVk) {
    g_modVk = modVk;
    // VK -> scan code -> DIK (extended keys have 0x80 in the DIK code).
    const UINT sc = keyVk ? MapVirtualKeyW((UINT)keyVk, MAPVK_VK_TO_VSC_EX) : 0;
    g_keyDik = sc ? (int)((sc & 0x7F) | ((sc & 0xFF00) ? 0x80 : 0)) : 0;
}

}  // namespace qvfr::panelinput

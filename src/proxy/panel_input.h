// The tuning panel's keys away from the game (dlaa-plan D7.P). AMS2 reads the keyboard through DirectInput 8 (and a
// low-level keyboard hook gets no keys in it), so the proxy hooks the keyboard device's GetDeviceState and
// GetDeviceData in dinput8.dll and takes the panel's keys out of what the game gets: with the panel open the arrows,
// Enter (also the keypad's) and Esc, and always the hotkey's key while its modifier is held. A key the game lost at
// its press stays away from it until its release (so Esc that closes the panel does not reach the game), and a key
// held when the panel opens reads as released. The panel itself polls the keys (GetAsyncKeyState). Other devices
// (wheels, the mouse) pass unchanged.
#pragma once

namespace qvfr::panelinput {

// Once, render thread: the hooks (MinHook, hooks.h). False: not hooked (logged); the game then gets the keys too.
bool Start();
// The panel's state for the hooks (any thread).
void SetOpen(bool open);
void SetHotkey(int modVk, int keyVk);

}  // namespace qvfr::panelinput

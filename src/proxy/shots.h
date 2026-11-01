// Screenshots of game textures for checks the agent cannot do in the headset: a staging copy on the render thread,
// then PNG files (WIC, half resolution) and pixel differences between image pairs on a worker thread.
// Files go to Documents\ams2-qvfr\shots\<date>\<name>.png.
#pragma once

#include <d3d11.h>

namespace qvfr::shots {

struct Pair {
    int a, b;           // image indices
    const char* label;  // for the log line
    // rw > 0: image a shows the rectangle (rx, ry, rw, rh) of image b, in b's pixels (a focus view inside its outer).
    // a is then averaged over each b pixel's footprint and compared with the b pixels inside the rectangle.
    float rx = 0, ry = 0, rw = 0, rh = 0;
};

// One shot is up to kMaxImages images taken in the same frame. All calls come from the game's render thread.
constexpr int kMaxImages = 8;

// Starts a shot; false while the previous one is still being written. names and pairs must outlive the shot
// (string literals and static arrays).
bool Begin(const char* const* names, int count, const Pair* pairs, int pairCount);
bool Active();  // between Begin and End
// Copies src (a 2D texture, one sample) into a new staging texture. Call right after the game wrote it.
bool Grab(ID3D11DeviceContext* ctx, ID3D11Resource* src, int index);
// At least one frame after the grabs: maps the copies and hands them to the worker thread. Images that were not
// grabbed are listed in the log.
void End(ID3D11DeviceContext* ctx);
void Cancel();  // releases a shot that was begun but not ended (VR_Shutdown)

}  // namespace qvfr::shots

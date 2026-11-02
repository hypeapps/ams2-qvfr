// Screenshots of game textures for checks the agent cannot do in the headset: a staging copy on the render thread,
// then PNG files (WIC, half resolution) and pixel differences between image pairs on a worker thread.
// Files go to x64\qvfr\shots\<date>\<name>.png (BaseDir).
#pragma once

#include <d3d11.h>

namespace qvfr::shots {

struct Pair {
    int a, b;           // image indices
    const char* label;  // for the log line
    // rw > 0: image a shows the rectangle (rx, ry, rw, rh) of image b, in b's pixels (a focus view inside its outer).
    // a is then averaged over each b pixel's footprint and compared with the b pixels inside the rectangle, leaving
    // out `inset` b pixels at its edges (the blend band of a composed eye).
    float rx = 0, ry = 0, rw = 0, rh = 0, inset = 0;
};

// One shot is up to kMaxImages images taken in the same frame (the clones' dump takes most of them). All calls come
// from the game's render thread.
constexpr int kMaxImages = 40;

// Starts a shot; false while the previous one is still being written. names and pairs must outlive the shot
// (string literals and static arrays).
bool Begin(const char* const* names, int count, const Pair* pairs, int pairCount);
bool Active();  // between Begin and End
// Copies src (a 2D texture, one sample) into a new staging texture. Call right after the game wrote it. 8-bit and
// R10G10B10A2 images are saved as they are; HDR and single-channel formats (R11G11B10, RGBA16F, R16, R32, R8) are
// mapped to 8 bits (color x / (1 + x) with gamma 2.2, one channel clamped to 0..1, NaN and infinity magenta) and get a
// line with their value range and the NaN count.
bool Grab(ID3D11DeviceContext* ctx, ID3D11Resource* src, int index);
// At least one frame after the grabs: maps the copies and hands them to the worker thread. Images that were not
// grabbed are listed in the log.
void End(ID3D11DeviceContext* ctx);
void Cancel();  // releases a shot that was begun but not ended (VR_Shutdown)

}  // namespace qvfr::shots

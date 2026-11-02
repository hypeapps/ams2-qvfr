// Composition of the quad views (M4.4, M7): per eye, outer (the full FOV) is scaled bilinearly to the 1:1 grid and the
// focus view is placed into its window with a soft edge (shaders/compose.hlsl): pixel for pixel when the focus texture
// has the window's size, bilinearly otherwise. The result is a grid-size texture per eye that goes to SteamVR instead
// of the game texture.
// Draws on the game's immediate context at Submit; the pipeline state it touches is saved and restored.
// Not thread safe: the caller serializes all calls (quad mode calls under its lock on the render thread).
#pragma once

#include <d3d11.h>

#include <cstdint>

namespace qvfr::compose {

struct Layout {
    uint32_t gridW = 0, gridH = 0;          // output size (the SteamVR size)
    uint32_t outerW = 0, outerH = 0;        // outer texture (the whole FOV)
    uint32_t focusTexW = 0, focusTexH = 0;  // focus texture (the window's image)
    uint32_t focusW = 0, focusH = 0;        // focus window in grid pixels
    uint32_t x0 = 0, y0 = 0;                // focus window position in the grid
    float marginPx = 0;                     // hidden band at the focus edge (not in mbucchia)
    // Soft edge inside the margin. mbucchia's smoothen_focus_view_edges: a blend band per side as a fraction of the shown
    // focus, the focus weight at least 0.5 up to the edge (0 = hard edge). legacyEdge: the pre-M7 curve instead, from 0
    // to full focus over featherPx grid pixels (0 = hard edge).
    float smoothen = 0;
    bool legacyEdge = false;
    float featherPx = 0;
    // Debug overlay (M4.5): blend band edges and the gaze point.
    bool overlay = false;
    int gazeMode = 0;                 // 0 = no gaze, 1 = live, 2 = frozen
    float gazeX = 0, gazeY = 0;       // gaze point in grid pixels
};

// Composes one eye (0 = left, 1 = right) from outer and focus (2D textures of the layout's sizes, one sample, the same
// format) into that eye's output texture. Returns the output (owned by this module, valid until Release) or null;
// the reason is logged once per kind of failure.
ID3D11Texture2D* Draw(ID3D11DeviceContext* ctx, int eye, ID3D11Texture2D* outer, ID3D11Texture2D* focus,
                      const Layout& l);
void Release();  // all D3D11 objects (VR_Shutdown, device change)

}  // namespace qvfr::compose

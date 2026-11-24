// The game's passes right after the TAA resolve, in each eye group (quad mode, through sun_fx's D3D11 hooks).
//
// After the resolve (the game's TAA or DLSS in its target) the game runs a luma sharpen (pixel shader with the constant
// sharp_strength) into the scene color and then draws the windscreen layers (sun strip, frit, dirt, rain, scratches,
// reflections) on it, one sample per pixel.
//
// The game's sharpen ([dlaa] game_sharpen = fraction of its strength): 1 = as the game draws it; 0 = off (the TAA
// result copied into its target); in between the TAA result is copied and the game's draw blended over it with that
// factor, which for this sharpen is the same as a lower sharp_strength.
//
// The glass's depth ([dlaa] windscreen_depth = radius): TAA or DLSS anti-aliased the cockpit's outline against what is
// behind the glass, but the glass's depth test sorts those mixed pixels to the cockpit, so they keep the bright
// background where the glass darkens the rest: a light, stepped seam along the roll cage and the mirror. At the group's
// first glass draw the scene's depth is copied and made the farthest within the radius (shaders/windscreen.hlsl
// CSDilate, a compute pass: no draw state of the game's changes; D32 only); the vehicle glass draws (rainControlMap_tex)
// test against that copy, the other layers against the game's. The radius counts render pixels: with DLSS SR the depth
// after TAA is the render size's, upscaled by nearest.
//
// The layers' pixel shaders: those with the VehicleWindscreen constants, which sit in a buffer most of the game's
// shaders share, so only draws after the sharpen into its target, with a depth buffer, count.
// Diagnosis switch (quad switch "windscreen"): the layer draws skipped.
#pragma once

#include <d3d11.h>

#include <cstddef>

namespace qvfr::windscreen {

// sun_fx's CreatePixelShader detour: notes the sharpen and the windscreen layer shaders.
void OnPixelShaderMade(ID3D11PixelShader* ps, const void* code, size_t len);
// sun_fx's PSSetShader detour, immediate context.
void OnPixelShaderSet(ID3D11PixelShader* ps);
// sun_fx's Draw and DrawIndexed detours, immediate context, before the draw. `draw(arg)` runs the rest of the detour
// (the game's draw with the other modules around it). True: the draw was handled here (run with other state, replaced
// or skipped) and the detour returns.
using DrawFn = void (*)(void* arg);
bool OnDraw(ID3D11DeviceContext* c, DrawFn draw, void* arg);
// Render thread: frame start and the eye groups' boundaries.
void OnFrameStart();
void BeginGroup();
void EndGroup();

void SetLayers(bool on);
bool GetLayers();
// The quad switch "windscreen_depth": the glass's depth radius over the setting ([benchmark] sequences).
void SetDepth(int radius);
int GetDepth();

void LogStats();
void Shutdown();  // VR_Shutdown: the scratch textures go

}  // namespace qvfr::windscreen

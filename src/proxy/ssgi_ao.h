// Own ambient occlusion in place of the game's SSAO (docs/ssgi-plan.md G1; the game's passes in "Ustalenia G0").
//
// The game lights the scene in a light pre-pass: its SSAO (CSSAORender) ends in a half-size R16 texture that two
// full-screen draws read, the overlay (PS 15564: the sun's shadow and the AO into the G-buffer target's alpha, which the
// forward pass multiplies the sun by) and the ambient pass (PS 15549: sky irradiance x AO into the diffuse light target,
// which the forward pass multiplies by the material's albedo). Both sample it at UV x screenSpaceDimensions.xy, with no
// size of their own, so a texture of the G-buffer's size can stand in for it without changing the game's shaders.
//
// At the overlay draw this module runs a compute pass (shaders/ssgi_ao.hlsl: GTAO with a visibility bitmask, then a
// depth-aware blur) on the G-buffer normals and linear depth that draw binds, with the game's own constant windows of
// that draw (projection tangents, far, the view's share of the targets) bound to the compute shader, and binds the result
// in place of the game's SSAO for that draw; the ambient draw of the same view gets it too. The game's SSAO still runs.
// A view is its eye group, or focus's when its SSAO runs early in outer's group (mvr_pass.h EarlyEye).
// [ssgi] ao (live, common.h): 0 the game's SSAO, 1 own AO, 2 own AO on the ambient light only (the overlay gets 1),
// 3 as 1 plus a grey preview of the AO drawn over each view's MSAA resolve (PS 15179) or, without MSAA, over DLSS's
// output of the view (PreviewInto).
//
// One bounce of light (docs/ssgi-plan.md G2, [ssgi] gi, needs own AO): the same compute pass gives each pixel the light
// of the samples that occlude it, read from the view's history (its final image of the previous frame: the MSAA resolve
// or DLSS's output, half size), with the previous frame's camera of the view (the pass keeps each view's camera in a
// small buffer) for the world; vehicles (the deferred-ambient weight 0 in the overlay's target) at the same place of
// the screen. After the ambient draw a full-screen draw adds the blurred bounce into the diffuse light target (One,
// One), which the forward pass multiplies by the albedo, on the world and the cars alike. gi: 0 off, 1 on, 2 on plus
// the bounce alone drawn over the image (preview).
// The draws are recognised by the DXBC checksum in sun_fx's table (its hooks are the only ones on Draw/DrawIndexed), so
// the module works in quad mode only. Unknown checksums (another game build): nothing changes.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::ssgiao {

// Draws sun_fx recognises for this module.
enum DrawKind { DrawNone = 0, DrawOverlay = 1, DrawAmbient = 2, DrawResolve = 3, DrawAmbientLightmap = 4 };

// Render thread, every frame start (quad.cpp): the game's immediate context (null before the first Submit); `allowed`
// false after a quad fault. Reads the settings; drops this module's D3D11 objects when the device changes.
void OnFrameStart(ID3D11DeviceContext* ctx, bool allowed);
// From sun_fx's Draw and DrawIndexed hooks, before the game's draw `kind` with pixel shader `ps` in eye group `group`
// (-1 outside the groups). True: AfterDraw must follow the game's draw (restores a slot, draws the preview).
bool BeforeDraw(ID3D11DeviceContext* c, int kind, ID3D11PixelShader* ps, int group);
void AfterDraw(ID3D11DeviceContext* c);
// gi on, without MSAA: dlaa.cpp, after DLSS wrote eye group `group`'s TAA output into `target` (the view's final image
// of this frame): the view's history. Before PreviewInto (the history without the preview).
void CaptureHistory(ID3D11DeviceContext* c, int group, ID3D11Texture2D* target);
// ao=3 or gi=2 without MSAA: dlaa.cpp, after DLSS wrote eye group `group`'s TAA output into `target` (no MSAA resolve
// there).
void PreviewInto(ID3D11DeviceContext* c, int group, ID3D11Texture2D* target);

// Draw dump (diagnostics): after each change of ao or gi to on, for 2 frames, each overlay draw's target and the 8
// draws after it (pixel shader checksum, targets, shader resources t0..t7) go to the log. From sun_fx's draw hooks:
// DumpWanted cheap, DumpDraw before the game's draw of `count` vertices or indices.
bool DumpWanted();
void DumpDraw(ID3D11DeviceContext* c, UINT count);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
void Shutdown();             // VR_Shutdown: D3D11 objects released

// qvfr_proxy_smoke --ssgi-ao: the module on the test context with the modes set by hand (the rest from the defaults).
void TestInit(ID3D11DeviceContext* ctx, int mode, int gi = 0);
struct TestState {
    unsigned long long passes = 0, giPasses = 0, captures = 0, ambient = 0, ambientMissed = 0, composites = 0,
                       previews = 0, noHistory = 0, refused = 0, faults = 0;
    std::string probe;  // the latest probe line (read at a frame start, logged with the stats)
    unsigned long long dumpLines = 0;
};
TestState TestGet();

}  // namespace qvfr::ssgiao

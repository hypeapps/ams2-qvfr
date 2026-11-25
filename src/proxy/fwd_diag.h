// Diagnostics of MVR's live forward pass: the draws the game makes while outer's forward pass has no target at outer's
// size bound. MVR pauses there (one view), and a draw in the pause hands the whole forward pass back to the focus group
// ("draws into targets of other sizes"), which then draws its own forward and velocity passes over outer's G-buffer.
//
// Each such draw is described in the log: the draw call and the engine's return address, the shaders (DXBC checksum and
// the names in their reflection: constant buffers, textures, samplers, inputs, outputs), the bound targets, viewport,
// textures and output merger state, and the engine's call stack at the bind of the other-size target. Shaders are
// described once, when the game makes them (CreateVertexShader through mvr_vs, CreatePixelShader through sun_fx).
//
// Log lines start with "Forward diag". Full detail (every draw, up to kDetailDraws) for the first kDetailPasses passes
// of each eye with draws in a pause and again every kDetailEveryS seconds; each new kind of draw (VS, PS, targets) and
// each shader of such a draw once; a pause without draws once per bound target; counts in the periodic stats.
#pragma once

#include <d3d11.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace qvfr::fwddiag {

// Any thread: a shader the game made (stage 'V' or 'P'); its bytecode is read once for the log. Guards itself.
void OnShaderMade(void* shader, char stage, const void* code, size_t len);
// Render thread, right before mvrpass::BeforeDraw: the draw method, its counts and the engine's return address.
void NoteDrawCall(const char* method, unsigned count, unsigned instances, void* caller);

// mvr_pass, render thread. A forward pass of outer starts (eye 0 or 1; `live`: the live pass, B9d; outer's and focus's
// sizes).
void OnPassStart(int eye, uint64_t frame, bool live, unsigned w, unsigned h, unsigned fw, unsigned fh);
// A bind in it without a target at outer's size: MVR pauses.
void OnPauseBind(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv);
// A draw while paused, before the real draw.
void OnPausedDraw(ID3D11DeviceContext* c);
// The end of a live forward pass: why it went to the focus group (null: focus got it from outer).
void OnPassEnd(const char* why);

void LogStats(bool totals);  // a stats line when a forward pass ran since the last one, or the session's totals

// Smoke tests.
struct TestState {
    uint64_t shaders = 0;      // shaders described
    uint64_t pausedDraws = 0;  // draws seen in a pause
    uint64_t detailLines = 0;  // draw lines logged
    uint64_t kinds = 0;        // kinds of draws
    std::string lastDraw;      // the last draw's description
    std::string lastPs;        // the reflection summary of the last draw's PS
};
TestState TestGet();

}  // namespace qvfr::fwddiag

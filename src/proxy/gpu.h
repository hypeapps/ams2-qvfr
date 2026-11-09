// The game's D3D11 device and GPU timestamps per frame phase (M4.5).
// The device comes from the Submit textures and is checked at every Submit; the proxy holds one reference to the
// immediate context, so a cached pointer can never point at a destroyed or a new device. Every D3D11 object of the
// proxy (composition, copies, shots, queries) must be dropped when Generation() changes.
// Timestamps: one disjoint query and a timestamp per mark per frame, read without flushing a few frames later on the
// render thread. A span is GPU time between two points in the game's command stream, so it also holds GPU idle time
// when the game is CPU bound. Does not depend on SteamVR: works on any OpenVR runtime (OpenComposite included).
#pragma once

#include <d3d11.h>

#include <cstdint>

namespace qvfr::gpu {

// The game's immediate context (no extra reference for the caller) or null before the first D3D11 Submit.
ID3D11DeviceContext* Context();
uint32_t Generation();  // 0 before the first device; +1 at every device change
// Every Submit with a D3D11 texture (before the proxy draws for it). Returns true when the device changed.
bool OnSubmitTexture(void* d3dTexture);

enum Phase { OuterL, OuterR, FocusL, FocusR, ComposeL, ComposeR, kPhases };
const char* PhaseName(int p);  // CSV column part: outer_l, ..., compose_r

// Render thread only (other threads are ignored). FrameStart at the WaitGetPoses exit of frame `no`; FrameEnd right
// before the last Submit of the frame. A mark that was already set in the frame keeps its first value.
void FrameStart(uint64_t no);
void FrameEnd();
void Begin(Phase p);
void End(Phase p);

// Spans inside the eye groups, for the cost of the MVR steps: the engine's render tasks of each group (CPhase1Render
// with the proxy's work around it, CSSAORender through CPhase2Render, CPhase3Render) and the proxy's parts of the MVR
// G-buffer (mvr_pass). Same rules as the phases: render thread only, the first Begin and End of a frame count.
enum Span {
    P1OuterL, P1OuterR, P1FocusL, P1FocusR,      // CPhase1Render (outer: with the MVR part; focus: skipped or own)
    LitOuterL, LitOuterR, LitFocusL, LitFocusR,  // CSSAORender start -> CPhase2Render end
    P3OuterL, P3OuterR, P3FocusL, P3FocusR,      // CPhase3Render
    EarlyL, EarlyR,                              // focus's SSAO and lights run early in outer's group (B9a)
    CopyBackL, CopyBackR,                        // slice 0 of the copies into the game's targets
    ExtractL, ExtractR,                          // slice 1's corners into the stash (colors and the depth pass)
    MaskStashL, MaskStashR,                      // the masks into the stash
    FillL, FillR,                                // the stash into focus's clones and the masks in the window
    FwdOutL, FwdOutR,                            // B9d: the live forward pass's corners into focus's clones
    kSpans
};
const char* SpanName(int s);  // "p1 outer L", ...
void SpanBegin(Span s);
void SpanEnd(Span s);

struct Result {
    uint64_t frame = 0;
    float frameMs = -1;          // FrameStart -> FrameEnd, < 0 when not measured
    float phaseMs[kPhases] = {};  // < 0 when the phase did not run in that frame
    float spanMs[kSpans] = {};    // < 0 when the span did not run in that frame
};
// Frames measured since the last call, oldest first; returns how many were written to out (at most max).
int Take(Result* out, int max);
bool Enabled();  // [timing] gpu=1 and a device

void Release();  // VR_Shutdown: queries and the context reference

}  // namespace qvfr::gpu

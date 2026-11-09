// O2 step 3 (docs/mvr_optimizations.md): the focus groups' forward pass recording without its velocity sub-passes.
//
// CForwardBuildCommandThread queues the forward pass recording (0x9cf790) on an engine worker, and the render thread
// waits for it. The recording walks the draw queue once per sub-pass and submits every entry through the engine device
// (vtable +0x660). Two of those walks are the velocity pass (pass keys DAT_142670870 and DAT_142670868, targets from
// device+0x2e8), and with MVR focus's velocity draws are dropped anyway (B11): the velocity pass's first draw fills the
// target with outer's vectors and the rest go nowhere. While armed for a focus group, the recording of view 2 submits
// velocity entries only until one of them went into the recording (the submit's result, low byte; its replay is the
// draw B11 fills at) and returns at once for the rest. The engine records into its own command buffer, so the D3D11
// draws only happen at the replay.
// Everything else of the recording (the forward sub-passes with the transparent objects, the jitter, the velocity
// pass's binds and its other calls) stays as it is.
#pragma once

#include <cstdint>

namespace qvfr::fwdrec {

// Hooks the forward recording and the device submit (anchors ForwardDraw and DeviceSubmit). False when one is
// missing; then Arm does nothing.
bool Install();
bool Installed();

// Quad, right before a focus group's CForwardBuildCommandThread whose velocity comes from outer in this frame: the
// next recording of view 2 skips its velocity submits after the first draw. False (reason in why) when the engine's
// pass keys cannot be checked.
bool Arm(int eye, const char** why);

// What the armed recording did; Disarm at the focus group's CEndEye returns it and ends the arming.
struct Result {
    bool armed = false;
    bool seen = false;        // a recording of view 2 ran while armed
    uint32_t kept = 0;        // velocity submits that went through (up to and with the first one that drew)
    uint32_t skipped = 0;     // velocity submits skipped
    uint32_t drew = 0;        // kept velocity submits that went into the recording (at most 1 per sub-pass walk)
};
Result Disarm();

// Smoke test: the detours on a fake device, pass table and forward recording (no engine). Text into out; true = OK.
bool SelfTest(char* out, size_t n);

}  // namespace qvfr::fwdrec

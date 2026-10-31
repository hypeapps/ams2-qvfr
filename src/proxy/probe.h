// M4.1 probe ([quad] mode=probe): logging hooks in the engine task graph (AMS2AVX.exe) and in the D3D11 immediate
// context. Answers what the focus groups need: in which order and on which threads the eye tasks run, whether their
// draws reach the immediate context between CBeginEye and CEndEye, which texture is a view's output, what the render
// context holds per view. The image does not change.
#pragma once

#include <openvr.h>  // 0.9.12

namespace qvfr::probe {

void Install();                                            // VR_Init (engine hooks); D3D11 hooks follow at the first Submit
void OnFrameStart();                                       // WaitGetPoses entry on the game's render thread
void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex);  // around the real Submit
void OnSubmitEnd(vr::EVREye eye);
void LogStats(bool totals);
void Shutdown();                                           // VR_Shutdown

}  // namespace qvfr::probe

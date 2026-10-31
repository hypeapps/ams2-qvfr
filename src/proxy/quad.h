// Quad mode ([quad] mode=quad): two more eye groups in the engine's VR task graph render the focus views.
// The builder loop runs for views 2, 3 and 4 (4-byte patch); views 3 and 4 do not fit the render context arrays,
// so their tasks get view 2, and the focus groups swap the view 2 fields of the context for their own camera data
// while they run. M4.2: the focus groups render with the eye cameras (focus L with the left eye's data), so the image
// in the headset does not change; shots compare the four views.
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::quad {

void Install();       // VR_Init: engine hooks and the builder patch (only on the supported exe)
void OnFrameStart();  // WaitGetPoses entry (render thread): frame bookkeeping, shot hotkey, shot read-back
void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex);  // the first Submit gives the D3D11 device
void LogStats(bool totals);
std::string Describe();   // one line for summary.txt
const char* RunSuffix();  // "_quad" while the focus groups run, else ""
void Shutdown();          // VR_Shutdown: hooks off, builder patch removed

}  // namespace qvfr::quad

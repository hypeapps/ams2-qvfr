// Diagnostics for players (M8.5): what the mod detected and what the player should change.
// - The "Detected" block: one summary in the log about 20 s after the first frame (exe, game settings, path, backend,
//   OpenXR runtime and layers, headset, gaze, quad state, known modules, warnings), or at VR_Shutdown if earlier.
// - Findings: logged once each as "WARNING:" or "Note:". Serious ones (the mod cannot work, or the image will be wrong)
//   also go to a Windows message box ([log] warning_box), shown on its own thread so the game never waits for it.
#pragma once

#include <openvr.h>  // 0.9.12

#include <string>

namespace qvfr::diag {

enum class Level { Info, Warning, Serious };

// Logs the finding the first time `id` comes; later calls with the same id do nothing.
void Report(Level level, const char* id, const char* fmt, ...);
// A line of the Detected block (the last value of a key wins). Thread-safe.
void Set(const char* key, const std::string& value);

// Once at startup, after the backend is loaded: exe and its build (quad mode), game graphics settings (TAA), then the
// first message box with the serious findings so far.
void CheckStartup();
void NoteOpenVrSystem(vr::IVRSystem* real);  // path=openvr, after VR_Init: headset names from SteamVR
void CheckModule(const std::wstring& fileNameLower);  // every new module in the process (LibMagic)
// Shows the serious findings not shown yet in one message box (nothing when there are none or warning_box=0).
void ShowPendingBox();
// The Quad line of the Detected block (quad::Describe; registered so this file does not depend on quad.cpp).
void SetQuadDescribe(std::string (*describe)());

// Render thread, every proxy frame: gaze source (nullptr = no gaze read), its validity and the focus window source
// (quad::WindowSource). Writes the Detected block after kDetectedS seconds.
void OnFrame(const char* gazeSource, bool gazeOk, int windowSrc);
void OnShutdown();  // the Detected block if it was not written yet

}  // namespace qvfr::diag

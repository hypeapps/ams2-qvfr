// Sampler log: every distinct D3D11_SAMPLER_DESC created on the game's device, with the module and address of its first
// caller, its number of call sites and of creations (the runtime returns the same object for an equal description).
// Shows the game's texture filtering and MipLODBias. A hook on the device's CreateSamplerState, installed through a
// temporary WARP device at VR_Init; the stats interval logs the descriptions new since the last one, VR_Shutdown all
// of them. [debug] sampler_log=0 leaves the log out.
//
// Texture mip bias under DLSS ([dlaa] mip_bias, mip_bias_offset): the game's scene textures are sampled at the output
// resolution's detail, as the DLSS Programming Guide asks (3.5: bias = log2(render / output width) - 1 + epsilon; -1 with
// DLAA). Scene samplers are the ones the game's exe creates with no comparison or min/max reduction, linear minification
// with linear mips or anisotropic, and MaxLOD above 0 (shadow maps, point sampling and passes on render targets stay as
// they are). A sampler object is immutable, so the hook makes a copy with the bias added to the game's MipLODBias (per
// bias value in use, in steps of 1/8) and a hook on PSSetSamplers binds it in place of the game's object. Each eye
// group uses its view's bias (outer or focus, each from its render scale); from the group's TAA resolve to its end the
// game's samplers come back (the passes after TAA run at the output size). Each group start and the TAA resolve rebind
// the slots bound before them (the engine may not bind again what it thinks is bound). The table holds a reference on
// each object it lists, so a released sampler's address never comes back with another description.
// Without the quad eye groups (stereo) outer's bias holds for the whole frame, the passes after TAA included.
#pragma once

#include <d3d11.h>

#include <cstddef>
#include <cstdint>

namespace qvfr::samplers {

// VR_Init, once per process ([debug] sampler_log or [dlaa] mode). Guards itself.
void Install();
// Stats interval: the descriptions new since the last call (sampler_log) and the binds replaced by the mip bias;
// `totals`: all of them.
void LogStats(bool totals);

// Render thread, every frame start (dlaa.cpp): `dlss` DLSS replaces the game's TAA in this frame, `ini` [dlaa]
// mip_bias; the bias of outer's and focus's views over the game's MipLODBias (stereo: outer's for both eyes).
void SetViewBias(bool dlss, bool ini, float outer, float focus);
// Benchmark sequence and test switch dlss_mip_bias: -1 the ini's, 0 off, 1 on (still only while DLSS runs).
int GetSwitch();
void SetSwitch(int v);
// Eye groups on the render thread (0 outer L, 1 outer R, 2 focus L, 3 focus R), before the engine's CBeginEye and
// CEndEye Execute. BeginGroup rebinds the pixel shader sampler slots of `ctx` for the group.
void BeginGroup(int group, ID3D11DeviceContext* ctx);
void EndGroup();
// The group's TAA resolve (dlaa.cpp, DLSS in its place): the game's samplers until the group ends (no-op without the
// eye groups).
void SceneEnd(ID3D11DeviceContext* ctx);

// Smoke test (qvfr_proxy_smoke --samplers): the hooks on `dev`'s CreateSamplerState and its immediate context's
// PSSetSamplers, the distinct descriptions so far and how many of them have a negative MipLODBias.
bool TestInstall(ID3D11Device* dev);
size_t TestDistinct();
size_t TestNegativeBias();
uint64_t TestCreations();
size_t TestSceneSamplers();    // game samplers the bias applies to
uint64_t TestReplacedBinds();  // samplers replaced at bind so far (PSSetSamplers and the rebinds)

}  // namespace qvfr::samplers

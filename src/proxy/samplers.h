// Sampler log: every distinct D3D11_SAMPLER_DESC created on the game's device, with the module and address of its first
// caller, its number of call sites and of creations (the runtime returns the same object for an equal description).
// Shows the game's
// texture filtering and MipLODBias, the input for a mip bias under DLSS. A hook on the device's CreateSamplerState,
// installed through a temporary WARP device at VR_Init; the stats interval logs the descriptions new since the last
// one, VR_Shutdown all of them. [debug] sampler_log=0 leaves the hook out.
//
// Mip bias test (scenario switches dlss_mip_bias and sampler_aniso, off unless a switch sets them): for each scene
// sampler the game's exe creates (no comparison or min/max reduction, linear minification with linear mips or
// anisotropic, MaxLOD above 0) the hook makes three more with the original's description changed: MipLODBias - 1, the
// anisotropic filter at 16x, both. A sampler object is immutable, so a hook on PSSetSamplers binds the variant the
// current setting and eye group want in place of the game's object, and each eye group's start rebinds the slots
// bound before it (the engine may not bind again what it thinks is bound). The table holds a reference on each
// object it lists, so a released sampler's address never comes back with another description.
#pragma once

#include <d3d11.h>

#include <cstddef>
#include <cstdint>

namespace qvfr::samplers {

// VR_Init, once per process. Guards itself.
void Install();
// Stats interval: the descriptions new since the last call; `totals`: all of them with their counts.
void LogStats(bool totals);

// Mip bias test. Bias: 0 off, 1 every view, 2 the focus views only (eye groups 2 and 3). Aniso: the anisotropic 16x
// variants. Render thread; a change logs the binds replaced under the previous setting.
constexpr int kBiasModes = 3;
void SetBias(int mode);
int GetBias();
void SetAniso(bool on);
bool GetAniso();
// Eye groups on the render thread (0 outer L, 1 outer R, 2 focus L, 3 focus R), before the engine's CBeginEye and
// CEndEye Execute. BeginGroup rebinds the pixel shader sampler slots of `ctx` for the group.
void BeginGroup(int group, ID3D11DeviceContext* ctx);
void EndGroup();

// Smoke test (qvfr_proxy_smoke --samplers): the hooks on `dev`'s CreateSamplerState and its immediate context's
// PSSetSamplers, the distinct descriptions so far and how many of them have a negative MipLODBias.
bool TestInstall(ID3D11Device* dev);
size_t TestDistinct();
size_t TestNegativeBias();
uint64_t TestCreations();
size_t TestVariantSets();     // game descriptions with mip bias variants
uint64_t TestReplacedBinds();  // samplers replaced at bind so far (PSSetSamplers and the group rebinds)

}  // namespace qvfr::samplers

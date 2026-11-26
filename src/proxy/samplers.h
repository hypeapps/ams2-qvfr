// Sampler log: every distinct D3D11_SAMPLER_DESC created on the game's device, with the module and address of its first
// caller, its number of call sites and of creations (the runtime returns the same object for an equal description).
// Shows the game's
// texture filtering and MipLODBias, the input for a mip bias under DLSS. A hook on the device's CreateSamplerState,
// installed through a temporary WARP device at VR_Init; the stats interval logs the descriptions new since the last
// one, VR_Shutdown all of them. [debug] sampler_log=0 leaves the hook out.
#pragma once

#include <d3d11.h>

#include <cstddef>
#include <cstdint>

namespace qvfr::samplers {

// VR_Init, once per process. Guards itself.
void Install();
// Stats interval: the descriptions new since the last call; `totals`: all of them with their counts.
void LogStats(bool totals);

// Smoke test (qvfr_proxy_smoke --samplers): the hook on `dev`'s CreateSamplerState, the distinct descriptions so far
// and how many of them have a negative MipLODBias.
bool TestInstall(ID3D11Device* dev);
size_t TestDistinct();
size_t TestNegativeBias();
uint64_t TestCreations();

}  // namespace qvfr::samplers

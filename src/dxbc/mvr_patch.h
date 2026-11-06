// Adds NVIDIA Multi-View Rendering position outputs to a compiled vertex shader (SM4/SM5 DXBC).
//
// The patched shader writes view 0 as before and, for each extra view n (1..3), an output
// NV_POSITION_VIEW_n_SEMANTIC = M_n * position, where M_n is a 4x4 matrix (4 rows) in a new constant buffer.
// Writes to the SV_Position register go to a new temp register; before every ret the shader copies that temp to the
// position output and writes the extra views. Create the result with NvAPI_D3D11_CreateVertexShaderEx and the
// NV_POSITION_SEMANTIC custom semantic named "NV_POSITION".
//
// With clip, the shader also gets per-view clip distances, so a view can be cut to a part of its render target slice
// (all slices share one size and one viewport): SV_ClipDistance0 = (1, 1, 1, 1) for view 0 and
// NV_CLIP_DISTANCE_0_VIEW_n_SEMANTIC = C_n * position for view n, where C_n holds 4 clip planes (4 rows; a point is kept
// where all four dot products are >= 0). Both are float4, as the driver requires (B5). Create the result with the
// NV_CLIP_DISTANCE_0_SEMANTIC custom semantic named "NV_CLIP_DISTANCE_0" in addition to NV_POSITION.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qvfr::dxbc {

struct MvrVsPatchOptions {
    uint32_t cbSlot = 3;      // constant buffer slot for the matrices; must be unused by the shader
    uint32_t extraViews = 1;  // 1..3
    bool clip = false;        // add per-view clip distances
    // Buffer layout: view n (1..3) has M_n at rows R*(n-1) .. R*(n-1)+3 and, with clip, C_n at the next 4 rows,
    // where R = RowsPerView(clip).
    static constexpr uint32_t RowsPerView(bool clip) { return clip ? 8 : 4; }
};

struct MvrVsPatchInfo {
    uint32_t positionReg = 0;
    uint32_t firstNewOutputReg = 0;  // first register this patch added
    uint32_t viewPositionReg = 0;    // NV_POSITION_VIEW_1 (views 2, 3 follow)
    uint32_t clipReg = 0;            // SV_ClipDistance0 of view 0 (with clip)
    uint32_t viewClipReg = 0;        // NV_CLIP_DISTANCE_0_VIEW_1 (with clip; views 2, 3 follow)
    uint32_t tempReg = 0;
    uint32_t returns = 0;           // epilogues inserted
    uint32_t rewrittenOperands = 0;  // writes to the position register redirected to the temp
};

// The semantic name of view n's position output, as the HLSL MVR shaders declare it.
std::string MvrPositionSemantic(uint32_t view);
// The semantic name of view n's clip distance output (n >= 1).
std::string MvrClipSemantic(uint32_t view);

bool PatchVsForMvr(const void* code, size_t size, const MvrVsPatchOptions& opt, std::vector<uint8_t>& out,
                   MvrVsPatchInfo* info, std::string* err);

}  // namespace qvfr::dxbc

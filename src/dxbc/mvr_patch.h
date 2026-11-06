// Adds NVIDIA Multi-View Rendering position outputs to a compiled vertex shader (SM4/SM5 DXBC).
//
// The patched shader writes view 0 as before and, for each extra view n (1..3), an output
// NV_POSITION_VIEW_n_SEMANTIC = M_n * position, where M_n is a 4x4 matrix (4 rows) in a new constant buffer.
// Writes to the SV_Position register go to a new temp register; before every ret the shader copies that temp to the
// position output and writes the extra views. Create the result with NvAPI_D3D11_CreateVertexShaderEx and the
// NV_POSITION_SEMANTIC custom semantic named "NV_POSITION".
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qvfr::dxbc {

struct MvrVsPatchOptions {
    uint32_t cbSlot = 3;      // constant buffer slot for the matrices; must be unused by the shader
    uint32_t extraViews = 1;  // 1..3; view n uses rows 4*(n-1) .. 4*(n-1)+3 of the buffer
};

struct MvrVsPatchInfo {
    uint32_t positionReg = 0;
    uint32_t firstNewOutputReg = 0;
    uint32_t tempReg = 0;
    uint32_t returns = 0;           // epilogues inserted
    uint32_t rewrittenOperands = 0;  // writes to the position register redirected to the temp
};

// The semantic name of view n's position output, as the HLSL MVR shaders declare it.
std::string MvrPositionSemantic(uint32_t view);

bool PatchVsForMvr(const void* code, size_t size, const MvrVsPatchOptions& opt, std::vector<uint8_t>& out,
                   MvrVsPatchInfo* info, std::string* err);

}  // namespace qvfr::dxbc

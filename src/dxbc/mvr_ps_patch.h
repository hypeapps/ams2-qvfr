// Makes a compiled pixel shader (SM4/SM5 DXBC) draw an extra Multi-View Rendering view correctly (docs/mvr.md B9b).
//
// With MVR, every vertex shader output but the position reaches the pixel shader from view 0 (B1). The scene's
// forward shaders depend on the view in two ways (B2):
// - screen inputs: an input the shader divides by its own w (div r.xy, vN.xy, vN.w) is the screen UV times w of
//   view 0's position. The patch replaces each such input by a temp computed at the start:
//   v.xy = v.xy * S + v.w * O, where (S, O) is the identity in view 0 and row 0 of a new constant buffer otherwise;
// - screen textures (by resource name, e.g. globalTexture0..4: G-buffer color, normals, linear depth, deferred
//   lights): in an extra view every instruction that reads one reads the extra view's texture in slot texBase + i
//   instead, and coordinates in UV (sample, gather, lod) are multiplied by row 1 first. Texel coordinates (ld) stay.
//   resinfo and sampleinfo keep the original texture.
// The view number comes from SV_RenderTargetArrayIndex (MVR sends each view to its render target array slice; the
// pixel shader can read it even though no vertex shader writes it, B1 cases 10 and 11), added as a new input. It counts
// only while row 2.x of the constant buffer holds kMvrPsMagic: the game draws some passes into array slices through
// geometry shaders that write SV_RenderTargetArrayIndex themselves (B9b in the game: 3 such shaders), and those must
// keep drawing as the original.
//
// View 0, and every slice without the magic value, computes exactly what the original shader computes: S = 1 and
// O = 0 are immediates (movc on the gated view number), and the extra view's textures are read only in the branch of
// a gated non-zero view. The new constant buffer and textures need not be bound unless MVR draws more than one view.
//
// Layout of the extra view (one view, slice 1 of a target of the outer size W x H, the focus image of size fw x fh in
// its top left corner, with its screen textures of the focus size):
//   row 0 = (S.x, S.y, O.x, O.y): with view 1's clip position x_1 = s.x * x_0 + o.x * w and y_1 = s.y * y_0 + o.y * w
//           and screen inputs u = 0.5 * ssd.x * (x + w), v = 0.5 * ssd.y * (w - y):
//           S = s, O.x = 0.5 * ssd.x * (1 + o.x - s.x), O.y = 0.5 * ssd.y * (1 - o.y - s.y);
//   row 1 = (W / fw, H / fh, 1, 1), so a UV of the corner of the outer-size slice addresses the focus-size texture;
//   row 2 = (kMvrPsMagic as uint bits, 0, 0, 0): the view number counts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qvfr::dxbc {

struct MvrPsPatchOptions {
    uint32_t cbSlot = 3;    // constant buffer slot of the two rows; must be unused by the shader
    uint32_t texBase = 21;  // the extra view's screen texture i goes to slot texBase + i; must be unused
    std::vector<std::string> screenTextures = {"globalTexture0", "globalTexture1", "globalTexture2", "globalTexture3",
                                               "globalTexture4"};
    uint32_t viewReg = 0;  // input register of SV_RenderTargetArrayIndex; 0 = the one after the last input
};

struct MvrPsPatchInfo {
    uint32_t screenInputs = 0;     // inputs transformed
    uint32_t screenInputMask = 0;  // bit per input register
    uint32_t textureMask = 0;      // bit i: screen texture i is read
    uint32_t wrapped = 0;          // instructions that read a screen texture (each got a branch for the extra view)
    uint32_t scaled = 0;           // of them with UV coordinates (scaled by row 1)
    uint32_t viewReg = 0;          // input register of SV_RenderTargetArrayIndex
    uint32_t firstTemp = 0;        // first temp register the patch added
    uint32_t temps = 0;            // temps the patch added
    // Per screen texture (option order): the shader's own texture register, kNoRegister when it reads none with a
    // branch. The extra view's copy then has to be bound at texBase + i.
    static constexpr uint32_t kNoRegister = 0xFFFFFFFFu;
    std::vector<uint32_t> textureRegs;
};

enum class MvrPsPatchResult {
    Patched,    // `out` holds the patched shader
    NotNeeded,  // no screen input and no screen texture: the original shader is right for every view
    Refused,    // `err` says why
};

// The number of rows in the constant buffer, and the value of row 2.x (uint) that turns the view number on ("QVFR").
constexpr uint32_t kMvrPsRows = 3;
constexpr uint32_t kMvrPsMagic = 0x52465651;

MvrPsPatchResult PatchPsForMvr(const void* code, size_t size, const MvrPsPatchOptions& opt, std::vector<uint8_t>& out,
                               MvrPsPatchInfo* info, std::string* err);

}  // namespace qvfr::dxbc

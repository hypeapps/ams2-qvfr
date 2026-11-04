// Focus culling (M9.3): the engine culls the scene once per frame with view 1's frustum (both eyes) and every eye group,
// the focus groups too, draws that set. Before a focus group the proxy runs the engine's scene gather and processor
// again, inline on the render thread, with a frustum block whose four side planes are the focus window's, and copies the
// result into the render context's draw queue. The eye groups run one after another, so the eye's set is no longer needed
// by then; the next frame culls it again. Engine functions and layout: docs/ams2-renderer.md (M9.2), AMS2AVX.exe aa43bd68.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qvfr::focus_cull {

constexpr size_t kBlockSize = 0x1a0;  // engine frustum block (camera+0x260, ctx+0x190 + view*0x1a0)

// Frustum block layout, learned from a block the engine built (0xcf6b80): 6 world-space planes (float4 n, d; inside is
// n.p + d >= 0), per plane a sign mask pair (x, y, z lanes) and 8 corners (intersections of three planes, w = 1).
struct Layout {
    int side[4] = {};        // block plane of the clip-space left, right, bottom and top edge
    uint8_t corner[8][3] = {};  // planes each corner is the intersection of
    uint32_t maskSet = 0, maskClear = 0;  // mask lane bits for a normal component >= 0 and < 0
};

// Learns the layout from an engine block of an eye and that eye's view (row vectors, world to view) and projection.
// False (reason in why) when the block does not look like the engine format this code was written for.
bool Learn(const uint8_t* block, const float* view, const float* eyeProj, Layout& out, char* why, size_t whySize);

// The eye block with the four side planes of view * focusProj (each edge pushed out by `margin` in NDC), masks and
// corners recomputed. Near and far stay the eye's.
void Build(const uint8_t* eyeBlock, const Layout& l, const float* view, const float* focusProj, float margin,
           uint8_t* out);

struct Result {
    double ms = 0;               // gather + processor + queue copy on the render thread
    uint32_t setEye = 0, setFocus = 0;      // renderables set (ctx+0x670) before and after
    uint32_t queueEye = 0, queueFocus = 0;  // render context draw queue (ctx+0x698) before and after
    bool notYet = false;  // with false from Run: the engine has not culled this graph yet (loading), not a failure
};

// Engine functions resolved (anchors matched).
bool Available();

// Re-culls the scene into the render context's draw queue with `block` (persistent: the processor keeps a pointer to it
// until the next frame). Checks the engine objects first and changes nothing when one fails (false, reason in why).
// `fault` is set when the engine faulted inside: the objects were put back to "done", culling must stay off.
bool Run(uint8_t* ctx, const uint8_t* block, Result& r, bool& fault, char* why, size_t whySize);

}  // namespace qvfr::focus_cull

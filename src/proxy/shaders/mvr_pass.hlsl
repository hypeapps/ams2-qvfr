// MVR gate B8 (mvr_pass.h): the module's own passes, one full-screen triangle each.
// - PSDepth: the depth of view 1 (slice 1 of the pass's two-slice copy, top-left corner) into the eye's stash at the
//   focus size: D3D11 copies no part of a depth texture. MVR's view 1 differs from focus's own pass by the GPU's
//   sub-pixel snapping (tens to hundreds of float steps on sloped surfaces, another surface on some silhouette pixels),
//   and the game's later passes (forward, velocity) draw focus's own geometry with LESS_EQUAL against this depth. So it
//   can be made conservative: dilate = the farthest of the 3 x 3 neighbourhood, plus slopeScale pixels' worth of the
//   local slope, plus biasUlps float steps (never past 1, the far plane). The forward pass writes depth, so the exact
//   depth of focus's geometry is back after it.
// - PSMask: a sun mask of outer (1/4 size, drawn at the end of the G-buffer pass) read in the focus window into the
//   focus group's mask target: outer UV = origin + focus UV * scale.
// - PSDepthIn (gate B9c): a focus-size depth (focus's clone) as it is into the top-left corner of slice 1 of a
//   two-slice copy, where the MVR forward pass then draws view 1 against it.

cbuffer Params : register(b0) {
    float2 origin;   // the focus window's top left corner in outer UV
    float2 scale;    // the focus window's size in outer UV
    float2 invSize;  // 1 / the target's size, pixels
    float2 pad;
};

cbuffer DepthParams : register(b1) {
    uint biasUlps;     // float steps farther
    float slopeScale;  // plus this many pixels' worth of the local depth slope
    uint2 cornerSize;  // focus's corner of slice 1 (neighbours are read inside it)
    uint dilate;       // 1: the farthest depth of the 3 x 3 neighbourhood first
    uint3 pad1;
};

Texture2DArray<float> depthArr : register(t0);
Texture2D<float> mask : register(t0);
Texture2D<float> depthIn : register(t0);
SamplerState linearClamp : register(s0);

float4 VSMain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

float DepthAt(int2 p) {
    return depthArr.Load(int4(clamp(p, int2(0, 0), int2(cornerSize) - 1), 1, 0));
}

// The smaller one-sided difference per axis: a surface's slope without the jump at a silhouette.
float Slope(int2 p, float d, int2 axis) {
    const float a = DepthAt(p + axis), b = DepthAt(p - axis);
    const float fa = a < 1.0 ? abs(a - d) : 1.0, fb = b < 1.0 ? abs(d - b) : 1.0;
    return min(fa, fb);
}

float PSDepth(float4 pos : SV_Position) : SV_Depth {
    const int2 p = int2(pos.xy);
    float d = DepthAt(p);
    if (d >= 1.0) return d;
    if (dilate) {
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x) d = max(d, DepthAt(p + int2(x, y)));
        if (d >= 1.0) return d;
    }
    float moved = d;
    if (slopeScale > 0) {
        const float slope = max(Slope(p, d, int2(1, 0)), Slope(p, d, int2(0, 1)));
        moved += slopeScale * (slope < 1.0 ? slope : 0.0);
    }
    return min(asfloat(asuint(moved) + biasUlps), 1.0);
}

float PSMask(float4 pos : SV_Position) : SV_Target {
    return mask.SampleLevel(linearClamp, origin + pos.xy * invSize * scale, 0);
}

float PSDepthIn(float4 pos : SV_Position) : SV_Depth {
    return depthIn.Load(int3(pos.xy, 0));
}

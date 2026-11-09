// MVR (docs/mvr.md B10): the focus view's motion vectors from outer's of the same eye.
//
// With MVR the focus image (G-buffer, forward, linear depth) is outer's geometry drawn through the window map, while the
// focus group's own velocity pass draws focus's geometry, whose object matrices differ from outer's by float rounding
// (up to 0.4 px near the cockpit, changing from frame to frame). This pass gives DLSS motion vectors of the geometry the
// focus image shows: outer's, read in the window and moved into focus's uv.
//
// The engine stores uv now - uv before (y down). Per focus pixel:
//   uo = outer uv of the pixel centre (inverse of this frame's window map);
//   vo = outer's motion vector at uo, from the 2 x 2 outer texels around it: bilinear when their depths (outer's
//        linear depth, the same rasterization as its motion vectors) all match the pixel's (focus's linear depth), else
//        the one whose depth is closest (no blending across a silhouette);
//   motion = uf - prevMap(uo - vo): the point's position in the last frame's focus image (the window may have moved).
// The window maps are uv = scale * outer uv + offset per axis (mvr_motion.h).

cbuffer Params : register(b0) {
    float4 cur;    // this frame's map: x scale u, y offset u, z scale v, w offset v
    float4 prev;   // the last frame's map
    float4 sizes;  // focus width, height, outer width, height
    float4 opt;    // x: relative depth tolerance for blending, y: 1 = nearest texel only (diagnosis)
};

Texture2D<float2> outerMotion : register(t0);
Texture2D<float> focusDepth : register(t1);
Texture2D<float> outerDepth : register(t2);
RWTexture2D<float2> outMotion : register(u0);

[numthreads(8, 8, 1)]
void CSFocusMotion(uint3 id : SV_DispatchThreadID) {
    const uint2 fs = (uint2)sizes.xy;
    if (id.x >= fs.x || id.y >= fs.y) return;
    const int2 os = (int2)sizes.zw;
    const float2 uf = (id.xy + 0.5) / sizes.xy;
    const float2 uo = (uf - cur.yw) / cur.xz;
    const float2 t = uo * sizes.zw - 0.5;
    const int2 b = (int2)floor(t);
    const float2 fr = t - b;
    const float dp = focusDepth.Load(int3(id.xy, 0));
    float2 vo = 0;
    float2 sum = 0;
    float bestErr = 1e30, bestW = -1, maxErr = 0;
    float2 best = 0;
    [unroll] for (int i = 0; i < 4; ++i) {
        const int2 o = int2(i & 1, i >> 1);
        const int2 c = clamp(b + o, int2(0, 0), os - 1);
        const float w = (o.x ? fr.x : 1 - fr.x) * (o.y ? fr.y : 1 - fr.y);
        const float2 m = outerMotion.Load(int3(c, 0));
        const float dc = outerDepth.Load(int3(c, 0));
        const float err = abs(dc - dp) / max(abs(dp), 1e-6);
        sum += w * m;
        maxErr = max(maxErr, err);
        if (err < bestErr || (err == bestErr && w > bestW)) {
            bestErr = err;
            bestW = w;
            best = m;
        }
    }
    vo = (opt.y == 0 && maxErr <= opt.x) ? sum : best;
    const float2 ufPrev = prev.xz * (uo - vo) + prev.yw;
    outMotion[id.xy] = uf - ufPrev;
}

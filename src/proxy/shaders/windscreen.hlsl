// The depth the windscreen glass tests against (windscreen.h, [dlaa] windscreen_depth): the scene's depth, each pixel
// the farthest of its neighbourhood within `radius`. TAA or DLSS anti-aliased the cockpit's outline against what is
// behind the glass, but the glass's depth test sorts those mixed pixels to the cockpit, so they kept the bright
// background where the glass darkens the rest (a light, stepped seam along the roll cage and the mirror). With the
// farther depth the glass covers that outer ring of the outline too. Depth grows away from the camera (1 = far plane).

// A compute pass (no draw state of the game's touched): a copy of the depth in, the slice written whole into a texture
// that is then copied into a depth buffer; outside the viewport the depth as it was.

cbuffer DepthParams : register(b0) {
    uint2 dOrigin;  // the viewport's corner in the depth texture
    uint2 dSize;    // the viewport's size
    uint2 dTex;     // the texture's size
    uint dSlice;
    int dRadius;
};

Texture2DArray<float> depthSrc : register(t0);
RWTexture2DArray<float> depthOut : register(u0);

[numthreads(8, 8, 1)]
void CSDilate(uint3 id : SV_DispatchThreadID) {
    if (id.x >= dTex.x || id.y >= dTex.y) return;
    const int2 p = int2(id.xy);
    const int2 lo = int2(dOrigin), hi = int2(dOrigin + dSize) - 1;
    float d = depthSrc.Load(int4(p, dSlice, 0));
    if (all(p >= lo) && all(p <= hi))
        for (int y = -dRadius; y <= dRadius; ++y)
            for (int x = -dRadius; x <= dRadius; ++x)
                d = max(d, depthSrc.Load(int4(clamp(p + int2(x, y), lo, hi), dSlice, 0)));
    depthOut[uint3(id.xy, dSlice)] = d;
}

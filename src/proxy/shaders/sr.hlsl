// DLSS SR (sr.h, dlaa.cpp): at the TAA resolve the textures the passes after TAA need are carried from their clones at
// the render size into the originals at the eye size. One full-screen triangle into the original: colors through the
// sampler bound (linear for color formats, point for one-channel data such as linear depth), depth with SV_Depth from
// the nearest texel (no blending of depths at edges).

Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);

struct VsOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VsOut VSFull(uint id : SV_VertexID) {
    VsOut o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float4 PSColor(VsOut i) : SV_Target {
    return src.SampleLevel(samp, i.uv, 0);
}

float PSDepth(VsOut i) : SV_Depth {
    return src.SampleLevel(samp, i.uv, 0).r;
}

// DLSS SR in quad views (sr_cb.h, dlaa-plan S2.1): the passes after the TAA resolve were recorded while the scene's
// render size was in the engine objects, so their constants hold it where the targets now have the view size (soft
// particles and light glows read the depth at SV_Position / size and saw it 1.5 times too far out). A window of such a
// constant buffer, copied into `fixWindow`, gets each pair of floats at an even index that equals a render-size pair
// (width and height, their reciprocals and their fractions) replaced by the view-size pair; the table is in FixPairs.
cbuffer FixPairs : register(b0) {
    uint fixPairCount;   // entries used in fixPairs
    uint fixUnused;      // (the window's size comes from the buffer)
    uint2 fixPad;
    float4 fixPairs[48];  // xy: the render-size pair, zw: the view-size pair
};
RWByteAddressBuffer fixWindow : register(u0);

[numthreads(64, 1, 1)]
void CSFixSizes(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x * 2;
    uint bytes;
    fixWindow.GetDimensions(bytes);
    if (i + 1 >= bytes / 4) return;
    const float2 v = asfloat(fixWindow.Load2(i * 4));
    for (uint p = 0; p < fixPairCount; ++p) {
        const float4 e = fixPairs[p];
        if (all(abs(v - e.xy) <= abs(e.xy) * 2e-6)) {
            fixWindow.Store2(i * 4, asuint(e.zw));
            return;
        }
    }
}

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

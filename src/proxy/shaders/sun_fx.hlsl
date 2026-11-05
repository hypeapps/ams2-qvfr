// Screen-space sun effects of a focus view taken from outer (sun_fx.h): one full-screen triangle into the focus view's
// HDR target, added on top (blend one, one). Each focus pixel reads the outer accumulation (crepuscular rays and lens
// flares as outer drew them) at the same direction: outer UV = origin + focus UV * scale.

cbuffer Params : register(b0) {
    float2 origin;      // the focus window's top left corner in outer UV
    float2 scale;       // the focus window's size in outer UV
    float2 vpOrigin;    // the bound viewport's top left corner, pixels
    float2 vpInvSize;   // 1 / the bound viewport's size
};

Texture2D accum : register(t0);
SamplerState linearClamp : register(s0);

float4 VSMain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float2 uvFocus = (pos.xy - vpOrigin) * vpInvSize;
    return float4(accum.SampleLevel(linearClamp, origin + uvFocus * scale, 0).rgb, 0);
}

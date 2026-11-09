// DLAA (dlaa.h): DLSS's bias-current-color mask from the game's stencil. The game's TAA resolve (PS 16508) reads the
// stencil as normal/stencil .w * 65535 >> 8 and keeps a fixed, short history (weight 0.75) where bit 2 is set; these
// pixels get `biasValue` in the mask (DLSS leans on the current frame there), all others 0.

cbuffer Params : register(b0) {
    float biasValue;
    float3 pad;
};

Texture2D<float4> normalStencil : register(t0);
RWTexture2D<unorm float> mask : register(u0);

[numthreads(8, 8, 1)]
void CSMask(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    mask.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    uint s = (uint)(normalStencil.Load(int3(id.xy, 0)).w * 65535.0) >> 8;
    mask[id.xy] = (s & 2) ? biasValue : 0.0;
}

// Displays and mirrors from the game's TAA ([dlaa] cockpit_taa): after DLSS, the pixels whose stencil high byte has all of
// `bits` take the game's TAA result, which handles changing digits without trails. Stencil high bytes seen in the game:
// 0x88 = displays and mirrors (DDU, timer, rear-view mirror), 0x86 = the rest of the player's car; trees 0x1A and sky
// 0x28 share single bits with them and stay with DLSS.
cbuffer CockpitParams : register(b1) {
    uint bits;
    uint3 pad2;
};
Texture2D<float4> taaColor : register(t1);
RWTexture2D<float3> outColor : register(u0);

[numthreads(8, 8, 1)]
void CSCockpit(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    outColor.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    uint s = (uint)(normalStencil.Load(int3(id.xy, 0)).w * 65535.0) >> 8;
    if ((s & bits) == bits) outColor[id.xy] = taaColor.Load(int3(id.xy, 0)).rgb;
}

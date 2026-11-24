// DLAA (dlaa.h): DLSS's bias-current-color mask from the game's stencil (normal/stencil .w * 65535 >> 8, the high byte
// is the pixel's class). The pixels whose high byte has all of `maskBits` (0x88 displays and mirrors, 0x80 the whole
// player's car) get `biasValue` in the mask (DLSS leans on the current frame there), all others 0. The mask has the
// render size; the stencil may be larger (a view in the corner of its slice), its corner is read. The stencil comes
// through our own one-slice array view (dlaa.cpp), so a slice of an array (focus's scene clones with MVR) reads right.

cbuffer Params : register(b0) {
    float biasValue;
    uint maskBits;
    float2 pad;
};

Texture2DArray<float4> maskStencil : register(t2);
RWTexture2D<unorm float> mask : register(u0);

[numthreads(8, 8, 1)]
void CSMask(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    mask.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    uint s = (uint)(maskStencil.Load(int4(id.xy, 0, 0)).w * 65535.0) >> 8;
    mask[id.xy] = (s & maskBits) == maskBits ? biasValue : 0.0;
}

Texture2D<float4> normalStencil : register(t0);

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

// Displays after DLSS ([dlaa] cockpit_taa; DLSS SR, and DLAA with display_pass; S2.4): DLSS (preset K) ignores the
// bias-current-color mask and keeps old digits in its history (trails). After DLSS every output pixel is copied from its result; the pixels whose stencil
// class (read at the render size, maskStencil) has all of `dispBits` instead take the current frame alone, bilinear at
// the pixel's position without the jitter: no history, no trails (dlaa-plan S2.4 A2, run 2). Clamping DLSS's result to
// the current 3 x 3 neighbourhood was not enough: on 2-3 px digit strokes it holds both colours.
// The current frame has no anti-aliasing and the stencil outline is a hard pixel step, so the outline of a display or
// mirror would show stair steps. `edge` render pixels along the outline keep DLSS's anti-aliased result instead, and
// the next render pixel blends into the current frame (edge 0: the hard switch at the outline).
// The output is a second texture: the cs_5_0 typed UAV cannot be read (DLSS's R11G11B10 result).
cbuffer DisplayParams : register(b2) {
    uint dispBits;
    uint edge;          // render pixels along the outline that keep DLSS's result (0..2)
    float2 jitter;      // DLSS's jitter offset in render pixels
    float2 toRender;    // render size / output size
    uint2 renderSize;
};
Texture2D<float3> dlssColor : register(t3);
Texture2D<float3> curColor : register(t4);
RWTexture2D<float3> dispOut : register(u1);

float3 CurAt(int2 p) { return curColor.Load(int3(clamp(p, int2(0, 0), int2(renderSize) - 1), 0)); }

bool IsDisplay(int2 p) {
    p = clamp(p, int2(0, 0), int2(renderSize) - 1);
    const uint s = (uint)(maskStencil.Load(int4(p, 0, 0)).w * 65535.0) >> 8;
    return (s & dispBits) == dispBits;
}

// 1 when the render texel and every texel within `edge` of it are displays: away from the outline.
float Inner(int2 p) {
    const int r = (int)edge;
    for (int y = -r; y <= r; ++y)
        for (int x = -r; x <= r; ++x)
            if (!IsDisplay(p + int2(x, y))) return 0.0;
    return 1.0;
}

[numthreads(8, 8, 1)]
void CSDisplays(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    dispOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float2 rp = (float2(id.xy) + 0.5) * toRender;  // the pixel's centre in render pixels
    const int2 ri = min(int2(rp), int2(renderSize) - 1);
    const float3 dlss = dlssColor.Load(int3(id.xy, 0));
    if (!IsDisplay(ri)) {
        dispOut[id.xy] = dlss;
        return;
    }
    // Weight of the current frame: the inner texels bilinearly at the pixel's centre, so it ramps over one render pixel.
    float k = 1.0;
    if (edge > 0) {
        const float2 q = rp - 0.5;
        const int2 a = int2(floor(q));
        const float2 g = q - float2(a);
        k = lerp(lerp(Inner(a), Inner(a + int2(1, 0)), g.x), lerp(Inner(a + int2(0, 1)), Inner(a + int2(1, 1)), g.x), g.y);
        if (k <= 0.0) {
            dispOut[id.xy] = dlss;
            return;
        }
    }
    const float2 p = rp - jitter - 0.5;  // the current frame's texel grid, the scene point of this pixel
    const int2 b = int2(floor(p));
    const float2 f = p - float2(b);
    const float3 cur =
        lerp(lerp(CurAt(b), CurAt(b + int2(1, 0)), f.x), lerp(CurAt(b + int2(0, 1)), CurAt(b + int2(1, 1)), f.x), f.y);
    dispOut[id.xy] = lerp(dlss, cur, k);
}

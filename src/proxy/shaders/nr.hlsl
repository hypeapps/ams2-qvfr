// NR bridge (nr_bridge.h): the DLAA result (R11G11B10_FLOAT) to and from the half-float textures shared with the
// D3D12 device, and the game's motion vectors in the model's units.
// With inputScale 1 and tonemap 0 the round trip changes no bit: every R11G11B10 value is exact in half floats (same
// exponent range, more mantissa bits). With tonemap 1 the model sees color * inputScale through a max-channel Reinhard
// (c / (1 + max(c)), into [0, 1)), with tonemap 2 also a 2.2 gamma (like a finished display image); CSUnpack applies the
// exact inverse.

cbuffer Params : register(b0) {
    float inputScale;  // color into the model = HDR color * inputScale
    uint tonemap;      // 1: max-channel Reinhard around the model; 2: Reinhard and 2.2 gamma
    float mvScale;     // motion: the game's uv now - uv before * -mvScale * size (0 = zero vectors)
    uint transfer;     // 0: the exact inverse of the model's output; 1: the original HDR * (output / input) per channel
    uint depthMode;    // CSDepth: 1 the game's linear depth as it is, 2 reversed device-like depth depthK / linear
    float depthK;      // near / far of the reversed depth (2)
    uint debug;        // CSUnpack: 1 the model's edit instead of the image (needs t1)
    float pad;         // CSUnpack: strength, the model's edit x this (1 = as the model made it)
    // CSMotion, CSDepth, the region shaders: the region of the slot they use (the whole slot, or one image in the
    // atlas of all images that go through the model in one evaluation).
    uint offsetX;
    uint offsetY;
    uint regionW;
    uint regionH;
    // CSMotion, CSDepth, CSPackRegion, CSUpEdit: source pixels per region pixel in each axis (an image at a lower
    // resolution in the atlas: 2 = half) and where the region starts in the source (only a centre part of it in the
    // atlas)
    uint srcStep;
    uint srcX;
    uint srcY;
    uint fade;  // CSUpEdit: with a centre part, the model's edit fades out over this many source pixels to its edge
};

// transfer 1: the model's input (t1) and the original HDR image (t2). The ratio is taken in linear display values
// (after the gamma, before Reinhard), so a small edit in the highlights stays small in HDR; inverting Reinhard there
// turned a 2% dimming of the sky into 55% (dlaa-plan D7, test 2).
Texture2D<float4> modelIn : register(t1);
Texture2D<float4> original : register(t2);

Texture2D<float4> src : register(t0);
RWTexture2D<float4> shared4 : register(u0);

[numthreads(8, 8, 1)]
void CSPack(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    shared4.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    float3 c = src.Load(int3(id.xy, 0)).rgb;
    if (inputScale != 1.0) c *= inputScale;
    if (tonemap) c /= 1.0 + max(c.r, max(c.g, c.b));
    if (tonemap >= 2) c = pow(max(c, 0.0), 1.0 / 2.2);
    shared4[id.xy] = float4(c, 1.0);
}

// The model's output (t0) back into the image's units. `strength` (pad slot) above 1 multiplies the model's edit in its
// units first (DXL's "Self Layers"; needs t1).
// The display-like model units back into the image's (the inverse of CSPack's input_scale and tonemap).
float3 FromModel(float3 v) {
    if (tonemap >= 2) v = pow(v, 2.2);
    if (tonemap) v /= max(1.0 - max(v.r, max(v.g, v.b)), 1.0 / 65504.0);
    if (inputScale != 1.0) v /= inputScale;
    return v;
}

// The model's output at `m` of `outT` (its input at the same place in t1); `o`: the pixel's place in `original` (t2).
float3 Unpacked(Texture2D<float4> outT, uint2 m, uint2 o) {
    float3 c = max(outT.Load(int3(m, 0)).rgb, 0.0);
    const float strength = pad;
    float3 i0 = 0;
    [branch] if (debug == 1 || strength != 1.0 || transfer == 1) i0 = max(modelIn.Load(int3(m, 0)).rgb, 0.0);
    float3 result;
    if (debug == 1) {
        // The edit as a grey-centred picture in the model's units, made into the image's units (so the game's tonemapper
        // shows grey as mid grey): grey = unchanged, lighter = the model brightened, darker = it darkened.
        result = FromModel(clamp(0.5 + (c - i0) * 20.0, 0.0, 0.97));
    } else {
        if (strength != 1.0) {
            c = max(i0 + (c - i0) * strength, 0.0);
            if (tonemap) c = min(c, 0.999);  // Reinhard's inverse needs < 1
        }
        if (transfer == 1) {
            if (all(c == i0)) {
                result = original.Load(int3(o, 0)).rgb;  // untouched by the model: DLAA's pixel bit for bit
            } else {
                float3 cl = c, il = i0;
                if (tonemap >= 2) {
                    cl = pow(cl, 2.2);
                    il = pow(il, 2.2);
                }
                result = original.Load(int3(o, 0)).rgb * clamp((cl + 1e-4) / (il + 1e-4), 0.125, 8.0);
            }
        } else {
            result = FromModel(c);
        }
    }
    return result;
}

RWTexture2D<float3> color : register(u0);

[numthreads(8, 8, 1)]
void CSUnpack(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    color.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    color[id.xy] = Unpacked(src, id.xy, id.xy);
}

// The same into an image with alpha (the final focus image, R8G8B8A8): its alpha comes from the copy of the image (t2).
RWTexture2D<float4> color4 : register(u0);

[numthreads(8, 8, 1)]
void CSUnpack4(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    color4.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    color4[id.xy] = float4(Unpacked(src, id.xy, id.xy), original.Load(int3(id.xy, 0)).a);
}

// The game's TAA depth is linear, 0 at the camera to 1 at the far plane (the sky is >= 0.99999, docs/ams2-renderer.md).
RWTexture2D<float> depthOut : register(u0);

[numthreads(8, 8, 1)]
void CSDepth(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    const float z = src.Load(int3(uint2(srcX, srcY) + id.xy * srcStep, 0)).r;
    depthOut[id.xy + uint2(offsetX, offsetY)] = depthMode == 2 ? saturate(depthK / max(z, 1e-7)) : saturate(z);
}

// The game stores uv now - uv before (as DLAA reads it); the model, like DLSS, wants the offset from the current pixel
// to its previous position in pixels (of the region: the source's size / srcStep). In the atlas the previous position
// is held inside the pixel's own region, so the model's history never reaches into the neighbouring one.
RWTexture2D<float2> motion : register(u0);

[numthreads(8, 8, 1)]
void CSMotion(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    uint w, h, sw, sh;
    motion.GetDimensions(w, h);
    src.GetDimensions(sw, sh);
    float2 m = 0;
    if (mvScale != 0) {
        m = src.Load(int3(uint2(srcX, srcY) + id.xy * srcStep, 0)).xy * -mvScale * float2(sw, sh) / (float)srcStep;
        if (regionW != w || regionH != h)
            m = clamp(float2(id.xy) + m, 0.0, float2(regionW, regionH) - 1.0) - float2(id.xy);
    }
    motion[id.xy + uint2(offsetX, offsetY)] = m;
}

// The atlas of final images (display-encoded, UNORM views): each image goes straight into its region of the model's
// input and comes straight back out of the model's output, without a copy of the atlas. An outer image may go in at a
// lower resolution or only its centre part (the periphery needs the model's tone and color, not its finest detail, and
// its far edge none of it).
// CSPackRegion: the image (t0) from (srcX, srcY), averaged over srcStep x srcStep pixels, into the regionW x regionH
// region at (offsetX, offsetY) of the model's input (u0), through input_scale and tonemap like CSPack.
[numthreads(8, 8, 1)]
void CSPackRegion(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    uint w, h;
    src.GetDimensions(w, h);
    const uint2 base = uint2(srcX, srcY) + id.xy * srcStep;
    float3 c = 0;
    for (uint y = 0; y < srcStep; ++y)
        for (uint x = 0; x < srcStep; ++x) c += src.Load(int3(min(base + uint2(x, y), uint2(w - 1, h - 1)), 0)).rgb;
    c /= (float)(srcStep * srcStep);
    if (inputScale != 1.0) c *= inputScale;
    if (tonemap) c /= 1.0 + max(c.r, max(c.g, c.b));
    if (tonemap >= 2) c = pow(max(c, 0.0), 1.0 / 2.2);
    shared4[id.xy + uint2(offsetX, offsetY)] = float4(c, 1.0);
}

// The model's output (its region at offsetX/offsetY) for the region shaders.
Texture2D<float4> modelOut : register(t3);

// CSUnpackRegion (an image at full resolution, whole): the region of the model's output (t3) back into the image's
// units, into the image's copy (u0, regionW x regionH) with the image's (t2) alpha.
[numthreads(8, 8, 1)]
void CSUnpackRegion(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    color4[id.xy] = float4(Unpacked(modelOut, id.xy + uint2(offsetX, offsetY), id.xy), original.Load(int3(id.xy, 0)).a);
}

// CSUpEdit (an image at a lower resolution or only its centre part): the image (t0) plus the model's edit of its
// region, bilinearly enlarged: the region of the model's output (t3) minus its input (t1), both back in the image's
// units (transfer 0 only: there is no original pixel for a region pixel). With a centre part (fade > 0) the edit fades
// out smoothly over `fade` source pixels inside the part's edge and is zero outside it. Without an edit the image stays
// bit for bit; alpha stays.
float3 RegionEdit(int2 p) {
    const uint2 m = uint2(p) + uint2(offsetX, offsetY);
    return Unpacked(modelOut, m, m) - FromModel(max(modelIn.Load(int3(m, 0)).rgb, 0.0));
}

[numthreads(8, 8, 1)]
void CSUpEdit(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    color4.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float4 o = src.Load(int3(id.xy, 0));
    const float2 local = float2(id.xy) + 0.5 - float2(srcX, srcY);  // pixel centre in the part, source pixels
    float weight = 1;
    if (fade > 0) {
        const float2 size = float2(regionW, regionH) * (float)srcStep;
        const float2 inside = min(local, size - local);
        weight = smoothstep(0.0, 1.0, saturate(min(inside.x, inside.y) / (float)fade));
    }
    if (weight <= 0) {
        color4[id.xy] = o;
        return;
    }
    const float2 last = float2(regionW - 1, regionH - 1);
    const float2 q = clamp(local / (float)srcStep - 0.5, 0.0, last);
    const int2 a = int2(q);
    const int2 b = int2(min(float2(a + 1), last));
    const float2 f = q - float2(a);
    const float3 e = lerp(lerp(RegionEdit(a), RegionEdit(int2(b.x, a.y)), f.x),
                          lerp(RegionEdit(int2(a.x, b.y)), RegionEdit(b), f.x), f.y);
    color4[id.xy] = float4(saturate(o.rgb + e * weight), o.a);
}

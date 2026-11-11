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
    // CSMotion, CSDepth: the region of the output they write (the whole output, or one image in the atlas of all images
    // that go through the model in one evaluation); the source is the region's size.
    uint offsetX;
    uint offsetY;
    uint regionW;
    uint regionH;
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

float3 Unpacked(uint2 id) {
    float3 c = max(src.Load(int3(id, 0)).rgb, 0.0);
    const float3 i0 = max(modelIn.Load(int3(id, 0)).rgb, 0.0);  // read only where bound (debug, strength, ratio)
    const float strength = pad;
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
                result = original.Load(int3(id, 0)).rgb;  // untouched by the model: DLAA's pixel bit for bit
            } else {
                float3 cl = c, il = i0;
                if (tonemap >= 2) {
                    cl = pow(cl, 2.2);
                    il = pow(il, 2.2);
                }
                result = original.Load(int3(id, 0)).rgb * clamp((cl + 1e-4) / (il + 1e-4), 0.125, 8.0);
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
    color[id.xy] = Unpacked(id.xy);
}

// The same into an image with alpha (the final focus image, R8G8B8A8): its alpha comes from the copy of the image (t2).
RWTexture2D<float4> color4 : register(u0);

[numthreads(8, 8, 1)]
void CSUnpack4(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    color4.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    color4[id.xy] = float4(Unpacked(id.xy), original.Load(int3(id.xy, 0)).a);
}

// The game's TAA depth is linear, 0 at the camera to 1 at the far plane (the sky is >= 0.99999, docs/ams2-renderer.md).
RWTexture2D<float> depthOut : register(u0);

[numthreads(8, 8, 1)]
void CSDepth(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    const float z = src.Load(int3(id.xy, 0)).r;
    depthOut[id.xy + uint2(offsetX, offsetY)] = depthMode == 2 ? saturate(depthK / max(z, 1e-7)) : saturate(z);
}

// The game stores uv now - uv before (as DLAA reads it); the model, like DLSS, wants the offset from the current pixel
// to its previous position in pixels. In the atlas the previous position is held inside the pixel's own image, so the
// model's history never reaches into the neighbouring one.
RWTexture2D<float2> motion : register(u0);

[numthreads(8, 8, 1)]
void CSMotion(uint3 id : SV_DispatchThreadID) {
    if (id.x >= regionW || id.y >= regionH) return;
    uint w, h;
    motion.GetDimensions(w, h);
    const float2 size = float2(regionW, regionH);
    float2 m = 0;
    if (mvScale != 0) {
        m = src.Load(int3(id.xy, 0)).xy * -mvScale * size;
        if (regionW != w || regionH != h) m = clamp(float2(id.xy) + m, 0.0, size - 1.0) - float2(id.xy);
    }
    motion[id.xy + uint2(offsetX, offsetY)] = m;
}

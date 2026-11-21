// Own ambient occlusion in place of the game's SSAO and one bounce of indirect light (ssgi_ao.h). The algorithm:
// ground-truth ambient occlusion with a visibility bitmask (O. Therrien, Y. Levesque, G. Gilet, "Screen Space Indirect
// Lighting with Visibility Bitmask", 2023): per slice, every depth sample marks the sectors of the hemisphere it covers,
// from its front face to a back face `thickness` behind it, instead of raising a single horizon. The sectors a sample
// covers first (none nearer covered them) bring its light: the view's image of the previous frame at that point. The
// code is written here from the paper.
//
// Inputs are the game's own: the G-buffer's linear depth (R32_FLOAT, view depth / far, 1 = sky), view-space normals
// (R16G16B16A16_UNORM, xy spheremap-encoded, w the flags), the deferred-ambient weight (R8G8B8A8_UNORM, rgb: 1 the
// world, 0 vehicles) and the constant windows the overlay draw (PS 15564) binds, so the view's projection, camera and
// size need no CPU copy. View space: x right, y up, z forward (visible surfaces have n.z < 0); world = viewI applied to
// it as the game's shaders do (docs/ssgi/ssgi-plan-g0.md "Ustalenia G0").

// The game's constant windows at the overlay draw (COMMON_BUFFER2 and that shader's _GLOBAL_, offsets from the
// capture).
cbuffer GameCommon : register(b12) {
    float4 gcPad0[4];
    float4 viewI[4];                 // 64: world from view, rows: the view's axes and its origin in the world
    float4 gcPad1[16];
    float4 screenSpaceDimensions;    // 384: xy = the view's share of its targets (UV scale)
    float4 frustumDimensions;        // 400
    float4 gcPad2[3];
    float4 viewportPixelDimensions;  // 464: zw = the targets' size in pixels
};
cbuffer GameGlobal : register(b0) {
    float4 cProjectionTanAngles;  // 0: tangents left, right, up, down (all positive)
    float4 ggPad[19];
    float frustumDepth;           // 320: linear depth 1 in view units
};
cbuffer Params : register(b1) {
    float radius;       // view units (metres)
    float thickness;    // view units: how far behind its front face a sample still occludes
    float power;        // the AO to this power
    float maxRadiusPx;  // screen radius limit in pixels
    uint sliceCount;
    uint stepCount;     // per side of a slice
    float blurDepth;    // the blur's depth tolerance, relative
    uint giOn;          // 1: the bounce is computed (history and the previous camera valid)
    uint camCur;        // this view's camera entry of this frame (written here)
    uint camPrev;       // the entry of the previous frame (read when giOn)
    float giClamp;      // the history's luminance at most this (0: no limit)
    float giVehicle;    // the bounce on vehicles (weight 0) times this
};

Texture2D<float> depthTex : register(t0);
Texture2D<float4> normalTex : register(t1);
Texture2D<float4> rawIn : register(t2);
Texture2DArray<float4> weightTex : register(t4);  // one slice (the overlay's target may be an array slice)
Texture2D<float4> histTex : register(t5);
RWStructuredBuffer<float4> cams : register(u1);  // per entry: viewI rows 0..3, tangents
RWTexture2D<float4> rawOut : register(u3);       // rgb the bounce, a the AO
RWTexture2D<float> aoOut : register(u0);
RWTexture2D<float4> giOut : register(u2);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);

static const float kPi = 3.14159265;
static const float kHalfPi = 1.57079633;
static const uint kCamSize = 5;

// The view's region in the AO texture (the same size as the G-buffer): the game samples it at uv * ssd.xy.
float2 Region() {
    float w, h;
    depthTex.GetDimensions(w, h);
    return screenSpaceDimensions.xy * float2(w, h);
}

// The G-buffer pixel of a view UV, as the game's shaders load it.
int2 GbufPixel(float2 uv) {
    float2 p = floor(uv * screenSpaceDimensions.xy * viewportPixelDimensions.zw);
    return (int2)clamp(p, 0, viewportPixelDimensions.zw - 1);
}

float Depth01(float2 uv) { return depthTex.Load(int3(GbufPixel(uv), 0)); }
bool IsSky(float d) { return d <= 0 || d >= 0.99999; }

// The depth at an exact view UV: bilinear in 1 / depth between the four G-buffer texels around it (exact on a plane, so
// a sample stays on its slice's line); the nearest texel where they straddle a depth edge or the sky.
float Depth01At(float2 uv) {
    float w, h;
    depthTex.GetDimensions(w, h);
    const float2 g = uv * screenSpaceDimensions.xy * viewportPixelDimensions.zw - 0.5;
    const float2 i = floor(g), f = g - i;
    const float4 d = depthTex.GatherRed(pointClamp, (i + 1) / float2(w, h));  // w (i), z (i + x), x (i + y), y
    const float lo = min(min(d.x, d.y), min(d.z, d.w)), hi = max(max(d.x, d.y), max(d.z, d.w));
    const bool smooth = lo > 0 && hi < 0.99999 && hi <= lo * 1.1;
    const float4 r = 1 / max(d, 1e-9);
    return smooth ? 1 / lerp(lerp(r.w, r.z, f.x), lerp(r.x, r.y, f.x), f.y) : Depth01(uv);
}

float3 ViewPos(float2 uv, float d) {
    const float4 t = cProjectionTanAngles;
    return d * frustumDepth * float3(lerp(-t.x, t.y, uv.x), lerp(t.z, -t.w, uv.y), 1);
}

float3 DecodeNormal(float2 enc) {
    const float2 e = enc * 2 - 1;
    const float w = 1 - dot(e, e);
    if (w < 0) return float3(0, 0, -1);
    return float3(e * (2 * sqrt(w)), 2 * w - 1);
}

float3 ViewNormal(float2 uv) { return DecodeNormal(normalTex.Load(int3(GbufPixel(uv), 0)).xy); }

// Interleaved gradient noise (Jimenez 2014): fixed per pixel, the same in both eyes and in every frame.
float Noise(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

// The sectors between a and b (0..1 of the slice's hemisphere) as bits, both ends rounded to the nearest sector
// boundary: a sample on the pixel's own tangent plane (a at 1 or b at 0 up to rounding errors) marks nothing.
uint Sectors(float a, float b) {
    const uint s = (uint)round(saturate(a) * 32), e = (uint)round(saturate(b) * 32);
    if (e <= s) return 0;
    const uint count = e - s;  // s < 32 here
    const uint bits = count >= 32 ? 0xFFFFFFFFu : ((1u << count) - 1u);
    return bits << s;
}

// The previous frame's camera of this view (cams[camPrev]).
struct Camera {
    float3 axis[3];
    float3 origin;
    float4 tan;
    bool valid;
};

Camera PrevCamera() {
    Camera c;
    const uint b = camPrev * kCamSize;
    c.valid = true;
    [unroll] for (int i = 0; i < 3; ++i) {
        const float3 a = cams[b + i].xyz;
        const float l = dot(a, a);
        c.valid = c.valid && l > 1e-12;
        c.axis[i] = a / max(l, 1e-12);  // a uniform scale of viewI divides out
    }
    c.origin = cams[b + 3].xyz;
    c.tan = cams[b + 4];
    c.valid = c.valid && c.tan.x + c.tan.y > 0 && c.tan.z + c.tan.w > 0;
    return c;
}

// A view point of this frame in the previous frame's view UV (outside 0..1: not on that image).
float2 Reproject(float3 p, Camera c) {
    const float3 w = p.x * viewI[0].xyz + p.y * viewI[1].xyz + p.z * viewI[2].xyz + viewI[3].xyz - c.origin;
    const float3 q = float3(dot(w, c.axis[0]), dot(w, c.axis[1]), dot(w, c.axis[2]));
    if (q.z <= 1e-4) return float2(-1, -1);
    return float2((q.x / q.z + c.tan.x) / (c.tan.x + c.tan.y), (c.tan.z - q.y / q.z) / (c.tan.z + c.tan.w));
}

// The light leaving a sample point towards this pixel: the view's image of the previous frame there. The world stays in
// place, so its points go through both cameras; vehicles (the deferred-ambient weight 0: the player's own car, which
// moves with the camera) are read at the same place of the screen.
float3 Radiance(float2 suv, float3 sp, Camera c) {
    float2 huv = suv;
    if (weightTex.Load(int4(GbufPixel(suv), 0, 0)).r >= 0.25) {
        huv = Reproject(sp, c);
        if (any(huv < 0) || any(huv > 1)) return 0;
    }
    float3 l = max(histTex.SampleLevel(linearClamp, huv, 0).rgb, 0);
    if (giClamp > 0) {
        const float y = dot(l, float3(0.2126, 0.7152, 0.0722));
        if (y > giClamp) l *= giClamp / y;
    }
    return l;
}

[numthreads(8, 8, 1)]
void CSAo(uint3 id : SV_DispatchThreadID) {
    if (all(id.xy == 0)) {  // this frame's camera, for the next frame's reprojection
        const uint b = camCur * kCamSize;
        [unroll] for (int i = 0; i < 4; ++i) cams[b + i] = viewI[i];
        cams[b + 4] = cProjectionTanAngles;
    }
    const float2 region = Region();
    if (any((float2)id.xy >= region)) return;
    const float2 px = id.xy + 0.5;
    const float2 uv = px / region;
    const float d = Depth01(uv);
    if (IsSky(d)) {
        rawOut[id.xy] = float4(0, 0, 0, 1);
        return;
    }
    const float3 P = ViewPos(uv, d);
    const float4 nraw = normalTex.Load(int3(GbufPixel(uv), 0));
    const float3 N = DecodeNormal(nraw.xy);
    const float3 V = normalize(-P);
    const float4 t = cProjectionTanAngles;
    const float pxPerUnit = region.x / ((t.x + t.y) * P.z);
    const float radiusPx = min(radius * pxPerUnit, maxRadiusPx);
    if (radiusPx < 1.5) {
        rawOut[id.xy] = float4(0, 0, 0, 1);
        return;
    }
    // The bounce: not on pixels without lights (flag 32 in the normals' w, high byte).
    const uint flags = ((uint)(nraw.w * 65535 + 0.5)) >> 8;
    const Camera cam = PrevCamera();
    const bool gi = giOn != 0 && (flags & 32) == 0 && cam.valid;
    const float n0 = Noise(px), n1 = Noise(px + float2(23.7, 51.3));
    float visible = 0;
    float3 light = 0;
    [loop] for (uint s = 0; s < sliceCount; ++s) {
        const float phi = (s + n0) * kPi / sliceCount;
        const float2 omega = float2(cos(phi), sin(phi));  // screen pixels, y down
        // The same direction in view space (the slice plane holds the eye, P and the screen line): a pixel spans
        // tangentSpan / region per axis, which differs between the axes.
        const float3 dir = float3(omega.x * (t.x + t.y) / region.x, -omega.y * (t.z + t.w) / region.y, 0);
        const float3 ortho = normalize(dir - dot(dir, V) * V);
        const float3 axis = cross(ortho, V);
        const float3 projN = N - axis * dot(N, axis);
        const float lenN = max(length(projN), 1e-4);
        const float n = (dot(ortho, projN) < 0 ? -1 : 1) * acos(saturate(dot(projN, V) / lenN));
        uint mask = 0;
        [unroll] for (int side = -1; side <= 1; side += 2) {
            [loop] for (uint j = 0; j < stepCount; ++j) {
                float f = (j + n1) / stepCount;
                const float dist = max(f * f * radiusPx, j + 1.0);
                const float2 sp = px + side * omega * dist;
                if (any(sp < 0) || any(sp >= region)) break;
                const float2 suv = sp / region;
                const float sd = Depth01At(suv);
                if (IsSky(sd)) continue;
                const float3 S = ViewPos(suv, sd);
                const float3 D = S - P;
                const float len = length(D);
                if (len > radius || len < 1e-6) continue;
                const float front = acos(clamp(dot(D / len, V), -1, 1));
                const float back = acos(clamp(dot(normalize(D - V * thickness), V), -1, 1));
                // Signed angles from V, positive towards `ortho`: side +1 covers [front, back], side -1 the mirror.
                const float2 th = side > 0 ? float2(min(front, back), max(front, back))
                                           : float2(-max(front, back), -min(front, back));
                const float2 u = (th - n + kHalfPi) / kPi;
                const uint bits = Sectors(u.x, u.y);
                const uint fresh = bits & ~mask;
                mask |= bits;
                if (gi && fresh != 0) {
                    // Cosine-weighted share of the slice's half circle (the sectors are even in angle from the
                    // projected normal: 32 of them at cos 1 make pi / 2 * 2 / pi = 1), the sample's front side only:
                    // 0 for a sample whose surface runs along the ray (the pixel's own curved surface, e.g. a car's
                    // roof, would light itself), full from 15 degrees on.
                    const float a = 0.5 * (th.x + th.y) - n;
                    const float facing = saturate(dot(ViewNormal(suv), -D / len) * 4);
                    light += Radiance(suv, S, cam) * (countbits(fresh) * (kPi / 64) * max(cos(a), 0) * facing);
                }
            }
        }
        visible += 1 - countbits(mask) / 32.0;
    }
    const float vehicle = lerp(giVehicle, 1, saturate(weightTex.Load(int4(GbufPixel(uv), 0, 0)).r));
    rawOut[id.xy] = float4(gi ? light / sliceCount * vehicle : 0, pow(saturate(visible / sliceCount), power));
}

// 5 x 5 blur that keeps depth edges: removes the per-pixel noise pattern of the slice rotation, from the AO and the
// bounce alike.
[numthreads(8, 8, 1)]
void CSBlur(uint3 id : SV_DispatchThreadID) {
    const float2 region = Region();
    if (any((float2)id.xy >= region)) return;
    const float dc = Depth01((id.xy + 0.5) / region);
    if (IsSky(dc)) {
        aoOut[id.xy] = 1;
        if (giOn) giOut[id.xy] = 0;
        return;
    }
    const int2 last = (int2)ceil(region) - 1;
    float4 sum = 0;
    float wsum = 0;
    [unroll] for (int y = -2; y <= 2; ++y) {
        [unroll] for (int x = -2; x <= 2; ++x) {
            const int2 p = clamp((int2)id.xy + int2(x, y), 0, last);
            const float ds = Depth01((p + 0.5) / region);
            const float w = IsSky(ds) ? 0 : saturate(1 - abs(ds - dc) / (dc * blurDepth));
            sum += rawIn.Load(int3(p, 0)) * w;
            wsum += w;
        }
    }
    const float4 r = wsum > 0 ? sum / wsum : rawIn.Load(int3(id.xy, 0));
    aoOut[id.xy] = r.a;
    if (giOn) giOut[id.xy] = float4(r.rgb, 0);
}

// The history: the view's final image (DLSS output or the MSAA resolve, a rectangle of the source) at the history's
// size, four bilinear taps per texel.
cbuffer HistParams : register(b3) {
    float2 srcOrigin;  // pixels
    float2 srcSize;    // pixels
    float2 srcInvTex;  // 1 / the source texture's size
    float2 histPad;
};
Texture2D<float4> histSrc : register(t7);
RWTexture2D<float4> histOut : register(u4);

[numthreads(8, 8, 1)]
void CSHistory(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    histOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float2 size = float2(w, h);
    const float2 p = srcOrigin + (id.xy + 0.5) / size * srcSize;
    const float2 o = 0.25 * srcSize / size;
    float3 c = 0;
    [unroll] for (int i = 0; i < 4; ++i) {
        const float2 q = p + o * float2((i & 1) ? 1 : -1, (i & 2) ? 1 : -1);
        c += histSrc.SampleLevel(linearClamp, q * srcInvTex, 0).rgb;
    }
    c *= 0.25;
    histOut[id.xy] = float4(all(c == c) ? max(c, 0) : 0, 1);  // NaN to 0
}

// Probe (diagnostics in the log): a 64 x 64 grid over the view, the world and vehicles (the deferred-ambient weight
// below 0.25) apart. Per 8 x 8 group: the world's sums of the luminances of the bounce, of the diffuse light target
// before the bounce is added and of the history, the bounce's maximum; the world's samples and those where the bounce
// is more than 1 % of the light; the vehicles' sums of the bounce and of the history, their samples. Compiled with
// SSGI_PROBE only (groupshared).
#ifdef SSGI_PROBE
cbuffer ProbeParams : register(b4) {
    float2 probeRegion;  // the view's texels in the AO and light targets
    float2 probePad;
};
Texture2D<float4> probeGi : register(t8);
Texture2D<float4> probeLight : register(t9);
Texture2D<float4> probeHist : register(t10);
Texture2D<float4> probeWeight : register(t11);
RWStructuredBuffer<float4> probeOut : register(u5);  // 3 per group
groupshared float4 gsA[64];
groupshared float4 gsB[64];
groupshared float4 gsC[64];

float Luma(float3 c) { return dot(max(c, 0), float3(0.2126, 0.7152, 0.0722)); }

[numthreads(8, 8, 1)]
void CSProbe(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 g : SV_GroupID) {
    const float2 uv = (id.xy + 0.5) / 64;
    const int3 t = int3(clamp(uv * probeRegion, 0, probeRegion - 1), 0);
    const float yb = Luma(probeGi.Load(t).rgb), yl = Luma(probeLight.Load(t).rgb);
    const float yh = Luma(probeHist.SampleLevel(linearClamp, uv, 0).rgb);
    const bool vehicle = probeWeight.Load(t).r < 0.25;
    gsA[gi] = vehicle ? float4(0, 0, 0, yb) : float4(yb, yl, yh, yb);
    gsB[gi] = vehicle ? 0 : float4(1, yb > 0.01 * yl && yb > 0 ? 1 : 0, 0, 0);
    gsC[gi] = vehicle ? float4(yb, yh, 1, 0) : 0;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 32; s > 0; s >>= 1) {
        if (gi < s) {
            const float4 a = gsA[gi], b = gsA[gi + s];
            gsA[gi] = float4(a.xyz + b.xyz, max(a.w, b.w));
            gsB[gi] += gsB[gi + s];
            gsC[gi] += gsC[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        const uint k = (g.y * 8 + g.x) * 3;
        probeOut[k] = gsA[0];
        probeOut[k + 1] = gsB[0];
        probeOut[k + 2] = gsC[0];
    }
}
#endif

// Full-screen draws over a view: the AO preview (ao=3), the bounce into the diffuse light target and its preview.
cbuffer PreviewParams : register(b2) {
    float2 vpOrigin;
    float2 vpInvSize;
    float2 aoRegion;  // the AO's texels of the view (the overlay draw's viewport)
    float level;
    float alpha;
};
Texture2D<float> aoTex : register(t3);
Texture2D<float4> giTex : register(t6);

float4 VSFull(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

int2 RegionTexel(float2 p) {
    const float2 uv = (p - vpOrigin) * vpInvSize;
    return (int2)clamp(uv * aoRegion, 0, aoRegion - 1);
}

float4 PSPreview(float4 p : SV_Position) : SV_Target {
    return float4(aoTex.Load(int3(RegionTexel(p.xy), 0)).xxx * level, 1);
}

float4 PSGi(float4 p : SV_Position) : SV_Target {
    return float4(giTex.Load(int3(RegionTexel(p.xy), 0)).rgb * level, alpha);
}

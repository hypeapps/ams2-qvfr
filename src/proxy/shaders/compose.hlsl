// Quad views composition (M4.4, M7): one full-screen triangle per eye over the 1:1 grid. Outer covers the whole grid and
// is scaled bilinearly; focus fills its window (pixel for pixel when its texture has the window's size, bilinearly
// otherwise) and is blended into outer at the window edge, inside the hidden margin. Views are _SRGB where the format
// allows, so filtering and blending run on linear values and untouched focus pixels come out bit for bit.
// Soft edge: the curve of the Quad-Views-Foveated layer by Matthieu Bucchianeri (ProjectionPS.hlsl, MIT License,
// https://github.com/mbucchia/Quad-Views-Foveated), or the pre-M7 one (legacyEdge).
// Debug overlay (M4.5): the edges of the blend band (yellow: where the focus starts to show, green: from where it is
// fully sharp) and the gaze point (red ring, blue when frozen).

cbuffer Params : register(b0) {
    float2 gridSize;     // output size in pixels (the 1:1 grid)
    float2 focusOrigin;  // top left corner of the focus window, grid pixels
    float2 focusSize;    // focus window size, grid pixels
    float margin;        // hidden band at the focus edge (guard band), grid pixels
    float feather;       // legacyEdge: blend band inside the margin, grid pixels (> 0)
    float2 gaze;         // gaze point, grid pixels
    float gazeMode;      // 0 = no gaze, 1 = live, 2 = frozen
    float overlay;       // 1 = draw the debug overlay
    float smoothen;      // mbucchia's smoothen_focus_view_edges (0 = hard edge)
    float legacyEdge;    // 1 = the pre-M7 curve over `feather`
    float exact;         // 1 = the focus texture has the window's size: pixels are loaded, not filtered
    float pad;
};

Texture2D outerTex : register(t0);
Texture2D focusTex : register(t1);
SamplerState linearClamp : register(s0);

float4 VSMain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

// 1 on a rectangle outline at distance d (per axis) from the focus window edge (lines `width` pixels wide), else 0.
float Outline(float2 q, float2 d, float width) {
    float2 lo = d, hi = focusSize - d;
    bool inside = all(q >= lo - width * 0.5) && all(q <= hi + width * 0.5);
    float2 e = min(abs(q - lo), abs(q - hi));
    return inside && min(e.x, e.y) <= width * 0.5 ? 1 : 0;
}

// Focus weight at q (pixel center in window pixels).
float FocusWeight(float2 q) {
    if (legacyEdge > 0) {
        float2 e = min(q, focusSize - q);
        float w = saturate((min(e.x, e.y) - margin) / feather);
        return w * w * (3 - 2 * w);
    }
    float2 uv = (q - margin) / (focusSize - 2 * margin);  // 0..1 over the shown focus
    if (any(uv <= 0) || any(uv >= 1)) return 0;
    if (smoothen <= 0) return 1;
    float2 s = smoothstep(0, smoothen, uv) - smoothstep(1 - smoothen, 1, uv);
    return max(0.5, s.x * s.y);
}

float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 c = outerTex.SampleLevel(linearClamp, pos.xy / gridSize, 0);
    float2 q = pos.xy - focusOrigin;  // pixel center in window pixels
    float w = FocusWeight(q);
    [branch] if (w > 0) {
        float4 f = exact > 0 ? focusTex.Load(int3(q, 0)) : focusTex.SampleLevel(linearClamp, q / focusSize, 0);
        c = w < 1 ? lerp(c, f, w) : f;  // exact focus pixels where the weight is 1
    }
    [branch] if (overlay > 0) {
        const float lineW = 6;
        float2 sharp = legacyEdge > 0 ? margin + feather : margin + smoothen * (focusSize - 2 * margin);
        if (Outline(q, margin, lineW) > 0) c = float4(1, 0.85, 0, 1);
        if (Outline(q, sharp, lineW) > 0) c = float4(0, 1, 0.2, 1);
        if (gazeMode > 0) {
            float r = length(pos.xy - gaze);
            if (abs(r - 36) <= 5 || r <= 6) c = gazeMode > 1 ? float4(0.1, 0.4, 1, 1) : float4(1, 0.1, 0.1, 1);
        }
    }
    return c;
}

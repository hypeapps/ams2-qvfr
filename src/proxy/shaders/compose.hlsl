// Quad views composition (M4.4): one full-screen triangle per eye over the 1:1 grid. Outer covers the whole grid at a
// lower density and is upscaled bilinearly; focus is copied 1:1 into its window and blended into outer over the
// feather band inside the hidden margin. Views are _SRGB where the format allows, so filtering and blending run on
// linear values and untouched focus pixels come out bit for bit.
// Debug overlay (M4.5): the edges of the blend band (yellow: where the focus starts to show, green: from where it is
// fully sharp) and the gaze point (red ring, blue when frozen).

cbuffer Params : register(b0) {
    float2 gridSize;     // output size in pixels (the 1:1 grid)
    float2 focusOrigin;  // top left corner of the focus window, grid pixels
    float2 focusSize;    // focus window size, grid pixels (= focus texture size)
    float margin;        // hidden band at the focus edge (guard band), grid pixels
    float feather;       // blend band inside the margin, grid pixels (> 0)
    float2 gaze;         // gaze point, grid pixels
    float gazeMode;      // 0 = no gaze, 1 = live, 2 = frozen
    float overlay;       // 1 = draw the debug overlay
};

Texture2D outerTex : register(t0);
Texture2D focusTex : register(t1);
SamplerState linearClamp : register(s0);

float4 VSMain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

// 1 on a rectangle outline at distance d from the focus window edge (lines `width` pixels wide), else 0.
float Outline(float2 q, float d, float width) {
    float2 lo = d, hi = focusSize - d;
    bool inside = all(q >= lo - width * 0.5) && all(q <= hi + width * 0.5);
    float2 e = min(abs(q - lo), abs(q - hi));
    return inside && min(e.x, e.y) <= width * 0.5 ? 1 : 0;
}

float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 c = outerTex.SampleLevel(linearClamp, pos.xy / gridSize, 0);
    float2 q = pos.xy - focusOrigin;  // pixel center in focus pixels
    float2 e = min(q, focusSize - q);
    float w = saturate((min(e.x, e.y) - margin) / feather);
    w = w * w * (3 - 2 * w);
    [branch] if (w > 0) {
        float4 f = focusTex.Load(int3(q, 0));
        c = w < 1 ? lerp(c, f, w) : f;  // exact focus pixels inside the band
    }
    [branch] if (overlay > 0) {
        const float lineW = 6;
        if (Outline(q, margin, lineW) > 0) c = float4(1, 0.85, 0, 1);
        if (Outline(q, margin + feather, lineW) > 0) c = float4(0, 1, 0.2, 1);
        if (gazeMode > 0) {
            float r = length(pos.xy - gaze);
            if (abs(r - 36) <= 5 || r <= 6) c = gazeMode > 1 ? float4(0.1, 0.4, 1, 1) : float4(1, 0.1, 0.1, 1);
        }
    }
    return c;
}

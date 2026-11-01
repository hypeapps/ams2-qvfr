// Quad views composition (M4.4): one full-screen triangle per eye over the 1:1 grid. Outer covers the whole grid at a
// lower density and is upscaled bilinearly; focus is copied 1:1 into its window and blended into outer over the
// feather band inside the hidden margin. Views are _SRGB where the format allows, so filtering and blending run on
// linear values and untouched focus pixels come out bit for bit.

cbuffer Params : register(b0) {
    float2 gridSize;     // output size in pixels (the 1:1 grid)
    float2 focusOrigin;  // top left corner of the focus window, grid pixels
    float2 focusSize;    // focus window size, grid pixels (= focus texture size)
    float margin;        // hidden band at the focus edge (guard band), grid pixels
    float feather;       // blend band inside the margin, grid pixels (> 0)
};

Texture2D outerTex : register(t0);
Texture2D focusTex : register(t1);
SamplerState linearClamp : register(s0);

float4 VSMain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
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
    return c;
}

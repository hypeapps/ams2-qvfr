// --sun-fx: the proxy's sun effects module (sun_fx.h) on a real D3D11 device, without SteamVR and without the proxy DLL.
// One frame of four eye groups on a shared 4x MSAA HDR target, as the game renders without clones: outer L and R draw a
// "lens flare" (a rectangle, alpha 0.5, blend SrcAlpha + One) and "rays" (a constant) with registered pixel shaders,
// then an anchor pass reads the HDR target into the group's output. Focus L draws the effects (skipped, the module's
// pass adds outer's accumulation instead), focus R draws none (the pass runs at the anchor). Checks the outputs pixel by
// pixel, the module's counters and the debug layer; then a frame with the module off (the focus draws its own effects).
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "sun_fx.h"

namespace {

const char kHlsl[] = R"(
cbuffer P : register(b0) { float2 invSize; float2 pad; };
Texture2DMS<float4> hdr : register(t0);
float4 VS(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 Flare(float4 p : SV_Position) : SV_Target {
    float2 uv = p.xy * invSize;
    if (uv.x < 0.2 || uv.x > 0.6 || uv.y < 0.3 || uv.y > 0.7) discard;
    return float4(uv.x, uv.y, 0.25, 0.5);
}
float4 Rays(float4 p : SV_Position) : SV_Target { return float4(0.1, 0.05, 0.02, 1); }
float4 Anchor(float4 p : SV_Position) : SV_Target { return hdr.Load(int2(p.xy), 0); }
)";

// Car paint of the bonnet reflection as the game's shaders declare it (bonnetReflectionAndClamp of COMMON_BUFFER2 used,
// motionBlurMap_tex), and a shader with the same declarations that does not use the constant.
const char kPaintHlsl[] = R"(
cbuffer COMMON_BUFFER2 : register(b12) { float4 pad0[43]; float4 bonnetReflectionAndClamp; };
Texture2D<float4> motionBlurMap_tex : register(t1);
float4 Paint(float4 p : SV_Position) : SV_Target { return motionBlurMap_tex.Load(int3(0, 0, 0)) * bonnetReflectionAndClamp.x; }
float4 Decoy(float4 p : SV_Position) : SV_Target { return motionBlurMap_tex.Load(int3(0, 0, 0)) * pad0[0].x; }
)";

constexpr UINT kW = 64, kH = 48;
constexpr float kBase = 0.2f;
constexpr float kOrigin[2] = {0.25f, 0.3f}, kScale[2] = {0.5f, 0.4f};

float Half(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e ? std::ldexp(1.0f + m / 1024.0f, (int)e - 15) : std::ldexp(m / 1024.0f, -14);
    return s ? -v : v;
}

// What outer's effects add at the pixel center uv (flare at alpha 0.5, rays).
void Effects(float u, float v, float out[3]) {
    const bool in = u >= 0.2f && u <= 0.6f && v >= 0.3f && v <= 0.7f;
    out[0] = 0.1f + (in ? 0.5f * u : 0);
    out[1] = 0.05f + (in ? 0.5f * v : 0);
    out[2] = 0.02f + (in ? 0.5f * 0.25f : 0);
}

// Bilinear read of outer's accumulation (texel centers, clamp), as the module's sampler does.
void Bilinear(float u, float v, float out[3]) {
    const float x = std::clamp(u * kW - 0.5f, 0.0f, kW - 1.0f), y = std::clamp(v * kH - 0.5f, 0.0f, kH - 1.0f);
    const int x0 = (int)x, y0 = (int)y, x1 = std::min(x0 + 1, (int)kW - 1), y1 = std::min(y0 + 1, (int)kH - 1);
    const float fx = x - x0, fy = y - y0;
    float a[3], b[3], c[3], d[3];
    Effects((x0 + 0.5f) / kW, (y0 + 0.5f) / kH, a);
    Effects((x1 + 0.5f) / kW, (y0 + 0.5f) / kH, b);
    Effects((x0 + 0.5f) / kW, (y1 + 0.5f) / kH, c);
    Effects((x1 + 0.5f) / kW, (y1 + 0.5f) / kH, d);
    for (int i = 0; i < 3; ++i)
        out[i] = (a[i] * (1 - fx) + b[i] * fx) * (1 - fy) + (c[i] * (1 - fx) + d[i] * fx) * fy;
}

}  // namespace

int RunSunFx() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const bool debugLayer = SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                        D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                                        nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);
    if (!qvfr::sunfx::TestInstall(dev, ctx)) {
        printf("Sun effects hooks could not be installed\n");
        return 1;
    }

    // Shaders (compiled here: the effect shaders are registered by pointer, not recognised by checksum).
    auto blob = [&](const char* entry, const char* target) {
        ID3DBlob *code = nullptr, *err = nullptr;
        if (FAILED(D3DCompile(kHlsl, sizeof(kHlsl) - 1, "sun_fx_smoke", nullptr, nullptr, entry, target, 0, 0, &code,
                              &err))) {
            printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
            code = nullptr;
        }
        if (err) err->Release();
        return code;
    };
    ID3DBlob* vsCode = blob("VS", "vs_5_0");
    ID3DBlob* flareCode = blob("Flare", "ps_5_0");
    ID3DBlob* raysCode = blob("Rays", "ps_5_0");
    ID3DBlob* anchorCode = blob("Anchor", "ps_5_0");
    if (!vsCode || !flareCode || !raysCode || !anchorCode) return 1;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader *flare = nullptr, *rays = nullptr, *anchor = nullptr;
    dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(flareCode->GetBufferPointer(), flareCode->GetBufferSize(), nullptr, &flare);
    dev->CreatePixelShader(raysCode->GetBufferPointer(), raysCode->GetBufferSize(), nullptr, &rays);
    dev->CreatePixelShader(anchorCode->GetBufferPointer(), anchorCode->GetBufferSize(), nullptr, &anchor);
    for (ID3DBlob* b : {vsCode, flareCode, raysCode, anchorCode}) b->Release();
    qvfr::sunfx::TestRegister(flare, qvfr::sunfx::KindFlare);
    qvfr::sunfx::TestRegister(rays, qvfr::sunfx::KindRays);

    // HDR target (shared by the groups), outputs per group, constants, blend, index buffer.
    D3D11_TEXTURE2D_DESC d{};
    d.Width = kW;
    d.Height = kH;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    d.SampleDesc = {4, 0};
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D* hdr = nullptr;
    ID3D11RenderTargetView* hdrRtv = nullptr;
    ID3D11ShaderResourceView* hdrSrv = nullptr;
    bool made = SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &hdr)) &&
                SUCCEEDED(dev->CreateRenderTargetView(hdr, nullptr, &hdrRtv)) &&
                SUCCEEDED(dev->CreateShaderResourceView(hdr, nullptr, &hdrSrv));
    d.SampleDesc = {1, 0};
    d.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* out[4] = {};
    ID3D11RenderTargetView* outRtv[4] = {};
    for (int g = 0; g < 4; ++g)
        made = made && SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &out[g])) &&
               SUCCEEDED(dev->CreateRenderTargetView(out[g], nullptr, &outRtv[g]));
    const float cb[4] = {1.0f / kW, 1.0f / kH, 0, 0};
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(cb);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA init{cb, 0, 0};
    ID3D11Buffer* cbuf = nullptr;
    made = made && SUCCEEDED(dev->CreateBuffer(&bd, &init, &cbuf));
    const uint16_t idx[3] = {0, 1, 2};
    bd.ByteWidth = sizeof(idx);
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    init.pSysMem = idx;
    ID3D11Buffer* ib = nullptr;
    made = made && SUCCEEDED(dev->CreateBuffer(&bd, &init, &ib));
    D3D11_BLEND_DESC blend{};
    blend.RenderTarget[0].BlendEnable = TRUE;
    blend.RenderTarget[0].SrcBlend = blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blend.RenderTarget[0].DestBlend = blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    blend.RenderTarget[0].BlendOp = blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    ID3D11BlendState* additive = nullptr;
    made = made && SUCCEEDED(dev->CreateBlendState(&blend, &additive));
    if (!made || !vs || !flare || !rays || !anchor) {
        printf("Cannot create the sun effects test objects\n");
        return 1;
    }

    // The game's draws of one eye group.
    const D3D11_VIEWPORT vp = {0, 0, (float)kW, (float)kH, 0, 1};
    auto bind = [&](ID3D11RenderTargetView* rtv) {
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        qvfr::sunfx::OnSetTargets(ctx, 1, &rtv);  // quad's OMSetRenderTargets detour does this in the game
    };
    auto group = [&](int g, bool effects, bool focusWindow) {
        if (focusWindow)
            qvfr::sunfx::BeginGroup(g, kOrigin[0], kOrigin[1], kScale[0], kScale[1]);
        else
            qvfr::sunfx::BeginGroup(g, 0, 0, 1, 1);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->IASetIndexBuffer(ib, DXGI_FORMAT_R16_UINT, 0);
        ctx->VSSetShader(vs, nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, &cbuf);
        ctx->RSSetViewports(1, &vp);
        bind(hdrRtv);
        const float base[4] = {kBase, kBase, kBase, 1};
        ctx->ClearRenderTargetView(hdrRtv, base);
        if (effects) {
            ctx->OMSetBlendState(additive, nullptr, 0xFFFFFFFF);
            ctx->PSSetShader(flare, nullptr, 0);
            ctx->DrawIndexed(3, 0, 0);
            ctx->PSSetShader(rays, nullptr, 0);
            ctx->DrawIndexed(3, 0, 0);
        }
        ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        bind(outRtv[g]);
        ctx->PSSetShader(anchor, nullptr, 0);
        ctx->PSSetShaderResources(0, 1, &hdrSrv);
        ctx->Draw(3, 0);
        ID3D11ShaderResourceView* none = nullptr;
        ctx->PSSetShaderResources(0, 1, &none);
        qvfr::sunfx::EndGroup();
    };
    // Reads one output and compares it with `expect(u, v, rgb)` at every pixel center.
    auto check = [&](ID3D11Texture2D* tex, auto expect, float& maxErr) {
        D3D11_TEXTURE2D_DESC s = d;
        s.Usage = D3D11_USAGE_STAGING;
        s.BindFlags = 0;
        s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        D3D11_MAPPED_SUBRESOURCE m{};
        maxErr = 1e9f;
        if (FAILED(dev->CreateTexture2D(&s, nullptr, &staging))) return false;
        ctx->CopyResource(staging, tex);
        if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
            maxErr = 0;
            for (UINT y = 0; y < kH; ++y) {
                const uint16_t* row = (const uint16_t*)((const uint8_t*)m.pData + (size_t)y * m.RowPitch);
                for (UINT x = 0; x < kW; ++x) {
                    float e[3];
                    expect((x + 0.5f) / kW, (y + 0.5f) / kH, e);
                    for (int c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::fabs(Half(row[x * 4 + c]) - e[c]));
                }
            }
            ctx->Unmap(staging, 0);
        }
        staging->Release();
        return maxErr < 0.01f;
    };
    auto outerExpect = [](float u, float v, float e[3]) {
        Effects(u, v, e);
        for (int c = 0; c < 3; ++c) e[c] += kBase;
    };
    auto focusExpect = [](float u, float v, float e[3]) {
        Bilinear(kOrigin[0] + u * kScale[0], kOrigin[1] + v * kScale[1], e);
        for (int c = 0; c < 3; ++c) e[c] += kBase;
    };

    // Frame 1, module on: outer L, outer R, focus L with effect draws, focus R without.
    qvfr::sunfx::OnFrameStart(ctx, true);
    group(0, true, false);
    group(1, true, false);
    group(2, true, true);
    group(3, false, true);
    float errOuter = 0, errFocusDraw = 0, errFocusAnchor = 0, errOff = 0;
    const bool outerOk = check(out[0], outerExpect, errOuter);
    const bool focusDrawOk = check(out[2], focusExpect, errFocusDraw);
    const bool focusAnchorOk = check(out[3], focusExpect, errFocusAnchor);
    const qvfr::sunfx::TestState st = qvfr::sunfx::TestGet();
    const bool countsOk = st.replays == 4 && st.skipped == 2 && st.composites == 1 && st.anchorComposites == 1 &&
                          st.missed == 0 && st.faults == 0;
    // Frame 2, module off: focus L draws its own effects at its own UV.
    qvfr::sunfx::OnFrameStart(ctx, false);
    group(0, true, false);
    group(1, true, false);
    group(2, true, true);
    const bool offOk = check(out[2], outerExpect, errOff);
    const qvfr::sunfx::TestState st2 = qvfr::sunfx::TestGet();
    const bool offCountsOk = st2.replays == st.replays && st2.skipped == st.skipped;

    // DLAA in quad mode: a replaced DrawIndexed does not reach D3D11; a draw of DLSS itself passes untouched.
    static int s_replaceCalls = 0;
    static bool s_inside = false;
    const bool replaceSet = qvfr::sunfx::SetDrawReplace(
        [](ID3D11DeviceContext*, UINT count) {
            ++s_replaceCalls;
            return count == 3;
        },
        [] { return s_inside; }, nullptr);
    auto drawInto0 = [&] {
        const float zero[4] = {0, 0, 0, 0};
        bind(outRtv[0]);
        ctx->ClearRenderTargetView(outRtv[0], zero);
        ctx->OMSetBlendState(additive, nullptr, 0xFFFFFFFF);
        ctx->PSSetShader(flare, nullptr, 0);
        ctx->DrawIndexed(3, 0, 0);
        ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    };
    auto zeroExpect = [](float, float, float e[3]) { e[0] = e[1] = e[2] = 0; };
    float errReplaced = 0, errInside = 0;
    drawInto0();
    const bool replacedOk = check(out[0], zeroExpect, errReplaced) && s_replaceCalls == 1;
    s_inside = true;
    drawInto0();
    const bool insideOk = !check(out[0], zeroExpect, errInside) && s_replaceCalls == 1;
    qvfr::sunfx::SetDrawReplace(nullptr, nullptr, nullptr);
    const bool replaceOk = replaceSet && replacedOk && insideOk;

    // Bonnet reflection: outer L and R read their own image (red, green), the focus groups their own (blue); the focus
    // groups' car paint gets outer's image of the eye, and the game's view is back in the slot after the draw.
    bool bonnetOk = false;
    char bonnetText[256] = "not run";
    {
        auto paintBlob = [&](const char* entry) {
            ID3DBlob *code = nullptr, *err = nullptr;
            if (FAILED(D3DCompile(kPaintHlsl, sizeof(kPaintHlsl) - 1, "sun_fx_smoke_paint", nullptr, nullptr, entry,
                                  "ps_5_0", 0, 0, &code, &err))) {
                printf("D3DCompile %s: %s\n", entry, err ? (const char*)err->GetBufferPointer() : "?");
                code = nullptr;
            }
            if (err) err->Release();
            return code;
        };
        const unsigned long long madeBefore = qvfr::sunfx::TestGet().bonnetMade;
        ID3DBlob* paintCode = paintBlob("Paint");
        ID3DBlob* decoyCode = paintBlob("Decoy");
        ID3D11PixelShader *paint = nullptr, *decoy = nullptr;
        if (paintCode) dev->CreatePixelShader(paintCode->GetBufferPointer(), paintCode->GetBufferSize(), nullptr, &paint);
        if (decoyCode) dev->CreatePixelShader(decoyCode->GetBufferPointer(), decoyCode->GetBufferSize(), nullptr, &decoy);
        for (ID3DBlob* b : {paintCode, decoyCode})
            if (b) b->Release();
        const unsigned long long paintMade = qvfr::sunfx::TestGet().bonnetMade - madeBefore;
        const float colors[3][4] = {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
        ID3D11Texture2D* img[3] = {};
        ID3D11ShaderResourceView* imgSrv[3] = {};
        D3D11_TEXTURE2D_DESC id{};
        id.Width = id.Height = 1;
        id.MipLevels = id.ArraySize = 1;
        id.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        id.SampleDesc = {1, 0};
        id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bool ok2 = paint && decoy;
        for (int i = 0; i < 3 && ok2; ++i) {
            D3D11_SUBRESOURCE_DATA px{colors[i], sizeof(colors[i]), 0};
            ok2 = SUCCEEDED(dev->CreateTexture2D(&id, &px, &img[i])) &&
                  SUCCEEDED(dev->CreateShaderResourceView(img[i], nullptr, &imgSrv[i]));
        }
        float cb12[44][4] = {};
        cb12[0][0] = 1;
        cb12[43][0] = 1;
        D3D11_BUFFER_DESC pb{};
        pb.ByteWidth = sizeof(cb12);
        pb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA pinit{cb12, 0, 0};
        ID3D11Buffer* paintCb = nullptr;
        ok2 = ok2 && SUCCEEDED(dev->CreateBuffer(&pb, &pinit, &paintCb));
        bool restored = true;
        auto paintGroup = [&](int g, int image, ID3D11PixelShader* ps) {
            qvfr::sunfx::BeginGroup(g, g < 2 ? 0 : kOrigin[0], g < 2 ? 0 : kOrigin[1], g < 2 ? 1 : kScale[0],
                                    g < 2 ? 1 : kScale[1]);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx->VSSetShader(vs, nullptr, 0);
            ctx->RSSetViewports(1, &vp);
            ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
            bind(outRtv[g]);
            ctx->PSSetShader(ps, nullptr, 0);
            ctx->PSSetConstantBuffers(12, 1, &paintCb);
            ctx->PSSetShaderResources(1, 1, &imgSrv[image]);
            ctx->Draw(3, 0);
            ID3D11ShaderResourceView* now = nullptr;
            ctx->PSGetShaderResources(1, 1, &now);
            if (now != imgSrv[image]) restored = false;
            if (now) now->Release();
            ID3D11ShaderResourceView* none = nullptr;
            ctx->PSSetShaderResources(1, 1, &none);
            qvfr::sunfx::EndGroup();
        };
        auto solid = [](const float* rgb) {
            return [rgb](float, float, float e[3]) {
                for (int c = 0; c < 3; ++c) e[c] = rgb[c];
            };
        };
        float err[6] = {};
        bool px[6] = {};
        unsigned long long focusGiven = 0, missed = 0;
        if (ok2) {
            const qvfr::sunfx::TestState s0 = qvfr::sunfx::TestGet();
            qvfr::sunfx::SetBonnetFromOuter(true);
            qvfr::sunfx::OnFrameStart(ctx, true);
            paintGroup(0, 0, paint);
            paintGroup(1, 1, paint);
            paintGroup(2, 2, paint);
            px[0] = check(out[2], solid(colors[0]), err[0]);  // focus L: outer L's image
            paintGroup(3, 2, paint);
            px[1] = check(out[3], solid(colors[1]), err[1]);  // focus R: outer R's image
            paintGroup(2, 2, decoy);
            px[2] = check(out[2], solid(colors[2]), err[2]);  // not a car paint shader: its own image
            qvfr::sunfx::OnFrameStart(ctx, true);  // a frame without outer's car paint: the game's image stays
            paintGroup(2, 2, paint);
            px[3] = check(out[2], solid(colors[2]), err[3]);
            qvfr::sunfx::SetBonnetFromOuter(false);  // off: the game's image
            qvfr::sunfx::OnFrameStart(ctx, true);
            paintGroup(0, 0, paint);
            paintGroup(2, 2, paint);
            px[4] = check(out[2], solid(colors[2]), err[4]);
            qvfr::sunfx::SetBonnetFromOuter(true);
            const qvfr::sunfx::TestState s1 = qvfr::sunfx::TestGet();
            focusGiven = s1.bonnetFocus - s0.bonnetFocus;
            missed = s1.bonnetMissed - s0.bonnetMissed;
            bonnetOk = paintMade == 1 && px[0] && px[1] && px[2] && px[3] && px[4] && restored && focusGiven == 2 &&
                       missed == 1 && s1.bonnetOuter - s0.bonnetOuter == 2;
        }
        snprintf(bonnetText, sizeof(bonnetText),
                 "car paint shaders recognised %llu of 1 (decoy not), focus L / R got outer L / R's image %d / %d "
                 "(max %.4f / %.4f), decoy kept its own %d, frame without outer's kept the game's %d, off %d, game's "
                 "view back %d, focus draws given %llu, missed %llu",
                 paintMade, px[0], px[1], err[0], err[1], px[2], px[3], px[4], restored, focusGiven, missed);
        for (IUnknown* u : {(IUnknown*)paint, (IUnknown*)decoy, (IUnknown*)paintCb})
            if (u) u->Release();
        for (int i = 0; i < 3; ++i) {
            if (imgSrv[i]) imgSrv[i]->Release();
            if (img[i]) img[i]->Release();
        }
    }
    qvfr::sunfx::Shutdown();

    int warnings = 0;
    if (info) {
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> buf(size);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (FAILED(info->GetMessage(i, msg, &size)) || msg->Severity > D3D11_MESSAGE_SEVERITY_WARNING) continue;
            ++warnings;
            printf("D3D11 debug layer: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
        }
        info->Release();
    }
    for (IUnknown* u : {(IUnknown*)vs, (IUnknown*)flare, (IUnknown*)rays, (IUnknown*)anchor, (IUnknown*)hdr,
                        (IUnknown*)hdrRtv, (IUnknown*)hdrSrv, (IUnknown*)cbuf, (IUnknown*)ib, (IUnknown*)additive})
        u->Release();
    for (int g = 0; g < 4; ++g) {
        outRtv[g]->Release();
        out[g]->Release();
    }
    ctx->Release();
    dev->Release();
    const bool ok = outerOk && focusDrawOk && focusAnchorOk && countsOk && offOk && offCountsOk && replaceOk &&
                    bonnetOk && !warnings;
    printf("Bonnet reflection check %s: %s\n", bonnetOk ? "OK" : "FAILED", bonnetText);
    printf("Sun effects check %s: outer max error %.4f, focus with effect draws %.4f, focus at the anchor %.4f, module "
           "off %.4f; outer draws repeated %llu, focus draws skipped %llu, passes at an effect draw %llu, at the anchor "
           "%llu, missed %llu, faults %llu; module off counts unchanged %d; DLAA draw replace: replaced draw left the "
           "target empty %d (max %.4f), DLSS's own draw went through %d (max %.4f), callback calls %d; debug layer %s, "
           "%d warnings\n",
           ok ? "OK" : "FAILED", errOuter, errFocusDraw, errFocusAnchor, errOff, st.replays, st.skipped, st.composites,
           st.anchorComposites, st.missed, st.faults, offCountsOk, replacedOk, errReplaced, insideOk, errInside,
           s_replaceCalls, debugLayer ? "on" : "not installed", warnings);
    return ok ? 0 : 7;
}

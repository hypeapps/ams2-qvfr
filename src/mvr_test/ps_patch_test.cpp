// B9b: the DXBC pixel shader patch for Multi-View Rendering (src/dxbc/mvr_ps_patch.h), without the game.
//
// --ps-patch draws a scene with a pixel shader shaped like the game's forward shaders (a screen input divided by its
// own w and turned into texel coordinates, a Gather on SV_Position times 1 / size, a UV sample of a screen texture and
// a material texture) and checks:
// - the patched shader without MVR draws exactly what the original draws (the game today, gate B9b);
// - with MVR (2 views, the patched vertex shader of B4 with clip distances): view 0 is exactly the original image and
//   the top left corner of slice 1 is the focus image, compared with the original shader drawn as the focus pass of
//   the game draws it today (focus projection, focus-size target, textures and constants);
// - the original pixel shader under MVR gets the corner wrong (the test sees a broken patch).
// --ps-dir runs the patch over a folder of compiled pixel shaders (ps_*.dxbc from a RenderDoc capture), creates each
// result and counts what the patch did.

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "mvr_patch.h"
#include "mvr_ps_patch.h"

using Microsoft::WRL::ComPtr;

namespace {

void Out(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

struct Dev {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<ID3D11InfoQueue> info;
};

bool MakeDevice(Dev& g, bool debug) {
    ComPtr<IDXGIFactory1> f;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) return false;
    ComPtr<IDXGIAdapter1> nv;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> a;
        if (f->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        if (d.VendorId == 0x10DE) {
            nv = a;
            break;
        }
    }
    if (!nv) return Out("  no NVIDIA adapter\n"), false;
    const D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDevice(nv.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, debug ? D3D11_CREATE_DEVICE_DEBUG : 0,
                                   &fl, 1, D3D11_SDK_VERSION, &g.dev, nullptr, &g.ctx);
    if (FAILED(hr) && debug)
        hr = D3D11CreateDevice(nv.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &g.dev,
                               nullptr, &g.ctx);
    if (FAILED(hr)) return Out("  D3D11CreateDevice 0x%08lX\n", static_cast<unsigned long>(hr)), false;
    if (debug) g.dev.As(&g.info);
    return true;
}

// Debug layer messages at warning level or worse; returns how many were not the expected one. The patched pixel
// shader reads SV_RenderTargetArrayIndex, which no vertex shader writes: the debug layer reports a linkage error at
// each draw, the draw still runs (B1 cases 10 and 11), and only the debug layer checks the linkage.
int Drain(Dev& g) {
    if (!g.info) return 0;
    int n = 0, expected = 0;
    const UINT64 count = g.info->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T len = 0;
        g.info->GetMessage(i, nullptr, &len);
        std::vector<char> buf(len);
        auto* m = reinterpret_cast<D3D11_MESSAGE*>(buf.data());
        if (SUCCEEDED(g.info->GetMessage(i, m, &len)) && m->Severity <= D3D11_MESSAGE_SEVERITY_WARNING) {
            const std::string text(m->pDescription, strnlen(m->pDescription, m->DescriptionByteLength));
            // Also: behind a geometry shader that writes the index, the register the patch chose may differ from its.
            if (text.find("linkage error") != std::string::npos &&
                (text.find("(SV_RenderTargetArrayIndex,0) as input, but it is not provided") != std::string::npos ||
                 text.find("'SV_RenderTargetArrayIndex' is defined for mismatched hardware registers") !=
                     std::string::npos)) {
                ++expected;
                continue;
            }
            Out("    [d3d11 debug] %s\n", text.c_str());
            ++n;
        }
    }
    g.info->ClearStoredMessages();
    if (expected)
        Out("  debug layer: %d expected linkage reports (SV_RenderTargetArrayIndex not written by the VS, or in another "
            "register of the GS)\n",
            expected);
    return n;
}

ComPtr<ID3DBlob> Build(const char* src, const char* target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(src, strlen(src), "ps_patch_test", nullptr, nullptr, "main", target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        Out("  compile failed: %s\n", err ? static_cast<const char*>(err->GetBufferPointer()) : "?");
        return nullptr;
    }
    return code;
}

std::string Disasm(const void* code, size_t size) {
    ComPtr<ID3DBlob> d;
    if (FAILED(D3DDisassemble(code, size, 0, nullptr, &d))) return {};
    return std::string(static_cast<const char*>(d->GetBufferPointer()),
                       strnlen(static_cast<const char*>(d->GetBufferPointer()), d->GetBufferSize()));
}

std::string LineWith(const std::string& text, const char* what) {
    const size_t at = text.find(what);
    if (at == std::string::npos) return "(none)";
    const size_t b = text.rfind('\n', at), e = text.find('\n', at);
    std::string s = text.substr(b == std::string::npos ? 0 : b + 1, e == std::string::npos ? std::string::npos : e - b - 1);
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ')) s.pop_back();
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    return s;
}

// The scene: a quad tilted in depth (perspective-correct interpolation matters) covering the whole target.
const char* kVs =
    "cbuffer Proj : register(b0) { float4 R[4]; };\n"
    "struct O { float4 pos : SV_Position; float4 scr : TEXCOORD0; float4 col : TEXCOORD1; };\n"
    "O main(float4 p : POSITION) {\n"
    "  O o;\n"
    "  o.pos = float4(dot(R[0], p), dot(R[1], p), dot(R[2], p), dot(R[3], p));\n"
    "  o.scr = float4(0.5 * (o.pos.x + o.pos.w), 0.5 * (o.pos.w - o.pos.y), o.pos.z, o.pos.w);\n"
    "  o.col = p;\n"
    "  return o;\n"
    "}\n";
// Shaped like the game's forward shaders (docs/mvr.md B2).
const char* kPs =
    "Texture2D materialTex : register(t0);\n"
    "Texture2D globalTexture0 : register(t3);\n"
    "Texture2D globalTexture2 : register(t5);\n"
    "SamplerState pointSampler : register(s0);\n"
    "SamplerState linearSampler : register(s1);\n"
    "cbuffer G : register(b1) { float4 viewportPixelDimensions; float4 screenSpaceDimensions; };\n"
    "float4 main(float4 pos : SV_Position, float4 scr : TEXCOORD0, centroid float4 col : TEXCOORD1) : SV_Target {\n"
    "  float2 uv = scr.xy / scr.w;\n"
    "  int2 p = int2(min(uv * viewportPixelDimensions.zw, viewportPixelDimensions.zw - 1));\n"
    "  float4 a = globalTexture0.Load(int3(p, 0));\n"
    "  float4 g = globalTexture2.GatherRed(pointSampler, pos.xy * screenSpaceDimensions.zw);\n"
    "  float d = globalTexture2.SampleLevel(linearSampler, uv, 0).y;\n"
    "  float m = materialTex.SampleLevel(linearSampler, col.xy * 0.05 + 0.5, 0).x;\n"
    "  return float4(a.x + a.y * 1000, g.x + g.y + g.z + g.w, d, a.z + m);\n"
    "}\n";
const char* kRtPs = "float4 main(float4 pos : SV_Position, uint rt : SV_RenderTargetArrayIndex) : SV_Target"
                    " { return rt; }\n";
// Like the game's layered passes: a geometry shader sends the scene to slice 1 itself.
const char* kLayeredGs =
    "struct O { float4 pos : SV_Position; float4 scr : TEXCOORD0; float4 col : TEXCOORD1; };\n"
    "struct G { float4 pos : SV_Position; float4 scr : TEXCOORD0; float4 col : TEXCOORD1;"
    " uint rt : SV_RenderTargetArrayIndex; };\n"
    "[maxvertexcount(3)] void main(triangle O v[3], inout TriangleStream<G> s) {\n"
    "  for (int k = 0; k < 3; ++k) { G g; g.pos = v[k].pos; g.scr = v[k].scr; g.col = v[k].col; g.rt = 1; s.Append(g); }\n"
    "}\n";

constexpr int kW = 96, kH = 80, kFw = 64, kFh = 48;  // outer and focus sizes (B8 smoke)
constexpr UINT kVsPatchSlot = 2;                     // the test VS uses b0
// Focus NDC from outer clip: x_f = a x + b w, y_f = c y + d w (outer NDC x in [-0.6, 0.4], y in [-0.3, 0.5]).
constexpr float kA = 2.0f, kB = 0.2f, kC = 2.5f, kD = -0.25f;

struct Tex {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11ShaderResourceView> srv;
};

// RGBA32F texture, texel (x, y) = fn(x, y).
template <class F>
Tex MakeTex(Dev& g, int w, int h, F fn) {
    std::vector<float> px(static_cast<size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) fn(x, y, &px[(static_cast<size_t>(y) * w + x) * 4]);
    D3D11_TEXTURE2D_DESC d = {};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_IMMUTABLE;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = {px.data(), static_cast<UINT>(w * 16), 0};
    Tex t;
    g.dev->CreateTexture2D(&d, &sd, &t.tex);
    if (t.tex) g.dev->CreateShaderResourceView(t.tex.Get(), nullptr, &t.srv);
    return t;
}

ComPtr<ID3D11Buffer> MakeCb(Dev& g, const void* data, UINT bytes) {
    D3D11_BUFFER_DESC d = {};
    d.ByteWidth = (bytes + 15) & ~15u;
    d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    d.Usage = D3D11_USAGE_IMMUTABLE;
    std::vector<uint8_t> b(d.ByteWidth, 0);
    memcpy(b.data(), data, bytes);
    D3D11_SUBRESOURCE_DATA sd = {b.data(), 0, 0};
    ComPtr<ID3D11Buffer> cb;
    g.dev->CreateBuffer(&d, &sd, &cb);
    return cb;
}

struct Target {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11RenderTargetView> all, s0;
    int w = 0, h = 0, slices = 0;
};

bool MakeTarget(Dev& g, int w, int h, int slices, Target& t) {
    t.w = w;
    t.h = h;
    t.slices = slices;
    D3D11_TEXTURE2D_DESC d = {};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = slices;
    d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(g.dev->CreateTexture2D(&d, nullptr, &t.tex))) return false;
    D3D11_RENDER_TARGET_VIEW_DESC r = {};
    r.Format = d.Format;
    r.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
    r.Texture2DArray.ArraySize = slices;
    if (FAILED(g.dev->CreateRenderTargetView(t.tex.Get(), &r, &t.all))) return false;
    r.Texture2DArray.ArraySize = 1;
    return SUCCEEDED(g.dev->CreateRenderTargetView(t.tex.Get(), &r, &t.s0));
}

std::vector<std::vector<float>> Read(Dev& g, const Target& t) {
    D3D11_TEXTURE2D_DESC d = {};
    t.tex->GetDesc(&d);
    d.BindFlags = 0;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> st;
    g.dev->CreateTexture2D(&d, nullptr, &st);
    g.ctx->CopyResource(st.Get(), t.tex.Get());
    std::vector<std::vector<float>> out(t.slices, std::vector<float>(static_cast<size_t>(t.w) * t.h * 4));
    for (int s = 0; s < t.slices; ++s) {
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(g.ctx->Map(st.Get(), s, D3D11_MAP_READ, 0, &m))) continue;
        for (int y = 0; y < t.h; ++y)
            memcpy(&out[s][static_cast<size_t>(y) * t.w * 4], static_cast<const char*>(m.pData) + y * m.RowPitch,
                   static_cast<size_t>(t.w) * 16);
        g.ctx->Unmap(st.Get(), s);
    }
    return out;
}

struct Diff {
    size_t pixels = 0, differ = 0;
    float maxAbs = 0;
    int firstX = -1, firstY = -1;
};

// Compares the w x h top left corner of two images; tol = 0 means bit-exact.
Diff Compare(const std::vector<float>& a, int strideA, const std::vector<float>& b, int strideB, int w, int h, float tol) {
    Diff d;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* pa = &a[(static_cast<size_t>(y) * strideA + x) * 4];
            const float* pb = &b[(static_cast<size_t>(y) * strideB + x) * 4];
            bool bad = false;
            for (int c = 0; c < 4; ++c) {
                const float e = std::fabs(pa[c] - pb[c]);
                d.maxAbs = std::max(d.maxAbs, e);
                bad = bad || (tol == 0 ? memcmp(&pa[c], &pb[c], 4) != 0 : !(e <= tol));
            }
            ++d.pixels;
            if (bad) {
                if (d.firstX < 0) d.firstX = x, d.firstY = y;
                ++d.differ;
            }
        }
    return d;
}

std::string DiffText(const Diff& d) {
    char b[160];
    snprintf(b, sizeof(b), "%zu of %zu pixels differ (max %.6g)", d.differ, d.pixels, d.maxAbs);
    std::string s = b;
    if (d.differ) {
        snprintf(b, sizeof(b), ", first at (%d, %d)", d.firstX, d.firstY);
        s += b;
    }
    return s;
}

bool CreateVsEx(Dev& g, const std::vector<uint8_t>& code, ComPtr<ID3D11VertexShader>& vs) {
    NV_CUSTOM_SEMANTIC sem[2] = {};
    sem[0].version = sem[1].version = NV_CUSTOM_SEMANTIC_VERSION;
    sem[0].NVCustomSemanticType = NV_POSITION_SEMANTIC;
    strcpy_s(sem[0].NVCustomSemanticNameString, "NV_POSITION");
    sem[1].NVCustomSemanticType = NV_CLIP_DISTANCE_0_SEMANTIC;
    strcpy_s(sem[1].NVCustomSemanticNameString, "NV_CLIP_DISTANCE_0");
    NvAPI_D3D11_CREATE_VERTEX_SHADER_EX args = {};
    args.version = NVAPI_D3D11_CREATEVERTEXSHADEREX_VERSION;
    args.NumCustomSemantics = 2;
    args.pCustomSemantics = sem;
    return NvAPI_D3D11_CreateVertexShaderEx(g.dev.Get(), code.data(), code.size(), nullptr, &args, &vs) == NVAPI_OK &&
           vs;
}

NvAPI_Status MultiView(Dev& g, int views) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = static_cast<NvU32>(views);
    for (int v = 0; v < NV_MULTIVIEW_MAX_SUPPORTED_VIEWS; ++v) p.renderTargetIndexOffset[v] = static_cast<NvU32>(v);
    return NvAPI_D3D_SetMultiViewMode(g.ctx.Get(), &p);
}

// Rows of a clip-space transform x' = sx x + ox w, y' = sy y + oy w.
void AffineRows(float sx, float ox, float sy, float oy, float* r) {
    const float m[16] = {sx, 0, 0, ox, 0, sy, 0, oy, 0, 0, 1, 0, 0, 0, 0, 1};
    memcpy(r, m, sizeof(m));
}

// Rows of P' = A * P (A affine as above, P 4 rows).
void Times(float sx, float ox, float sy, float oy, const float* p, float* r) {
    for (int k = 0; k < 4; ++k) {
        r[k] = sx * p[k] + ox * p[12 + k];
        r[4 + k] = sy * p[4 + k] + oy * p[12 + k];
        r[8 + k] = p[8 + k];
        r[12 + k] = p[12 + k];
    }
}

bool RunCase(bool debug, bool rtRegHigh) {
    Dev g;
    if (!MakeDevice(g, debug)) return false;
    Out("\n[B9b patched PS%s]\n", rtRegHigh ? ", view number in v31" : "");
    auto vsCode = Build(kVs, "vs_5_0");
    auto psCode = Build(kPs, "ps_5_0");
    auto rtCode = Build(kRtPs, "ps_5_0");
    auto gsCode = Build(kLayeredGs, "gs_5_0");
    if (!vsCode || !psCode || !rtCode || !gsCode) return false;
    if (!rtRegHigh)
        Out("  fxc declares the view number as: %s\n",
            LineWith(Disasm(rtCode->GetBufferPointer(), rtCode->GetBufferSize()), "rendertarget_array_index").c_str());

    qvfr::dxbc::MvrPsPatchOptions popt;
    popt.viewReg = rtRegHigh ? 31 : 0;
    std::vector<uint8_t> ps2;
    qvfr::dxbc::MvrPsPatchInfo pinf;
    std::string err;
    const auto pr = qvfr::dxbc::PatchPsForMvr(psCode->GetBufferPointer(), psCode->GetBufferSize(), popt, ps2, &pinf, &err);
    if (pr != qvfr::dxbc::MvrPsPatchResult::Patched) {
        Out("  PS patch: %s %s\n", pr == qvfr::dxbc::MvrPsPatchResult::NotNeeded ? "not needed" : "refused",
            err.c_str());
        return false;
    }
    const std::string dis = Disasm(ps2.data(), ps2.size());
    Out("  PS patch: %u screen inputs (mask 0x%x), textures 0x%x, %u reads wrapped (%u in UV), view number v%u, "
        "temps r%u..r%u, %zu -> %zu bytes, disassembly %s\n",
        pinf.screenInputs, pinf.screenInputMask, pinf.textureMask, pinf.wrapped, pinf.scaled, pinf.viewReg,
        pinf.firstTemp, pinf.firstTemp + pinf.temps - 1, static_cast<size_t>(psCode->GetBufferSize()), ps2.size(),
        dis.empty() ? "FAILED" : "ok");
    if (!rtRegHigh) Out("  patch declares it as: %s\n", LineWith(dis, "rendertarget_array_index").c_str());
    if (getenv("QVFR_PS_LISTING")) Out("%s\n", dis.c_str());

    qvfr::dxbc::MvrVsPatchOptions vopt;
    vopt.cbSlot = kVsPatchSlot;
    vopt.clip = true;
    std::vector<uint8_t> vs2;
    if (!qvfr::dxbc::PatchVsForMvr(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), vopt, vs2, nullptr, &err))
        return Out("  VS patch failed: %s\n", err.c_str()), false;

    ComPtr<ID3D11VertexShader> vsPlain, vsMvr;
    ComPtr<ID3D11PixelShader> psOrig, psPatched;
    ComPtr<ID3D11GeometryShader> gsLayered;
    g.dev->CreateGeometryShader(gsCode->GetBufferPointer(), gsCode->GetBufferSize(), nullptr, &gsLayered);
    g.dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vsPlain);
    g.dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &psOrig);
    const HRESULT hrP = g.dev->CreatePixelShader(ps2.data(), ps2.size(), nullptr, &psPatched);
    if (!vsPlain || !psOrig || FAILED(hrP) || !CreateVsEx(g, vs2, vsMvr))
        return Out("  shader creation failed (patched PS 0x%08lX)\n", static_cast<unsigned long>(hrP)), false;
    ComPtr<ID3D11InputLayout> layout;
    const D3D11_INPUT_ELEMENT_DESC ie = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,
                                         D3D11_INPUT_PER_VERTEX_DATA, 0};
    g.dev->CreateInputLayout(&ie, 1, vsCode->GetBufferPointer(), vsCode->GetBufferSize(), &layout);

    // Projection: x, y as they are, w = z, depth 1 - 1 / z.
    float proj[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 1, 0};
    float projF[16];
    Times(kA, kB, kC, kD, proj, projF);
    // View 1 in the corner of an outer-size slice: x_1 = (fw / W)(x_f + w) - w, y_1 = (fh / H)(y_f - w) + w.
    const float rx = static_cast<float>(kFw) / kW, ry = static_cast<float>(kFh) / kH;
    const float sx = rx * kA, ox = rx * (kB + 1) - 1, sy = ry * kC, oy = ry * (kD - 1) + 1;
    float vsPatch[32];
    AffineRows(sx, ox, sy, oy, vsPatch);
    float* c = vsPatch + 16;
    for (int k = 0; k < 4; ++k) {
        c[k] = (2 * rx - 1) * vsPatch[12 + k] - vsPatch[k];      // x_1 <= (2 fw / W - 1) w
        c[4 + k] = vsPatch[4 + k] - (1 - 2 * ry) * vsPatch[12 + k];  // y_1 >= (1 - 2 fh / H) w
        c[8 + k] = c[12 + k] = vsPatch[12 + k];
    }
    float psRows[12] = {sx, sy, 0.5f * (1 + ox - sx), 0.5f * (1 - oy - sy),
                        static_cast<float>(kW) / kFw, static_cast<float>(kH) / kFh, 1, 1};
    memcpy(&psRows[8], &qvfr::dxbc::kMvrPsMagic, 4);
    // A buffer of the game's in PS slot 3 (the patch's slot) during a layered pass: anything but the magic value.
    float gameRows[12];
    for (int i = 0; i < 12; ++i) gameRows[i] = 0.37f * (i + 1);
    const float gOuter[8] = {0, 0, kW, kH, 0, 0, 1.0f / kW, 1.0f / kH};
    const float gFocus[8] = {0, 0, kFw, kFh, 0, 0, 1.0f / kFw, 1.0f / kFh};
    auto cbProj = MakeCb(g, proj, sizeof(proj)), cbProjF = MakeCb(g, projF, sizeof(projF));
    auto cbVsPatch = MakeCb(g, vsPatch, sizeof(vsPatch)), cbPs = MakeCb(g, psRows, sizeof(psRows));
    auto cbGame = MakeCb(g, gameRows, sizeof(gameRows));
    auto cbOuter = MakeCb(g, gOuter, sizeof(gOuter)), cbFocus = MakeCb(g, gFocus, sizeof(gFocus));

    // Screen textures: texel coordinates and which image (0 outer, 1 focus).
    auto lin = [](int id) {
        return [id](int x, int y, float* p) {
            p[0] = static_cast<float>(x);
            p[1] = static_cast<float>(y);
            p[2] = static_cast<float>(id);
            p[3] = 0;
        };
    };
    auto gat = [](int id) {
        return [id](int x, int y, float* p) {
            p[0] = static_cast<float>(x + 1000 * id);
            p[1] = static_cast<float>(y + 100 * id);
            p[2] = p[3] = 0;
        };
    };
    Tex mat = MakeTex(g, 16, 16, [](int x, int y, float* p) { p[0] = (x * 7 + y * 3) % 11 / 11.0f, p[1] = p[2] = p[3] = 0; });
    Tex t0o = MakeTex(g, kW, kH, lin(0)), t2o = MakeTex(g, kW, kH, gat(0));
    Tex t0f = MakeTex(g, kFw, kFh, lin(1)), t2f = MakeTex(g, kFw, kFh, gat(1));
    ComPtr<ID3D11SamplerState> smp[2];
    for (int i = 0; i < 2; ++i) {
        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = i ? D3D11_FILTER_MIN_MAG_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        g.dev->CreateSamplerState(&sd, &smp[i]);
    }
    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    ComPtr<ID3D11RasterizerState> rs;
    g.dev->CreateRasterizerState(&rd, &rs);

    const float z0 = 1.5f, z1 = 4.0f, e = 1.2f;
    const float quad[6][4] = {{-e * z0, -e * z0, z0, 1}, {e * z0, -e * z0, z0, 1}, {-e * z1, e * z1, z1, 1},
                              {e * z0, -e * z0, z0, 1},  {e * z1, e * z1, z1, 1},   {-e * z1, e * z1, z1, 1}};
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = sizeof(quad);
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA vd = {quad, 0, 0};
    ComPtr<ID3D11Buffer> vb;
    g.dev->CreateBuffer(&bd, &vd, &vb);

    struct Pass {
        ID3D11VertexShader* vs;
        ID3D11PixelShader* ps;
        bool focus;  // the game's focus pass: focus projection, size, textures and constants
        int views;   // 1 or 2 (MVR)
        ID3D11GeometryShader* gs = nullptr;  // layered: draws into both slices without MVR
        ID3D11Buffer* slot3 = nullptr;       // PS slot 3 when not the MVR rows
    };
    auto draw = [&](const Pass& p, Target& t) {
        const bool twoSlices = p.views == 2 || p.gs;
        if (!MakeTarget(g, p.focus ? kFw : kW, p.focus ? kFh : kH, twoSlices ? 2 : 1, t)) return false;
        g.ctx->ClearState();
        const float clear[4] = {-1, -1, -1, -1};
        g.ctx->ClearRenderTargetView(t.all.Get(), clear);
        const UINT stride = 16, offset = 0;
        g.ctx->IASetInputLayout(layout.Get());
        g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g.ctx->IASetVertexBuffers(0, 1, vb.GetAddressOf(), &stride, &offset);
        ID3D11Buffer* vcb[3] = {p.focus ? cbProjF.Get() : cbProj.Get(), nullptr, cbVsPatch.Get()};
        g.ctx->VSSetConstantBuffers(0, 3, vcb);
        g.ctx->VSSetShader(p.vs, nullptr, 0);
        g.ctx->GSSetShader(p.gs, nullptr, 0);
        ID3D11Buffer* pcb[4] = {nullptr, p.focus ? cbFocus.Get() : cbOuter.Get(), nullptr,
                                p.gs ? p.slot3 : cbPs.Get()};
        g.ctx->PSSetConstantBuffers(0, 4, pcb);
        ID3D11ShaderResourceView* srv[6] = {mat.srv.Get(), nullptr, nullptr, p.focus ? t0f.srv.Get() : t0o.srv.Get(),
                                            nullptr, p.focus ? t2f.srv.Get() : t2o.srv.Get()};
        g.ctx->PSSetShaderResources(0, 6, srv);
        ID3D11ShaderResourceView* extra[5] = {t0f.srv.Get(), nullptr, t2f.srv.Get(), nullptr, nullptr};
        g.ctx->PSSetShaderResources(popt.texBase, 5, extra);
        ID3D11SamplerState* ss[2] = {smp[0].Get(), smp[1].Get()};
        g.ctx->PSSetSamplers(0, 2, ss);
        g.ctx->PSSetShader(p.ps, nullptr, 0);
        g.ctx->RSSetState(rs.Get());
        const D3D11_VIEWPORT vp = {0, 0, static_cast<float>(t.w), static_cast<float>(t.h), 0, 1};
        g.ctx->RSSetViewports(1, &vp);
        g.ctx->OMSetRenderTargets(1, twoSlices ? t.all.GetAddressOf() : t.s0.GetAddressOf(), nullptr);
        if (p.views == 2 && MultiView(g, 2) != NVAPI_OK) return false;
        g.ctx->Draw(6, 0);
        if (p.views == 2) MultiView(g, 1);
        return true;
    };
    Target tO, tP0, tMv, tF, tU;
    if (!draw({vsPlain.Get(), psOrig.Get(), false, 1}, tO) || !draw({vsPlain.Get(), psPatched.Get(), false, 1}, tP0) ||
        !draw({vsMvr.Get(), psPatched.Get(), false, 2}, tMv) || !draw({vsPlain.Get(), psOrig.Get(), true, 1}, tF) ||
        !draw({vsMvr.Get(), psOrig.Get(), false, 2}, tU))
        return Out("  RESULT: FAIL (setup)\n"), false;
    const auto o = Read(g, tO), p0 = Read(g, tP0), mv = Read(g, tMv), f = Read(g, tF), u = Read(g, tU);

    bool ok = true;
    auto check = [&](const char* what, const Diff& d, bool wantSame) {
        const bool good = wantSame ? d.differ == 0 : d.differ > d.pixels / 2;
        Out("  %-58s %s -> %s\n", what, DiffText(d).c_str(), good ? "OK" : "WRONG");
        ok = ok && good;
    };
    check("patched PS without MVR vs original (bit-exact):", Compare(p0[0], kW, o[0], kW, kW, kH, 0), true);
    check("MVR view 0 vs original (bit-exact):", Compare(mv[0], kW, o[0], kW, kW, kH, 0), true);
    // Load and Gather give whole texel numbers: an error is at least 1. The UV sample may round in its last bits.
    check("MVR view 1, slice corner vs the focus pass (tolerance 0.01):", Compare(mv[1], kW, f[0], kFw, kFw, kFh, 0.01f),
          true);
    {
        Diff d;
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                if (x < kFw && y < kFh) continue;
                ++d.pixels;
                const float* q = &mv[1][(static_cast<size_t>(y) * kW + x) * 4];
                if (q[0] != -1 || q[1] != -1 || q[2] != -1 || q[3] != -1) {
                    if (d.firstX < 0) d.firstX = x, d.firstY = y;
                    ++d.differ;
                }
            }
        check("MVR view 1 outside the corner untouched:", d, true);
    }
    check("original PS under MVR, corner vs the focus pass (must differ):", Compare(u[1], kW, f[0], kFw, kFw, kFh, 0.01f),
          false);
    // A layered pass of the game: a geometry shader writes slice 1, MVR off, PS slot 3 empty or the game's buffer.
    {
        Target lo, ln, lg, lm;
        if (!draw({vsPlain.Get(), psOrig.Get(), false, 1, gsLayered.Get(), nullptr}, lo) ||
            !draw({vsPlain.Get(), psPatched.Get(), false, 1, gsLayered.Get(), nullptr}, ln) ||
            !draw({vsPlain.Get(), psPatched.Get(), false, 1, gsLayered.Get(), cbGame.Get()}, lg) ||
            !draw({vsPlain.Get(), psPatched.Get(), false, 1, gsLayered.Get(), cbPs.Get()}, lm))
            return Out("  RESULT: FAIL (setup of the layered pass)\n"), false;
        const auto ro = Read(g, lo), rn = Read(g, ln), rg = Read(g, lg), rm = Read(g, lm);
        check("layered GS (slice 1), slot 3 empty, vs original (bit-exact):", Compare(rn[1], kW, ro[1], kW, kW, kH, 0),
              true);
        check("layered GS (slice 1), a game buffer in slot 3 (bit-exact):", Compare(rg[1], kW, ro[1], kW, kW, kH, 0),
              true);
        check("layered GS with the MVR rows bound (must differ):", Compare(rm[1], kW, ro[1], kW, kW, kH, 0.01f), false);
    }
    if (Drain(g)) ok = false;
    Out("  RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int RunPsPatchTests(bool debug) {
    int failed = 0;
    failed += RunCase(debug, false) ? 0 : 1;
    failed += RunCase(debug, true) ? 0 : 1;
    Out("\nB9b PS patch checks: %d failed.\n", failed);
    return failed ? 1 : 0;
}

int RunPsPatchDir(const std::string& dir, const std::string& outDir, bool verbose) {
    namespace fs = std::filesystem;
    Dev g;
    if (!MakeDevice(g, false)) return 2;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".dxbc" && e.path().filename().string().rfind("ps_", 0) == 0) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (!outDir.empty()) fs::create_directories(outDir);
    Out("\n[PS patch dir] %zu pixel shaders in %s\n", files.size(), dir.c_str());
    int patched = 0, notNeeded = 0, refused = 0, createFailed = 0, disFailed = 0;
    unsigned inputs = 0, wraps = 0, scaled = 0, withInputs = 0, withTex = 0, maxTemps = 0;
    unsigned texCount[32] = {};
    std::map<std::string, int> reasons;
    qvfr::dxbc::MvrPsPatchOptions opt;
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        std::vector<uint8_t> code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string name = f.stem().string();
        std::vector<uint8_t> out;
        qvfr::dxbc::MvrPsPatchInfo inf;
        std::string err;
        const auto r = qvfr::dxbc::PatchPsForMvr(code.data(), code.size(), opt, out, &inf, &err);
        if (r == qvfr::dxbc::MvrPsPatchResult::NotNeeded) {
            ++notNeeded;
            continue;
        }
        if (r == qvfr::dxbc::MvrPsPatchResult::Refused) {
            ++refused;
            ++reasons[err.substr(0, err.find(" at token"))];
            Out("  %s: REFUSED: %s\n", name.c_str(), err.c_str());
            continue;
        }
        ++patched;
        ComPtr<ID3D11PixelShader> ps;
        const HRESULT hr = g.dev->CreatePixelShader(out.data(), out.size(), nullptr, &ps);
        const std::string dis = Disasm(out.data(), out.size());
        createFailed += FAILED(hr);
        disFailed += dis.empty();
        if (!dis.empty() && !outDir.empty()) std::ofstream(fs::path(outDir) / (name + ".patched.txt")) << dis;
        inputs += inf.screenInputs;
        wraps += inf.wrapped;
        scaled += inf.scaled;
        withInputs += inf.screenInputs > 0;
        withTex += inf.textureMask != 0;
        for (int k = 0; k < 32; ++k) texCount[k] += (inf.textureMask >> k) & 1;
        maxTemps = std::max(maxTemps, inf.firstTemp + inf.temps);
        if (verbose || FAILED(hr) || dis.empty())
            Out("  %s: inputs 0x%x, textures 0x%x, wrapped %u (UV %u), v%u, temps %u, CreatePixelShader %s, "
                "disassembly %s\n",
                name.c_str(), inf.screenInputMask, inf.textureMask, inf.wrapped, inf.scaled, inf.viewReg,
                inf.firstTemp + inf.temps, SUCCEEDED(hr) ? "ok" : "FAILED", dis.empty() ? "FAILED" : "ok");
    }
    Out("  patched %d (with screen inputs %u, %u inputs; with screen textures %u, %u reads wrapped, %u in UV), not "
        "needed %d, refused %d; CreatePixelShader failed %d, disassembly failed %d; most temps %u\n",
        patched, withInputs, inputs, withTex, wraps, scaled, notNeeded, refused, createFailed, disFailed, maxTemps);
    Out("  shaders per screen texture:");
    for (size_t k = 0; k < opt.screenTextures.size(); ++k) Out(" %s %u", opt.screenTextures[k].c_str(), texCount[k]);
    Out("\n");
    for (const auto& [why, n] : reasons) Out("  refused %d x: %s\n", n, why.c_str());
    return createFailed || disFailed ? 1 : 0;
}

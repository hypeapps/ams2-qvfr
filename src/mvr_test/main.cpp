// qvfr_mvr_test: checks NVIDIA Multi-View Rendering (MVR) in D3D11 on the local GPU, without the game.
//
// Correctness cases draw one rectangle with up to 4 views in a single draw. View 0 is the input position; every other
// view gets an affine transform of it in clip space (the same shape as "focus = sub-frustum of outer"). Each view lands
// in its own render target array slice (renderTargetIndexOffset) or, in the viewport-mask case, in its own viewport of a
// wide texture. The pixel shader writes the view id it received through a per-view generic attribute, so the readback
// shows both where each view landed and whether the pixel shader can tell the views apart.
//
// The performance case draws the same object list once per view (sequential passes, as the game does today) and once
// with MVR, and reports the CPU time of the submission, the CPU time of the whole process and the GPU time.
//
// B4 cases take an ordinary single-view vertex shader, add the extra view positions with the DXBC patch
// (src/dxbc/mvr_patch.h) and compare the image with the HLSL MVR shader pixel by pixel. --patch-dir runs the patch
// over a folder of compiled vertex shaders (*.dxbc, e.g. dumped from a RenderDoc capture) and creates each result.

#include <windows.h>
#include <timeapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "dxbc.h"
#include "mvr_patch.h"

using Microsoft::WRL::ComPtr;

namespace {

bool g_verbose = false;

void Log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

std::string NvErr(NvAPI_Status s) {
    NvAPI_ShortString msg = {};
    NvAPI_GetErrorMessage(s, msg);
    return std::string(msg) + " (" + std::to_string(static_cast<int>(s)) + ")";
}

struct Gpu {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<ID3D11InfoQueue> info;  // only with --debug
};

// Affine clip-space transform per view: x' = ax*x + bx*w, y' = ay*y + by*w, z and w unchanged.
struct ViewXf {
    float ax, bx, ay, by;
};
// View 1 is a focus window covering clip [0, 0.5] x [0, 0.5] of view 0, scaled to the full target.
constexpr ViewXf kViews[4] = {{1, 0, 1, 0}, {4, -1, 4, -1}, {-1, 0, 1, 0}, {2, 0.5f, 2, -0.5f}};
constexpr float kRectX0 = -0.5f, kRectX1 = 0.25f, kRectY0 = -0.5f, kRectY1 = 0.25f;

void ViewRows(const ViewXf& v, float* rows /* 16 floats */) {
    const float r[16] = {v.ax, 0, 0, v.bx, 0, v.ay, 0, v.by, 0, 0, 1, 0, 0, 0, 0, 1};
    memcpy(rows, r, sizeof(r));
}

void DrainDebugMessages(Gpu& g) {
    if (!g.info) return;
    const UINT64 n = g.info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T len = 0;
        g.info->GetMessage(i, nullptr, &len);
        std::vector<char> buf(len);
        auto* m = reinterpret_cast<D3D11_MESSAGE*>(buf.data());
        if (SUCCEEDED(g.info->GetMessage(i, m, &len)) && m->Severity <= D3D11_MESSAGE_SEVERITY_WARNING)
            Log("    [d3d11 debug] %.*s\n", static_cast<int>(m->DescriptionByteLength), m->pDescription);
    }
    g.info->ClearStoredMessages();
}

bool CreateDevice(Gpu& g, bool debug) {
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    ComPtr<IDXGIAdapter1> nvidia;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> a;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        if (d.VendorId == 0x10DE && !nvidia) {
            nvidia = a;
            static bool logged = false;
            if (!logged)
                Log("Adapter: %ls (VRAM %llu MB)\n", d.Description,
                    static_cast<unsigned long long>(d.DedicatedVideoMemory >> 20));
            logged = true;
        }
    }
    if (!nvidia) {
        Log("No NVIDIA adapter found.\n");
        return false;
    }
    const D3D_FEATURE_LEVEL fl[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    UINT flags = debug ? D3D11_CREATE_DEVICE_DEBUG : 0;
    HRESULT hr = D3D11CreateDevice(nvidia.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, fl, 2, D3D11_SDK_VERSION,
                                   &g.dev, nullptr, &g.ctx);
    if (FAILED(hr) && debug) {
        Log("Debug layer unavailable (0x%08lX), continuing without it.\n", static_cast<unsigned long>(hr));
        hr = D3D11CreateDevice(nvidia.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, fl, 2, D3D11_SDK_VERSION, &g.dev,
                               nullptr, &g.ctx);
    }
    if (FAILED(hr)) {
        Log("D3D11CreateDevice failed: 0x%08lX\n", static_cast<unsigned long>(hr));
        return false;
    }
    if (debug) g.dev.As(&g.info);
    return true;
}

ComPtr<ID3DBlob> Compile(const std::string& src, const char* target, const char* name) {
    ComPtr<ID3DBlob> code, err;
    HRESULT hr = D3DCompile(src.data(), src.size(), name, nullptr, nullptr, "main", target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr)) {
        Log("  Compile of %s failed: %s\n", name, err ? static_cast<const char*>(err->GetBufferPointer()) : "?");
        return nullptr;
    }
    return code;
}

void SetSemantic(NV_CUSTOM_SEMANTIC& s, NV_CUSTOM_SEMANTIC_TYPE type, const char* name) {
    memset(&s, 0, sizeof(s));
    s.version = NV_CUSTOM_SEMANTIC_VERSION;
    s.NVCustomSemanticType = type;
    strcpy_s(s.NVCustomSemanticNameString, name);
}

// Vertex shader for N views. Outputs for view 0 come first so a normal pixel shader links against them.
std::string MvrVsSource(int views, bool viewportMask, bool viewId) {
    std::string s =
        "cbuffer Views : register(b0) { float4 Rows[16]; };\n"
        "cbuffer Obj : register(b1) { float4 Offset; };\n"
        "float4 Xf(uint v, float4 p) { return float4(dot(Rows[v*4+0], p), dot(Rows[v*4+1], p),"
        " dot(Rows[v*4+2], p), dot(Rows[v*4+3], p)); }\n"
        "struct VSOut {\n"
        "  float4 pos0 : SV_Position;\n";
    if (viewId) s += "  float4 vid0 : NV_VIEWID_VIEW_0_SEMANTIC;\n";
    if (viewportMask) s += "  uint vpm0 : NV_VIEWPORT_MASK_VIEW_0_SEMANTIC;\n";
    for (int v = 1; v < views; ++v) {
        const std::string n = std::to_string(v);
        s += "  float4 pos" + n + " : NV_POSITION_VIEW_" + n + "_SEMANTIC;\n";
        if (viewId) s += "  float4 vid" + n + " : NV_VIEWID_VIEW_" + n + "_SEMANTIC;\n";
        if (viewportMask) s += "  uint vpm" + n + " : NV_VIEWPORT_MASK_VIEW_" + n + "_SEMANTIC;\n";
    }
    s += "};\nVSOut main(float4 pos : POSITION) {\n  VSOut o;\n  float4 p = pos + Offset;\n";
    for (int v = 0; v < views; ++v) {
        const std::string n = std::to_string(v);
        s += "  o.pos" + n + " = Xf(" + n + ", p);\n";
        if (viewId) s += "  o.vid" + n + " = float4(" + n + ", 0, 0, 1);\n";
        if (viewportMask) s += "  o.vpm" + n + " = " + std::to_string(1u << v) + ";\n";
    }
    s += "  return o;\n}\n";
    return s;
}

// Pixel shader: R = view id * 64, G = 255 marks coverage.
const char* kMvrPs =
    "struct PSIn { float4 pos : SV_Position; float4 vid : NV_VIEWID_VIEW_0_SEMANTIC; };\n"
    "float4 main(PSIn i) : SV_Target { return float4(i.vid.x * 64.0 / 255.0, 1, 0, 1); }\n";

const char* kPlainVs =
    "cbuffer Views : register(b0) { float4 Rows[16]; };\n"
    "cbuffer Obj : register(b1) { float4 Offset; };\n"
    "float4 main(float4 pos : POSITION) : SV_Position { float4 p = pos + Offset;\n"
    "  return float4(dot(Rows[0], p), dot(Rows[1], p), dot(Rows[2], p), dot(Rows[3], p)); }\n";
const char* kPlainPs = "float4 main(float4 pos : SV_Position) : SV_Target { return float4(0, 1, 0, 1); }\n";
// Pixel shader reading the render target array index the rasterizer used.
const char* kRtIndexPs =
    "float4 main(float4 pos : SV_Position, uint rt : SV_RenderTargetArrayIndex) : SV_Target"
    " { return float4(rt * 64.0 / 255.0, 1, 0, 1); }\n";

// B4: a single-view vertex shader shaped like the game's: more outputs, the position written in parts, read back
// and written on two paths with their own ret.
const char* kBranchyVs =
    "cbuffer Views : register(b0) { float4 Rows[16]; };\n"
    "cbuffer Obj : register(b1) { float4 Offset; };\n"
    "struct VSOut { float4 pos : SV_Position; float4 col : COLOR0; float2 uv : TEXCOORD0; };\n"
    "VSOut main(float4 pos : POSITION, uint id : SV_VertexID) {\n"
    "  VSOut o; float4 p = pos + Offset;\n"
    "  o.col = float4(id, 0, 0, 1); o.uv = p.xy;\n"
    "  [branch] if (id == 5) {\n"
    "    o.pos = float4(dot(Rows[0], p), dot(Rows[1], p), dot(Rows[2], p), dot(Rows[3], p));\n"
    "    return o;\n"
    "  }\n"
    "  o.pos.xy = float2(dot(Rows[0], p), dot(Rows[1], p));\n"
    "  o.uv += o.pos.xy;\n"
    "  o.pos.zw = float2(dot(Rows[2], p), dot(Rows[3], p));\n"
    "  return o;\n"
    "}\n";
const char* kBranchyRtIndexPs =
    "struct PSIn { float4 pos : SV_Position; float4 col : COLOR0; float2 uv : TEXCOORD0;"
    " uint rt : SV_RenderTargetArrayIndex; };\n"
    "float4 main(PSIn i) : SV_Target { return float4(i.rt * 64.0 / 255.0, 1, 0, 1); }\n";

// Constant buffer slot of the patch matrices in the B4 cases (the test shaders use b0 and b1).
constexpr UINT kPatchCbSlot = 2;

struct Shaders {
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11InputLayout> layout;
};

// How the pixel shader learns the view. "Attr*" modes pass the view id as a per-view attribute (NV_VIEWID_VIEW_n) and
// differ in which custom semantics are registered with NvAPI for the VS and the PS (Gen = NV_POSITION + NV_VIEWID,
// Pos = NV_POSITION only, Plain = PS created by plain CreatePixelShader). RtIndex reads SV_RenderTargetArrayIndex.
// AttrVsGenPsGen hung the GPU (TDR) on driver 581.80, so it runs only with --include-hang.
enum class ViewIdMode { None, AttrVsGenPsGen, AttrVsGenPsPlain, AttrVsGenPsPos, AttrVsPosPsPos, AttrVsPosPsPlain, RtIndex };

bool CreateMvrShaders(Gpu& g, int views, bool viewportMask, ViewIdMode idMode, Shaders& out) {
    using M = ViewIdMode;
    const bool viewId = idMode != M::None && idMode != M::RtIndex;
    const bool vsGen = idMode == M::AttrVsGenPsGen || idMode == M::AttrVsGenPsPlain || idMode == M::AttrVsGenPsPos;
    const bool psEx = idMode == M::AttrVsGenPsGen || idMode == M::AttrVsGenPsPos || idMode == M::AttrVsPosPsPos;
    const bool psGen = idMode == M::AttrVsGenPsGen;
    auto vsCode = Compile(MvrVsSource(views, viewportMask, viewId), "vs_5_0", "mvr_vs");
    auto psCode = Compile(viewId ? kMvrPs : idMode == M::RtIndex ? kRtIndexPs : kPlainPs, "ps_5_0", "mvr_ps");
    if (!vsCode || !psCode) return false;

    NV_CUSTOM_SEMANTIC sem[3];
    NvU32 n = 0;
    SetSemantic(sem[n++], NV_POSITION_SEMANTIC, "NV_POSITION");
    if (vsGen) SetSemantic(sem[n++], NV_GENERIC_ATTRIBUTE_SEMANTIC, "NV_VIEWID");
    if (viewportMask) SetSemantic(sem[n++], NV_VIEWPORT_MASK_SEMANTIC, "NV_VIEWPORT_MASK");

    NvAPI_D3D11_CREATE_VERTEX_SHADER_EX vsArgs = {};
    vsArgs.version = NVAPI_D3D11_CREATEVERTEXSHADEREX_VERSION;
    vsArgs.NumCustomSemantics = n;
    vsArgs.pCustomSemantics = sem;
    NvAPI_Status st = NvAPI_D3D11_CreateVertexShaderEx(g.dev.Get(), vsCode->GetBufferPointer(),
                                                       vsCode->GetBufferSize(), nullptr, &vsArgs, &out.vs);
    if (st != NVAPI_OK || !out.vs) {
        Log("  NvAPI_D3D11_CreateVertexShaderEx failed: %s\n", NvErr(st).c_str());
        return false;
    }
    if (psEx) {
        NV_CUSTOM_SEMANTIC psSem[1];
        if (psGen)
            SetSemantic(psSem[0], NV_GENERIC_ATTRIBUTE_SEMANTIC, "NV_VIEWID");
        else
            SetSemantic(psSem[0], NV_POSITION_SEMANTIC, "NV_POSITION");
        NvAPI_D3D11_CREATE_PIXEL_SHADER_EX psArgs = {};
        psArgs.version = NVAPI_D3D11_CREATEPIXELSHADEREX_VERSION;
        psArgs.NumCustomSemantics = 1;
        psArgs.pCustomSemantics = psSem;
        st = NvAPI_D3D11_CreatePixelShaderEx_2(g.dev.Get(), psCode->GetBufferPointer(), psCode->GetBufferSize(),
                                               nullptr, &psArgs, &out.ps);
        if (st != NVAPI_OK || !out.ps) {
            Log("  NvAPI_D3D11_CreatePixelShaderEx_2 failed: %s\n", NvErr(st).c_str());
            return false;
        }
    } else if (FAILED(g.dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr,
                                               &out.ps))) {
        Log("  CreatePixelShader failed\n");
        return false;
    }
    const D3D11_INPUT_ELEMENT_DESC ie = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,
                                         D3D11_INPUT_PER_VERTEX_DATA, 0};
    return SUCCEEDED(
        g.dev->CreateInputLayout(&ie, 1, vsCode->GetBufferPointer(), vsCode->GetBufferSize(), &out.layout));
}

// Which vertex shader a case uses: the HLSL MVR shader, or a single-view shader patched in DXBC (B4).
enum class VsKind { Hlsl, PatchedPlain, PatchedBranchy };

bool CreateVsEx(Gpu& g, const void* code, size_t size, ComPtr<ID3D11VertexShader>& vs) {
    NV_CUSTOM_SEMANTIC sem;
    SetSemantic(sem, NV_POSITION_SEMANTIC, "NV_POSITION");
    NvAPI_D3D11_CREATE_VERTEX_SHADER_EX args = {};
    args.version = NVAPI_D3D11_CREATEVERTEXSHADEREX_VERSION;
    args.NumCustomSemantics = 1;
    args.pCustomSemantics = &sem;
    const NvAPI_Status st = NvAPI_D3D11_CreateVertexShaderEx(g.dev.Get(), code, size, nullptr, &args, &vs);
    if (st != NVAPI_OK || !vs) {
        Log("  NvAPI_D3D11_CreateVertexShaderEx failed: %s\n", NvErr(st).c_str());
        return false;
    }
    return true;
}

bool CreatePatchedShaders(Gpu& g, int views, VsKind kind, bool rtIndexPs, Shaders& out) {
    auto vsCode = Compile(kind == VsKind::PatchedBranchy ? kBranchyVs : kPlainVs, "vs_5_0", "patched_vs");
    auto psCode = Compile(kind == VsKind::PatchedBranchy ? kBranchyRtIndexPs
                          : rtIndexPs                    ? kRtIndexPs
                                                         : kPlainPs,
                          "ps_5_0", "patched_ps");
    if (!vsCode || !psCode) return false;
    qvfr::dxbc::MvrVsPatchOptions opt;
    opt.cbSlot = kPatchCbSlot;
    opt.extraViews = static_cast<uint32_t>(views - 1);
    std::vector<uint8_t> patched;
    qvfr::dxbc::MvrVsPatchInfo info;
    std::string err;
    if (!qvfr::dxbc::PatchVsForMvr(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), opt, patched, &info, &err)) {
        Log("  patch failed: %s\n", err.c_str());
        return false;
    }
    Log("  patch: position o%u -> r%u (%u writes), %u epilogue(s), new outputs from o%u, %zu -> %zu bytes\n",
        info.positionReg, info.tempReg, info.rewrittenOperands, info.returns, info.firstNewOutputReg,
        static_cast<size_t>(vsCode->GetBufferSize()), patched.size());
    if (!CreateVsEx(g, patched.data(), patched.size(), out.vs)) return false;
    if (FAILED(g.dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &out.ps)))
        return false;
    const D3D11_INPUT_ELEMENT_DESC ie = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,
                                         D3D11_INPUT_PER_VERTEX_DATA, 0};
    return SUCCEEDED(
        g.dev->CreateInputLayout(&ie, 1, vsCode->GetBufferPointer(), vsCode->GetBufferSize(), &out.layout));
}

bool CreatePlainShaders(Gpu& g, Shaders& out) {
    auto vsCode = Compile(kPlainVs, "vs_5_0", "plain_vs");
    auto psCode = Compile(kPlainPs, "ps_5_0", "plain_ps");
    if (!vsCode || !psCode) return false;
    if (FAILED(g.dev->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &out.vs)))
        return false;
    if (FAILED(g.dev->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &out.ps)))
        return false;
    const D3D11_INPUT_ELEMENT_DESC ie = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,
                                         D3D11_INPUT_PER_VERTEX_DATA, 0};
    return SUCCEEDED(
        g.dev->CreateInputLayout(&ie, 1, vsCode->GetBufferPointer(), vsCode->GetBufferSize(), &out.layout));
}

ComPtr<ID3D11Buffer> MakeBuffer(Gpu& g, UINT bytes, UINT bind, D3D11_USAGE usage, const void* init) {
    D3D11_BUFFER_DESC d = {};
    d.ByteWidth = bytes;
    d.BindFlags = bind;
    d.Usage = usage;
    d.CPUAccessFlags = usage == D3D11_USAGE_DYNAMIC ? D3D11_CPU_ACCESS_WRITE : 0;
    D3D11_SUBRESOURCE_DATA sd = {init, 0, 0};
    ComPtr<ID3D11Buffer> b;
    g.dev->CreateBuffer(&d, init ? &sd : nullptr, &b);
    return b;
}

void WriteBuffer(Gpu& g, ID3D11Buffer* b, const void* data, size_t bytes) {
    D3D11_MAPPED_SUBRESOURCE m;
    if (SUCCEEDED(g.ctx->Map(b, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        memcpy(m.pData, data, bytes);
        g.ctx->Unmap(b, 0);
    }
}

NvAPI_Status SetMultiView(Gpu& g, int views, bool independentViewportMask, bool offsetsPerView) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = static_cast<NvU32>(views);
    for (int v = 0; v < NV_MULTIVIEW_MAX_SUPPORTED_VIEWS; ++v)
        p.renderTargetIndexOffset[v] = offsetsPerView ? static_cast<NvU32>(v) : 0;
    p.independentViewportMaskEnable = independentViewportMask ? 1 : 0;
    return NvAPI_D3D_SetMultiViewMode(g.ctx.Get(), &p);
}

ComPtr<ID3D11RasterizerState> NoCull(Gpu& g) {
    D3D11_RASTERIZER_DESC r = {};
    r.FillMode = D3D11_FILL_SOLID;
    r.CullMode = D3D11_CULL_NONE;
    r.DepthClipEnable = TRUE;
    r.MultisampleEnable = TRUE;
    ComPtr<ID3D11RasterizerState> rs;
    g.dev->CreateRasterizerState(&r, &rs);
    return rs;
}

// Render target: a texture array (arraySize slices of w x h) with an RTV over all slices and one RTV per slice.
struct Target {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11RenderTargetView> all;
    std::vector<ComPtr<ID3D11RenderTargetView>> slice;
    int w = 0, h = 0, slices = 0, samples = 1;
};

bool MakeTarget(Gpu& g, int w, int h, int slices, int samples, Target& t) {
    t.w = w;
    t.h = h;
    t.slices = slices;
    t.samples = samples;
    D3D11_TEXTURE2D_DESC d = {};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = slices;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = samples;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(g.dev->CreateTexture2D(&d, nullptr, &t.tex))) return false;
    auto rtv = [&](int first, int count, ComPtr<ID3D11RenderTargetView>& out) {
        D3D11_RENDER_TARGET_VIEW_DESC r = {};
        r.Format = d.Format;
        if (samples > 1) {
            r.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
            r.Texture2DMSArray.FirstArraySlice = first;
            r.Texture2DMSArray.ArraySize = count;
        } else {
            r.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
            r.Texture2DArray.FirstArraySlice = first;
            r.Texture2DArray.ArraySize = count;
        }
        return SUCCEEDED(g.dev->CreateRenderTargetView(t.tex.Get(), &r, &out));
    };
    if (!rtv(0, slices, t.all)) return false;
    t.slice.resize(slices);
    for (int s = 0; s < slices; ++s)
        if (!rtv(s, 1, t.slice[s])) return false;
    return true;
}

// Copies the target (resolving MSAA) to CPU memory, one RGBA8 image per slice.
std::vector<std::vector<uint32_t>> Readback(Gpu& g, const Target& t) {
    D3D11_TEXTURE2D_DESC d = {};
    t.tex->GetDesc(&d);
    ComPtr<ID3D11Texture2D> src = t.tex;
    if (t.samples > 1) {
        D3D11_TEXTURE2D_DESC r = d;
        r.SampleDesc.Count = 1;
        r.BindFlags = 0;
        ComPtr<ID3D11Texture2D> resolved;
        g.dev->CreateTexture2D(&r, nullptr, &resolved);
        for (int s = 0; s < t.slices; ++s) {
            const UINT sub = D3D11CalcSubresource(0, s, 1);
            g.ctx->ResolveSubresource(resolved.Get(), sub, t.tex.Get(), sub, d.Format);
        }
        src = resolved;
    }
    D3D11_TEXTURE2D_DESC sd = d;
    sd.SampleDesc.Count = 1;
    sd.BindFlags = 0;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    g.dev->CreateTexture2D(&sd, nullptr, &staging);
    g.ctx->CopyResource(staging.Get(), src.Get());
    std::vector<std::vector<uint32_t>> out(t.slices, std::vector<uint32_t>(static_cast<size_t>(t.w) * t.h));
    for (int s = 0; s < t.slices; ++s) {
        D3D11_MAPPED_SUBRESOURCE m;
        const UINT sub = D3D11CalcSubresource(0, s, 1);
        if (FAILED(g.ctx->Map(staging.Get(), sub, D3D11_MAP_READ, 0, &m))) continue;
        for (int y = 0; y < t.h; ++y)
            memcpy(&out[s][static_cast<size_t>(y) * t.w], static_cast<const char*>(m.pData) + y * m.RowPitch,
                   static_cast<size_t>(t.w) * 4);
        g.ctx->Unmap(staging.Get(), sub);
    }
    return out;
}

struct Box {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // [x0, x1) x [y0, y1)
    bool empty = true;
};

// Pixels whose centers fall inside the transformed rectangle (top-left rule is irrelevant at 1 px tolerance).
Box ExpectedBox(const ViewXf& v, int w, int h) {
    float xa = v.ax * kRectX0 + v.bx, xb = v.ax * kRectX1 + v.bx;
    float ya = v.ay * kRectY0 + v.by, yb = v.ay * kRectY1 + v.by;
    float nx0 = std::max(-1.0f, std::min(xa, xb)), nx1 = std::min(1.0f, std::max(xa, xb));
    float ny0 = std::max(-1.0f, std::min(ya, yb)), ny1 = std::min(1.0f, std::max(ya, yb));
    Box b;
    if (nx0 >= nx1 || ny0 >= ny1) return b;
    b.empty = false;
    b.x0 = static_cast<int>(std::lround((nx0 + 1) * 0.5f * w));
    b.x1 = static_cast<int>(std::lround((nx1 + 1) * 0.5f * w));
    b.y0 = static_cast<int>(std::lround((1 - ny1) * 0.5f * h));
    b.y1 = static_cast<int>(std::lround((1 - ny0) * 0.5f * h));
    return b;
}

// Measures the covered box in a w x h region of an image (region starts at column xOff) and the view ids seen.
Box MeasureBox(const std::vector<uint32_t>& img, int stride, int xOff, int w, int h, std::set<int>& ids) {
    Box b;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint32_t px = img[static_cast<size_t>(y) * stride + xOff + x];
            const int r = px & 0xFF, gch = (px >> 8) & 0xFF;
            if (gch < 128) continue;
            ids.insert((r + 32) / 64);
            if (b.empty) {
                b = {x, y, x + 1, y + 1, false};
            } else {
                b.x0 = std::min(b.x0, x);
                b.y0 = std::min(b.y0, y);
                b.x1 = std::max(b.x1, x + 1);
                b.y1 = std::max(b.y1, y + 1);
            }
        }
    return b;
}

bool Near(const Box& a, const Box& b) {
    if (a.empty || b.empty) return a.empty == b.empty;
    return std::abs(a.x0 - b.x0) <= 1 && std::abs(a.x1 - b.x1) <= 1 && std::abs(a.y0 - b.y0) <= 1 &&
           std::abs(a.y1 - b.y1) <= 1;
}

struct CaseCfg {
    const char* name;
    int views;
    bool viewportMask;  // views side by side in one wide texture instead of array slices
    int samples;
    ViewIdMode idMode;
    bool enableMvr;  // false: MVR shaders, but multi-view mode stays off (only view 0 must appear)
    VsKind vs = VsKind::Hlsl;
};

constexpr int kW = 256, kH = 256;

// Draws the rectangle with the case's shaders and returns one image per slice.
bool DrawCase(Gpu& g, const CaseCfg& c, std::vector<std::vector<uint32_t>>& img, Target& t) {
    Shaders sh;
    const bool ok = c.vs == VsKind::Hlsl
                        ? CreateMvrShaders(g, c.views, c.viewportMask, c.idMode, sh)
                        : CreatePatchedShaders(g, c.views, c.vs, c.idMode == ViewIdMode::RtIndex, sh);
    if (!ok) {
        DrainDebugMessages(g);
        Log("  shader creation failed\n");
        return false;
    }
    if (!MakeTarget(g, c.viewportMask ? kW * c.views : kW, kH, c.viewportMask ? 1 : c.views, c.samples, t)) {
        Log("  target creation failed\n");
        return false;
    }
    const float verts[6][4] = {{kRectX0, kRectY0, 0.5f, 1}, {kRectX0, kRectY1, 0.5f, 1}, {kRectX1, kRectY1, 0.5f, 1},
                               {kRectX0, kRectY0, 0.5f, 1}, {kRectX1, kRectY1, 0.5f, 1}, {kRectX1, kRectY0, 0.5f, 1}};
    auto vb = MakeBuffer(g, sizeof(verts), D3D11_BIND_VERTEX_BUFFER, D3D11_USAGE_IMMUTABLE, verts);
    // HLSL shaders: Rows[v*4 + r] = row r of view v. The patched shaders read only view 0 from there and get
    // M_n = (view n) * (view 0)^-1 in the patch buffer; view 0 is the identity, so M_n is view n.
    float rows[64] = {};
    for (int v = 0; v < 4; ++v) ViewRows(kViews[v], &rows[v * 16]);
    auto viewCb = MakeBuffer(g, sizeof(rows), D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_IMMUTABLE, rows);
    auto patchCb = MakeBuffer(g, sizeof(rows) - 64, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_IMMUTABLE, rows + 16);
    const float zero[4] = {};
    auto objCb = MakeBuffer(g, sizeof(zero), D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_IMMUTABLE, zero);
    auto rs = NoCull(g);

    const UINT stride = 16, offset = 0;
    g.ctx->ClearState();
    g.ctx->IASetInputLayout(sh.layout.Get());
    g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g.ctx->IASetVertexBuffers(0, 1, vb.GetAddressOf(), &stride, &offset);
    ID3D11Buffer* cbs[3] = {viewCb.Get(), objCb.Get(), patchCb.Get()};
    g.ctx->VSSetConstantBuffers(0, c.vs == VsKind::Hlsl ? 2 : kPatchCbSlot + 1, cbs);
    g.ctx->VSSetShader(sh.vs.Get(), nullptr, 0);
    g.ctx->PSSetShader(sh.ps.Get(), nullptr, 0);
    g.ctx->RSSetState(rs.Get());
    std::vector<D3D11_VIEWPORT> vps(c.viewportMask ? c.views : 1);
    for (size_t i = 0; i < vps.size(); ++i)
        vps[i] = {static_cast<float>(i * kW), 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
    g.ctx->RSSetViewports(static_cast<UINT>(vps.size()), vps.data());
    g.ctx->OMSetRenderTargets(1, t.all.GetAddressOf(), nullptr);
    const float clear[4] = {0, 0, 0, 0};
    g.ctx->ClearRenderTargetView(t.all.Get(), clear);

    if (c.enableMvr) {
        NvAPI_Status st = SetMultiView(g, c.views, c.viewportMask, !c.viewportMask);
        if (st != NVAPI_OK) Log("  NvAPI_D3D_SetMultiViewMode: %s\n", NvErr(st).c_str());
    }
    g.ctx->Draw(6, 0);
    if (c.enableMvr) SetMultiView(g, 1, false, false);
    g.ctx->Flush();
    DrainDebugMessages(g);
    img = Readback(g, t);
    return true;
}

bool RunCase(Gpu& g, const CaseCfg& c) {
    static const char* kIdNames[] = {"none",
                                     "attribute, VS NV_POSITION+NV_VIEWID, PS NV_VIEWID",
                                     "attribute, VS NV_POSITION+NV_VIEWID, plain PS",
                                     "attribute, VS NV_POSITION+NV_VIEWID, PS NV_POSITION",
                                     "attribute, VS NV_POSITION, PS NV_POSITION",
                                     "attribute, VS NV_POSITION, plain PS",
                                     "SV_RenderTargetArrayIndex in PS"};
    static const char* kVsNames[] = {"HLSL MVR", "patched plain", "patched branchy"};
    Log("\n[%s] views=%d layout=%s samples=%d view id=%s mvr=%s vs=%s\n", c.name, c.views,
        c.viewportMask ? "viewports" : "array-slices", c.samples, kIdNames[static_cast<int>(c.idMode)],
        c.enableMvr ? "on" : "off", kVsNames[static_cast<int>(c.vs)]);
    std::vector<std::vector<uint32_t>> img;
    Target t;
    if (!DrawCase(g, c, img, t)) {
        Log("  RESULT: FAIL (setup)\n");
        return false;
    }
    bool ok = true;
    for (int v = 0; v < c.views; ++v) {
        std::set<int> ids;
        const int slice = c.viewportMask ? 0 : v;
        const Box got = MeasureBox(img[slice], t.w, c.viewportMask ? v * kW : 0, kW, kH, ids);
        const Box exp = (c.enableMvr || v == 0) ? ExpectedBox(kViews[v], kW, kH) : Box{};
        const bool boxOk = Near(got, exp);
        const bool idOk = c.idMode == ViewIdMode::None ? true
                          : exp.empty                  ? ids.empty()
                                                       : (ids.size() == 1 && *ids.begin() == v);
        std::string idStr;
        for (int id : ids) idStr += (idStr.empty() ? "" : ",") + std::to_string(id);
        Log("  view %d: expected [%d,%d)x[%d,%d)%s, got [%d,%d)x[%d,%d)%s -> box %s, view id seen {%s} -> %s\n", v,
            exp.x0, exp.x1, exp.y0, exp.y1, exp.empty ? " (empty)" : "", got.x0, got.x1, got.y0, got.y1,
            got.empty ? " (empty)" : "", boxOk ? "OK" : "WRONG", idStr.c_str(), idOk ? "OK" : "WRONG");
        ok = ok && boxOk;
        ok = ok && idOk;
    }
    if (c.vs != VsKind::Hlsl) {
        // The patched shader must give exactly the image of the HLSL MVR shader with the same pixel shader output.
        CaseCfg ref = c;
        ref.vs = VsKind::Hlsl;
        std::vector<std::vector<uint32_t>> refImg;
        Target refT;
        Log("  reference (HLSL MVR shader):\n");
        if (!DrawCase(g, ref, refImg, refT)) {
            Log("  RESULT: FAIL (reference)\n");
            return false;
        }
        size_t diff = 0, total = 0;
        for (size_t s = 0; s < img.size(); ++s)
            for (size_t i = 0; i < img[s].size(); ++i, ++total) diff += img[s][i] != refImg[s][i];
        Log("  vs HLSL MVR shader: %zu of %zu pixels differ -> %s\n", diff, total, diff == 0 ? "OK" : "WRONG");
        ok = ok && diff == 0;
    }
    Log("  RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// B4: DXBC patch checks without drawing

std::vector<uint8_t> BlobBytes(ID3DBlob* b) {
    const auto* p = static_cast<const uint8_t*>(b->GetBufferPointer());
    return std::vector<uint8_t>(p, p + b->GetBufferSize());
}

// Checksum and rebuild against compiler output, and the patched output signature against the HLSL MVR shader's.
bool RunPatchChecks() {
    Log("\n[DXBC patch checks]\n");
    bool ok = true;
    const struct {
        const char* name;
        std::string src;
        const char* target;
    } shaders[] = {{"plain VS", kPlainVs, "vs_5_0"},
                   {"branchy VS", kBranchyVs, "vs_5_0"},
                   {"MVR VS, 2 views", MvrVsSource(2, false, false), "vs_5_0"},
                   {"MVR VS, 4 views", MvrVsSource(4, false, false), "vs_5_0"},
                   {"plain VS (vs_4_0)", kPlainVs, "vs_4_0"},
                   {"plain PS", kPlainPs, "ps_5_0"}};
    for (const auto& s : shaders) {
        auto code = Compile(s.src, s.target, s.name);
        if (!code) return false;
        const auto bytes = BlobBytes(code.Get());
        const bool sumOk = qvfr::dxbc::ChecksumMatches(bytes.data(), bytes.size());
        std::vector<qvfr::dxbc::Chunk> chunks;
        std::string err;
        const bool parsed = qvfr::dxbc::Parse(bytes.data(), bytes.size(), chunks, &err);
        const bool rebuildOk = parsed && qvfr::dxbc::Build(chunks) == bytes;
        Log("  %-20s %5zu bytes, %zu chunks: checksum %s, rebuild %s\n", s.name, bytes.size(), chunks.size(),
            sumOk ? "OK" : "WRONG", rebuildOk ? "identical" : ("DIFFERENT " + err).c_str());
        ok = ok && sumOk && rebuildOk;
    }

    // The patched output signature must declare the extra views as the compiler does for the HLSL MVR shader.
    for (int views : {2, 4}) {
        auto plain = Compile(kPlainVs, "vs_5_0", "plain_vs");
        auto ref = Compile(MvrVsSource(views, false, false), "vs_5_0", "mvr_vs");
        if (!plain || !ref) return false;
        qvfr::dxbc::MvrVsPatchOptions opt;
        opt.cbSlot = kPatchCbSlot;
        opt.extraViews = static_cast<uint32_t>(views - 1);
        std::vector<uint8_t> patched;
        std::string err;
        if (!qvfr::dxbc::PatchVsForMvr(plain->GetBufferPointer(), plain->GetBufferSize(), opt, patched, nullptr,
                                       &err)) {
            Log("  patch failed: %s\n", err.c_str());
            return false;
        }
        auto sig = [](const void* p, size_t n) {
            std::vector<qvfr::dxbc::Chunk> ch;
            std::vector<qvfr::dxbc::SigElement> el;
            qvfr::dxbc::Parse(p, n, ch, nullptr);
            if (auto* o = qvfr::dxbc::Find(ch, qvfr::dxbc::FourCC('O', 'S', 'G', 'N')))
                qvfr::dxbc::ReadSignature(*o, el, nullptr);
            return el;
        };
        const auto a = sig(patched.data(), patched.size()), b = sig(ref->GetBufferPointer(), ref->GetBufferSize());
        bool same = a.size() == b.size();
        for (size_t i = 0; same && i < a.size(); ++i)
            same = a[i].name == b[i].name && a[i].semanticIndex == b[i].semanticIndex &&
                   a[i].systemValue == b[i].systemValue && a[i].componentType == b[i].componentType &&
                   a[i].reg == b[i].reg && a[i].mask == b[i].mask && a[i].rwMask == b[i].rwMask;
        Log("  output signature, %d views: %s\n", views, same ? "same as HLSL MVR shader" : "DIFFERENT");
        if (!same) {
            for (const auto* list : {&a, &b}) {
                Log("    %s:\n", list == &a ? "patched" : "HLSL");
                for (const auto& e : *list)
                    Log("      %s%u sv=%u type=%u o%u mask=%x rw=%x\n", e.name.c_str(), e.semanticIndex, e.systemValue,
                        e.componentType, e.reg, e.mask, e.rwMask);
            }
        }
        ok = ok && same;
        if (views == 2 && g_verbose) {
            ComPtr<ID3DBlob> dis;
            if (SUCCEEDED(D3DDisassemble(patched.data(), patched.size(), 0, nullptr, &dis)))
                Log("  patched plain VS, 2 views:\n%s\n", static_cast<const char*>(dis->GetBufferPointer()));
            else
                Log("  D3DDisassemble of the patched shader failed\n");
        }
    }
    Log("  RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Patches every *.dxbc vertex shader in a folder and creates the result; writes the disassembly of each patched
// shader to outDir when given.
int RunPatchDir(Gpu& g, const std::string& dir, uint32_t slot, const std::string& outDir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".dxbc") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (!outDir.empty()) fs::create_directories(outDir);
    Log("\n[patch dir] %zu shaders in %s, constant buffer slot %u\n", files.size(), dir.c_str(), slot);
    int okCount = 0, skipped = 0;
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        std::vector<uint8_t> code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string name = f.filename().string();
        if (code.size() < 8 || (code[0] != 'D' || code[1] != 'X')) {
            Log("  %s: not DXBC, skipped\n", name.c_str());
            ++skipped;
            continue;
        }
        const bool sumOk = qvfr::dxbc::ChecksumMatches(code.data(), code.size());
        ComPtr<ID3D11VertexShader> orig;
        const bool origOk = SUCCEEDED(g.dev->CreateVertexShader(code.data(), code.size(), nullptr, &orig));
        qvfr::dxbc::MvrVsPatchOptions opt;
        opt.cbSlot = slot;
        std::vector<uint8_t> patched;
        qvfr::dxbc::MvrVsPatchInfo info;
        std::string err;
        if (!qvfr::dxbc::PatchVsForMvr(code.data(), code.size(), opt, patched, &info, &err)) {
            Log("  %s: checksum %s, original %s, PATCH FAILED: %s\n", name.c_str(), sumOk ? "ok" : "WRONG",
                origOk ? "ok" : "REJECTED", err.c_str());
            continue;
        }
        ComPtr<ID3D11VertexShader> vs;
        const bool exOk = CreateVsEx(g, patched.data(), patched.size(), vs);
        ComPtr<ID3DBlob> dis;
        const bool disOk = SUCCEEDED(D3DDisassemble(patched.data(), patched.size(), 0, nullptr, &dis));
        if (disOk && !outDir.empty()) {
            std::ofstream o(fs::path(outDir) / (f.stem().string() + ".patched.txt"));
            o.write(static_cast<const char*>(dis->GetBufferPointer()),
                    static_cast<std::streamsize>(strnlen(static_cast<const char*>(dis->GetBufferPointer()),
                                                         dis->GetBufferSize())));
        }
        const bool ok = sumOk && origOk && exOk && disOk;
        okCount += ok;
        if (!ok || g_verbose)
            Log("  %s: checksum %s, original %s, patch o%u->r%u (%u writes, %u ret), CreateVertexShaderEx %s, "
                "disassembly %s\n",
                name.c_str(), sumOk ? "ok" : "WRONG", origOk ? "ok" : "REJECTED", info.positionReg, info.tempReg,
                info.rewrittenOperands, info.returns, exOk ? "ok" : "FAILED", disOk ? "ok" : "FAILED");
    }
    DrainDebugMessages(g);
    Log("  patched and created: %d of %zu (%d skipped)\n", okCount, files.size() - skipped, skipped);
    return okCount == static_cast<int>(files.size()) - skipped ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Performance

struct PerfCfg {
    int views;
    int samples;
    bool mvr;
    int draws;
    int trisPerDraw;
    int frames;
    int w, h;
};

struct PerfResult {
    double submitMs = 0;   // median CPU time of the submission on this thread
    double processMs = 0;  // CPU time of the whole process (all threads, incl. the driver's) per frame
    double gpuMs = 0;      // median GPU time of the frame
};

// CPU time of all threads of the process (ours and the driver's), from the cycle counter. GetProcessTimes only moves
// on scheduler ticks, which is too coarse for frames of a fraction of a millisecond.
double CyclesPerMs() {
    static double cpm = 0;
    if (cpm == 0) {
        LARGE_INTEGER f, a, b;
        QueryPerformanceFrequency(&f);
        ULONG64 c0 = 0, c1 = 0;
        QueryThreadCycleTime(GetCurrentThread(), &c0);
        QueryPerformanceCounter(&a);
        do QueryPerformanceCounter(&b);
        while (b.QuadPart - a.QuadPart < f.QuadPart / 5);  // busy 200 ms
        QueryThreadCycleTime(GetCurrentThread(), &c1);
        cpm = static_cast<double>(c1 - c0) / (1000.0 * static_cast<double>(b.QuadPart - a.QuadPart) / f.QuadPart);
    }
    return cpm;
}

double ProcessCpuMs() {
    ULONG64 cycles = 0;
    QueryProcessCycleTime(GetCurrentProcess(), &cycles);
    return static_cast<double>(cycles) / CyclesPerMs();
}

// Waits for a query without burning CPU (the process CPU time is part of the measurement).
template <typename T>
void WaitData(Gpu& g, ID3D11Query* q, T& out) {
    while (g.ctx->GetData(q, &out, sizeof(out), 0) == S_FALSE) Sleep(1);
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool RunPerf(Gpu& g, const PerfCfg& c, PerfResult& r) {
    Shaders sh;
    if (c.mvr ? !CreateMvrShaders(g, c.views, false, ViewIdMode::None, sh) : !CreatePlainShaders(g, sh)) return false;
    Target t;
    if (!MakeTarget(g, c.w, c.h, c.views, c.samples, t)) return false;

    // One small grid mesh, drawn at many offsets (one draw per "object").
    std::vector<float> verts;
    const int quads = std::max(1, c.trisPerDraw / 2);
    const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(quads))));
    const float size = 0.04f, cell = size / side;
    for (int i = 0; i < quads; ++i) {
        const float x = (i % side) * cell, y = (i / side) * cell;
        const float q[6][2] = {{x, y}, {x, y + cell}, {x + cell, y + cell}, {x, y}, {x + cell, y + cell}, {x + cell, y}};
        for (auto& p : q) verts.insert(verts.end(), {p[0], p[1], 0.5f, 1.0f});
    }
    const UINT vertCount = static_cast<UINT>(verts.size() / 4);
    auto vb = MakeBuffer(g, static_cast<UINT>(verts.size() * 4), D3D11_BIND_VERTEX_BUFFER, D3D11_USAGE_IMMUTABLE,
                         verts.data());
    std::vector<float> offsets(static_cast<size_t>(c.draws) * 4);
    unsigned seed = 12345;
    auto rnd = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / static_cast<float>(1u << 24);
    };
    for (int i = 0; i < c.draws; ++i) {
        offsets[i * 4 + 0] = rnd() * 1.8f - 0.92f;
        offsets[i * 4 + 1] = rnd() * 1.8f - 0.92f;
    }
    auto viewCb = MakeBuffer(g, 64 * 4, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, nullptr);
    auto objCb = MakeBuffer(g, 16, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, nullptr);
    auto rs = NoCull(g);
    // A texture and a sampler bound per draw, like a material, so the driver has bindings to track.
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = 64;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> matTex;
    ComPtr<ID3D11ShaderResourceView> matSrv;
    g.dev->CreateTexture2D(&td, nullptr, &matTex);
    g.dev->CreateShaderResourceView(matTex.Get(), nullptr, &matSrv);
    D3D11_SAMPLER_DESC smp = {};
    smp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    smp.AddressU = smp.AddressV = smp.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    smp.MaxLOD = D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler;
    g.dev->CreateSamplerState(&smp, &sampler);

    D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    ComPtr<ID3D11Query> disjoint, ts0, ts1;
    g.dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    g.dev->CreateQuery(&qd, &ts0);
    g.dev->CreateQuery(&qd, &ts1);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    std::vector<double> submit, gpu;
    const int warmup = 30;
    double cpu0 = 0;
    const UINT stride = 16, offset = 0;
    const D3D11_VIEWPORT vp = {0, 0, static_cast<float>(c.w), static_cast<float>(c.h), 0, 1};
    const float clear[4] = {0, 0, 0, 0};

    for (int f = 0; f < warmup + c.frames; ++f) {
        if (f == warmup) cpu0 = ProcessCpuMs();
        g.ctx->Begin(disjoint.Get());
        g.ctx->End(ts0.Get());
        LARGE_INTEGER a, b;
        QueryPerformanceCounter(&a);

        g.ctx->ClearRenderTargetView(t.all.Get(), clear);
        g.ctx->IASetInputLayout(sh.layout.Get());
        g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g.ctx->VSSetShader(sh.vs.Get(), nullptr, 0);
        g.ctx->PSSetShader(sh.ps.Get(), nullptr, 0);
        g.ctx->RSSetState(rs.Get());
        g.ctx->RSSetViewports(1, &vp);
        const int passes = c.mvr ? 1 : c.views;
        if (c.mvr) SetMultiView(g, c.views, false, true);
        for (int pass = 0; pass < passes; ++pass) {
            float rows[64] = {};
            if (c.mvr) {
                for (int v = 0; v < c.views; ++v) ViewRows(kViews[v], &rows[v * 16]);
            } else {
                ViewRows(kViews[pass], rows);
            }
            WriteBuffer(g, viewCb.Get(), rows, sizeof(rows));
            g.ctx->OMSetRenderTargets(1, c.mvr ? t.all.GetAddressOf() : t.slice[pass].GetAddressOf(), nullptr);
            for (int i = 0; i < c.draws; ++i) {
                // Per-draw work similar to a replayed engine command list: object constants, bindings, draw.
                WriteBuffer(g, objCb.Get(), &offsets[i * 4], 16);
                ID3D11Buffer* cbs[2] = {viewCb.Get(), objCb.Get()};
                g.ctx->VSSetConstantBuffers(0, 2, cbs);
                g.ctx->PSSetConstantBuffers(0, 1, objCb.GetAddressOf());
                g.ctx->PSSetShaderResources(0, 1, matSrv.GetAddressOf());
                g.ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
                g.ctx->IASetVertexBuffers(0, 1, vb.GetAddressOf(), &stride, &offset);
                g.ctx->Draw(vertCount, 0);
            }
        }
        if (c.mvr) SetMultiView(g, 1, false, false);

        QueryPerformanceCounter(&b);
        g.ctx->End(ts1.Get());
        g.ctx->End(disjoint.Get());
        g.ctx->Flush();
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
        WaitData(g, disjoint.Get(), dj);
        UINT64 t0 = 0, t1 = 0;
        WaitData(g, ts0.Get(), t0);
        WaitData(g, ts1.Get(), t1);
        if (f >= warmup) {
            submit.push_back(1000.0 * static_cast<double>(b.QuadPart - a.QuadPart) / static_cast<double>(freq.QuadPart));
            if (!dj.Disjoint && dj.Frequency) gpu.push_back(1000.0 * static_cast<double>(t1 - t0) / dj.Frequency);
        }
    }
    r.processMs = (ProcessCpuMs() - cpu0) / c.frames;
    r.submitMs = Median(submit);
    r.gpuMs = Median(gpu);
    DrainDebugMessages(g);
    return true;
}

int ArgInt(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    return def;
}

bool HasArg(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], name) == 0) return true;
    return false;
}

std::string ArgStr(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (HasArg(argc, argv, "--help")) {
        Log("qvfr_mvr_test [--debug] [--verbose] [--case N] [--include-hang] [--skip-perf] [--draws N] [--tris N]\n"
            "              [--frames N] [--width N] [--height N]\n"
            "qvfr_mvr_test --patch-dir DIR [--patch-slot N] [--patch-out DIR] [--verbose]\n");
        return 0;
    }
    const bool debug = HasArg(argc, argv, "--debug");
    g_verbose = HasArg(argc, argv, "--verbose");
    timeBeginPeriod(1);

    NvAPI_Status st = NvAPI_Initialize();
    if (st != NVAPI_OK) {
        Log("NvAPI_Initialize failed: %s\n", NvErr(st).c_str());
        return 2;
    }
    NvU32 driver = 0;
    NvAPI_ShortString branch = {};
    if (NvAPI_SYS_GetDriverAndBranchVersion(&driver, branch) == NVAPI_OK)
        Log("NVIDIA driver %u.%02u (%s)\n", driver / 100, driver % 100, branch);

    Gpu g;
    if (!CreateDevice(g, debug)) return 2;

    NV_QUERY_MULTIVIEW_SUPPORT_PARAMS q = {};
    q.version = NV_QUERY_MULTIVIEW_SUPPORT_PARAMS_VER;
    st = NvAPI_D3D_QueryMultiViewSupport(g.dev.Get(), &q);
    Log("NvAPI_D3D_QueryMultiViewSupport: %s, MultiView=%u, SinglePassStereo=%u, SinglePassStereoXYZW=%u\n",
        NvErr(st).c_str(), q.bMultiViewSupported, q.bSinglePassStereoSupported, q.bSinglePassStereoXYZWSupported);
    if (st != NVAPI_OK || !q.bMultiViewSupported) {
        Log("Multi-View Rendering is not supported here; stopping.\n");
        return 1;
    }

    const std::string patchDir = ArgStr(argc, argv, "--patch-dir");
    if (!patchDir.empty())
        return RunPatchDir(g, patchDir, static_cast<uint32_t>(ArgInt(argc, argv, "--patch-slot", 3)),
                           ArgStr(argc, argv, "--patch-out"));

    // Correctness, simplest first. Every case gets a fresh device, so a GPU hang in one does not hide the others.
    const CaseCfg cases[] = {
        {"MVR shader, mode off", 2, false, 1, ViewIdMode::None, false},
        {"2 views, positions only", 2, false, 1, ViewIdMode::None, true},
        {"4 views, positions only", 4, false, 1, ViewIdMode::None, true},
        {"2 views, MSAA 8x", 2, false, 8, ViewIdMode::None, true},
        {"4 views, MSAA 8x", 4, false, 8, ViewIdMode::None, true},
        {"2 views, viewports", 2, true, 1, ViewIdMode::None, true},
        {"view id: VS gen, plain PS", 2, false, 1, ViewIdMode::AttrVsGenPsPlain, true},
        {"view id: VS gen, PS pos", 2, false, 1, ViewIdMode::AttrVsGenPsPos, true},
        {"view id: VS pos, PS pos", 2, false, 1, ViewIdMode::AttrVsPosPsPos, true},
        {"view id: VS pos, plain PS", 2, false, 1, ViewIdMode::AttrVsPosPsPlain, true},
        {"view id: render target index", 2, false, 1, ViewIdMode::RtIndex, true},
        {"view id: render target index, 4 views, MSAA 8x", 4, false, 8, ViewIdMode::RtIndex, true},
        {"view id: VS gen, PS gen (hangs)", 2, false, 1, ViewIdMode::AttrVsGenPsGen, true},
        {"B4 patched VS, mode off", 2, false, 1, ViewIdMode::RtIndex, false, VsKind::PatchedPlain},
        {"B4 patched VS, 2 views", 2, false, 1, ViewIdMode::RtIndex, true, VsKind::PatchedPlain},
        {"B4 patched VS, 4 views, MSAA 8x", 4, false, 8, ViewIdMode::RtIndex, true, VsKind::PatchedPlain},
        {"B4 patched branchy VS, 2 views", 2, false, 1, ViewIdMode::RtIndex, true, VsKind::PatchedBranchy},
        {"B4 patched branchy VS, 4 views, MSAA 8x", 4, false, 8, ViewIdMode::RtIndex, true, VsKind::PatchedBranchy},
    };
    const int only = ArgInt(argc, argv, "--case", -1);
    const bool includeHang = HasArg(argc, argv, "--include-hang");
    int passed = 0, run = 0;
    if (only < 0) {
        passed += RunPatchChecks() ? 1 : 0;
        ++run;
    }
    for (int i = 0; i < static_cast<int>(std::size(cases)); ++i) {
        if (only >= 0 && i != only) continue;
        if (cases[i].idMode == ViewIdMode::AttrVsGenPsGen && !includeHang && only != i) continue;
        Gpu cg;
        if (!CreateDevice(cg, debug)) return 2;
        Log("\n#%d", i);
        passed += RunCase(cg, cases[i]) ? 1 : 0;
        ++run;
        if (cg.dev->GetDeviceRemovedReason() != S_OK) {
            Log("  Device removed: 0x%08lX\n", static_cast<unsigned long>(cg.dev->GetDeviceRemovedReason()));
            Sleep(3000);  // let the driver finish the recovery before the next device
        }
    }
    Log("\nCorrectness: %d of %d cases passed.\n", passed, run);
    const bool allPassed = passed == run;

    if (HasArg(argc, argv, "--skip-perf")) return allPassed ? 0 : 1;

    // Performance: the same object list drawn once per view (sequential) or once with MVR.
    PerfCfg base = {};
    base.draws = ArgInt(argc, argv, "--draws", 2000);
    base.trisPerDraw = ArgInt(argc, argv, "--tris", 400);
    base.frames = ArgInt(argc, argv, "--frames", 400);
    base.w = ArgInt(argc, argv, "--width", 2048);
    base.h = ArgInt(argc, argv, "--height", 1720);
    Log("\nPerformance: %d draws x %d triangles per view, %d frames, target %d x %d per view\n", base.draws,
        base.trisPerDraw, base.frames, base.w, base.h);
    Log("%-28s %14s %16s %10s\n", "case", "submit CPU ms", "process CPU ms", "GPU ms");
    const struct {
        int views, samples;
    } perfCases[] = {{2, 1}, {2, 8}, {4, 1}, {4, 8}};
    for (const auto& pc : perfCases) {
        for (bool mvr : {false, true}) {
            PerfCfg c = base;
            c.views = pc.views;
            c.samples = pc.samples;
            c.mvr = mvr;
            PerfResult r;
            char name[64];
            snprintf(name, sizeof(name), "%d views, %dx, %s", c.views, c.samples, mvr ? "MVR" : "sequential");
            if (!RunPerf(g, c, r)) {
                Log("%-28s failed\n", name);
                continue;
            }
            Log("%-28s %14.3f %16.3f %10.3f\n", name, r.submitMs, r.processMs, r.gpuMs);
        }
    }
    return allPassed ? 0 : 1;
}

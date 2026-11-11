// NR bridge (nr_bridge.h). The device, shared fence and texture setup and the NGX contract (feature 18, the driver
// core's capability parameter block, DLSSNR.* parameters, tuning at creation, the model driven directly through a
// forwarder when the core refuses the feature) follow nr-bridge's direct engine
// (https://github.com/lowpoplabs/nr-bridge, src/nr_direct.h, MIT License, Copyright (c) 2026 LowPopLabs).
#include "nr_bridge.h"

#include <windows.h>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef QVFR_HAVE_NGX_HEADERS
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#endif

#include "common.h"
#include "nr_depth_cs.h"   // kNrDepthCS (fxc from shaders/nr.hlsl)
#include "nr_down_cs.h"    // kNrDownCS
#include "nr_motion_cs.h"  // kNrMotionCS
#include "nr_pack_cs.h"    // kNrPackCS
#include "nr_unpack4_cs.h"  // kNrUnpack4CS
#include "nr_unpack_cs.h"   // kNrUnpackCS
#include "nr_upedit_cs.h"   // kNrUpEditCS

namespace qvfr::nrbridge {

namespace {

constexpr UINT kRing = 8;  // D3D12 submissions in flight before Process waits on the CPU
constexpr DWORD kWaitMs = 2000;
constexpr int kFeatureNr = 18;   // NVSDK_NGX_Feature_Reserved18 in the 310.9 SDK: Neural Rendering
constexpr int kNgxSdk = 0x15;    // NVSDK_NGX_Version_API of the 310.9 SDK
constexpr int kMaxModelFails = 20;  // consecutive failed evaluations before the bridge gives up
constexpr int kDiagAfter = 60;      // images after a model instance is made: its input and output measured once

float HalfToFloat(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = m * (1.0f / 16777216.0f);  // 2^-24
    else if (e == 31) v = m ? 0.0f : 65504.0f;  // NaN as 0, Inf as the largest half
    else v = ldexpf(1.0f + m / 1024.0f, (int)e - 15);
    return s ? -v : v;
}

// Mean, minimum and maximum of R, G, B and Rec. 709 luminance of an RGBA16F staging texture.
struct Mean {
    double rgb[3] = {}, lum = 0, lo = 1e30, hi = -1e30;
};

Mean Measure(ID3D11DeviceContext* c, ID3D11Texture2D* staging, UINT w, UINT h) {
    Mean r;
    D3D11_MAPPED_SUBRESOURCE map{};
    if (FAILED(c->Map(staging, 0, D3D11_MAP_READ, 0, &map))) return r;
    double n = 0;
    for (UINT y = 0; y < h; y += 2)
        for (UINT x = 0; x < w; x += 2) {
            const uint16_t* p = (const uint16_t*)((const uint8_t*)map.pData + (size_t)y * map.RowPitch) + x * 4;
            const float cr = HalfToFloat(p[0]), cg = HalfToFloat(p[1]), cb = HalfToFloat(p[2]);
            const double l = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
            r.rgb[0] += cr;
            r.rgb[1] += cg;
            r.rgb[2] += cb;
            r.lum += l;
            r.lo = l < r.lo ? l : r.lo;
            r.hi = l > r.hi ? l : r.hi;
            ++n;
        }
    c->Unmap(staging, 0);
    if (n > 0) {
        for (double& v : r.rgb) v /= n;
        r.lum /= n;
    }
    return r;
}

// R11G11B10_FLOAT channel: `bits` mantissa bits under a 5-bit exponent (bias 15).
float SmallFloat(uint32_t v, int bits) {
    const uint32_t m = v & ((1u << bits) - 1), e = (v >> bits) & 31;
    if (e == 0) return ldexpf((float)m, -14 - bits);
    if (e == 31) return m ? 0.0f : 65000.0f;
    return ldexpf(1.0f + (float)m / (float)(1u << bits), (int)e - 15);
}

// A mapped staging texture read as linear RGB (RGBA16F or R11G11B10_FLOAT; other formats: not readable).
struct Pixels {
    ID3D11DeviceContext* c = nullptr;
    ID3D11Texture2D* t = nullptr;
    D3D11_MAPPED_SUBRESOURCE map{};
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    bool ok = false;
    Pixels(ID3D11DeviceContext* ctx, ID3D11Texture2D* tex) : c(ctx), t(tex) {
        if (!t) return;
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        fmt = d.Format;
        if (fmt != DXGI_FORMAT_R16G16B16A16_FLOAT && fmt != DXGI_FORMAT_R11G11B10_FLOAT) return;
        ok = SUCCEEDED(c->Map(t, 0, D3D11_MAP_READ, 0, &map));
    }
    ~Pixels() {
        if (ok) c->Unmap(t, 0);
    }
    void Rgb(UINT x, UINT y, float* v) const {
        const uint8_t* row = (const uint8_t*)map.pData + (size_t)y * map.RowPitch;
        if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            const uint16_t* p = (const uint16_t*)row + (size_t)x * 4;
            for (int i = 0; i < 3; ++i) v[i] = HalfToFloat(p[i]);
        } else {
            const uint32_t p = ((const uint32_t*)row)[x];
            v[0] = SmallFloat(p & 0x7FF, 6);
            v[1] = SmallFloat((p >> 11) & 0x7FF, 6);
            v[2] = SmallFloat(p >> 22, 5);
        }
    }
};

// The model's edit: in its units (input -> output) and, when both are readable, in the HDR image (DLAA's result before
// and after the bridge). Every second pixel in x and y.
struct Edit {
    double n = 0, absLum = 0, signedLum = 0, absRgb[3] = {}, p99Lum = 0, maxLum = 0, over1 = 0, over8 = 0;
    double hdrN = 0, hdrBefore = 0, hdrAfter = 0, hdrAbs = 0, hdrP50 = 0, hdrP99 = 0, hdrOver2 = 0, hdrOver10 = 0;
};

double Percentile(const std::vector<uint32_t>& hist, double n, double q, double binWidth) {
    const double want = n * q;
    double sum = 0;
    for (size_t i = 0; i < hist.size(); ++i) {
        sum += hist[i];
        if (sum >= want) return (double)i * binWidth;  // the bin's lower edge: no change reads 0
    }
    return (double)hist.size() * binWidth;
}

Edit MeasureEdit(ID3D11DeviceContext* c, ID3D11Texture2D* in, ID3D11Texture2D* out, ID3D11Texture2D* before,
                 ID3D11Texture2D* after, UINT w, UINT h) {
    Edit e;
    const Pixels pi(c, in), po(c, out);
    if (!pi.ok || !po.ok) return e;
    const Pixels pb(c, before), pa(c, after);
    const bool hdr = pb.ok && pa.ok;
    constexpr double kLumBin = 1.0 / 4096, kRelBin = 0.0005;  // model units; HDR relative change (0.05%)
    std::vector<uint32_t> lumHist(4096), relHist(2000);
    // HDR: the per-pixel change relative to the pixel, with a floor of 1% of the frame's mean (black stays calm).
    double hdrMean = 0;
    if (hdr) {
        double s = 0, k = 0;
        for (UINT y = 0; y < h; y += 8)
            for (UINT x = 0; x < w; x += 8) {
                float b[3];
                pb.Rgb(x, y, b);
                s += 0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2];
                ++k;
            }
        hdrMean = k > 0 ? s / k : 0;
    }
    for (UINT y = 0; y < h; y += 2)
        for (UINT x = 0; x < w; x += 2) {
            float a[3], b[3];
            pi.Rgb(x, y, a);
            po.Rgb(x, y, b);
            const double la = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
            const double lb = 0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2];
            const double d = lb - la, ad = fabs(d);
            e.signedLum += d;
            e.absLum += ad;
            for (int i = 0; i < 3; ++i) e.absRgb[i] += fabs((double)b[i] - (double)a[i]);
            if (ad > e.maxLum) e.maxLum = ad;
            if (ad > 1.0 / 255) ++e.over1;
            if (ad > 8.0 / 255) ++e.over8;
            ++lumHist[std::min<size_t>((size_t)(ad / kLumBin), lumHist.size() - 1)];
            ++e.n;
            if (hdr) {
                float o[3], f[3];
                pb.Rgb(x, y, o);
                pa.Rgb(x, y, f);
                const double lo = 0.2126 * o[0] + 0.7152 * o[1] + 0.0722 * o[2];
                const double lf = 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
                const double rel = fabs(lf - lo) / std::max(lo, 0.01 * hdrMean + 1e-12);
                e.hdrBefore += lo;
                e.hdrAfter += lf;
                e.hdrAbs += fabs(lf - lo);
                if (rel > 0.02) ++e.hdrOver2;
                if (rel > 0.10) ++e.hdrOver10;
                ++relHist[std::min<size_t>((size_t)(rel / kRelBin), relHist.size() - 1)];
                ++e.hdrN;
            }
        }
    if (e.n > 0) {
        e.p99Lum = Percentile(lumHist, e.n, 0.99, kLumBin);
        e.absLum /= e.n;
        e.signedLum /= e.n;
        for (double& v : e.absRgb) v /= e.n;
        e.over1 /= e.n;
        e.over8 /= e.n;
    }
    if (e.hdrN > 0) {
        e.hdrP50 = Percentile(relHist, e.hdrN, 0.50, kRelBin);
        e.hdrP99 = Percentile(relHist, e.hdrN, 0.99, kRelBin);
        e.hdrBefore /= e.hdrN;
        e.hdrAfter /= e.hdrN;
        e.hdrAbs /= e.hdrN;
        e.hdrOver2 /= e.hdrN;
        e.hdrOver10 /= e.hdrN;
    }
    return e;
}

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

using CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

void Barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cl->ResourceBarrier(1, &b);
}

struct Consts {
    float inputScale;
    uint32_t tonemap;
    float mvScale;
    uint32_t transfer;
    uint32_t depthMode;
    float depthK;
    uint32_t debug;
    float pad;
    uint32_t offsetX, offsetY, regionW, regionH;  // CSMotion, CSDepth, CSDown, CSUpEdit: the region they use
    uint32_t srcStep, srcX, srcY, fade;  // source pixels per region pixel, the part's origin, CSUpEdit's fade (px)
};

#ifdef QVFR_HAVE_NGX_HEADERS
// The driver's NGX core (D3D12 entry points) and the forwarder's.
using CoreInitFn = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
                                                  const NVSDK_NGX_FeatureCommonInfo*);
using CoreShutdownFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
using CoreCapsFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using CoreCreateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*,
                                                    NVSDK_NGX_Handle**);
using CoreEvalFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                                                  const NVSDK_NGX_Parameter*, void*);
using CoreReleaseFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using FwdLoadFn = int(__cdecl*)(const wchar_t*);
using FwdInitFn = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
using FwdCreateFn = int(__cdecl*)(ID3D12GraphicsCommandList*, int, void*, void**);
using FwdEvalFn = int(__cdecl*)(ID3D12GraphicsCommandList*, void*, void*);
using FwdReleaseFn = int(__cdecl*)(void*);

std::atomic<int> g_ngxLines{0};

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    if (!message) return;
    std::string s(message);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty()) return;
    // The core dumps its whole configuration at init; keep what can explain a failure (as nr-bridge).
    std::string low = s;
    for (char& ch : low) ch = (char)tolower((unsigned char)ch);
    const bool keep = low.find("error") != std::string::npos || low.find("fail") != std::string::npos ||
                      low.find("warn") != std::string::npos || low.find("not supported") != std::string::npos ||
                      low.find("feature") != std::string::npos || low.find("snippet") != std::string::npos ||
                      low.find("dlssnr") != std::string::npos || low.find("called from module") != std::string::npos;
    if (!keep || g_ngxLines.fetch_add(1) >= 200) return;
    Log("NR NGX: %s", s.c_str());
}

const char* ResultName(unsigned r) {
    switch (r) {
        case 0x1: return "Success";
        case 0xBAD00000: return "Fail";
        case 0xBAD00001: return "FeatureNotSupported";
        case 0xBAD00002: return "PlatformError";
        case 0xBAD00003: return "FeatureAlreadyExists";
        case 0xBAD00004: return "FeatureNotFound";
        case 0xBAD00005: return "InvalidParameter";
        case 0xBAD00006: return "ScratchBufferTooSmall";
        case 0xBAD00007: return "NotInitialized";
        case 0xBAD00008: return "UnsupportedInputFormat";
        case 0xBAD00009: return "RWFlagMissing";
        case 0xBAD0000A: return "MissingInput";
        case 0xBAD0000B: return "UnableToInitializeFeature";
        case 0xBAD0000C: return "OutOfDate";
        case 0xBAD0000D: return "OutOfGPUMemory";
        case 0xBAD0000E: return "UnsupportedFormat";
        case 0xBAD0000F: return "UnableToWriteToAppDataPath";
        case 0xBAD00010: return "UnsupportedParameter";
        case 0xBAD00011: return "Denied";
        case 0xBAD00012: return "NotImplemented";
        default: return "?";
    }
}
#endif

}  // namespace

struct Bridge::Impl {
    ID3D11Device5* dev11 = nullptr;
    ID3D11DeviceContext* ctx = nullptr;  // not referenced: the context ctx4 was taken from
    ID3D11DeviceContext4* ctx4 = nullptr;
    ID3D11ComputeShader* pack = nullptr;
    ID3D11ComputeShader* unpack = nullptr;
    ID3D11ComputeShader* unpack4 = nullptr;  // into an image with alpha
    ID3D11ComputeShader* down = nullptr;     // an image into its atlas region at a lower resolution
    ID3D11ComputeShader* upEdit = nullptr;   // the model's edit of such a region onto the full image
    ID3D11ComputeShader* motionCs = nullptr;
    ID3D11ComputeShader* depthCs = nullptr;
    ID3D11Buffer* cb = nullptr;
    Consts cbValue{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    HMODULE d3d12 = nullptr;
    ID3D12Device* dev12 = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc[kRing] = {};
    UINT64 allocValue[kRing] = {};  // fence BA value of the submission that used it
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* ab12 = nullptr;  // D3D11 -> D3D12: the input is ready
    ID3D12Fence* ba12 = nullptr;  // D3D12 -> D3D11: the output is ready
    ID3D11Fence* ab11 = nullptr;
    ID3D11Fence* ba11 = nullptr;
    UINT64 vAB = 0, vBA = 0;
    HANDLE event = nullptr;
    ID3D12QueryHeap* tsHeap = nullptr;
    ID3D12Resource* tsRead = nullptr;
    UINT64 tsFreq = 0;
    bool tsPending[kRing] = {};
    int tsSlot[kRing] = {};  // the slot whose submission wrote the timestamps
    UINT ring = 0;
    struct Slot {
        UINT w = 0, h = 0;
        UINT guideW = 0, guideH = 0;  // the last Guides call (0: none)
        bool guideDepth = false;      // it wrote the game's depth
        ID3D11Texture2D* color11 = nullptr;   // written by CSPack
        ID3D11Texture2D* out11 = nullptr;     // read by CSUnpack
        ID3D11Texture2D* motion11 = nullptr;  // written by CSMotion
        ID3D11Texture2D* depth11 = nullptr;   // cleared to the constant depth
        ID3D11Texture2D* orig11 = nullptr;    // transfer 1: the image as DLAA left it (not shared)
        ID3D11ShaderResourceView* origSrv = nullptr;
        ID3D11ShaderResourceView* colorSrv = nullptr;  // the model's input, read back by CSUnpack (transfer 1)
        ID3D11UnorderedAccessView* colorUav = nullptr;
        ID3D11UnorderedAccessView* motionUav = nullptr;
        ID3D11UnorderedAccessView* depthUav = nullptr;
        ID3D11ShaderResourceView* outSrv = nullptr;
        ID3D12Resource* color12 = nullptr;
        ID3D12Resource* out12 = nullptr;
        ID3D12Resource* motion12 = nullptr;
        ID3D12Resource* depth12 = nullptr;
        float depthValue = -1;
        void* feature = nullptr;  // NVSDK_NGX_Handle*
        bool fresh = true;
        int fails = 0;
        int diagLeft = -1;  // images until the model's input and output are measured once (-1: not planned)
    } slots[kSlots];
    // NGX
    bool ngxTried = false, ngxOk = false;
    int path = 0;  // 0 none yet, 1 the driver's core, 2 the model through the forwarder
    HMODULE core = nullptr, fwd = nullptr;
    bool coreInited = false, modelInited = false;
    void* caps = nullptr;  // NVSDK_NGX_Parameter* (the core's capability block)
#ifdef QVFR_HAVE_NGX_HEADERS
    CoreInitFn coreInit = nullptr;
    CoreShutdownFn coreShutdown = nullptr;
    CoreCapsFn coreCaps = nullptr;
    CoreCreateFn coreCreate = nullptr;
    CoreEvalFn coreEval = nullptr;
    CoreReleaseFn coreRelease = nullptr;
    FwdLoadFn fwdLoad = nullptr;
    FwdInitFn fwdInit = nullptr;
    FwdCreateFn fwdCreate = nullptr;
    FwdEvalFn fwdEval = nullptr;
    FwdReleaseFn fwdRelease = nullptr;
    std::wstring searchDir;
    const wchar_t* paths[1] = {};
    NVSDK_NGX_FeatureCommonInfo info{};
#endif

    void ReleaseSlot(Slot& s) {
        Rel(s.colorUav);
        Rel(s.motionUav);
        Rel(s.depthUav);
        Rel(s.outSrv);
        Rel(s.origSrv);
        Rel(s.colorSrv);
        Rel(s.orig11);
        Rel(s.color11);
        Rel(s.out11);
        Rel(s.motion11);
        Rel(s.depth11);
        Rel(s.color12);
        Rel(s.out12);
        Rel(s.motion12);
        Rel(s.depth12);
        s.w = s.h = 0;
        s.depthValue = -1;
    }
};

bool Bridge::Fail(const char* what, long hr) {
    if (!failed_) {
        char buf[320];
        if (hr)
            snprintf(buf, sizeof(buf), "%s (0x%08lX)", what, (unsigned long)hr);
        else
            snprintf(buf, sizeof(buf), "%s", what);
        why_ = buf;
        Log("NR bridge: DISABLED: %s", buf);
    }
    failed_ = true;
    ready_ = false;
    return false;
}

const char* Bridge::modelPath() const {
    if (!set_.model) return "passthrough";
    if (!impl_ || !impl_->path) return "not created";
    return impl_->path == 1 ? "the driver's NGX core" : "the model DLL directly (forwarder)";
}

bool Bridge::Init(ID3D11Device* dev) {
    if (ready_) return true;
    if (failed_ || !dev) return false;
    impl_ = new Impl;
    Impl& m = *impl_;
    HRESULT hr = dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&m.dev11);
    if (FAILED(hr) || !m.dev11) return Fail("no ID3D11Device5 (shared fences need Windows 10 1703 or newer)", hr);

    IDXGIDevice* dxgi = nullptr;
    IDXGIAdapter* adapter = nullptr;
    hr = dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi);
    if (SUCCEEDED(hr)) hr = dxgi->GetAdapter(&adapter);
    Rel(dxgi);
    if (FAILED(hr) || !adapter) return Fail("the D3D11 device's adapter", hr);
    DXGI_ADAPTER_DESC ad{};
    adapter->GetDesc(&ad);
    char luid[32];
    snprintf(luid, sizeof(luid), "%08lX%08lX", (unsigned long)ad.AdapterLuid.HighPart, ad.AdapterLuid.LowPart);
    adapter_ = Narrow(ad.Description) + " (LUID " + luid + ")";

    m.d3d12 = LoadLibraryW(L"d3d12.dll");
    auto create = m.d3d12 ? (CreateDeviceFn)GetProcAddress(m.d3d12, "D3D12CreateDevice") : nullptr;
    if (!create) {
        Rel(adapter);
        return Fail("d3d12.dll or D3D12CreateDevice not found");
    }
    hr = create(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&m.dev12);
    Rel(adapter);
    if (FAILED(hr) || !m.dev12) return Fail("D3D12CreateDevice", hr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = m.dev12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&m.queue)))
        return Fail("CreateCommandQueue", hr);
    for (UINT i = 0; i < kRing; ++i)
        if (FAILED(hr = m.dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                                        (void**)&m.alloc[i])))
            return Fail("CreateCommandAllocator", hr);
    if (FAILED(hr = m.dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m.alloc[0], nullptr,
                                               __uuidof(ID3D12GraphicsCommandList), (void**)&m.list)))
        return Fail("CreateCommandList", hr);
    m.list->Close();

    // Both fences are made on D3D12 and opened on D3D11 (nr-bridge's CreateFences).
    auto fence = [&](ID3D12Fence*& f12, ID3D11Fence*& f11, const char* name) {
        HRESULT r = m.dev12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&f12);
        if (FAILED(r)) return Fail(name, r);
        HANDLE h = nullptr;
        r = m.dev12->CreateSharedHandle(f12, nullptr, GENERIC_ALL, nullptr, &h);
        if (FAILED(r)) return Fail("fence CreateSharedHandle", r);
        r = m.dev11->OpenSharedFence(h, __uuidof(ID3D11Fence), (void**)&f11);
        CloseHandle(h);
        if (FAILED(r)) return Fail("D3D11 OpenSharedFence", r);
        return true;
    };
    if (!fence(m.ab12, m.ab11, "fence D3D11 -> D3D12") || !fence(m.ba12, m.ba11, "fence D3D12 -> D3D11")) return false;
    m.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m.event) return Fail("CreateEvent");

    D3D12_QUERY_HEAP_DESC qh{};
    qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh.Count = kRing * 2;
    if (SUCCEEDED(m.dev12->CreateQueryHeap(&qh, __uuidof(ID3D12QueryHeap), (void**)&m.tsHeap))) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = kRing * 2 * sizeof(UINT64);
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(m.dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                    nullptr, __uuidof(ID3D12Resource), (void**)&m.tsRead)) ||
            FAILED(m.queue->GetTimestampFrequency(&m.tsFreq)))
            Rel(m.tsHeap);
    }

    if (FAILED(hr = dev->CreateComputeShader(kNrPackCS, sizeof(kNrPackCS), nullptr, &m.pack)) ||
        FAILED(hr = dev->CreateComputeShader(kNrUnpackCS, sizeof(kNrUnpackCS), nullptr, &m.unpack)) ||
        FAILED(hr = dev->CreateComputeShader(kNrUnpack4CS, sizeof(kNrUnpack4CS), nullptr, &m.unpack4)) ||
        FAILED(hr = dev->CreateComputeShader(kNrMotionCS, sizeof(kNrMotionCS), nullptr, &m.motionCs)) ||
        FAILED(hr = dev->CreateComputeShader(kNrDepthCS, sizeof(kNrDepthCS), nullptr, &m.depthCs)) ||
        FAILED(hr = dev->CreateComputeShader(kNrDownCS, sizeof(kNrDownCS), nullptr, &m.down)) ||
        FAILED(hr = dev->CreateComputeShader(kNrUpEditCS, sizeof(kNrUpEditCS), nullptr, &m.upEdit)))
        return Fail("CreateComputeShader", hr);
    D3D11_BUFFER_DESC bdesc{};
    bdesc.ByteWidth = sizeof(Consts);
    bdesc.Usage = D3D11_USAGE_DEFAULT;
    bdesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(hr = dev->CreateBuffer(&bdesc, nullptr, &m.cb))) return Fail("constant buffer", hr);
    ready_ = true;
    Log("NR bridge: D3D12 device on %s, shared fences and queue ready%s", adapter_.c_str(),
        m.tsHeap ? ", GPU timestamps on" : ", no GPU timestamps");
    return true;
}

void Bridge::WaitIdle() {
    if (!impl_) return;
    Impl& m = *impl_;
    if (!m.queue || !m.ba12 || !m.event) return;
    m.queue->Signal(m.ba12, ++m.vBA);
    if (m.ba12->GetCompletedValue() < m.vBA) {
        m.ba12->SetEventOnCompletion(m.vBA, m.event);
        WaitForSingleObject(m.event, 5000);
    }
}

void Bridge::ReleaseFeatures() {
    if (!impl_) return;
    Impl& m = *impl_;
#ifdef QVFR_HAVE_NGX_HEADERS
    for (auto& s : m.slots) {
        if (!s.feature) continue;
        if (m.path == 1 && m.coreRelease) m.coreRelease((NVSDK_NGX_Handle*)s.feature);
        else if (m.path == 2 && m.fwdRelease) m.fwdRelease(s.feature);
        s.feature = nullptr;
        s.fresh = true;
    }
#endif
}

void Bridge::Configure(const Settings& s) {
    const bool tuning = s.intensity != set_.intensity || s.localStructure != set_.localStructure ||
                        s.localTone != set_.localTone || s.skinStructure != set_.skinStructure || s.style != set_.style ||
                        s.preset != set_.preset || s.autoMask != set_.autoMask || s.uiCorrection != set_.uiCorrection ||
                        s.model != set_.model ||
                        s.depthMode != set_.depthMode;  // depth: a new instance, its history and one more measurement
    // The rest changes only what goes in and comes back: the instances stay and measure again.
    const bool remeasure = s.inputScale != set_.inputScale || s.tonemap != set_.tonemap || s.transfer != set_.transfer ||
                           s.motion != set_.motion || s.debug != set_.debug || s.strength != set_.strength ||
                           s.depth != set_.depth ||
                           s.depthK != set_.depthK;
    if (remeasure && !tuning && impl_)
        for (auto& sl : impl_->slots)
            if (sl.feature) sl.diagLeft = kDiagAfter;
    const bool fixedChanged = impl_ && impl_->ngxTried &&
                              (s.modelPath != set_.modelPath || s.appId != set_.appId || s.ngx != set_.ngx);
    if (tuning && impl_) {
        bool any = false;
        for (auto& sl : impl_->slots) any |= sl.feature != nullptr;
        if (any) {
            WaitIdle();
            ReleaseFeatures();
            Log("NR bridge: settings changed; the model instances are made again (intensity %.2f, style %u, preset %u)",
                s.intensity, s.style, s.preset);
        }
    }
    if (fixedChanged) Log("NR bridge: [nr] model paths, app_id and ngx apply at the next game start");
    Settings n = s;
    if (fixedChanged) {
        n.modelPath = set_.modelPath;
        n.forwarderPath = set_.forwarderPath;
        n.appId = set_.appId;
        n.ngx = set_.ngx;
    }
    set_ = n;
}

bool Bridge::EnsureNgx() {
#ifndef QVFR_HAVE_NGX_HEADERS
    return Fail("this build has no NVIDIA NGX headers (DLSS SDK)");
#else
    Impl& m = *impl_;
    if (m.ngxTried) return m.ngxOk;
    m.ngxTried = true;
    if (GetFileAttributesW(set_.modelPath.c_str()) == INVALID_FILE_ATTRIBUTES)
        return Fail(("the model " + Narrow(set_.modelPath) + " is missing").c_str());
    // The driver's NGX core: already loaded by DLAA's D3D11 NGX, else from the driver store.
    m.core = GetModuleHandleW(L"_nvngx.dll");
    if (!m.core) {
        wchar_t dir[MAX_PATH] = {};
        DWORD sz = sizeof(dir);
        HKEY k = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"System\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore", 0, KEY_READ,
                          &k) == ERROR_SUCCESS) {
            RegQueryValueExW(k, L"NGXPath", nullptr, nullptr, (LPBYTE)dir, &sz);
            RegCloseKey(k);
        }
        if (dir[0]) m.core = LoadLibraryExW((std::wstring(dir) + L"\\_nvngx.dll").c_str(), nullptr,
                                            LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!m.core) return Fail("the driver's NGX core (_nvngx.dll) not found");
    m.coreInit = (CoreInitFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_Init_Ext");
    m.coreShutdown = (CoreShutdownFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_Shutdown1");
    m.coreCaps = (CoreCapsFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    m.coreCreate = (CoreCreateFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_CreateFeature");
    m.coreEval = (CoreEvalFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_EvaluateFeature");
    m.coreRelease = (CoreReleaseFn)GetProcAddress(m.core, "NVSDK_NGX_D3D12_ReleaseFeature");
    if (!m.coreInit || !m.coreCaps || !m.coreCreate || !m.coreEval) return Fail("the NGX core has no D3D12 entry points");

    m.searchDir = set_.searchDir;
    m.paths[0] = m.searchDir.c_str();
    m.info = {};
    m.info.PathListInfo.Path = m.paths;
    m.info.PathListInfo.Length = 1;
    m.info.LoggingInfo.LoggingCallback = NgxLog;
    m.info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    m.info.LoggingInfo.DisableOtherLoggingSinks = true;
    NVSDK_NGX_Result r = m.coreInit(set_.appId, set_.dataDir.c_str(), m.dev12, (NVSDK_NGX_Version)kNgxSdk, &m.info);
    Log("NR bridge: NGX core D3D12 Init_Ext (app 0x%llX) -> 0x%08X %s", set_.appId, (unsigned)r, ResultName((unsigned)r));
    if (NVSDK_NGX_FAILED(r)) return Fail("NGX core init");
    m.coreInited = true;
    NVSDK_NGX_Parameter* caps = nullptr;
    r = m.coreCaps(&caps);
    if (NVSDK_NGX_FAILED(r) || !caps) return Fail("NGX capability parameter block");
    m.caps = caps;
    // The parameter block's virtual setters (NVIDIA's header): one round trip each for uint and float, logged.
    unsigned ub = 0;
    float fb = 0;
    caps->Set("DLSSNR.ProbeU", 0xC0FFEEu);
    caps->Set("DLSSNR.ProbeF", 0.3125f);
    const bool uOk = caps->Get("DLSSNR.ProbeU", &ub) == NVSDK_NGX_Result_Success && ub == 0xC0FFEEu;
    const bool fOk = caps->Get("DLSSNR.ProbeF", &fb) == NVSDK_NGX_Result_Success && fb == 0.3125f;
    Log("NR bridge: parameter block %p: uint round trip %s, float round trip %s", (void*)caps, uOk ? "ok" : "FAILED",
        fOk ? "ok" : "FAILED");
    if (!uOk || !fOk) return Fail("the NGX parameter block does not keep values (its layout differs from the header)");

    if (set_.ngx != 1) {
        m.fwd = LoadLibraryExW(set_.forwarderPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (m.fwd) {
            m.fwdLoad = (FwdLoadFn)GetProcAddress(m.fwd, "qnr_load");
            m.fwdInit = (FwdInitFn)GetProcAddress(m.fwd, "qnr_init");
            m.fwdCreate = (FwdCreateFn)GetProcAddress(m.fwd, "qnr_create");
            m.fwdEval = (FwdEvalFn)GetProcAddress(m.fwd, "qnr_evaluate");
            m.fwdRelease = (FwdReleaseFn)GetProcAddress(m.fwd, "qnr_release");
        }
        Log("NR bridge: forwarder %s -> %s", Narrow(set_.forwarderPath).c_str(),
            (m.fwdLoad && m.fwdInit && m.fwdCreate && m.fwdEval) ? "loaded" : "NOT available (the model cannot be driven directly)");
    }
    m.ngxOk = true;
    return true;
#endif
}

bool Bridge::EnsureContext(ID3D11DeviceContext* c) {
    Impl& m = *impl_;
    if (c == m.ctx) return true;
    Rel(m.ctx4);
    m.ctx = c;
    HRESULT hr = c->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&m.ctx4);
    if (FAILED(hr) || !m.ctx4) return Fail("no ID3D11DeviceContext4", hr);
    return true;
}

// Shared textures of a slot in the image's size (made again when the size changes).
bool Bridge::EnsureSlot(int slot, UINT w, UINT h) {
    Impl& m = *impl_;
    Impl::Slot& s = m.slots[slot];
    if (s.w == w && s.h == h) return true;
    HRESULT hr = S_OK;
    {
        if (s.feature) {
            WaitIdle();
            ReleaseFeatures();
        }
        m.ReleaseSlot(s);
        s.guideW = s.guideH = 0;
        s.guideDepth = false;
        auto make = [&](DXGI_FORMAT fmt, ID3D11Texture2D*& t11, ID3D12Resource*& t12) {
            D3D11_TEXTURE2D_DESC d{};
            d.Width = w;
            d.Height = h;
            d.MipLevels = 1;
            d.ArraySize = 1;
            d.Format = fmt;
            d.SampleDesc.Count = 1;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            d.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
            HRESULT r = m.dev11->CreateTexture2D(&d, nullptr, &t11);
            if (FAILED(r)) return Fail("shared texture CreateTexture2D", r);
            IDXGIResource1* res = nullptr;
            r = t11->QueryInterface(__uuidof(IDXGIResource1), (void**)&res);
            if (FAILED(r)) return Fail("IDXGIResource1", r);
            HANDLE hnd = nullptr;
            r = res->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &hnd);
            res->Release();
            if (FAILED(r)) return Fail("texture CreateSharedHandle", r);
            r = m.dev12->OpenSharedHandle(hnd, __uuidof(ID3D12Resource), (void**)&t12);
            CloseHandle(hnd);
            if (FAILED(r)) return Fail("D3D12 OpenSharedHandle (texture)", r);
            return true;
        };
        if (!make(DXGI_FORMAT_R16G16B16A16_FLOAT, s.color11, s.color12) ||
            !make(DXGI_FORMAT_R16G16B16A16_FLOAT, s.out11, s.out12) ||
            !make(DXGI_FORMAT_R16G16_FLOAT, s.motion11, s.motion12) || !make(DXGI_FORMAT_R32_FLOAT, s.depth11, s.depth12))
            return false;
        if (FAILED(hr = m.dev11->CreateUnorderedAccessView(s.color11, nullptr, &s.colorUav)) ||
            FAILED(hr = m.dev11->CreateUnorderedAccessView(s.motion11, nullptr, &s.motionUav)) ||
            FAILED(hr = m.dev11->CreateUnorderedAccessView(s.depth11, nullptr, &s.depthUav)) ||
            FAILED(hr = m.dev11->CreateShaderResourceView(s.out11, nullptr, &s.outSrv)) ||
            FAILED(hr = m.dev11->CreateShaderResourceView(s.color11, nullptr, &s.colorSrv)))
            return Fail("shared texture views", hr);
        s.w = w;
        s.h = h;
        s.fresh = true;
        if (set_.testShaders) s.diagLeft = kDiagAfter;  // smoke test: the measurement without the model (all zero)
        Log("NR bridge: slot %d: shared textures %ux%u (color and output RGBA16F, motion RG16F, depth R32F)", slot, w, h);
    }
    return true;
}

bool Bridge::Guides(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* motion,
                    ID3D11ShaderResourceView* depth, UINT w, UINT h, UINT x0, UINT y0, UINT fullW, UINT fullH,
                    UINT step, UINT srcX, UINT srcY) {
    if (!fullW || !fullH) {
        fullW = w;
        fullH = h;
    }
    if (!ready_ || failed_ || !c || slot < 0 || slot >= kSlots || !w || !h || x0 + w > fullW || y0 + h > fullH ||
        step < 1 || step > 4)
        return false;
    Impl& m = *impl_;
    if (!EnsureContext(c) || !EnsureSlot(slot, fullW, fullH)) return false;
    Impl::Slot& s = m.slots[slot];
    const bool gameDepth = set_.depthMode > 0 && depth;
    const Consts k{1.0f, 0u, (set_.motion && motion) ? 1.0f : 0.0f, 0u, gameDepth ? (uint32_t)set_.depthMode : 0u,
                   set_.depthK, 0u, 1.0f, x0, y0, w, h, step, srcX, srcY, 0u};
    if (memcmp(&k, &m.cbValue, sizeof(k)) != 0) {
        c->UpdateSubresource(m.cb, 0, nullptr, &k, 0, 0);
        m.cbValue = k;
    }
    ID3D11ShaderResourceView* noSrv = nullptr;
    ID3D11UnorderedAccessView* noUav = nullptr;
    c->CSSetConstantBuffers(0, 1, &m.cb);
    c->CSSetShader(m.motionCs, nullptr, 0);
    c->CSSetShaderResources(0, 1, motion ? &motion : &noSrv);  // without vectors the shader writes zero (mvScale 0)
    c->CSSetUnorderedAccessViews(0, 1, &s.motionUav, nullptr);
    c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    if (gameDepth) {
        c->CSSetShader(m.depthCs, nullptr, 0);
        c->CSSetShaderResources(0, 1, &depth);
        c->CSSetUnorderedAccessViews(0, 1, &s.depthUav, nullptr);
        c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        s.depthValue = -1;
    } else if (s.depthValue != set_.depth) {
        const float v[4] = {set_.depth, set_.depth, set_.depth, set_.depth};
        c->ClearUnorderedAccessViewFloat(s.depthUav, v);
        s.depthValue = set_.depth;
    }
    c->CSSetShaderResources(0, 1, &noSrv);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    c->CSSetShader(nullptr, nullptr, 0);
    s.guideW = fullW;
    s.guideH = fullH;
    s.guideDepth = gameDepth;
    return true;
}

bool Bridge::Downsample(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11UnorderedAccessView* dst, UINT x0,
                        UINT y0, UINT w, UINT h, UINT step, UINT srcX, UINT srcY) {
    if (!ready_ || failed_ || !c || !src || !dst || !w || !h || step < 1 || step > 4) return false;
    Impl& m = *impl_;
    const Consts k{1.0f, 0u, 0.0f, 0u, 0u, 0.0f, 0u, 1.0f, x0, y0, w, h, step, srcX, srcY, 0u};
    if (memcmp(&k, &m.cbValue, sizeof(k)) != 0) {
        c->UpdateSubresource(m.cb, 0, nullptr, &k, 0, 0);
        m.cbValue = k;
    }
    ID3D11ShaderResourceView* noSrv = nullptr;
    ID3D11UnorderedAccessView* noUav = nullptr;
    c->CSSetConstantBuffers(0, 1, &m.cb);
    c->CSSetShader(m.down, nullptr, 0);
    c->CSSetShaderResources(0, 1, &src);
    c->CSSetUnorderedAccessViews(0, 1, &dst, nullptr);
    c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    c->CSSetShaderResources(0, 1, &noSrv);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    c->CSSetShader(nullptr, nullptr, 0);
    return true;
}

bool Bridge::UpsampleEdit(ID3D11DeviceContext* c, ID3D11ShaderResourceView* full, ID3D11ShaderResourceView* before,
                          ID3D11ShaderResourceView* after, UINT x0, UINT y0, UINT w, UINT h, UINT step,
                          ID3D11UnorderedAccessView* out, UINT fullW, UINT fullH, UINT srcX, UINT srcY,
                          UINT fade) {
    if (!ready_ || failed_ || !c || !full || !before || !after || !out || !w || !h || step < 1 || step > 4) return false;
    Impl& m = *impl_;
    const Consts k{1.0f, 0u, 0.0f, 0u, 0u, 0.0f, 0u, 1.0f, x0, y0, w, h, step, srcX, srcY, fade};
    if (memcmp(&k, &m.cbValue, sizeof(k)) != 0) {
        c->UpdateSubresource(m.cb, 0, nullptr, &k, 0, 0);
        m.cbValue = k;
    }
    ID3D11ShaderResourceView* in[3] = {full, before, after};
    ID3D11ShaderResourceView* noIn[3] = {};
    ID3D11UnorderedAccessView* noUav = nullptr;
    c->CSSetConstantBuffers(0, 1, &m.cb);
    c->CSSetShader(m.upEdit, nullptr, 0);
    c->CSSetShaderResources(0, 3, in);
    c->CSSetUnorderedAccessViews(0, 1, &out, nullptr);
    c->Dispatch((fullW + 7) / 8, (fullH + 7) / 8, 1);
    c->CSSetShaderResources(0, 3, noIn);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    c->CSSetShader(nullptr, nullptr, 0);
    return true;
}

bool Bridge::Process(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav,
                     ID3D11ShaderResourceView* motion, ID3D11ShaderResourceView* depth, UINT w, UINT h, bool reset,
                     bool guidesReady) {
    if (!ready_ || failed_ || !c || !srv || !uav || slot < 0 || slot >= kSlots || !w || !h) return false;
    Impl& m = *impl_;
    LARGE_INTEGER t0, t1, qpf;
    QueryPerformanceCounter(&t0);
    HRESULT hr = S_OK;

    if (!EnsureContext(c)) return false;
    if (set_.model && !EnsureNgx()) return false;
    Impl::Slot& s = m.slots[slot];
    if (guidesReady && (s.guideW != w || s.guideH != h)) return false;  // no guides of this size: the image stays
    if (!EnsureSlot(slot, w, h)) return false;
    // The game's depth: from this call, or what the slot's Guides wrote.
    const bool gameDepth = set_.model && set_.depthMode > 0 && (guidesReady ? s.guideDepth : depth != nullptr);

    // A free allocator: the submission that used it last has finished (its timestamps can be read then).
    const UINT r = m.ring % kRing;
    if (m.allocValue[r] && m.ba12->GetCompletedValue() < m.allocValue[r]) {
        m.ba12->SetEventOnCompletion(m.allocValue[r], m.event);
        if (WaitForSingleObject(m.event, kWaitMs) != WAIT_OBJECT_0) return Fail("the D3D12 queue does not finish");
    }
    if (m.tsPending[r] && m.tsHeap && m.tsFreq) {
        const D3D12_RANGE range{r * 2 * sizeof(UINT64), (r + 1) * 2 * sizeof(UINT64)};
        UINT64* p = nullptr;
        if (SUCCEEDED(m.tsRead->Map(0, &range, (void**)&p))) {
            const UINT64* q = p + r * 2;
            if (q[1] >= q[0]) {
                const double ms = (double)(q[1] - q[0]) * 1000.0 / (double)m.tsFreq;
                stats_.gpuMs += ms;
                ++stats_.gpuSamples;
                stats_.slotGpuMs[m.tsSlot[r]] += ms;
                ++stats_.slotGpuSamples[m.tsSlot[r]];
            }
            const D3D12_RANGE none{0, 0};
            m.tsRead->Unmap(0, &none);
        }
        m.tsPending[r] = false;
    }

    // D3D12 commands first: a recording error leaves the image untouched.
    ID3D12CommandAllocator* a = m.alloc[r];
    ID3D12GraphicsCommandList* cl = m.list;
    if (FAILED(hr = a->Reset()) || FAILED(hr = cl->Reset(a, nullptr))) return Fail("command list Reset", hr);
    const bool ts = m.tsHeap != nullptr;
    if (ts) cl->EndQuery(m.tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, r * 2);
    bool modelRan = false;
#ifdef QVFR_HAVE_NGX_HEADERS
    if (set_.model) {
        auto* caps = (NVSDK_NGX_Parameter*)m.caps;
        auto setTuning = [&] {
            caps->Set("DLSSNR.Intensity", set_.intensity);
            caps->Set("DLSSNR.Style", set_.style);
            caps->Set("DLSSNR.LocalStructureStrength", set_.localStructure);
            caps->Set("DLSSNR.LocalToneStrength", set_.localTone);
            caps->Set("DLSSNR.SkinStructureStrength", set_.skinStructure);
            caps->Set("DLSSNR.UseAutoMask", set_.autoMask ? 1u : 0u);
        };
        if (!s.feature) {
            caps->Set("DLSSNR.Enabled", 1u);
            caps->Set("DLSSNR.Width", w);
            caps->Set("DLSSNR.Height", h);
            caps->Set("CreationNodeMask", 1u);
            caps->Set("VisibilityNodeMask", 1u);
            caps->Set("DLSSNR.Hint.Render.Preset", set_.preset);
            setTuning();
            caps->Set("DLSSNR.UICorrection", set_.uiCorrection ? 1u : 0u);
            // Once a path has worked, it stays (nr-bridge).
            if ((set_.ngx == 0 || set_.ngx == 1) && m.path != 2) {
                NVSDK_NGX_Handle* hnd = nullptr;
                const NVSDK_NGX_Result cr = m.coreCreate(cl, (NVSDK_NGX_Feature)kFeatureNr, caps, &hnd);
                Log("NR bridge: slot %d: core CreateFeature(18) %ux%u -> 0x%08X %s", slot, w, h, (unsigned)cr,
                    ResultName((unsigned)cr));
                if (NVSDK_NGX_SUCCEED(cr) && hnd) {
                    s.feature = hnd;
                    m.path = 1;
                }
            }
            if (!s.feature && (set_.ngx == 0 || set_.ngx == 2) && m.path != 1 && m.fwdCreate) {
                if (!m.modelInited) {
                    const int lr = m.fwdLoad(set_.modelPath.c_str());
                    const int ir = lr > 0 ? m.fwdInit(set_.appId, set_.dataDir.c_str(), m.dev12, kNgxSdk, &m.info) : 0;
                    Log("NR bridge: model %s: load %d, Init_Ext -> 0x%08X %s", Narrow(set_.modelPath).c_str(), lr,
                        (unsigned)ir, ResultName((unsigned)ir));
                    m.modelInited = lr > 0 && NVSDK_NGX_SUCCEED((NVSDK_NGX_Result)ir);
                }
                if (m.modelInited) {
                    void* hnd = nullptr;
                    const int cr = m.fwdCreate(cl, kFeatureNr, caps, &hnd);
                    Log("NR bridge: slot %d: model CreateFeature(18) %ux%u -> 0x%08X %s", slot, w, h, (unsigned)cr,
                        ResultName((unsigned)cr));
                    if (NVSDK_NGX_SUCCEED((NVSDK_NGX_Result)cr) && hnd) {
                        s.feature = hnd;
                        m.path = 2;
                    }
                }
            }
            if (!s.feature) {
                cl->Close();
                return Fail("no NGX path could create the Neural Rendering feature (see the lines above)");
            }
            s.fresh = true;
            s.diagLeft = kDiagAfter;
        }
        // The output starts as the input: pixels the model leaves alone (a mask) keep the image, not old content.
        Barrier(cl, s.color12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cl, s.out12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->CopyResource(s.out12, s.color12);
        Barrier(cl, s.color12, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(cl, s.out12, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cl, s.motion12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(cl, s.depth12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        caps->Set("DLSSNR.Color", (void*)s.color12);
        caps->Set("DLSSNR.Depth", (void*)s.depth12);
        caps->Set("DLSSNR.MVec", (void*)s.motion12);
        caps->Set("DLSSNR.Output", (void*)s.out12);
        caps->Set("DLSSNR.Enabled", 1u);
        caps->Set("DLSSNR.Width", w);
        caps->Set("DLSSNR.Height", h);
        caps->Set("DLSSNR.DepthInverted", (set_.depthMode == 2 && gameDepth) ? 1u : 0u);
        caps->Set("DLSSNR.Reset", (reset || s.fresh) ? 1u : 0u);
        for (const char* p : {"Color", "Output", "Depth", "MVec"}) {
            const std::string b = std::string("DLSSNR.") + p + "Subrect";
            caps->Set((b + "BaseX").c_str(), 0u);
            caps->Set((b + "BaseY").c_str(), 0u);
            caps->Set((b + "Width").c_str(), w);
            caps->Set((b + "Height").c_str(), h);
        }
        caps->Set("DLSSNR.MVecScaleX", 1.0f);
        caps->Set("DLSSNR.MVecScaleY", 1.0f);
        setTuning();
        const unsigned er = m.path == 1 ? (unsigned)m.coreEval(cl, (NVSDK_NGX_Handle*)s.feature, caps, nullptr)
                                        : (unsigned)m.fwdEval(cl, s.feature, caps);
        Barrier(cl, s.out12, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        Barrier(cl, s.depth12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        Barrier(cl, s.motion12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        Barrier(cl, s.color12, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (er == NVSDK_NGX_Result_Success) {
            modelRan = true;
            s.fresh = false;
            s.fails = 0;
        } else {
            ++stats_.modelFails;
            if (s.fails < 5) Log("NR bridge: slot %d: EvaluateFeature -> 0x%08X %s", slot, er, ResultName(er));
            if (++s.fails >= kMaxModelFails) {
                cl->Close();
                return Fail("the model keeps failing to evaluate");
            }
        }
    }
#endif
    if (!modelRan) {  // passthrough (or a failed evaluation): the input as it was
        Barrier(cl, s.color12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cl, s.out12, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->CopyResource(s.out12, s.color12);
        Barrier(cl, s.out12, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        Barrier(cl, s.color12, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    if (ts) {
        cl->EndQuery(m.tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, r * 2 + 1);
        cl->ResolveQueryData(m.tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, r * 2, 2, m.tsRead, r * 2 * sizeof(UINT64));
    }
    if (FAILED(hr = cl->Close())) return Fail("command list Close", hr);

    // D3D11: color and motion into the shared inputs, then hand them over.
    const bool shaped = set_.model || set_.testShaders;  // the image goes to the model in its shape
    const bool ratio = shaped && set_.transfer == 1;
    const bool debugView = shaped && set_.debug == 1;
    const float strength = shaped ? set_.strength : 1.0f;
    const bool measured = modelRan || set_.testShaders;
    const bool diagNow = measured && s.diagLeft == 1;  // this call measures the model's edit (below)
    // An image with alpha (the final focus image): CSUnpack4 keeps its alpha from the copy of the image.
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    uav->GetDesc(&ud);
    const bool four = ud.Format == DXGI_FORMAT_R8G8B8A8_UNORM || ud.Format == DXGI_FORMAT_R10G10B10A2_UNORM ||
                      ud.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    const Consts k{shaped ? set_.inputScale : 1.0f, shaped ? (uint32_t)set_.tonemap : 0u, set_.motion ? 1.0f : 0.0f,
                   ratio ? 1u : 0u, gameDepth ? (uint32_t)set_.depthMode : 0u, set_.depthK, debugView ? 1u : 0u,
                   strength, 0u, 0u, w, h, 1u, 0u, 0u, 0u};
    if (ratio || diagNow || four) {  // the image as it came, for CSUnpack and the measurement
        ID3D11Resource* img = nullptr;
        srv->GetResource(&img);
        ID3D11Texture2D* tex = nullptr;
        if (img) img->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex);
        if (img) img->Release();
        if (!tex) return Fail("the image is not a 2D texture");
        D3D11_TEXTURE2D_DESC d{};
        tex->GetDesc(&d);
        if (s.orig11) {
            D3D11_TEXTURE2D_DESC o{};
            s.orig11->GetDesc(&o);
            if (o.Width != d.Width || o.Height != d.Height || o.Format != d.Format) {
                Rel(s.origSrv);
                Rel(s.orig11);
            }
        }
        if (!s.orig11) {
            D3D11_TEXTURE2D_DESC o = d;
            o.MipLevels = 1;
            o.ArraySize = 1;
            o.Usage = D3D11_USAGE_DEFAULT;
            o.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            o.CPUAccessFlags = 0;
            o.MiscFlags = 0;
            // The view in the image's view format (a typeless image needs one).
            D3D11_SHADER_RESOURCE_VIEW_DESC vd{};
            srv->GetDesc(&vd);
            vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            vd.Texture2D.MostDetailedMip = 0;
            vd.Texture2D.MipLevels = 1;
            if (d.MipLevels != 1 || d.ArraySize != 1 || FAILED(hr = m.dev11->CreateTexture2D(&o, nullptr, &s.orig11)) ||
                FAILED(hr = m.dev11->CreateShaderResourceView(s.orig11, &vd, &s.origSrv))) {
                tex->Release();
                return Fail("copy of the image for transfer=ratio", hr);
            }
        }
        c->CopyResource(s.orig11, tex);
        tex->Release();
    }
    if (memcmp(&k, &m.cbValue, sizeof(k)) != 0) {
        c->UpdateSubresource(m.cb, 0, nullptr, &k, 0, 0);
        m.cbValue = k;
    }
    if (!guidesReady && !gameDepth && s.depthValue != set_.depth) {
        const float v[4] = {set_.depth, set_.depth, set_.depth, set_.depth};
        c->ClearUnorderedAccessViewFloat(s.depthUav, v);
        s.depthValue = set_.depth;
    }
    ID3D11ShaderResourceView* noSrv = nullptr;
    ID3D11UnorderedAccessView* noUav = nullptr;
    c->CSSetConstantBuffers(0, 1, &m.cb);
    c->CSSetShader(m.pack, nullptr, 0);
    c->CSSetShaderResources(0, 1, &srv);
    c->CSSetUnorderedAccessViews(0, 1, &s.colorUav, nullptr);
    c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    if (set_.model && !guidesReady) {
        // Without the game's vectors the shader writes zero (mvScale 0).
        if (!motion && k.mvScale != 0) {
            Consts z = k;
            z.mvScale = 0.0f;
            c->UpdateSubresource(m.cb, 0, nullptr, &z, 0, 0);
            m.cbValue = z;
        }
        c->CSSetShader(m.motionCs, nullptr, 0);
        c->CSSetShaderResources(0, 1, motion ? &motion : &srv);
        c->CSSetUnorderedAccessViews(0, 1, &s.motionUav, nullptr);
        c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        if (gameDepth) {
            c->CSSetShader(m.depthCs, nullptr, 0);
            c->CSSetShaderResources(0, 1, &depth);
            c->CSSetUnorderedAccessViews(0, 1, &s.depthUav, nullptr);
            c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            s.depthValue = -1;  // a later switch to the constant clears again
        }
    }
    c->CSSetShaderResources(0, 1, &noSrv);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    m.ctx4->Signal(m.ab11, ++m.vAB);
    c->Flush();

    // D3D12: wait for the inputs, run, signal the output.
    m.queue->Wait(m.ab12, m.vAB);
    ID3D12CommandList* lists[] = {cl};
    m.queue->ExecuteCommandLists(1, lists);
    m.queue->Signal(m.ba12, ++m.vBA);
    m.allocValue[r] = m.vBA;
    m.tsPending[r] = ts;
    m.tsSlot[r] = slot;
    ++m.ring;
    if (FAILED(hr = m.dev12->GetDeviceRemovedReason())) {
        // No D3D11 wait on a signal that may never come; the image keeps its D3D11 content.
        c->CSSetShader(nullptr, nullptr, 0);
        return Fail("D3D12 device removed", hr);
    }

    // D3D11: wait on the GPU for the output and write it back into the image.
    m.ctx4->Wait(m.ba11, m.vBA);
    ID3D11ShaderResourceView* back[3] = {s.outSrv, s.colorSrv, s.origSrv};
    ID3D11ShaderResourceView* noBack[3] = {};
    c->CSSetShader(four ? m.unpack4 : m.unpack, nullptr, 0);
    c->CSSetShaderResources(0, 3, back);  // the model's input always (CSUnpack reads it); the copy when made
    c->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    c->CSSetShaderResources(0, 3, noBack);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    c->CSSetShader(nullptr, nullptr, 0);

    QueryPerformanceCounter(&t1);
    QueryPerformanceFrequency(&qpf);
    stats_.cpuMs += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)qpf.QuadPart;
    ++stats_.calls;
    if (modelRan) ++stats_.modelCalls;

    // After each new instance and each [nr] change: what the model got and gave back (in its units, after input_scale
    // and tonemap) and what reached the HDR image.
    if (measured && s.diagLeft > 0 && --s.diagLeft == 0) {
        ID3D11Resource* img = nullptr;
        srv->GetResource(&img);
        ID3D11Texture2D* imgTex = nullptr;
        if (img) img->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&imgTex);
        Rel(img);
        auto staging = [&](ID3D11Texture2D* from) {
            ID3D11Texture2D* t = nullptr;
            if (!from) return t;
            D3D11_TEXTURE2D_DESC d{};
            from->GetDesc(&d);
            d.MipLevels = 1;
            d.ArraySize = 1;
            d.Usage = D3D11_USAGE_STAGING;
            d.BindFlags = 0;
            d.MiscFlags = 0;
            d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            if (FAILED(m.dev11->CreateTexture2D(&d, nullptr, &t))) return (ID3D11Texture2D*)nullptr;
            c->CopySubresourceRegion(t, 0, 0, 0, 0, from, 0, nullptr);
            return t;
        };
        // With debug=1 the image holds the edit picture, not the result: no HDR measurement.
        ID3D11Texture2D *in = staging(s.color11), *out = staging(s.out11), *before = debugView ? nullptr : staging(s.orig11),
                        *after = debugView ? nullptr : staging(imgTex);
        if (in && out) {
            const Mean mi = Measure(c, in, w, h), mo = Measure(c, out, w, h);
            Log("NR bridge: slot %d after %d model images (intensity %.2f, style %u, preset %u, auto_mask %d, "
                "ui_correction %d, depth %d, tonemap %d, input_scale %g, transfer %s): model "
                "input mean RGB (%.4f, %.4f, %.4f) luminance %.4f [darkest %.4f, brightest %.4f], output (%.4f, %.4f, "
                "%.4f) luminance %.4f [%.4f, %.4f], output / input %.3f",
                slot, kDiagAfter, set_.intensity, set_.style, set_.preset, set_.autoMask, set_.uiCorrection,
                gameDepth ? set_.depthMode : 0, set_.tonemap, set_.inputScale, set_.transfer ? "ratio" : "inverse",
                mi.rgb[0], mi.rgb[1], mi.rgb[2], mi.lum, mi.lo, mi.hi, mo.rgb[0], mo.rgb[1], mo.rgb[2], mo.lum, mo.lo,
                mo.hi, mi.lum > 0 ? mo.lum / mi.lum : 0.0);
            const Edit e = MeasureEdit(c, in, out, before, after, w, h);
            if (e.n > 0) {
                Log("NR bridge: slot %d: model edit in its units: mean |change| luminance %.4f (signed %+.4f), RGB "
                    "(%.4f, %.4f, %.4f), 99th percentile %.4f, max %.4f; pixels changed > 1/255: %.1f%%, > 8/255: %.1f%%",
                    slot, e.absLum, e.signedLum, e.absRgb[0], e.absRgb[1], e.absRgb[2], e.p99Lum, e.maxLum,
                    e.over1 * 100.0, e.over8 * 100.0);
                if (e.hdrN > 0)
                    Log("NR bridge: slot %d: in the HDR image: mean luminance %.4g -> %.4g (x%.3f), mean |change| %.2f%% "
                        "of the mean, per pixel 50th / 99th percentile %.2f%% / %.2f%%, pixels changed > 2%%: %.1f%%, "
                        "> 10%%: %.1f%%",
                        slot, e.hdrBefore, e.hdrAfter, e.hdrBefore > 0 ? e.hdrAfter / e.hdrBefore : 0.0,
                        e.hdrBefore > 0 ? e.hdrAbs / e.hdrBefore * 100.0 : 0.0, e.hdrP50 * 100.0, e.hdrP99 * 100.0,
                        e.hdrOver2 * 100.0, e.hdrOver10 * 100.0);
            }
        }
        Rel(in);
        Rel(out);
        Rel(before);
        Rel(after);
        Rel(imgTex);
    }
    return true;
}

void Bridge::Release() {
    if (impl_) {
        Impl& m = *impl_;
        WaitIdle();  // the D3D12 queue finishes before anything it uses goes
        ReleaseFeatures();
#ifdef QVFR_HAVE_NGX_HEADERS
        if (m.coreInited && m.coreShutdown && m.dev12) m.coreShutdown(m.dev12);
#endif
        for (auto& s : m.slots) m.ReleaseSlot(s);
        Rel(m.tsRead);
        Rel(m.tsHeap);
        Rel(m.list);
        for (auto*& a : m.alloc) Rel(a);
        Rel(m.ab11);
        Rel(m.ba11);
        Rel(m.ab12);
        Rel(m.ba12);
        Rel(m.queue);
        Rel(m.dev12);
        Rel(m.pack);
        Rel(m.unpack);
        Rel(m.unpack4);
        Rel(m.down);
        Rel(m.upEdit);
        Rel(m.motionCs);
        Rel(m.depthCs);
        Rel(m.cb);
        Rel(m.ctx4);
        Rel(m.dev11);
        if (m.event) CloseHandle(m.event);
        delete impl_;
        impl_ = nullptr;
    }
    ready_ = false;
}

}  // namespace qvfr::nrbridge

#include "shots.h"

#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

namespace qvfr::shots {

namespace {

using Microsoft::WRL::ComPtr;

struct Image {
    std::string name;
    UINT w = 0, h = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::vector<uint8_t> rgb;  // full resolution, 8 bits per channel
};

struct Job {
    std::string stamp;
    std::vector<Image> images;
    std::vector<Pair> pairs;
    std::vector<int> index;  // shot image index -> position in images, -1 = missing
};

std::atomic<bool> g_busy{false};  // a shot is begun or its worker is still writing
bool g_active = false;
const char* const* g_names = nullptr;
int g_count = 0;
const Pair* g_pairs = nullptr;
int g_pairCount = 0;
ComPtr<ID3D11Texture2D> g_staging[kMaxImages];
D3D11_TEXTURE2D_DESC g_desc[kMaxImages];

bool Supported(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM:
            return true;
        default:
            return false;
    }
}

// One mapped row into 8-bit RGB. The bytes are stored as the game wrote them (sRGB-encoded for the Submit textures),
// so the PNG shows what the compositor gets.
void ConvertRow(DXGI_FORMAT f, const uint8_t* src, uint8_t* dst, UINT w) {
    switch (f) {
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            for (UINT x = 0; x < w; ++x, src += 4, dst += 3) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
            }
            break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM:
            for (UINT x = 0; x < w; ++x, src += 4, dst += 3) {
                uint32_t v;
                memcpy(&v, src, 4);
                dst[0] = (uint8_t)((v & 0x3ff) >> 2);
                dst[1] = (uint8_t)(((v >> 10) & 0x3ff) >> 2);
                dst[2] = (uint8_t)(((v >> 20) & 0x3ff) >> 2);
            }
            break;
        default:
            for (UINT x = 0; x < w; ++x, src += 4, dst += 3) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
            break;
    }
}

// 24-bit BGR PNG at half resolution (2 x 2 box filter).
bool WritePng(const fs::path& path, const Image& img, std::string& err) {
    UINT w = std::max(1u, img.w / 2), h = std::max(1u, img.h / 2);
    std::vector<uint8_t> bgr((size_t)w * h * 3);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* r0 = img.rgb.data() + (size_t)std::min(2 * y, img.h - 1) * img.w * 3;
        const uint8_t* r1 = img.rgb.data() + (size_t)std::min(2 * y + 1, img.h - 1) * img.w * 3;
        uint8_t* d = bgr.data() + (size_t)y * w * 3;
        for (UINT x = 0; x < w; ++x, d += 3) {
            size_t x0 = (size_t)std::min(2 * x, img.w - 1) * 3, x1 = (size_t)std::min(2 * x + 1, img.w - 1) * 3;
            for (int c = 0; c < 3; ++c)
                d[2 - c] = (uint8_t)((r0[x0 + c] + r0[x1 + c] + r1[x0 + c] + r1[x1 + c] + 2) / 4);
        }
    }
    ComPtr<IWICImagingFactory> factory;
    ComPtr<IWICStream> stream;
    ComPtr<IWICBitmapEncoder> enc;
    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> props;
    WICPixelFormatGUID pf = GUID_WICPixelFormat24bppBGR;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) hr = factory->CreateStream(&stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc);
    if (SUCCEEDED(hr)) hr = enc->Initialize(stream.Get(), WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = enc->CreateNewFrame(&frame, &props);
    if (SUCCEEDED(hr)) hr = frame->Initialize(props.Get());
    if (SUCCEEDED(hr)) hr = frame->SetSize(w, h);
    if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&pf);
    if (SUCCEEDED(hr) && pf != GUID_WICPixelFormat24bppBGR) hr = E_FAIL;
    if (SUCCEEDED(hr)) hr = frame->WritePixels(h, w * 3, (UINT)bgr.size(), bgr.data());
    if (SUCCEEDED(hr)) hr = frame->Commit();
    if (SUCCEEDED(hr)) hr = enc->Commit();
    if (FAILED(hr)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "HRESULT 0x%08lX", (unsigned long)hr);
        err = buf;
        return false;
    }
    return true;
}

struct DiffStats {
    uint64_t sum = 0, over8 = 0, over32 = 0, n = 0;
    int maxd = 0;
    void Add(const uint8_t* p, const uint8_t* q) {
        int d = 0;
        for (int c = 0; c < 3; ++c) {
            int v = std::abs((int)p[c] - (int)q[c]);
            sum += v;
            d = std::max(d, v);
        }
        maxd = std::max(maxd, d);
        over8 += d > 8;
        over32 += d > 32;
        ++n;
    }
};

// Weights of the source pixels covering [s, e) along one axis (source pixel i covers [i, i + 1)).
struct Span {
    int i0 = 0;
    std::vector<float> w;
};

Span Footprint(double s, double e, int size) {
    Span sp;
    sp.i0 = std::max(0, (int)std::floor(s));
    int i1 = std::min(size, (int)std::ceil(e));
    double total = 0;
    for (int i = sp.i0; i < i1; ++i) {
        double w = std::min(e, (double)i + 1) - std::max(s, (double)i);
        sp.w.push_back((float)std::max(0.0, w));
        total += std::max(0.0, w);
    }
    for (float& w : sp.w) w = total > 0 ? (float)(w / total) : 0.0f;
    return sp;
}

// a shows the rectangle r of b: average a over the footprint of each b pixel fully inside r, compare with that pixel.
void DiffRegion(const Image& a, const Image& b, const Pair& r, DiffStats& st) {
    const double kx = a.w / (double)r.rw, ky = a.h / (double)r.rh;  // a pixels per b pixel
    const int u0 = std::max(0, (int)std::ceil(r.rx)), u1 = std::min((int)b.w, (int)std::floor(r.rx + r.rw));
    const int v0 = std::max(0, (int)std::ceil(r.ry)), v1 = std::min((int)b.h, (int)std::floor(r.ry + r.rh));
    if (u0 >= u1 || v0 >= v1) return;
    std::vector<Span> cols(u1 - u0);
    for (int u = u0; u < u1; ++u) cols[u - u0] = Footprint((u - r.rx) * kx, (u + 1 - r.rx) * kx, (int)a.w);
    for (int v = v0; v < v1; ++v) {
        Span row = Footprint((v - r.ry) * ky, (v + 1 - r.ry) * ky, (int)a.h);
        for (int u = u0; u < u1; ++u) {
            const Span& col = cols[u - u0];
            float acc[3] = {};
            for (size_t j = 0; j < row.w.size(); ++j) {
                const uint8_t* line = a.rgb.data() + (size_t)(row.i0 + j) * a.w * 3;
                for (size_t i = 0; i < col.w.size(); ++i) {
                    const uint8_t* p = line + (size_t)(col.i0 + i) * 3;
                    float w = row.w[j] * col.w[i];
                    for (int c = 0; c < 3; ++c) acc[c] += w * p[c];
                }
            }
            uint8_t avg[3];
            for (int c = 0; c < 3; ++c) avg[c] = (uint8_t)std::clamp((int)std::lround(acc[c]), 0, 255);
            st.Add(avg, b.rgb.data() + ((size_t)v * b.w + u) * 3);
        }
    }
}

void LogDiff(const Image& a, const Image& b, const Pair& pair) {
    DiffStats st;
    char what[160] = "";
    if (pair.rw > 0 && pair.rh > 0) {
        DiffRegion(a, b, pair, st);
        snprintf(what, sizeof(what), " (first image averaged into %.1f x %.1f pixels at (%.1f, %.1f) of the second)",
                 pair.rw, pair.rh, pair.rx, pair.ry);
    } else if (a.w != b.w || a.h != b.h) {
        Log("Shot: %s: sizes differ (%u x %u and %u x %u)", pair.label, a.w, a.h, b.w, b.h);
        return;
    } else {
        const size_t n = (size_t)a.w * a.h;
        for (size_t i = 0; i < n; ++i) st.Add(a.rgb.data() + i * 3, b.rgb.data() + i * 3);
    }
    if (!st.n) {
        Log("Shot: %s: no pixels to compare%s", pair.label, what);
        return;
    }
    const double n = (double)st.n;
    Log("Shot: %s: mean abs difference %.3f (0..255), max %d, pixels off by > 8: %.3f%%, > 32: %.3f%%%s", pair.label,
        (double)st.sum / (3.0 * n), st.maxd, 100.0 * st.over8 / n, 100.0 * st.over32 / n, what);
}

void RunJob(std::unique_ptr<Job> job) {
    HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    double t0 = NowMs();
    for (const Pair& p : job->pairs) {
        int a = job->index[p.a], b = job->index[p.b];
        if (a >= 0 && b >= 0) LogDiff(job->images[a], job->images[b], p);
    }
    fs::path dir = BaseDir() / L"shots" / Widen(job->stamp);
    std::error_code ec;
    fs::create_directories(dir, ec);
    int written = 0;
    for (const Image& img : job->images) {
        std::string err;
        if (WritePng(dir / Widen(img.name + ".png"), img, err))
            ++written;
        else
            Log("Shot: cannot write %s.png (%s)", img.name.c_str(), err.c_str());
    }
    Log("Shot: %d PNG files (half resolution) in %s, %.1f s", written, Narrow(dir.wstring()).c_str(),
        (NowMs() - t0) / 1000.0);
    if (SUCCEEDED(co)) CoUninitialize();
}

void ReleaseStaging() {
    for (auto& s : g_staging) s.Reset();
}

}  // namespace

bool Begin(const char* const* names, int count, const Pair* pairs, int pairCount) {
    if (g_busy.exchange(true)) return false;
    g_active = true;
    g_names = names;
    g_count = std::min(count, kMaxImages);
    g_pairs = pairs;
    g_pairCount = pairCount;
    ReleaseStaging();
    return true;
}

bool Active() { return g_active; }

bool Grab(ID3D11DeviceContext* ctx, ID3D11Resource* src, int index) {
    if (!g_active || !ctx || !src || index < 0 || index >= g_count || g_staging[index]) return false;
    ComPtr<ID3D11Texture2D> tex;
    if (FAILED(src->QueryInterface(IID_PPV_ARGS(&tex)))) return false;
    D3D11_TEXTURE2D_DESC d{};
    tex->GetDesc(&d);
    if (d.SampleDesc.Count != 1 || !Supported(d.Format)) {
        Log("Shot: %s: texture %u x %u format %d samples %u is not supported", g_names[index], d.Width, d.Height,
            (int)d.Format, d.SampleDesc.Count);
        return false;
    }
    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(&dev);
    D3D11_TEXTURE2D_DESC s = d;
    s.Usage = D3D11_USAGE_STAGING;
    s.BindFlags = 0;
    s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    s.MiscFlags = 0;
    HRESULT hr = dev ? dev->CreateTexture2D(&s, nullptr, &g_staging[index]) : E_FAIL;
    if (FAILED(hr)) {
        Log("Shot: %s: cannot create a staging texture %u x %u (0x%08lX)", g_names[index], d.Width, d.Height,
            (unsigned long)hr);
        return false;
    }
    g_desc[index] = d;
    ctx->CopyResource(g_staging[index].Get(), src);
    return true;
}

void End(ID3D11DeviceContext* ctx) {
    if (!g_active) return;
    g_active = false;
    auto job = std::make_unique<Job>();
    job->stamp = LocalTimeStamp();
    job->index.assign(g_count, -1);
    std::string missing;
    double t0 = NowMs();
    for (int i = 0; i < g_count; ++i) {
        if (!g_staging[i] || !ctx) {
            missing += (missing.empty() ? "" : ", ") + std::string(g_names[i]);
            continue;
        }
        const D3D11_TEXTURE2D_DESC& d = g_desc[i];
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(g_staging[i].Get(), 0, D3D11_MAP_READ, 0, &m))) {
            missing += (missing.empty() ? "" : ", ") + std::string(g_names[i]) + " (Map failed)";
            continue;
        }
        Image img;
        img.name = g_names[i];
        img.w = d.Width;
        img.h = d.Height;
        img.format = d.Format;
        img.rgb.resize((size_t)d.Width * d.Height * 3);
        for (UINT y = 0; y < d.Height; ++y)
            ConvertRow(d.Format, (const uint8_t*)m.pData + (size_t)y * m.RowPitch, img.rgb.data() + (size_t)y * d.Width * 3,
                       d.Width);
        ctx->Unmap(g_staging[i].Get(), 0);
        job->index[i] = (int)job->images.size();
        job->images.push_back(std::move(img));
    }
    ReleaseStaging();
    for (int i = 0; i < g_pairCount; ++i) job->pairs.push_back(g_pairs[i]);
    Log("Shot %s: %zu of %d images read back in %.0f ms%s%s", job->stamp.c_str(), job->images.size(), g_count,
        NowMs() - t0, missing.empty() ? "" : "; missing: ", missing.c_str());
    if (job->images.empty()) {
        g_busy = false;
        return;
    }
    try {
        std::thread([j = std::move(job)]() mutable {
            try {
                RunJob(std::move(j));
            } catch (...) {
                Log("Shot: the writer thread failed (out of memory?)");
            }
            g_busy = false;
        }).detach();
    } catch (...) {
        Log("Shot: cannot start the writer thread");
        g_busy = false;
    }
}

void Cancel() {
    if (!g_active) return;
    g_active = false;
    ReleaseStaging();
    g_busy = false;
}

}  // namespace qvfr::shots

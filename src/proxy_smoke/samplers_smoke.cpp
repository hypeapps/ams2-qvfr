// qvfr_proxy_smoke --samplers: the sampler log's CreateSamplerState hook (samplers.h) on a WARP device. Equal
// descriptions count as one, a negative MipLODBias is counted, and the stats lines go to the proxy log.
// qvfr_proxy_smoke --ngx-log: the NGX log filter (ngx_log.h) on lines NGX 310.9.16 wrote in a game session.
#include <windows.h>

#include <d3d11.h>

#include <cstdio>
#include <cstring>

#include "ngx_log.h"
#include "samplers.h"

int RunSamplers() {
    ID3D11Device* dev = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                 nullptr, nullptr)) ||
        !dev) {
        printf("Samplers check FAILED: no WARP device\n");
        return 2;
    }
    if (!qvfr::samplers::TestInstall(dev)) {
        printf("Samplers check FAILED: hook not installed\n");
        dev->Release();
        return 3;
    }
    D3D11_SAMPLER_DESC d{};
    d.Filter = D3D11_FILTER_ANISOTROPIC;
    d.AddressU = d.AddressV = d.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    d.MaxAnisotropy = 16;
    d.ComparisonFunc = D3D11_COMPARISON_NEVER;
    d.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* s[4] = {};
    HRESULT hr = dev->CreateSamplerState(&d, &s[0]);
    if (SUCCEEDED(hr)) hr = dev->CreateSamplerState(&d, &s[1]);  // the same description: one entry, two creations
    D3D11_SAMPLER_DESC b = d;
    b.MipLODBias = -1.0f;
    if (SUCCEEDED(hr)) hr = dev->CreateSamplerState(&b, &s[2]);
    D3D11_SAMPLER_DESC c = d;
    c.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    c.AddressU = c.AddressV = c.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    c.ComparisonFunc = D3D11_COMPARISON_GREATER_EQUAL;
    c.MaxAnisotropy = 1;
    if (SUCCEEDED(hr)) hr = dev->CreateSamplerState(&c, &s[3]);
    qvfr::samplers::LogStats(false);
    const size_t distinct = qvfr::samplers::TestDistinct(), negative = qvfr::samplers::TestNegativeBias();
    const unsigned long long creations = qvfr::samplers::TestCreations();
    qvfr::samplers::LogStats(true);
    for (auto*& x : s)
        if (x) x->Release();
    dev->Release();
    const bool ok = SUCCEEDED(hr) && distinct == 3 && negative == 1 && creations == 4;
    printf("Samplers check %s: created 0x%08lX, distinct %zu (want 3), negative MipLODBias %zu (want 1), creations %llu "
           "(want 4); lines \"Samplers:\" in the proxy log\n",
           ok ? "OK" : "FAILED", (unsigned long)hr, distinct, negative, creations);
    return ok ? 0 : 4;
}

int RunNgxLog() {
    using qvfr::ngxlog::Kind;
    struct Case {
        const char* line;
        const char* mode;
        Kind want;
    };
    const Case cases[] = {
        {"[NGXLoadLibrary:332] error: failed to load NGXCore: 126 (D:\\AMS2\\_nvngx.dll)", nullptr, Kind::Drop},
        {"[`anonymous-namespace'::SnippetLocationInfo::load:713] NGXLoadFromPath failed: -1160773628", nullptr,
         Kind::Drop},
        {"[NGXSecureLoadFeature:1345] Feature dlss failed to load (cmsid 141959980) from cache (location: C:\\x)",
         nullptr, Kind::Drop},
        {"[NGXLoadConfig:1150] app_E658703=2.14.16", nullptr, Kind::Drop},
        {"[tid:1][NGXInitLog:214] Build v310.9.16 CL 38955375", nullptr, Kind::Init},
        {"[tid:1][NGXCreateFeature:100] Error: invalid parameter", nullptr, Kind::Problem},
        {"[tid:1][Foo:1] Warning: resource state", "(DLAA)", Kind::Problem},
        {"[tid:1][NgxDltss::FillCreationParams:2763] Info: (DLAA) Using App hint Preset L", "(DLAA)", Kind::Preset},
        {"[tid:1][NgxDltss::FillCreationParams:2763] Info: (Quality) Using App hint Preset L", "(DLAA)", Kind::Drop},
        {"[tid:1][NgxDltss::FillCreationParams:2753] Info: (UltraQuality) App hint Preset Unspecified or Overridden, "
         "using title default Preset K",
         "(Quality)", Kind::Drop},
        {"[tid:1][NgxDltss::FillCreationParams:2753] Info: (Quality) App hint Preset Unspecified or Overridden, using "
         "title default Preset K",
         "(Quality)", Kind::Preset},
        {"[tid:1][NgxDltss::FillCreationParams:2763] Info: (Perf) Using App hint Preset F", "(UltraPerf)", Kind::Drop},
        {"[tid:1][NgxDltss::FillCreationParams:2763] Info: (DLAA) Using App hint Preset F", nullptr, Kind::Drop},
        {"[tid:1][NgxRayReconstruction::FillCreationParams:1733] Info: (DLAA) Using Default Overriden Preset Preset_D",
         "(DLAA)", Kind::Drop},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const Kind got = qvfr::ngxlog::Classify(c.line, c.mode);
        if (got != c.want) {
            ++bad;
            printf("  NGX log filter: \"%s\" with %s: got %d, want %d\n", c.line, c.mode ? c.mode : "no mode", (int)got,
                   (int)c.want);
        }
    }
    const bool tags = qvfr::ngxlog::ModeTag(5) && !strcmp(qvfr::ngxlog::ModeTag(5), "(DLAA)") &&
                      !strcmp(qvfr::ngxlog::ModeTag(2), "(Quality)") && !qvfr::ngxlog::ModeTag(9);
    const bool ok = !bad && tags;
    printf("NGX log filter check %s: %d of %zu lines classified wrong, mode tags %s\n", ok ? "OK" : "FAILED", bad,
           sizeof(cases) / sizeof(cases[0]), tags ? "OK" : "WRONG");
    return ok ? 0 : 4;
}

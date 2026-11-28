// qvfr_proxy_smoke --samplers: the sampler log's CreateSamplerState hook (samplers.h) on a WARP device. Equal
// descriptions count as one, a negative MipLODBias is counted, and the stats lines go to the proxy log.
// qvfr_proxy_smoke --ngx-log: the NGX log filter (ngx_log.h) on lines NGX 310.9.16 wrote in a game session.
#include <windows.h>

#include <d3d11.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ngx_log.h"
#include "samplers.h"

namespace {

D3D11_SAMPLER_DESC Desc(D3D11_FILTER f, D3D11_TEXTURE_ADDRESS_MODE a, float maxLod = D3D11_FLOAT32_MAX) {
    D3D11_SAMPLER_DESC d{};
    d.Filter = f;
    d.AddressU = d.AddressV = d.AddressW = a;
    d.MaxAnisotropy = 16;
    d.ComparisonFunc = D3D11_COMPARISON_NEVER;
    d.MinLOD = -D3D11_FLOAT32_MAX;
    d.MaxLOD = maxLod;
    return d;
}

// What slot `slot` holds now: its description, and whether it is `s` itself.
bool Bound(ID3D11DeviceContext* c, UINT slot, ID3D11SamplerState* s, D3D11_SAMPLER_DESC* d) {
    ID3D11SamplerState* b = nullptr;
    c->PSGetSamplers(slot, 1, &b);
    if (!b) {
        *d = {};
        return !s;
    }
    b->GetDesc(d);
    b->Release();
    return b == s;
}

// The texture mip bias: the game's scene samplers (the smoke exe counts as the game's exe) get copies with the bias;
// PSSetSamplers binds the copy of the current view's bias, the other samplers as they are. First as in stereo (no eye
// groups: outer's bias everywhere), then with the quad eye groups (each view's own bias, the game's samplers from the
// TAA resolve to the group's end, group starts rebinding).
int RunMipBias(ID3D11Device* dev) {
    using namespace qvfr::samplers;
    ID3D11DeviceContext* c = nullptr;
    dev->GetImmediateContext(&c);
    const size_t setsBefore = TestSceneSamplers();
    const D3D11_SAMPLER_DESC want[] = {
        Desc(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP),          // 0 scene, trilinear
        Desc(D3D11_FILTER_ANISOTROPIC, D3D11_TEXTURE_ADDRESS_WRAP),                   // 1 scene (MinLOD unlike --samplers)
        Desc(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP),     // 2 no linear mips: as it is
        Desc(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP, 0.0f),     // 3 LOD 0..0: as it is
        Desc(D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_BORDER),  // 4 shadow: as it is
    };
    constexpr UINT kN = sizeof(want) / sizeof(want[0]);
    ID3D11SamplerState* s[kN] = {};
    ID3D11SamplerState* late = nullptr;
    HRESULT hr = S_OK;
    for (UINT i = 0; i < kN && SUCCEEDED(hr); ++i) hr = dev->CreateSamplerState(&want[i], &s[i]);
    int bad = 0;
    auto check = [&](const char* what, bool ok) {
        if (!ok) {
            ++bad;
            printf("  mip bias: %s FAILED\n", what);
        }
    };
    D3D11_SAMPLER_DESC d{};
    auto all = [&](UINT from) {
        bool same = true;
        for (UINT i = from; i < kN; ++i) same &= Bound(c, i, s[i], &d);
        return same;
    };
    if (SUCCEEDED(hr)) {
        const uint64_t replaced0 = TestReplacedBinds();
        c->PSSetSamplers(0, kN, s);
        check("before any bias: the game's samplers bound", all(0) && TestReplacedBinds() == replaced0);

        // Stereo DLAA: -1 everywhere.
        SetViewBias(true, true, -1.0f, -1.0f);
        c->PSSetSamplers(0, kN, s);
        check("stereo: trilinear sampler with MipLODBias -1",
              !Bound(c, 0, s[0], &d) && d.MipLODBias == -1.0f && d.Filter == D3D11_FILTER_MIN_MAG_MIP_LINEAR);
        check("stereo: anisotropic sampler with MipLODBias -1",
              !Bound(c, 1, s[1], &d) && d.MipLODBias == -1.0f && d.Filter == D3D11_FILTER_ANISOTROPIC);
        check("stereo: mip point, LOD 0..0 and comparison samplers as they are", all(2));
        check("stereo: two samplers replaced", TestReplacedBinds() == replaced0 + 2);
        SceneEnd(c);  // no eye groups: the bias stays past the TAA resolve
        c->PSSetSamplers(0, 1, s);
        check("stereo: bias after the TAA resolve too", !Bound(c, 0, s[0], &d) && d.MipLODBias == -1.0f);

        SetViewBias(false, true, -1.0f, -1.0f);
        c->PSSetSamplers(0, kN, s);
        check("DLSS not running: the game's samplers", all(0));
        SetSwitch(0);
        SetViewBias(true, true, -1.0f, -1.0f);
        c->PSSetSamplers(0, kN, s);
        check("switch dlss_mip_bias=0: the game's samplers", all(0));
        SetSwitch(1);
        SetViewBias(true, false, -1.0f, -1.0f);
        c->PSSetSamplers(0, 1, s);
        check("switch dlss_mip_bias=1 over mip_bias=0: MipLODBias -1", !Bound(c, 0, s[0], &d) && d.MipLODBias == -1.0f);
        SetSwitch(-1);

        // Quad views with DLSS SR: outer at 50% (-2), focus at 77% (log2 0.77 - 1 = -1.377, step -1.375).
        SetViewBias(true, true, std::log2(0.5f) - 1, std::log2(0.77f) - 1);
        BeginGroup(0, c);
        check("outer L's start rebinds with outer's bias -2", !Bound(c, 0, s[0], &d) && d.MipLODBias == -2.0f);
        c->PSSetSamplers(0, kN, s);
        check("outer L binds with -2", !Bound(c, 1, s[1], &d) && d.MipLODBias == -2.0f && all(2));
        SceneEnd(c);
        check("outer L's TAA resolve puts the game's samplers back", Bound(c, 0, s[0], &d) && Bound(c, 1, s[1], &d));
        c->PSSetSamplers(0, kN, s);
        check("outer L after its TAA resolve binds the game's samplers", all(0));
        EndGroup();
        c->PSSetSamplers(0, kN, s);
        check("between the eye groups: the game's samplers", all(0));
        BeginGroup(2, c);
        check("focus L's start rebinds with focus's bias -1.375", !Bound(c, 0, s[0], &d) && d.MipLODBias == -1.375f);
        late = nullptr;
        D3D11_SAMPLER_DESC w = Desc(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_MIRROR);
        hr = dev->CreateSamplerState(&w, &late);
        if (SUCCEEDED(hr)) c->PSSetSamplers(5, 1, &late);
        check("a scene sampler created while the bias runs: focus's bias -1.375",
              SUCCEEDED(hr) && !Bound(c, 5, late, &d) && d.MipLODBias == -1.375f);
        EndGroup();
        SetViewBias(false, true, 0, 0);
        BeginGroup(3, c);
        check("DLSS off: focus R's start puts the game's samplers back", Bound(c, 0, s[0], &d) && Bound(c, 5, late, &d));
        EndGroup();
    }
    const size_t sets = TestSceneSamplers() - setsBefore;
    for (auto*& x : s)
        if (x) x->Release();
    if (late) late->Release();
    c->Release();
    // Three new scene samplers here (want[0], want[1], late); the others get no bias.
    const bool ok = SUCCEEDED(hr) && sets == 3 && !bad;
    printf("Mip bias check %s: created 0x%08lX, new game scene samplers %zu (want 3), %d checks failed, replaced at "
           "bind %llu; lines \"Samplers: texture mip bias\" and \"Samplers mip bias\" in the proxy log\n",
           ok ? "OK" : "FAILED", (unsigned long)hr, sets, bad, (unsigned long long)TestReplacedBinds());
    return ok ? 0 : 5;
}

}  // namespace

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
    const bool logOk = SUCCEEDED(hr) && distinct == 3 && negative == 1 && creations == 4;
    printf("Samplers check %s: created 0x%08lX, distinct %zu (want 3), negative MipLODBias %zu (want 1), creations %llu "
           "(want 4); lines \"Samplers:\" in the proxy log\n",
           logOk ? "OK" : "FAILED", (unsigned long)hr, distinct, negative, creations);
    const int bias = RunMipBias(dev);
    qvfr::samplers::LogStats(true);
    for (auto*& x : s)
        if (x) x->Release();
    dev->Release();
    return !logOk ? 4 : bias;
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

// NR bridge (dlaa-plan D7, prototype NR-P1): a D3D11 image through a private D3D12 device and back, where NVIDIA's
// Neural Rendering model (NGX feature 18, D3D12 only) runs on it.
//
// Per call, on the game's immediate context:
//   D3D11: CSPack color -> shared color (RGBA16F), CSMotion motion -> shared motion (RG16F), Signal(fence AB), Flush
//   D3D12: Wait(fence AB), the model (or a copy: passthrough) shared color -> shared output, Signal(fence BA)
//   D3D11: Wait(fence BA) (on the GPU), CSUnpack shared output -> color
// (an atlas: CSPackRegion per image, then the fences and the model once, then CSUnpackRegion or CSUpEdit per image)
// The D3D12 commands are recorded before any D3D11 work, so a failure there leaves the image as it was.
//
// The model: NGX's capability parameter block from the driver's NGX core (_nvngx.dll), then feature 18 created either
// by the core or, when the core refuses it, by the model DLL itself through the forwarder nvngx.dll_qvfr_nr.dll (the
// model only accepts callers whose path contains "nvngx.dll"). Parameters DLSSNR.*; tuning is read at creation.
// The device, shared fences and textures and the NGX contract follow nr-bridge's direct engine
// (https://github.com/lowpoplabs/nr-bridge, src/nr_direct.h, MIT License, Copyright (c) 2026 LowPopLabs), which follows
// the open-source OptiScaler DLSS-NR fork.
// No dependency on the rest of the proxy except Log (common.h), so qvfr_proxy_smoke --nr-bridge tests it directly.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::nrbridge {

// Independent images (one per view, or an atlas of several); each has its own shared textures and model instance.
constexpr int kSlots = 6;

struct Settings {
    bool model = false;  // false: passthrough (the D3D12 side only copies)
    std::wstring modelPath;      // nvngx_dlssnr.dll
    std::wstring forwarderPath;  // nvngx.dll_qvfr_nr.dll
    std::wstring searchDir;      // NGX feature search path
    std::wstring dataDir;        // NGX's data and log folder
    unsigned long long appId = 0x24480451ull;  // nr-bridge's default application id
    int ngx = 0;                 // 0 the driver's core first, then the model directly; 1 core only; 2 model only
    // Model tuning (read at creation; a change recreates the instances). Names and defaults as in nr-bridge.
    float intensity = 1.0f, localStructure = 1.0f, localTone = 1.0f, skinStructure = -1.0f;
    unsigned style = 0, preset = 0;
    bool autoMask = true;
    bool uiCorrection = true;
    // 1: the image shows the model's edit (0.5 + (output - input) * 20 per channel in the model's units, made back into
    // HDR like the image) instead of the result. Needs the model's input, so also with transfer 0.
    int debug = 0;
    float strength = 1.0f;       // the model's edit x this in its units before it goes back (DXL's "Self Layers")
    // Into the model: color * inputScale; tonemap 1 through a max-channel Reinhard, 2 also a 2.2 gamma (undone
    // afterwards).
    float inputScale = 1.0f;
    int tonemap = 0;
    // Back from the model: 0 the exact inverse of input_scale and tonemap; 1 the original HDR image * (output / input)
    // per channel in linear display values (a small edit of the highlights stays small).
    int transfer = 1;
    bool motion = true;          // the game's motion vectors (false: zero vectors)
    float depth = 0.5f;          // depthMode 0: constant depth (as nr-bridge)
    // Depth into the model: 0 the constant, 1 the game's linear TAA depth (0 camera .. 1 far), 2 reversed device-like
    // depth depthK / linear (DepthInverted).
    int depthMode = 0;
    float depthK = 2e-5f;        // near / far for depthMode 2
    bool testShaders = false;    // smoke test: input_scale, tonemap and transfer also without the model
};

struct Stats {
    uint64_t calls = 0;
    uint64_t modelCalls = 0;   // calls that ran the model (the rest copied)
    uint64_t modelFails = 0;   // model evaluations that failed (the image came back unchanged)
    double cpuMs = 0;          // sum of the CPU time of Process and Evaluate (includes the Flush)
    double gpuMs = 0;          // sum of the D3D12 queue time between the timestamps
    uint64_t gpuSamples = 0;
    double slotGpuMs[kSlots] = {};  // the same per slot
    uint64_t slotGpuSamples[kSlots] = {};
};

class Bridge {
public:
    Bridge() = default;
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    ~Bridge() { Release(); }

    // A D3D12 device on the D3D11 device's adapter, the command queue and the two shared fences. Shaders are D3D11
    // compute shaders on `dev`. False: failed() and why().
    bool Init(ID3D11Device* dev);
    // Settings for the next calls. A change of the tuning recreates the model instances (after the queue is idle); any
    // other change measures the model's edit again (log "NR bridge: slot N after ... images").
    void Configure(const Settings& s);
    // The image in `srv`/`uav` (float RGB, w x h, R11G11B10_FLOAT or RGBA16F) through D3D12 and back. `motion`: the
    // game's motion vectors (uv now - uv before, w x h), `depth` its linear depth (R32_FLOAT, w x h); either may be
    // null. reset: the image's history broke (new view, cut). `c` is the immediate context of the Init device; its compute shader state is changed (call it inside a
    // saved state). False: nothing was written to the image (a failure is permanent: failed()).
    // The image may also be R8G8B8A8_UNORM, R10G10B10A2_UNORM or RGBA16F with alpha (kept). guidesReady: the motion
    // vectors and depth come from the slot's last Guides call (same size), `motion` and `depth` are ignored.
    bool Process(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav,
                 ID3D11ShaderResourceView* motion, ID3D11ShaderResourceView* depth, UINT w, UINT h, bool reset,
                 bool guidesReady = false);
    // Only the motion vectors and depth of a slot (w x h), for a later Process with guidesReady (NR on the final image:
    // the guides at DLAA time, the color at the composition). With fullW x fullH (the slot's size) they go into the
    // region at (x0, y0) of an atlas of several images that go through the model in one evaluation; the previous
    // positions are held inside the region. step > 1: the region is the image at 1/step of its size in each axis (the
    // guides are point-sampled); (srcX, srcY): the region holds only the part of the image from there. False: nothing
    // written.
    bool Guides(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* motion, ID3D11ShaderResourceView* depth,
                UINT w, UINT h, UINT x0 = 0, UINT y0 = 0, UINT fullW = 0, UINT fullH = 0, UINT step = 1, UINT srcX = 0,
                UINT srcY = 0);
    // The atlas of final images (display-encoded, UNORM views; `c` as above), without a copy of the atlas: after the
    // slot's Guides (which give it its size), each image goes into its region with PackRegion, the slot through the
    // model once with Evaluate, and each image back out with UnpackRegion. False: nothing written (a failure is
    // permanent: failed()).
    // PackRegion: `src` from (srcX, srcY), averaged over step x step pixels, into the w x h region at (x0, y0) of the
    // slot's model input. The gaps between the regions are black (also after ClearInput, e.g. for a new layout).
    bool PackRegion(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* src, UINT x0, UINT y0, UINT w, UINT h,
                    UINT step = 1, UINT srcX = 0, UINT srcY = 0);
    void ClearInput(int slot);
    // Evaluate: the slot's input through the model (or copied: passthrough); false without guides of its size or an
    // image packed since it was made. reset as in Process.
    bool Evaluate(ID3D11DeviceContext* c, int slot, bool reset);
    // UnpackRegion: into `out` (fullW x fullH, the image's copy) the image `src` back from its region as PackRegion
    // put it there. step 1 and the whole image: the region of the model's output itself (alpha from `src`); otherwise
    // `src` plus the model's edit of the region bilinearly enlarged, with fade > 0 fading out over `fade` pixels
    // inside the part's edge (zero outside; without an edit `out` gets `src` bit for bit). transfer 1 only for the
    // whole image.
    bool UnpackRegion(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* src, ID3D11UnorderedAccessView* out,
                      UINT x0, UINT y0, UINT w, UINT h, UINT step, UINT fullW, UINT fullH, UINT srcX = 0,
                      UINT srcY = 0, UINT fade = 0);
    void Release();  // waits for the D3D12 queue, then releases everything

    bool ready() const { return ready_; }
    bool failed() const { return failed_; }
    const std::string& why() const { return why_; }
    const std::string& adapter() const { return adapter_; }
    const char* modelPath() const;  // "core", "model", "passthrough" or "not created"
    Stats stats() const { return stats_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    Settings set_;
    bool ready_ = false;
    bool failed_ = false;
    std::string why_;
    std::string adapter_;
    Stats stats_;

    bool Fail(const char* what, long hr = 0);
    bool EnsureContext(ID3D11DeviceContext* c);
    bool EnsureSlot(int slot, UINT w, UINT h);
    bool EnsureNgx();
    bool Record(int slot, UINT w, UINT h, bool reset, bool gameDepth, bool& modelRan);
    bool Submit(ID3D11DeviceContext* c, int slot);
    void LogEdit(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* srv, UINT w, UINT h, bool debugView,
                 bool gameDepth);
    void WaitIdle();
    void ReleaseFeatures();
};

}  // namespace qvfr::nrbridge

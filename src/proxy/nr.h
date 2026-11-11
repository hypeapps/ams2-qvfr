// NVIDIA Neural Rendering prototype NR-P1 (dlaa-plan D7, [nr] mode=focus): after DLAA in the focus views of quad
// views, the DLAA result goes through the NR bridge (nr_bridge.h) to a private D3D12 device, where the user's
// nvngx_dlssnr.dll (x64\qvfr\, never shipped) runs on it with the game's motion vectors, and back. [nr] model=0 is the
// bridge alone (passthrough: the image must stay the same). Not in the default ini; a prototype for local tests only
// (no release without NVIDIA's official SDK).
// [nr] mode=final: the model runs on the finished focus images (after the game's post-processing, display-encoded, as
// nr-bridge and DXL's compatibility mode do) at the composition; DLAA's evaluation only hands over the motion vectors and
// depth (Process), ProcessFinal the color.
#pragma once

#include <d3d11.h>

#include <string>

namespace qvfr::nr {

// Render thread, in DLAA's evaluation (dlaa.cpp, after DLSS wrote `out`): true when this view goes through the bridge.
// view: dlaa's view index (1 outer L, 2 outer R, 3 focus L, 4 focus R; outer only in mode=final with outer on).
bool Wanted(int view);
// The DLAA result (w x h, an SRV and a UAV of the same texture) through the bridge. motion: the motion vectors DLAA
// used (R16G16_FLOAT, uv now - uv before), depth: its linear depth (R32_FLOAT); either may be null. reset: DLAA reset
// its history in this evaluation. Guards itself (SEH); a failure turns NR off for the session and leaves the DLAA
// result.
void Process(ID3D11DeviceContext* c, int view, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav,
             ID3D11Resource* motion, ID3D11Resource* depth, UINT w, UINT h, bool reset);
// Render thread, at the composition (quad.cpp, before compose::Draw): true when the eyes' final images go through the
// model (mode=final; the outer ones too with [nr] outer).
bool WantedFinal();
// [quad] smoothen_focus_view_edges at the composition: `ini`, or the NR test step's.
float Smoothen(float ini);
// Eye e's finished focus and outer images through the model: `focus` and `outer` become the textures to compose from
// (the module's copies with the model's result, valid until the next frame or Shutdown) or stay. With [nr] batch=1 the
// left eye's call runs all images of the frame in one evaluation (rightFocus, rightOuter: the right eye's finished
// images; null skips the frame) and the right eye's call only takes its copies. The compute shader state of `c` is
// restored. Guards itself (SEH).
void ProcessFinal(ID3D11DeviceContext* c, int eye, ID3D11Texture2D*& focus, ID3D11Texture2D*& outer,
                  ID3D11Texture2D* rightFocus, ID3D11Texture2D* rightOuter);
// Live on/off ([nr] key, benchmark switch `nr`).
bool Active();
void SetActive(bool on);
// Benchmark sequence switches (quad.cpp): nr_outer 0/1, nr_batch 0/1, nr_outer_res (source pixels per outer pixel in
// the atlas, 1..4), nr_outer_area (the outer images' centre part through the model, percent 50..100), nr_preset 0..3,
// nr_style 0..3, nr_strength (percent 0..300). GetKnob: the value now (the ini's until set).
enum class Knob { Outer, Batch, OuterRes, OuterArea, Preset, Style, Strength, Count };
int GetKnob(Knob k);
void SetKnob(Knob k, int value);
// The last statistics window's (a benchmark run's) D3D12 GPU time of the model per frame and the atlas's size.
struct Window {
    bool any = false;  // images went through the model in it
    double gpuMsPerFrame = 0;
    double atlasMp = 0;  // 0 without the atlas
};
Window LastWindow();
void OnFrameStart();  // hotkey (render thread)
void LogStats(bool totals);
std::string Describe();
void Shutdown();  // before DLAA releases the device

}  // namespace qvfr::nr

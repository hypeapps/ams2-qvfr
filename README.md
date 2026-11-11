# AMS2 QVFR

**EXPERIMENTAL** Quad views foveated rendering for **Automobilista 2** in VR 

The game renders each eye in two parts: the whole field of view at a lower pixel density (the *outer* view) and a
smaller area at full density (the *focus* view). With eye tracking the focus follows your gaze; without it, it stays in
the middle of the view. The mod puts both together for each eye before the image goes to the headset. You get a sharp
image where you look and render far fewer pixels, so you have headroom for higher settings or a steadier frame rate.
MSAA keeps working, and so does the game's TAA. On NVIDIA RTX graphics cards the mod can also replace the game's TAA with
NVIDIA DLAA (experimental).

## Install

1. Download `ams2-qvfr-<version>.zip` of the latest release from
   [Releases](https://github.com/hypeapps/ams2-qvfr/releases/latest).
2. In Steam: right-click Automobilista 2 → **Manage** → **Browse local files**. This opens the folder with `AMS2.exe`.
3. Copy `x64\openvr_api.dll` from that folder to a safe place outside the game folder. If you already run AMS2 through
   your own OpenComposite, that file is your OpenComposite: keep the copy (see [VR API](#vr-api-openxr-or-openvr)).
4. Extract the zip into the folder with `AMS2.exe` and confirm overwriting `x64\openvr_api.dll`.
5. Customize the settings in `x64\qvfr\qvfr_config.exe`. The mod's files are not code-signed, so on the first start
   Windows SmartScreen may show a warning, and some antivirus programs may flag or quarantine `qvfr_config.exe` or
   `x64\openvr_api.dll` as a false positive. If that happens, restore the file and add an exception.
6. The mod starts on the **OpenXR** VR API (the default; OpenComposite comes with the mod). To use **OpenVR** instead,
   change it in `qvfr_config.exe` → **Settings** → **VR API**. Details: [VR API](#vr-api-openxr-or-openvr).
7. Then start the game as usual.

Everything the mod needs is in `x64\qvfr\`

**Update:** extract a new version the same way. Your `x64\qvfr\qvfr.ini` stays as it is.

**After Steam verifies the game files or updates the game,** Steam puts back its own `x64\openvr_api.dll` and the mod is
off. Open `qvfr_config.exe` → **Files** → **Repair**.

**Uninstall:** `qvfr_config.exe` → **Files** → **Turn the mod off** (puts the Valve DLL back), then delete `x64\qvfr`. Or
let Steam verify the game files and delete `x64\qvfr`.

## Settings

**IMPORTANT: The mod does its own quad views inside the game.** It does not ask the VR runtime for quad views, so runtime-side
  quad views (mbucchia's Quad-Views-Foveated layer, Pimax quad views, VectorXR) have no effect on AMS2 and need no change.

You can customize settings using the GUI: `x64\qvfr\qvfr_config.exe`

- **Settings:** quad views on/off, VR API, eye tracking (with the time the focus waits when the tracker loses your eyes,
  and a switch that shows your gaze and the focus edges in the game), anti-aliasing ([NVIDIA DLAA](#nvidia-dlaa-experimental)),
  focus position, debug options.
- **Resolution:** a to-scale preview of the left eye with the pixels the game renders, the presets and every size.
- **Status:** problems found at the last game start and what the mod detected (headset, runtime, eye tracking…).
- **Files:** checks the mod's files, repairs or turns off the mod. **Log:** the last log.

The sizes use the names, meaning and defaults of the Quad-Views-Foveated OpenXR layer by mbucchia, so its guides apply;
in the GUI they are percentages (multiplier × 100):

| GUI | `qvfr.ini` | Balanced preset |
|---|---|---|
| Preset | `[quad] preset` | `balanced` (also `quality`, `performance`) |
| Focus resolution | `focus_multiplier` | 100% |
| Peripheral resolution | `peripheral_multiplier` | 50% |
| Focus area width / height | `horizontal_focus_section` / `vertical_focus_section` | 35% / 35% |
| Width / height without eye tracking | `horizontal_fixed_section` / `vertical_fixed_section` | 50% / 45% |
| Transition thickness | `smoothen_focus_view_edges` | 20% |
| Enable eye tracking | `follow_gaze` | on |

- **Right Ctrl + F6** turns eye tracking on or off while you play (can be turned off in the GUI).
- Serious problems found at game start (Pimax foveated rendering on, unsupported game build) also show in a Windows
  message box; the game keeps running.

## VR API: OpenXR or OpenVR

Choose it in the GUI (**Settings → VR API**) or with `[runtime] path` in `qvfr.ini` (`openxr` or `openvr`); it applies at
the next game start.

- **OpenXR** (default): the game goes through OpenComposite, included with the mod in `x64\qvfr\opencomposite\`, to the OpenXR
  runtime active in your system (SteamVR, Pimax Play, Meta, Varjo Base…). The mod never changes which runtime is active.
  Eye tracking comes from the runtime's `XR_EXT_eye_gaze_interaction`, read by a small OpenXR layer of the mod that only
  the game process loads. To use your own OpenComposite build instead, set `[runtime] opencomposite=` to its full path.
- **OpenVR:** the game talks to SteamVR through the Valve DLL in `x64\qvfr\openvr\`. Eye tracking: what SteamVR
  gives games (tested with the Pimax Dream Air).

## NVIDIA DLAA (experimental)

NVIDIA DLSS without upscaling (DLAA) replaces the game's temporal anti-aliasing (TAA): calmer edges and a sharper image
than the game's TAA. The mod gives DLSS the game's own motion vectors (including the cars and
wheels), depth and camera jitter, and runs it for each of the four views. NVIDIA DLSS (`nvngx_dlss.dll`) comes with the
mod in `x64\qvfr\`.

- **Requirements:** an NVIDIA RTX graphics card, and in the game's graphics options **TAA on and MSAA off**. Without them
  the game's own anti-aliasing stays.
- **Turn it on:** `qvfr_config.exe` → **Settings** → **Rendering** → **NVIDIA DLAA**, then restart the game. Off by
  default.
- **DLSS preset:** the DLSS model (J, K, L, M).
- **DLSS exposure** (`[dlaa] auto_exposure` and `exposure` in `qvfr.ini`): fixed (default 0.002) keeps the brightness of
  lights and reflections; auto lets DLSS measure it, which dims them, most at night. Lower fixed values keep brighter
  highlights.
- Dashboard displays and mirrors keep the game's TAA, because DLSS leaves trails behind changing digits
  (`[dlaa] cockpit_taa` in `qvfr.ini`: 0 = DLSS everywhere, 2 = also the rest of your car).
- With **Debug hotkeys** on: right Ctrl + F1 switches between DLAA and the game's TAA, F12 steps through exposures and
  Home through the presets while you drive.
- The **Status** page shows whether DLAA ran at the last game start, and if not, why.

## NVIDIA DLSS 5 (alpha)

NVIDIA's DLSS 5 neural rendering model reworks the light and materials of the finished image, in the sharp area and in
the periphery. It works on the DLAA image (its motion vectors and depth), so turning it on also turns DLAA on.

- **Requirements:** everything DLAA needs, and NVIDIA's model `nvngx_dlssnr.dll` in `x64\qvfr\`. The mod does not include
  the model: when it is missing, the settings app shows a **Download** button under the DLSS 5 switch. The mod's own
  loader for it (`x64\qvfr\nvngx.dll_qvfr_nr.dll`) comes with the mod.
- **Turn it on:** `qvfr_config.exe` → **Settings** → **Rendering** → **NVIDIA DLSS 5**. Off by default. The **DLSS 5**
  card has its options: intensity, style, preset, strength, local tone and structure, auto mask, and how the periphery
  goes through the model (on or off, resolution, area, together with the sharp area or separately).
- **Tuning panel in the headset:** right Ctrl + F2 while you drive (always on, also without Debug hotkeys). Arrows choose
  and change an option, Enter switches, Esc closes; the game does not get these keys while the panel is open. Changes
  show at once, **Save to qvfr.ini** keeps them. The panel also shows the frame rate, the CPU and GPU frame time and
  the cost of DLSS 5.
- **Cost:** about 4 ms of GPU time per frame on an RTX 5090 with the periphery at half resolution (the default).

## NVIDIA DLSS Super Resolution in stereo (prototype)

With quad views off, the game renders at a lower resolution and DLSS upscales it to the full SteamVR resolution. Only in
`qvfr.ini` for now: `[quad] mode=off`, `[dlaa] mode=dlss` and `render_scale=0.67` (0.67 = DLSS Quality, 0.5 =
Performance), then restart the game. The same requirements as DLAA. With quad views on, the mod keeps DLAA.

## Requirements

- Automobilista 2 on Steam, in VR (DirectX 11), Windows 10 or 11, 64-bit.
- An OpenXR runtime (SteamVR, Pimax Play, Meta, Varjo Base…) for the default OpenXR path, or SteamVR for the OpenVR path.
- Eye tracking is optional. Without it you get fixed foveated rendering (the focus stays in the middle).
- MSAA or TAA in the game (MSAA is the most tested). With TAA the focus views keep their own TAA history.
- For NVIDIA DLAA: an NVIDIA RTX graphics card and TAA on in the game.
- For NVIDIA DLSS 5: the same, and NVIDIA's model `nvngx_dlssnr.dll` (not included).

## Compatibility notes

- **Pimax:** turn off **Foveated rendering** in Pimax Play for AMS2 (its LibMagic does not work with AMS2; the log warns).
- **Pimax's SteamVR driver and eye tracking:** with Pimax's own SteamVR driver, eye tracking works only on the OpenXR
  path. On the OpenVR path the driver gives the game a wrong gaze and the sharp area does not follow your eyes; the mod
  warns at the game start. Keep **OpenXR** in VR API (the default), or use the CustomHeadsetOpenVR driver, which works on
  both paths.
- **OpenXR Toolkit:** on the OpenXR path, turn off Eye tracking, Foveated rendering and upscaling for AMS2. If the game
  still closes at start with the OpenComposite error `XR_ERROR_ACTIONSETS_ALREADY_ATTACHED` (seen with some Toolkit forks),
  disable the Toolkit for AMS2. On the OpenVR path the Toolkit does not work at all.
- **Meta Link / Air Link:** AMS2 tries its Oculus mode before OpenVR. With the Meta software installed it can start in
  Oculus mode, where the mod is not loaded. A community workaround is a dummy `LibOVRRT64_1.dll` in the folder with
  `AMS2.exe` (https://github.com/rodolZ80/bypass-oculus-detector; not tested with this mod). Quest Pro eye tracking
  through Meta Link is not supported (Meta's PC runtime offers it only through its own extension, which the mod does not
  read).

## Troubleshooting

Turn these on in the GUI (**Settings → Debug**):

- **Benchmark hotkey right Ctrl + F9:** freezes the head pose and records the frame times: a 5 s countdown (keep your
  head still), 30 s of recording, beeps at the start and the end. Results in `x64\qvfr\bench\`.
- **Frame timing CSV for the whole session:** the same timings for the whole drive in `x64\qvfr\bench\`.
- **Show the gaze in the game** (Eye tracking): a white ring where you look, the focus edges in red (blend starts) and
  green (fully sharp).
- **Debug hotkeys:** right Ctrl + F3 sun effects from outer (crepuscular rays and lens flares), F4 focus culling, F5 clones, F7 freeze the gaze, F8 overlay, F10 screenshot of the views, F11 outer only, F1 NVIDIA DLAA or the game's TAA, F12 DLSS exposure steps, Home DLSS preset steps.

### Issue reporting

Report a problem by opening an issue on GitHub: https://github.com/hypeapps/ams2-qvfr/issues. Please attach:

- the newest log from `x64\qvfr\logs\` (`qvfr_<date>_<time>_AMS2AVX.log`). Drive at least 20 s: the block with what
  the mod detected is written then. Add `x64\qvfr\bench\` if you ran a benchmark;
- your headset, the VR API (OpenVR or OpenXR) and the OpenXR runtime, and what you saw.

## License

AMS2 QVFR is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): free to use, change and share for
any noncommercial purpose. Third-party software and its licenses: [THIRD_PARTY.md](THIRD_PARTY.md). OpenComposite is
included unmodified under the GPLv3; its source code is linked there and in `x64\qvfr\opencomposite\SOURCE.txt`.
NVIDIA DLSS is included unmodified under the NVIDIA RTX SDKs License (`x64\qvfr\licenses\nvidia-rtx-sdks-LICENSE.txt`).
NVIDIA's DLSS 5 model (`nvngx_dlssnr.dll`) is not included. NVIDIA and DLSS are trademarks of NVIDIA Corporation; this mod
is not sponsored or endorsed by NVIDIA.

Thanks to Matthieu Bucchianeri for Quad-Views-Foveated, whose settings and focus edge the mod follows, and to the
OpenComposite project.

Thanks to Dan Suzuki for the inspiration: his [LMU Quad View mod](https://github.com/DSuzuki35/LMU-Quad-View-Mod) showed
that this is possible today.

## Building from source

Windows, Visual Studio with the MSVC x64 tools, CMake 3.28 or newer and Ninja. All dependencies are in `third_party/`,
except the NVIDIA DLSS SDK, which `tools\fetch_dlss_sdk.ps1` downloads from NVIDIA (the release zip needs it).

```
.\tools\fetch_dlss_sdk.ps1   # the NVIDIA DLSS SDK files for DLAA (without them the proxy has no DLAA)
.\build.ps1                  # Release; .\build.ps1 -Config Debug for a debug build
.\tools\package.ps1          # the release zip in dist\
.\tools\install_proxy.ps1    # developer install into the Steam game folder (backs up the original DLL)
```

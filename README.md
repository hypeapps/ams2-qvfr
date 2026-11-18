# AMS2 QVFR

**Experimental** quad views foveated rendering for Automobilista 2 in VR.

The game renders each eye in two parts: the whole field of view at a lower pixel density (the *outer* view) and a smaller area at full density (the *focus* view). 

With eye tracking the focus view follows your gaze. Without eye tracking the focus view stays in the middle of the view. 

The mod puts both together for each eye before the image goes to the headset. 

You get a sharp image where you look and render far fewer pixels, so you have headroom for higher settings or a steadier frame rate.

On NVIDIA RTX graphics cards the mod can also replace the game's TAA with NVIDIA DLSS (DLAA or Super Resolution).

## Install

1. Download `ams2-qvfr-<version>.zip` of the latest release from
   [Releases](https://github.com/hypeapps/ams2-qvfr/releases/latest).
2. In Steam, right-click Automobilista 2, then choose **Manage** and **Browse local files**. Steam opens the folder with
   `AMS2.exe`.
3. Copy `x64\openvr_api.dll` from that folder to a safe place outside the game folder. If you already run AMS2 through
   your own OpenComposite, that file is your OpenComposite, so keep the copy. See [OpenXR or OpenVR](#openxr-or-openvr).
4. Extract the zip into the folder with `AMS2.exe` and confirm overwriting `x64\openvr_api.dll`.
5. Change the settings in the settings app, `x64\qvfr\qvfr_config.exe`. The mod's files are not code-signed. On the
   first start, Windows SmartScreen may show a warning. Some antivirus programs may flag or quarantine `qvfr_config.exe`
   or `x64\openvr_api.dll` as a false positive. If that happens, restore the file and add an exception.
6. The mod uses the OpenXR VR API by default, with the OpenComposite that comes with the mod. To use OpenVR instead, open
   the **Settings** page of the settings app and change **VR API**. Details are in [OpenXR or OpenVR](#openxr-or-openvr).
7. Start the game as usual.

Everything the mod needs is in `x64\qvfr\`.

**Update.** Extract a new version the same way. Your `x64\qvfr\qvfr.ini` stays as it is.

**Repair.** When Steam verifies or updates the game files, it puts back its own `x64\openvr_api.dll` and the mod stops
loading. To fix it, open the **Files** page of the settings app and click **Repair**.

**Uninstall.** Open the **Files** page of the settings app and click **Turn the mod off**, which puts the Valve DLL back.
Then delete `x64\qvfr`. You can also let Steam verify the game files and then delete `x64\qvfr`.

## Settings

**The mod makes its own quad views inside the game.** It does not ask the VR runtime for quad views. Runtime quad views,
such as mbucchia's Quad-Views-Foveated layer, Pimax quad views or VectorXR, have no effect on AMS2, and you do not need
to change them.

The settings app is `x64\qvfr\qvfr_config.exe`. It has these pages:

- **Settings** has quad views on or off, the VR API, eye tracking, anti-aliasing with
  [NVIDIA DLSS](#nvidia-dlaa-and-dlss-super-resolution), the focus position and the debug options. Eye tracking includes
  the time the focus waits when the tracker loses your eyes, and a switch that shows your gaze and the focus edges in
  the game.
- **Resolution** shows a to-scale preview of the left eye with the pixels the game renders, the presets and every size.
- **FPS counter** sets up a frame rate counter in the headset, on the OpenXR path.
- **Status** lists the problems found at the last game start and what the mod detected, such as the headset, the
  runtime and eye tracking.
- **Files** checks the mod's files, repairs the mod or turns it off.
- **Log** shows the last log.

The sizes have the names, meanings and defaults of mbucchia's Quad-Views-Foveated OpenXR layer, so its guides apply. The
settings app shows them as percentages, the multiplier times 100.

| Settings app | `qvfr.ini` | Balanced preset |
|---|---|---|
| Preset | `[quad] preset` | `balanced`; the other values are `quality` and `performance` |
| Focus resolution | `focus_multiplier` | 100% |
| Peripheral resolution | `peripheral_multiplier` | 50% |
| Focus area width and height | `horizontal_focus_section` and `vertical_focus_section` | 35% and 35% |
| Width and height without eye tracking | `horizontal_fixed_section` and `vertical_fixed_section` | 50% and 45% |
| Transition thickness | `smoothen_focus_view_edges` | 20% |
| Enable eye tracking | `follow_gaze` | on |

- Right Ctrl + F6 turns eye tracking on or off while you play. You can turn this hotkey off in the settings app.
- Serious problems found at game start, such as Pimax foveated rendering turned on or an unsupported game build, also
  show in a Windows message box. The game keeps running.

## OpenXR or OpenVR

Choose the VR API on the **Settings** page of the settings app, or set `[runtime] path` in `qvfr.ini` to `openxr` or
`openvr`. The change applies at the next game start.

- **OpenXR** is the default. The game goes through OpenComposite to the OpenXR runtime active in your system, such as
  SteamVR, Pimax Play, Meta or Varjo Base. OpenComposite comes with the mod in `x64\qvfr\opencomposite\`. The mod never
  changes which runtime is active. Eye tracking comes from the runtime's `XR_EXT_eye_gaze_interaction`. A small OpenXR
  layer of the mod reads it, and only the game process loads that layer. To use your own OpenComposite build, set
  `[runtime] opencomposite=` to its full path.
- **OpenVR** connects the game to SteamVR through the Valve DLL in `x64\qvfr\openvr\`. Eye tracking comes from what
  SteamVR gives games. We tested it with the Pimax Dream Air.

## NVIDIA DLAA and DLSS Super Resolution

NVIDIA DLSS replaces the game's temporal anti-aliasing, TAA. The mod gives DLSS the game's own motion vectors, which
include the cars and wheels, and the game's depth and camera jitter. It runs DLSS for each of the four views. The DLSS
file, `nvngx_dlss.dll`, comes with the mod in `x64\qvfr\`.

- **Requirements.** You need an NVIDIA RTX graphics card. In the game's graphics options, turn **TAA on and MSAA off**.
  Without them, the game keeps its own anti-aliasing.
- **Turn it on.** Open the **Settings** page of the settings app, go to **Rendering** and turn on **NVIDIA DLSS**. Then
  restart the game. DLSS is off by default and starts in the DLAA mode.
- **Mode.** DLAA renders at 100% and does not upscale. It gives calmer edges and a sharper image than the game's TAA.
  With Super Resolution, the game renders fewer pixels and DLSS upscales them to the size of each view. The Super
  Resolution modes are Quality at 67%, Balanced at 58%, Performance at 50% and Ultra Performance at 33%. You choose a
  mode under **Focus** for the focus and under **Peripheral** for the outer view. With quad views off, you choose one
  mode for the whole image. If the game started with an upscaling mode, a new mode applies while you drive, with a short
  hitch. If the game started in DLAA, the first upscaling mode needs a game restart.
- **Preset.** DLSS has four models, J, K, L and M. You choose one for the focus and one for the outer view.
- **Exposure.** `[dlaa] auto_exposure` and `exposure` in `qvfr.ini` set it. A fixed exposure, 0.002 by default, keeps
  the brightness of lights and reflections. With auto exposure, DLSS measures the exposure itself and dims lights and
  reflections, most at night. Lower fixed values keep brighter highlights.
- **Displays and mirrors.** Dashboard displays and mirrors take the current frame instead of DLSS, because DLSS leaves
  trails behind changing digits. `[dlaa] cockpit_taa` in `qvfr.ini` sets where this applies. The default, 1, covers the
  displays and mirrors, 0 uses DLSS everywhere, and 2 also covers the rest of your car. `[dlaa] display_pass=0` gives
  the displays and mirrors the game's TAA instead.
- **Hotkeys.** With **Debug hotkeys** on, right Ctrl + F1 switches between DLSS and the game's TAA while you drive. Right
  Ctrl + F12 steps through the exposures.
- The **Status** page shows whether DLSS ran at the last game start, and if not, why.

## NVIDIA DLSS 5 (alpha)

DLSS 5 is NVIDIA's neural rendering model. It reworks the light and materials of the finished image, in the focus and
in the outer view. It works on the DLSS image and uses its motion vectors and depth. Turning DLSS 5 on also turns DLSS
on, and turning DLSS off also turns DLSS 5 off.

- **Requirements.** DLSS 5 needs everything DLSS needs, and NVIDIA's model `nvngx_dlssnr.dll` in `x64\qvfr\`. The mod
  does not include the model. When the model is missing, the settings app shows a **Download** button under the DLSS 5
  switch. You can also download it here: https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.9.16
- **Turn it on.** Open the **Settings** page of the settings app, go to **Rendering** and turn on **NVIDIA DLSS 5**.
  DLSS 5 is off by default. The **DLSS 5** card has these options: intensity, style, preset, strength, local tone and
  structure, auto mask, and how the outer view goes through the model. For the outer view, you choose on or off, the
  resolution, the area, and evaluation together with the focus or separately.
- **Tuning panel in the headset.** Right Ctrl + F2 opens the panel while you drive, with or without Debug hotkeys. The
  arrow keys choose and change an option, Enter switches, and Esc closes the panel. The game does not receive these keys
  while the panel is open. The panel's **Neural Rendering** switch works whenever DLSS is running. Changes show at once,
  and **Save to qvfr.ini** keeps them. The panel also shows the frame rate, the CPU and GPU frame time, and the cost of
  DLSS 5.
- **Cost.** DLSS 5 takes about 4 ms of GPU time per frame on an RTX 5090, with the outer view at half resolution, which
  is the default.

## Requirements

- Automobilista 2 on Steam, in VR with DirectX 11, on 64-bit Windows 10 or 11.
- An OpenXR runtime, such as SteamVR, Pimax Play, Meta or Varjo Base, for the default OpenXR path. The OpenVR path needs
  SteamVR.
- Eye tracking is optional. Without it, you get fixed foveated rendering and the focus stays in the middle.
- MSAA or TAA in the game. MSAA has had the most testing. With TAA, the focus views keep their own TAA history.
- For DLSS: an NVIDIA RTX graphics card, and TAA on and MSAA off in the game.
- For DLSS 5: the same, and NVIDIA's model `nvngx_dlssnr.dll`, which the mod does not include.

## Compatibility notes

- **Pimax.** Turn off **Foveated rendering** in Pimax Play for AMS2. Its LibMagic does not work with AMS2, and the log
  warns about it.
- **Pimax's SteamVR driver and eye tracking.** With Pimax's own SteamVR driver, eye tracking works only on the OpenXR
  path. On the OpenVR path, the driver gives the game a wrong gaze and the focus does not follow your eyes. The mod warns
  about this at game start. Keep OpenXR as the VR API, which is the default, or use the CustomHeadsetOpenVR driver,
  which works on both paths.
- **OpenXR Toolkit.** On the OpenXR path, turn off Eye tracking, Foveated rendering and upscaling in the Toolkit for
  AMS2. If the game still closes at start with the OpenComposite error `XR_ERROR_ACTIONSETS_ALREADY_ATTACHED`, disable
  the Toolkit for AMS2. We have seen this error with some Toolkit forks. On the OpenVR path, the Toolkit does not work
  at all.
- **Meta Link and Air Link.** AMS2 tries its Oculus mode before OpenVR. With the Meta software installed, the game can
  start in Oculus mode, where the mod does not load. A community workaround is a dummy `LibOVRRT64_1.dll` in the folder
  with `AMS2.exe`, from https://github.com/rodolZ80/bypass-oculus-detector. We have not tested it with this mod. The mod
  does not support Quest Pro eye tracking through Meta Link. Meta's PC runtime offers it only through its own
  extension, and the mod does not read that extension.

## Troubleshooting

Turn these options on in the **Debug** section of the **Settings** page:

- **Benchmark hotkey right Ctrl + F9.** It freezes the head pose and records the frame times. A 5 second countdown comes
  first, so keep your head still. Recording takes 30 seconds, with beeps at the start and the end. The results go to
  `x64\qvfr\bench\`.
- **Frame timing CSV for the whole session.** It records the same timings for the whole drive in `x64\qvfr\bench\`.
- **Show the gaze in the game**, in the Eye tracking section. It draws a white ring where you look. The focus edges show
  in red where the blend starts and in green where the image is fully sharp.
- **Debug hotkeys.** With them on, these keys work with right Ctrl while you drive:
  - F1 switches between DLSS and the game's TAA.
  - F3 switches the focus between its own sun effects and those of the outer view. The sun effects are crepuscular
    rays and lens flares.
  - F4 turns focus culling on or off.
  - F5 turns clones on or off.
  - F7 freezes the gaze.
  - F8 turns the gaze ring and the focus edges on or off.
  - F10 saves a screenshot of the views.
  - F11 switches between quad views and the outer view alone.
  - F12 steps through the DLSS exposures.

### Issue reporting

To report a problem, open an issue on GitHub: https://github.com/hypeapps/ams2-qvfr/issues. Please attach:

- the newest log from `x64\qvfr\logs\`, named `qvfr_<date>_<time>_AMS2AVX.log`. Drive for at least 20 seconds first,
  because the mod writes the block with what it detected after that time. Add `x64\qvfr\bench\` if you ran a benchmark.
- your headset, the VR API and the OpenXR runtime, and what you saw.

## License

AMS2 QVFR is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You can use, change and share it for
any noncommercial purpose. [THIRD_PARTY.md](THIRD_PARTY.md) lists the third-party software and its licenses. The mod
includes OpenComposite unmodified, under the GPLv3. THIRD_PARTY.md and `x64\qvfr\opencomposite\SOURCE.txt` link to its
source code. The mod includes NVIDIA DLSS unmodified, under the NVIDIA RTX SDKs License. The license text is in
`x64\qvfr\licenses\nvidia-rtx-sdks-LICENSE.txt`. The mod does not include NVIDIA's DLSS 5 model, `nvngx_dlssnr.dll`.
NVIDIA and DLSS are trademarks of NVIDIA Corporation. NVIDIA does not sponsor or endorse this mod.

Thanks to Matthieu Bucchianeri for Quad-Views-Foveated. The mod follows its settings and its focus edge. Thanks also to
the OpenComposite project.

Thanks to Dan Suzuki for the inspiration. His [LMU Quad View mod](https://github.com/DSuzuki35/LMU-Quad-View-Mod) showed
that this is possible today.

## Building from source

You need Windows, Visual Studio with the MSVC x64 tools, CMake 3.28 or newer, and Ninja. All dependencies are in
`third_party/`, except the NVIDIA DLSS SDK. `tools\fetch_dlss_sdk.ps1` downloads it from NVIDIA. The release zip needs
the SDK.

```
.\tools\fetch_dlss_sdk.ps1   # downloads the NVIDIA DLSS SDK files; without them the proxy has no DLSS
.\build.ps1                  # Release build; .\build.ps1 -Config Debug makes a debug build
.\tools\package.ps1          # makes the release zip in dist\
.\tools\install_proxy.ps1    # developer install into the Steam game folder, with a backup of the original DLL
```

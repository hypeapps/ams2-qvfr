# AMS2 QVFR

**EXPERIMENTAL** Quad views foveated rendering for **Automobilista 2** in VR 

The game renders each eye in two parts: the whole field of view at a lower pixel density (the *outer* view) and a
smaller area at full density (the *focus* view). With eye tracking the focus follows your gaze; without it, it stays in
the middle of the view. The mod puts both together for each eye before the image goes to the headset. You get a sharp
image where you look and render far fewer pixels, so you have headroom for higher settings or a steadier frame rate.
MSAA keeps working.

## Install

1. In Steam: right-click Automobilista 2 → **Manage** → **Browse local files**. This opens the folder with `AMS2.exe`.
2. Copy `x64\openvr_api.dll` from that folder to a safe place outside the game folder. If you already run AMS2 through
   your own OpenComposite, that file is your OpenComposite: keep the copy (see [VR API](#vr-api-openvr-or-openxr)).
3. Extract the zip into the folder with `AMS2.exe` and confirm overwriting `x64\openvr_api.dll`.
4. Settings: `x64\qvfr\qvfr_config.exe`
5. Choose the VR API in `qvfr_config.exe` → **Settings** → **VR API**. The mod works
   with both: **OpenVR** (the default); **OpenXR** goes through OpenComposite, which comes with the mod,
   to the OpenXR runtime active in your system. Details: [VR API](#vr-api-openvr-or-openxr).
6. Then start the game as usual.

Everything the mod needs is in `x64\qvfr\`

**Update:** extract a new version the same way. Your `x64\qvfr\qvfr.ini` stays as it is.

**After Steam verifies the game files or updates the game,** Steam puts back its own `x64\openvr_api.dll` and the mod is
off. Open `qvfr_config.exe` → **Files** → **Repair**.

**Uninstall:** `qvfr_config.exe` → **Files** → **Turn the mod off** (puts the Valve DLL back), then delete `x64\qvfr`. Or
let Steam verify the game files and delete `x64\qvfr`.

## Settings

**IMPORTANT: The mod does its own quad views inside the game.** It does not ask the VR runtime for quad views, so runtime-side
  quad views (mbucchia's Quad-Views-Foveated layer, Pimax quad views, VectorXR) have no effect on AMS2 and need no change.

You can customize setting using GUI: x64\qvfr\qvfr_config.exe

- **Settings:** quad views on/off, VR API, eye tracking (with the time the focus waits when the tracker loses your eyes,
  and a switch that shows your gaze and the focus edges in the game), focus position, debug options.
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
- Serious problems found at game start (TAA on, Pimax foveated rendering on, unsupported game build) also show in a
  Windows message box; the game keeps running.

## VR API: OpenVR / OpenXR

Choose it in the GUI (**Settings → VR API**) or with `[runtime] path` in `qvfr.ini`; it applies at the next game start.

- **OpenVR** (default): the game talks to SteamVR through the Valve DLL in `x64\qvfr\openvr\`. Eye tracking: what SteamVR
  gives games (tested with the Pimax Dream Air).
- **OpenXR:** the game goes through OpenComposite, included with the mod in `x64\qvfr\opencomposite\`, to the OpenXR
  runtime active in your system (SteamVR, Pimax Play, Meta, Varjo Base…). The mod never changes which runtime is active.
  Eye tracking comes from the runtime's `XR_EXT_eye_gaze_interaction`, read by a small OpenXR layer of the mod that only
  the game process loads. To use your own OpenComposite build instead, set `[runtime] opencomposite=` to its full path.

## Requirements

- Automobilista 2 on Steam, in VR (DirectX 11), Windows 10 or 11, 64-bit.
- SteamVR, or any OpenXR runtime when you use the OpenXR path.
- Eye tracking is optional. Without it you get fixed foveated rendering (the focus stays in the middle).
- MSAA in the game. Temporal anti-aliasing (TAA) is not supported: with TAA the focus area can smear or flicker.

## Compatibility notes

- **Pimax:** turn off **Foveated rendering** in Pimax Play for AMS2 (its LibMagic does not work with AMS2; the log warns).
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
- **Debug hotkeys:** right Ctrl + F5 clones, F7 freeze the gaze, F8 overlay, F10 screenshot of the views, F11 outer only.

### Issue reporting

Report a problem by opening an issue on GitHub: https://github.com/hypeapps/ams2-qvfr/issues. Please attach:

- the newest log from `x64\qvfr\logs\` (`qvfr_<date>_<time>_AMS2AVX.log`). Drive at least 20 s: the block with what
  the mod detected is written then. Add `x64\qvfr\bench\` if you ran a benchmark;
- your headset, the VR API (OpenVR or OpenXR) and the OpenXR runtime, and what you saw.

## License

AMS2 QVFR is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): free to use, change and share for
any noncommercial purpose. Third-party software and its licenses: [THIRD_PARTY.md](THIRD_PARTY.md). OpenComposite is
included unmodified under the GPLv3; its source code is linked there and in `x64\qvfr\opencomposite\SOURCE.txt`.

Thanks to Matthieu Bucchianeri for Quad-Views-Foveated, whose settings and focus edge the mod follows, and to the
OpenComposite project.

Thanks to Dan Suzuki for the inspiration: his [LMU Quad View mod](https://github.com/DSuzuki35/LMU-Quad-View-Mod) showed
that this is possible today.

## Building from source

Windows, Visual Studio with the MSVC x64 tools, CMake 3.28 or newer and Ninja. All dependencies are in `third_party/`.

```
.\build.ps1                  # Release; .\build.ps1 -Config Debug for a debug build
.\tools\package.ps1          # the release zip in dist\
.\tools\install_proxy.ps1    # developer install into the Steam game folder (backs up the original DLL)
```

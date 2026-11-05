# Third-party software

AMS2 QVFR itself is licensed under the PolyForm Noncommercial License 1.0.0 ([LICENSE.md](LICENSE.md)). It ships with
and is built from the third-party software below. In the release package the license texts are in `x64\qvfr\licenses\`,
and OpenComposite's next to its DLL in `x64\qvfr\opencomposite\`. Versions and download details of the vendored sources:
[third_party/README.md](third_party/README.md).

## What each file of the package contains

| File in the game folder | Third-party parts | License |
|---|---|---|
| `x64\openvr_api.dll` (the mod, also as the spare copy `x64\qvfr\proxy\openvr_api.dll`) | MinHook 1.3.4; OpenVR SDK headers; OpenXR SDK headers; the soft focus edge curve of Quad-Views-Foveated; NVIDIA DLSS SDK 310.9.16 (NGX static library, object code) | BSD-2-Clause; BSD-3-Clause; Apache-2.0 OR MIT; MIT; NVIDIA RTX SDKs License |
| `x64\qvfr\nvngx_dlss.dll` | NVIDIA DLSS 310.9.16 runtime, unmodified | NVIDIA RTX SDKs License (see below) |
| `x64\qvfr\openvr\openvr_api.dll` | Valve OpenVR 0.9.12, unmodified (the same file AMS2 ships) | BSD-3-Clause |
| `x64\qvfr\opencomposite\openvr_api.dll` | OpenComposite 1.0.1539, unmodified official build | GPLv3 (see below) |
| `x64\qvfr\qvfr_config.exe` | RmlUi 6.3; FreeType 2.15.0; fonts Inter 4.1, Barlow Condensed and Tabler Icons 3.49.0 (cut to the used glyphs) | MIT; FreeType License; SIL Open Font License 1.1 (Inter, Barlow Condensed); MIT (Tabler Icons) |

## Components

- **MinHook** by Tsuda Kageyu, https://github.com/TsudaKageyu/minhook. BSD-2-Clause, `minhook-LICENSE.txt`.
- **OpenVR SDK** by Valve Corporation, https://github.com/ValveSoftware/openvr (0.9.12 DLL and headers, 2.15.6 headers).
  BSD-3-Clause, `openvr-LICENSE.txt`.
- **OpenXR SDK** headers by The Khronos Group, https://github.com/KhronosGroup/OpenXR-SDK. Apache-2.0 OR MIT,
  `openxr-sdk-LICENSE.txt`.
- **Quad-Views-Foveated** by Matthieu Bucchianeri, https://github.com/mbucchia/Quad-Views-Foveated. The curve of the soft
  edge between the focus and the outer view in the mod's composition shader follows its `ProjectionPS.hlsl`. MIT License,
  notice below.
- **OpenComposite** (OpenOVR) by ZNix and contributors, https://gitlab.com/znixian/OpenOVR. Shipped unmodified as the
  backend of `[runtime] path=openxr`: the official AppVeyor build 1.0.1539 of commit
  `a27e7e6a64bdcd1eff6b7fba1ea2ea34bcf1273d`. GNU General Public License v3 (`opencomposite\LICENSE.txt`), with parts under
  the licenses in the other files of that folder. The complete corresponding source code of this build is at
  https://gitlab.com/znixian/OpenOVR/-/tree/a27e7e6a64bdcd1eff6b7fba1ea2ea34bcf1273d (with its submodules). The mod only
  loads this DLL; it is not part of the mod's code and may be replaced by another OpenComposite build.
- **RmlUi** by Michael R. P. Ragazzon and contributors, https://github.com/mikke89/RmlUi. MIT, `rmlui-LICENSE.txt`.
- **FreeType**, https://freetype.org. FreeType License, `freetype-FTL.txt`. Portions of this software are copyright ©
  2026 The FreeType Project (https://freetype.org). All rights reserved.
- **Inter** by Rasmus Andersson, https://github.com/rsms/inter. SIL Open Font License 1.1, `inter-OFL.txt`.
- **Barlow Condensed** by Jeremy Tribby, https://github.com/google/fonts (`ofl/barlowcondensed`). SIL Open Font License
  1.1, `barlowcondensed-OFL.txt`.
- **Tabler Icons** by Paweł Kuna, https://github.com/tabler/tabler-icons. MIT, `tabler-icons-LICENSE.txt`.
- **NVIDIA DLSS SDK** 310.9.16 by NVIDIA Corporation, https://github.com/NVIDIA/DLSS (tag `v310.9.16`). The mod's optional
  DLAA (`[dlaa] mode=dlaa`) uses NVIDIA DLSS: the NGX static library is linked into `openvr_api.dll` and the DLSS runtime
  `nvngx_dlss.dll` ships unmodified in `x64\qvfr\`. Licensed under the NVIDIA RTX SDKs License
  (`nvidia-rtx-sdks-LICENSE.txt`), not under the mod's license: you may not reverse engineer, decompile, disassemble,
  modify or redistribute these NVIDIA parts on their own, and the license's limitations of warranty and liability apply.
  NVIDIA DLSS runs only on NVIDIA RTX graphics cards. NVIDIA and DLSS are trademarks of NVIDIA Corporation; the mod is not
  sponsored or endorsed by NVIDIA.

## Quad-Views-Foveated license notice

```
MIT License

Copyright (c) 2022-2023 Matthieu Bucchianeri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

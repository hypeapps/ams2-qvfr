# Third-party dependencies

Vendored into the repo (no submodules), only the files we need. Downloaded 2026-10-30.

| Directory | Source | Version (commit) | License | Used for |
|---|---|---|---|---|
| `openvr-2.15.6/` | https://github.com/ValveSoftware/openvr | v2.15.6 (`0924064316de3effbcd1acf1e309182a2deb1c05`) | BSD-3-Clause (`LICENSE`) | headers and `openvr_api.dll` for the tools; `IVRSystem_026`, `IVRInput_011` definitions |
| `openvr-0.9.12/` | https://github.com/ValveSoftware/openvr | v0.9.12 (`b4bf24578bec10d5dc6eb14f91b98e71d5f7ee1b`) | BSD-3-Clause (`LICENSE`) | interfaces used by AMS2; DLL for the gaze test in `legacy` mode and for the proxy |
| `minhook/` | https://github.com/TsudaKageyu/minhook | v1.3.4 (`c3fcafdc10146beb5919319d0683e44e3c30d537`) | BSD-2-Clause (`LICENSE.txt`) | hooks in the AMS2 executable (later milestones) |

`openvr-0.9.12/bin/win64/openvr_api.dll` has the same MD5 (`864be5a92276ce2b902b6f3a0f850703`)
as the game's original `x64\openvr_api.dll`, so the game ships the unmodified Valve DLL.

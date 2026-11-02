// OpenXR path tests of qvfr_proxy_smoke (M8.1): --xr-layer and --openxr-path.
#pragma once

#include <openvr.h>  // VR_CALLTYPE

#include <filesystem>

namespace fs = std::filesystem;

int RunXrLayer(const fs::path& dll, const fs::path& home);
int RunOpenXrPath(const fs::path& dll, const fs::path& home);

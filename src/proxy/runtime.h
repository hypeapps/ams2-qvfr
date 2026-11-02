// What stands behind the proxy (M8, [runtime]): the Valve openvr_api.dll (path=openvr) or OpenComposite (path=openxr).
// On the OpenXR path the proxy also prepares the process environment for OpenComposite's built-in OpenXR loader before
// the DLL is loaded: our API layer (a manifest that points at the proxy DLL itself) and optionally the runtime.
#pragma once

#include <windows.h>

#include <string>

namespace qvfr::runtime {

// After LoadConfig, before the backend is loaded. Returns the backend's full path (empty = could not build one).
std::wstring Prepare(HMODULE self);
// Startup log: backend, environment set for the loader, OpenXR implicit layers in the registry (with their state).
void LogStartup();
bool OpenXrPath();  // path=openxr was prepared

}  // namespace qvfr::runtime

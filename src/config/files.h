// Check and repair of the mod's files in the game's x64 folder (docs/architecture.md, "Instalacja: zip"). The zip
// overwrites x64\openvr_api.dll with the proxy and keeps a spare copy in x64\qvfr\proxy\, so the GUI can put the proxy
// back after Steam's file verification restored the Valve DLL, or put the Valve DLL back to turn the mod off.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qvfr::config {

enum class Status { Ok, Info, Warning, Error };

struct Check {
    Status status = Status::Ok;
    std::string item;    // what was checked
    std::string detail;  // what was found
};

enum class GameDll { Missing, Proxy, ProxyOther, Valve, OpenComposite, Unknown };

struct FileState {
    std::filesystem::path x64;          // the game's x64 folder (parent of the mod folder)
    bool gameFolder = false;            // ..\AMS2AVX.exe exists
    GameDll gameDll = GameDll::Missing; // x64\openvr_api.dll
    bool spare = false;                 // x64\qvfr\proxy\openvr_api.dll
    std::vector<Check> checks;
};

FileState CheckFiles(const std::filesystem::path& modDir, bool openxrPath);

// Both refuse while the game runs (the DLL is loaded). They return a message for the GUI.
// Puts the spare proxy into x64; a DLL there that is neither the Valve one nor a proxy goes to x64\qvfr\backup\ first.
std::string RepairProxy(const std::filesystem::path& modDir);
// Puts the Valve DLL from x64\qvfr\openvr\ into x64 (the mod is off; the files in x64\qvfr stay).
std::string RestoreValve(const std::filesystem::path& modDir);

bool GameRunning();  // AMS2AVX.exe or AMS2.exe is running

}  // namespace qvfr::config

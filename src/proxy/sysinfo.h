// Process and environment info: the exe and its SHA256, loaded modules, the game's graphics settings.
#pragma once

#include <string>

#include "common.h"

namespace qvfr::sysinfo {

std::string HashFile(const fs::path& path, const wchar_t* algorithm);  // BCRYPT_SHA256_ALGORITHM, BCRYPT_MD5_ALGORITHM

void LogProcessInfo();               // exe, command line, working directory, Windows; the exe SHA256 is computed in the background
void LogModules(const char* when);   // the full list the first time, then only new modules
void LogGameConfig();                // selected settings from Documents\Automobilista 2\*.xml
std::string GameSetting(const char* key);  // a value from graphicsconfigopenvrdx11.xml after LogGameConfig ("" = none)
std::string KnownModules();          // the known modules (ReShade, LibMagic...) seen so far, comma-separated
void CopyGameConfig(const fs::path& dir);  // copies the game configs into a measurement folder

}  // namespace qvfr::sysinfo

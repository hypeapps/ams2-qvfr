// The newest proxy log (x64\qvfr\logs\qvfr_<date>_<time>_<exe>.log): the SteamVR resolution, the Detected block and
// the WARNING/Note lines, for the GUI's status and resolution views.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qvfr::config {

struct LogInfo {
    bool found = false;
    std::filesystem::path file;
    std::string when;                      // "2026-11-02 21:24:24" from the file name
    unsigned gridW = 0, gridH = 0;         // SteamVR resolution per eye (0 = not in the log)
    bool openxr = false;                   // the run used path=openxr
    std::vector<std::string> detected;     // "Key: value" lines of the Detected block (without the time)
    std::vector<std::string> findings;     // "WARNING: ..." and "Note: ..." lines
    bool detectedWritten = false;          // false: the game ended before the block (or still runs)
};

LogInfo ReadNewestLog(const std::filesystem::path& logDir);

}  // namespace qvfr::config

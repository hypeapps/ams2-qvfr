#include "log_reader.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace qvfr::config {

namespace {

// "[    123.4] text" -> "text"
std::string StripTime(const std::string& line) {
    if (line.empty() || line[0] != '[') return line;
    const size_t close = line.find("] ");
    return close == std::string::npos ? line : line.substr(close + 2);
}

}  // namespace

LogInfo ReadNewestLog(const std::filesystem::path& logDir) {
    LogInfo info;
    std::error_code ec;
    std::filesystem::path newest;
    std::wstring newestName;
    for (const auto& e : std::filesystem::directory_iterator(logDir, ec)) {
        const std::wstring name = e.path().filename().wstring();
        // qvfr_YYYYMMDD_HHMMSS_<exe>.log sorts by time; the smoke test exe has its own name, the game is AMS2AVX.
        if (name.rfind(L"qvfr_", 0) != 0 || e.path().extension() != L".log") continue;
        if (name > newestName) {
            newestName = name;
            newest = e.path();
        }
    }
    if (newest.empty()) return info;
    std::ifstream in(newest, std::ios::binary);
    if (!in) return info;
    info.found = true;
    info.file = newest;
    if (newestName.size() >= 20) {
        const std::wstring d = newestName.substr(5, 15);  // YYYYMMDD_HHMMSS
        char buf[32];
        snprintf(buf, sizeof(buf), "%ls-%ls-%ls %ls:%ls:%ls", d.substr(0, 4).c_str(), d.substr(4, 2).c_str(),
                 d.substr(6, 2).c_str(), d.substr(9, 2).c_str(), d.substr(11, 2).c_str(), d.substr(13, 2).c_str());
        info.when = buf;
    }
    std::string line;
    bool inBlock = false;
    unsigned plainW = 0, plainH = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string t = StripTime(line);
        if (inBlock) {
            if (t.rfind("  ", 0) == 0) {
                info.detected.push_back(t.substr(2));
                continue;
            }
            inBlock = false;
        }
        if (t.rfind("Detected (", 0) == 0) {
            info.detected.clear();
            info.detectedWritten = true;
            inBlock = true;
            continue;
        }
        if (t.rfind("WARNING: ", 0) == 0 || t.rfind("Note: ", 0) == 0) info.findings.push_back(t);
        if (t.find("[runtime] path=openxr") != std::string::npos) info.openxr = true;
        // Quad mode: "... (quad focus size; SteamVR 8192 x 6864)"; otherwise the plain SteamVR answer.
        unsigned w = 0, h = 0;
        const size_t q = t.find("; SteamVR ");
        if (q != std::string::npos && sscanf(t.c_str() + q, "; SteamVR %u x %u", &w, &h) == 2 && !info.gridW) {
            info.gridW = w;
            info.gridH = h;
        }
        const char* tag = "GetRecommendedRenderTargetSize = ";
        const size_t r = t.find(tag);
        if (r != std::string::npos && t.find("quad focus size") == std::string::npos &&
            t.find("experiment") == std::string::npos && !plainW &&
            sscanf(t.c_str() + r + strlen(tag), "%u x %u", &w, &h) == 2) {
            plainW = w;
            plainH = h;
        }
    }
    if (!info.gridW) {
        info.gridW = plainW;
        info.gridH = plainH;
    }
    return info;
}

}  // namespace qvfr::config

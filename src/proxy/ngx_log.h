// Filter for NVIDIA NGX log callbacks. NGX writes hundreds of lines per DLSS feature (its whole configuration and
// every model search), so the proxy keeps only what can explain a problem, its init lines (DLL path, version) and,
// while a feature is created, the line with the preset NGX really took for that feature's quality mode.
#pragma once

#include <string>
#include <string_view>

namespace qvfr::ngxlog {

enum class Kind {
    Drop,
    Problem,  // an error, warning or failure
    Init,     // NGXInitLog: where nvngx_dlss.dll came from, its build
    Preset,   // "Using App hint Preset X" / "using title default Preset X" of the quality mode being created
};

inline std::string Lower(std::string_view s) {
    std::string low(s);
    for (char& ch : low) ch = (char)(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
    return low;
}

// NGX's searches before it finds its core and the model files fail on every start; they explain nothing.
inline bool Noise(const std::string& low) {
    return low.find("failed to load ngxcore") != std::string::npos || low.find("from cache") != std::string::npos ||
           low.find("ngxloadfrompath failed") != std::string::npos;
}

inline bool Problem(const std::string& low) {
    return low.find("error") != std::string::npos || low.find("warn") != std::string::npos ||
           low.find("fail") != std::string::npos;
}

// NGX's name of a DLSS quality mode as its preset lines write it, e.g. "(DLAA)", "(Quality)", "(UltraPerf)".
inline const char* ModeTag(int perfQuality) {
    switch (perfQuality) {  // NVSDK_NGX_PerfQuality_Value
        case 0: return "(Perf)";
        case 1: return "(Balanced)";
        case 2: return "(Quality)";
        case 3: return "(UltraPerf)";
        case 4: return "(UltraQuality)";
        case 5: return "(DLAA)";
        default: return nullptr;
    }
}

// `mode`: ModeTag of the feature being created, nullptr outside a creation. NGX writes a preset line for every quality
// mode at each creation; only the one of `mode` is the feature's. "(Quality)" is not part of "(UltraQuality)".
inline Kind Classify(std::string_view line, const char* mode) {
    const std::string low = Lower(line);
    if (low.find("using app hint preset") != std::string::npos ||
        low.find("using title default preset") != std::string::npos)
        return mode && low.find(Lower(mode)) != std::string::npos ? Kind::Preset : Kind::Drop;
    if (Noise(low)) return Kind::Drop;
    if (Problem(low)) return Kind::Problem;
    if (low.find("[ngxinitlog:") != std::string::npos) return Kind::Init;
    return Kind::Drop;
}

}  // namespace qvfr::ngxlog

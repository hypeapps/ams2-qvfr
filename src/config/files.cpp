#include "files.h"

#include <windows.h>

#include <bcrypt.h>
#include <tlhelp32.h>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

namespace qvfr::config {

namespace fs = std::filesystem;

namespace {

constexpr const char* kValveMd5 = "864be5a92276ce2b902b6f3a0f850703";  // official openvr_api.dll 0.9.12 (win64)
constexpr const char* kProxyMark = "ams2-qvfr openvr_api.dll proxy";  // the proxy's log header text

std::string ReadAll(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string Md5(const std::string& data) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h = nullptr;
    std::string out;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_MD5_ALGORITHM, nullptr, 0) == 0 &&
        BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0) {
        BCryptHashData(h, (PUCHAR)data.data(), (ULONG)data.size(), 0);
        unsigned char d[16];
        if (BCryptFinishHash(h, d, 16, 0) == 0) {
            char hex[3];
            for (unsigned char b : d) {
                snprintf(hex, sizeof(hex), "%02x", b);
                out += hex;
            }
        }
    }
    if (h) BCryptDestroyHash(h);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    return out;
}

GameDll Classify(const std::string& dll, const std::string& spare) {
    if (dll.empty()) return GameDll::Missing;
    if (!spare.empty() && dll == spare) return GameDll::Proxy;
    if (dll.find(kProxyMark) != std::string::npos) return spare.empty() ? GameDll::Proxy : GameDll::ProxyOther;
    if (Md5(dll) == kValveMd5) return GameDll::Valve;
    if (dll.find("OpenComposite") != std::string::npos) return GameDll::OpenComposite;
    return GameDll::Unknown;
}

std::string Copy(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    return ec ? "Copy failed: " + ec.message() : std::string();
}

std::string Stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

}  // namespace

bool GameRunning() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W e{sizeof(e)};
    bool found = false;
    for (BOOL ok = Process32FirstW(snap, &e); ok && !found; ok = Process32NextW(snap, &e))
        found = _wcsicmp(e.szExeFile, L"AMS2AVX.exe") == 0 || _wcsicmp(e.szExeFile, L"AMS2.exe") == 0;
    CloseHandle(snap);
    return found;
}

FileState CheckFiles(const fs::path& modDir, bool openxrPath, bool dlaa) {
    FileState st;
    st.x64 = modDir.parent_path();
    std::error_code ec;
    auto add = [&](Status s, const std::string& item, const std::string& detail) { st.checks.push_back({s, item, detail}); };
    st.gameFolder = fs::exists(st.x64.parent_path() / L"AMS2AVX.exe", ec);
    add(st.gameFolder ? Status::Ok : Status::Error, "Game folder",
        st.gameFolder ? st.x64.parent_path().string()
                      : "AMS2AVX.exe not found next to " + st.x64.string() + ": the mod folder must be <game>\\x64\\qvfr");

    const std::string spare = ReadAll(modDir / L"proxy" / L"openvr_api.dll");
    st.spare = !spare.empty();
    st.gameDll = Classify(ReadAll(st.x64 / L"openvr_api.dll"), spare);
    switch (st.gameDll) {
    case GameDll::Proxy: add(Status::Ok, "x64\\openvr_api.dll", "the mod (active)"); break;
    case GameDll::ProxyOther:
        add(Status::Warning, "x64\\openvr_api.dll", "the mod, but not the build in x64\\qvfr\\proxy (an older install?): Repair puts that one in");
        break;
    case GameDll::Valve:
        add(Status::Error, "x64\\openvr_api.dll",
            "the original Valve DLL: the mod is OFF (Steam's file verification restores it). Repair turns the mod on");
        break;
    case GameDll::OpenComposite:
        add(Status::Error, "x64\\openvr_api.dll",
            "OpenComposite: the mod is OFF. Repair puts the mod in (the DLL is kept in x64\\qvfr\\backup); the mod uses "
            "its own OpenComposite with path=openxr");
        break;
    case GameDll::Unknown:
        add(Status::Error, "x64\\openvr_api.dll", "an unknown DLL: the mod is OFF. Repair puts the mod in (the DLL is kept in x64\\qvfr\\backup)");
        break;
    case GameDll::Missing: add(Status::Error, "x64\\openvr_api.dll", "missing: VR does not start. Repair puts the mod in"); break;
    }
    add(st.spare ? Status::Ok : Status::Warning, "x64\\qvfr\\proxy\\openvr_api.dll",
        st.spare ? "spare copy of the mod (for Repair)" : "missing: Repair is not possible, unpack the mod's zip again");

    const std::string valve = ReadAll(modDir / L"openvr" / L"openvr_api.dll");
    add(valve.empty() ? Status::Error : Md5(valve) == kValveMd5 ? Status::Ok : Status::Warning, "x64\\qvfr\\openvr\\openvr_api.dll",
        valve.empty() ? "missing: path=openvr cannot start VR"
        : Md5(valve) == kValveMd5 ? "Valve OpenVR 0.9.12 (path=openvr)"
                                  : "not the official Valve 0.9.12 DLL");
    const bool oc = fs::exists(modDir / L"opencomposite" / L"openvr_api.dll", ec);
    add(oc ? Status::Ok : openxrPath ? Status::Error : Status::Info, "x64\\qvfr\\opencomposite\\openvr_api.dll",
        oc ? "OpenComposite (path=openxr)" : openxrPath ? "missing: path=openxr cannot start VR" : "missing (needed only for path=openxr)");
    const bool manifest = fs::exists(modDir / L"XR_APILAYER_AMS2QVFR_eye_gaze.json", ec);
    add(manifest ? Status::Ok : openxrPath ? Status::Error : Status::Info, "x64\\qvfr\\XR_APILAYER_AMS2QVFR_eye_gaze.json",
        manifest ? "OpenXR layer manifest (eye tracking on path=openxr)"
                 : openxrPath ? "missing: no eye tracking on path=openxr" : "missing (needed only for path=openxr)");
    st.dlss = fs::exists(modDir / L"nvngx_dlss.dll", ec);
    add(st.dlss ? Status::Ok : dlaa ? Status::Error : Status::Info, "x64\\qvfr\\nvngx_dlss.dll",
        st.dlss ? "NVIDIA DLSS (NVIDIA DLAA)" : dlaa ? "missing: NVIDIA DLAA cannot start, the game's TAA stays"
                                                    : "missing (needed only for NVIDIA DLAA)");
    return st;
}

std::string RepairProxy(const fs::path& modDir) {
    if (GameRunning()) return "Close the game first.";
    const fs::path x64 = modDir.parent_path(), spare = modDir / L"proxy" / L"openvr_api.dll", dll = x64 / L"openvr_api.dll";
    const std::string spareData = ReadAll(spare);
    if (spareData.empty()) return "x64\\qvfr\\proxy\\openvr_api.dll is missing: unpack the mod's zip again.";
    const GameDll now = Classify(ReadAll(dll), spareData);
    if (now == GameDll::OpenComposite || now == GameDll::Unknown) {
        std::error_code ec;
        fs::create_directories(modDir / L"backup", ec);
        const fs::path keep = modDir / L"backup" / (L"openvr_api_" + fs::path(Stamp()).wstring() + L".dll");
        const std::string err = Copy(dll, keep);
        if (!err.empty()) return "The current DLL could not be kept, nothing changed. " + err;
    }
    const std::string err = Copy(spare, dll);
    return err.empty() ? "The mod is on (x64\\openvr_api.dll is the mod's proxy)." : err;
}

std::string RestoreValve(const fs::path& modDir) {
    if (GameRunning()) return "Close the game first.";
    const fs::path valve = modDir / L"openvr" / L"openvr_api.dll";
    if (Md5(ReadAll(valve)) != kValveMd5) return "x64\\qvfr\\openvr\\openvr_api.dll is not the Valve DLL, nothing changed.";
    const std::string err = Copy(valve, modDir.parent_path() / L"openvr_api.dll");
    return err.empty() ? "The mod is off (x64\\openvr_api.dll is the original Valve DLL). Repair turns it on again." : err;
}

}  // namespace qvfr::config

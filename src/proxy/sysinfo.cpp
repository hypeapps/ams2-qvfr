#include "sysinfo.h"

#include <bcrypt.h>
#include <psapi.h>
#include <shlobj.h>

#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

namespace qvfr::sysinfo {

namespace {

std::mutex g_modMutex;
std::set<std::wstring> g_seenModules;

// Modules worth flagging in the log (file name fragment, lower case) and what they are.
struct Known {
    const wchar_t* part;
    const char* what;
};
const Known kKnown[] = {
    {L"reshade", "ReShade"},
    {L"rtsshooks", "RivaTuner Statistics Server"},
    {L"gameoverlayrenderer", "Steam overlay"},
    {L"libmagic", "Pimax LibMagic (DFR Pimax)"},
    {L"pimax", "Pimax module"},
    {L"fpsvr", "fpsVR"},
    {L"openxr", "OpenXR"},
    {L"vrclient", "SteamVR client"},
    {L"nvapi", "NVAPI"},
    {L"d3d11.dll", "Direct3D 11"},
    {L"dxgi.dll", "DXGI"},
    {L"opencomposite", "OpenComposite"},
    {L"discord", "Discord"},
    {L"overwolf", "Overwolf"},
    {L"nvcamera", "NVIDIA Ansel/Freestyle"},
    {L"bugsplat", "BugSplat"},
    {L"tobii", "Tobii"},
    {L"msiafterburner", "MSI Afterburner"},
};

std::wstring Lower(std::wstring s) {
    for (auto& c : s) c = (wchar_t)towlower(c);
    return s;
}

std::string FileVersion(const std::wstring& path) {
    DWORD h = 0;
    DWORD size = GetFileVersionInfoSizeW(path.c_str(), &h);
    if (!size) return {};
    std::vector<uint8_t> data(size);
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data())) return {};
    VS_FIXEDFILEINFO* fi = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(data.data(), L"\\", (void**)&fi, &len) || !fi) return {};
    char buf[64];
    snprintf(buf, sizeof(buf), "%u.%u.%u.%u", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS),
             HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
    return buf;
}

fs::path GameDocsDir() {
    PWSTR docs = nullptr;
    fs::path p;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &docs))) {
        p = fs::path(docs) / L"Automobilista 2";
        CoTaskMemFree(docs);
    }
    return p;
}

}  // namespace

std::string HashFile(const fs::path& path, const wchar_t* algorithm) {
    std::string result;
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) return "(file not found)";
    if (BCryptOpenAlgorithmProvider(&alg, algorithm, nullptr, 0) == 0 &&
        BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) == 0) {
        std::vector<uint8_t> buf(1 << 20);
        size_t n;
        while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) BCryptHashData(hash, buf.data(), (ULONG)n, 0);
        DWORD len = 0, cb = 0;
        BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, (PUCHAR)&len, sizeof(len), &cb, 0);
        std::vector<uint8_t> digest(len);
        if (BCryptFinishHash(hash, digest.data(), len, 0) == 0) {
            char hex[3];
            for (uint8_t b : digest) {
                snprintf(hex, sizeof(hex), "%02x", b);
                result += hex;
            }
        }
    }
    if (hash) BCryptDestroyHash(hash);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    fclose(f);
    return result;
}

void LogProcessInfo() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::error_code ec;
    auto size = fs::file_size(exe, ec);
    Log("Process: %s (PID %lu, %llu B)", Narrow(exe).c_str(), GetCurrentProcessId(), (unsigned long long)size);
    Log("Command line: %s", Narrow(GetCommandLineW()).c_str());
    Log("Working directory: %s", Narrow(fs::current_path(ec).wstring()).c_str());

    // RtlGetVersion reports the real Windows version (GetVersionEx lies without a manifest).
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    if (auto fn = (RtlGetVersionFn)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")) {
        OSVERSIONINFOW v{sizeof(v)};
        if (fn(&v) == 0) Log("Windows %lu.%lu build %lu", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber);
    }

    // The exe SHA256 (about 43 MB) is computed in the background so it does not delay VR_Init.
    std::wstring path = exe;
    std::thread([path] {
        std::string h = HashFile(path, BCRYPT_SHA256_ALGORITHM);
        Log("SHA256 %s = %s", Narrow(fs::path(path).filename().wstring()).c_str(), h.c_str());
    }).detach();
}

void LogModules(const char* when) {
    std::lock_guard lock(g_modMutex);
    HMODULE mods[1024];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return;
    size_t count = needed / sizeof(HMODULE);
    if (count > 1024) count = 1024;
    bool first = g_seenModules.empty();
    int added = 0;
    std::vector<std::string> known;
    for (size_t i = 0; i < count; ++i) {
        wchar_t path[MAX_PATH] = {};
        if (!GetModuleFileNameW(mods[i], path, MAX_PATH)) continue;
        std::wstring lower = Lower(path);
        if (!g_seenModules.insert(lower).second) continue;
        ++added;
        std::string ver = FileVersion(path);
        if (added == 1) Log("Modules in process (%s)%s:", when, first ? "" : ", new since the previous list");
        Log("  %p %s%s%s", (void*)mods[i], Narrow(path).c_str(), ver.empty() ? "" : "  v", ver.c_str());
        std::wstring name = Lower(fs::path(path).filename().wstring());
        for (const auto& k : kKnown) {
            if (name.find(k.part) != std::wstring::npos) {
                known.push_back(std::string(k.what) + " (" + Narrow(fs::path(path).filename().wstring()) + ")");
                break;
            }
        }
    }
    if (!added && !first) Log("Modules in process (%s): unchanged", when);
    if (!known.empty()) {
        std::string s;
        for (auto& k : known) s += (s.empty() ? "" : ", ") + k;
        Log("  known modules: %s", s.c_str());
    }
}

void LogGameConfig() {
    fs::path dir = GameDocsDir();
    for (const wchar_t* name : {L"graphicsconfigopenvrdx11.xml", L"openvrsettings.xml"}) {
        std::ifstream in(dir / name);
        if (!in) {
            Log("Game config %s: missing", Narrow(name).c_str());
            continue;
        }
        // The <data> part: lines like <prop name="X" key="value" ... />. Key=value pairs are logged.
        std::string line, out;
        bool data = false;
        while (std::getline(in, line)) {
            if (line.find("<data") != std::string::npos) data = true;
            if (!data) continue;
            size_t p = line.find("<prop name=\"");
            if (p == std::string::npos) continue;
            size_t q = line.find('"', p + 12);
            if (q == std::string::npos) continue;
            std::string rest = line.substr(q + 1);
            size_t end = rest.rfind("/>");
            if (end != std::string::npos) rest = rest.substr(0, end);
            while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
            while (!rest.empty() && rest.back() == ' ') rest.pop_back();
            if (rest.empty()) continue;
            // openvrsettings.xml uses <prop name="NearZ" data="0.1" />: "data" is replaced with the name.
            if (rest.rfind("data=", 0) == 0) rest = line.substr(p + 12, q - (p + 12)) + rest.substr(4);
            out += (out.empty() ? "" : " ") + rest;
        }
        Log("Game config %s: %s", Narrow(name).c_str(), out.c_str());
    }
}

void CopyGameConfig(const fs::path& target) {
    fs::path dir = GameDocsDir();
    std::error_code ec;
    for (const wchar_t* name : {L"graphicsconfigopenvrdx11.xml", L"openvrsettings.xml"}) {
        fs::copy_file(dir / name, target / name, fs::copy_options::overwrite_existing, ec);
    }
}

}  // namespace qvfr::sysinfo

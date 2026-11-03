#include "ini_file.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace qvfr::config {

namespace {

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}

bool SameNoCase(const std::string& a, const char* b) { return _stricmp(a.c_str(), b) == 0; }

// "[name]" -> name; empty when the line is not a section header.
std::string SectionOf(const std::string& line) {
    const std::string t = Trim(line);
    if (t.size() < 2 || t.front() != '[') return {};
    const size_t close = t.find(']');
    return close == std::string::npos ? std::string() : Trim(t.substr(1, close - 1));
}

// "key=value" -> key (trimmed); empty for comments, blank lines and sections.
std::string KeyOf(const std::string& line, std::string* value = nullptr) {
    const std::string t = Trim(line);
    if (t.empty() || t[0] == ';' || t[0] == '#' || t[0] == '[') return {};
    const size_t eq = t.find('=');
    if (eq == std::string::npos) return {};
    if (value) *value = Trim(t.substr(eq + 1));
    return Trim(t.substr(0, eq));
}

}  // namespace

bool IniFile::Load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        lines_.clear();
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    LoadText(ss.str());
    return true;
}

void IniFile::LoadText(const std::string& text) {
    lines_.clear();
    std::string line;
    std::istringstream in(text);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines_.push_back(line);
    }
}

bool IniFile::Get(const char* section, const char* key, std::string& value) const {
    bool in = false;
    for (const std::string& l : lines_) {
        const std::string sec = SectionOf(l);
        if (!sec.empty()) {
            in = SameNoCase(sec, section);
            continue;
        }
        std::string v;
        if (in && SameNoCase(KeyOf(l, &v), key)) {
            value = v;
            return true;
        }
    }
    return false;
}

std::string IniFile::GetOr(const char* section, const char* key, const char* def) const {
    std::string v;
    return Get(section, key, v) ? v : std::string(def);
}

bool IniFile::Set(const char* section, const char* key, const std::string& value) {
    bool in = false;
    int lastInSection = -1;  // last key line (or the header) of the section
    for (size_t i = 0; i < lines_.size(); ++i) {
        const std::string sec = SectionOf(lines_[i]);
        if (!sec.empty()) {
            in = SameNoCase(sec, section);
            if (in) lastInSection = (int)i;
            continue;
        }
        if (!in) continue;
        std::string v;
        const std::string k = KeyOf(lines_[i], &v);
        if (k.empty()) continue;
        lastInSection = (int)i;
        if (SameNoCase(k, key)) {
            if (v == value) return false;
            lines_[i] = k + "=" + value;
            return true;
        }
    }
    const std::string line = std::string(key) + "=" + value;
    if (lastInSection >= 0) {
        lines_.insert(lines_.begin() + lastInSection + 1, line);
    } else {
        if (!lines_.empty() && !Trim(lines_.back()).empty()) lines_.push_back("");
        lines_.push_back(std::string("[") + section + "]");
        lines_.push_back(line);
    }
    return true;
}

std::string IniFile::Text() const {
    std::string s;
    for (const std::string& l : lines_) s += l + "\n";
    return s;
}

bool IniFile::Save(const std::filesystem::path& path) {
    std::filesystem::path tmp = path;
    tmp += L".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        const std::string t = Text();
        out.write(t.data(), (std::streamsize)t.size());
        if (!out) return false;
    }
    // The proxy may have the file open for a moment (GetPrivateProfileString), so a few short retries.
    for (int i = 0; i < 20; ++i) {
        if (MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
        Sleep(5);
    }
    DeleteFileW(tmp.c_str());
    return false;
}

unsigned long long FileTime(const std::filesystem::path& path) {
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &a)) return 0;
    return (unsigned long long)a.ftLastWriteTime.dwHighDateTime << 32 | a.ftLastWriteTime.dwLowDateTime;
}

}  // namespace qvfr::config

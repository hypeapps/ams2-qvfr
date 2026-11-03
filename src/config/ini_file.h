// qvfr.ini as lines, so the GUI changes values and keeps every comment, order and unknown key. Saving writes a
// temporary file and moves it over the old one, so the proxy (which re-reads the file about once a second while the
// game runs) never sees a half-written file.
#pragma once

#include <windows.h>

#include <filesystem>
#include <string>
#include <vector>

namespace qvfr::config {

class IniFile {
public:
    bool Load(const std::filesystem::path& path);   // false when the file cannot be read (the document stays empty)
    void LoadText(const std::string& text);         // from memory (default ini, tests)
    // Value of [section] key; false when the key is not there. An empty value ("key=") counts as present but empty.
    bool Get(const char* section, const char* key, std::string& value) const;
    std::string GetOr(const char* section, const char* key, const char* def) const;
    // Changes the value in place, or adds the key at the end of its section (the section at the end of the file).
    // Returns true when the text changed.
    bool Set(const char* section, const char* key, const std::string& value);
    bool Save(const std::filesystem::path& path);   // atomic replace; false when it failed (retry later)
    std::string Text() const;

private:
    std::vector<std::string> lines_;
};

// Last write time of the file (0 when it does not exist).
unsigned long long FileTime(const std::filesystem::path& path);

}  // namespace qvfr::config

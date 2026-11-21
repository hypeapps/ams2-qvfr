#include "xr_layer_order.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <fstream>
#include <set>
#include <sstream>

#include "common.h"

namespace qvfr::xrorder {

namespace {

constexpr char kGazeExtension[] = "XR_EXT_eye_gaze_interaction";
// VIVE's layer provides the gaze on the Focus Vision; whether its manifest lists the extension is not known.
constexpr wchar_t kViveFacialManifest[] = L"viveopenxrfacialtracking";

// ---------------------------------------------------------------- a small JSON reader for layer manifests

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    std::string str;
    std::vector<std::string> keys;  // Object: keys[i] belongs to items[i]
    std::vector<Json> items;

    const Json* Field(const char* name) const {
        if (type != Object) return nullptr;
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == name) return &items[i];
        return nullptr;
    }
    // The field when it is a string, otherwise nullptr.
    const std::string* Text(const char* name) const {
        const Json* f = Field(name);
        return f && f->type == String ? &f->str : nullptr;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {
        if (s_.compare(0, 3, "\xEF\xBB\xBF") == 0) i_ = 3;
    }

    bool Parse(Json& out) {
        if (!Value(out, 0)) return false;
        Space();
        return i_ == s_.size();
    }

private:
    // The loader's JSON reader accepts comments, so they are skipped here as well.
    void Space() {
        for (;;) {
            while (i_ < s_.size() && strchr(" \t\r\n", s_[i_]) && s_[i_]) ++i_;
            if (s_.compare(i_, 2, "//") == 0) {
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
            } else if (s_.compare(i_, 2, "/*") == 0) {
                const size_t end = s_.find("*/", i_ + 2);
                i_ = end == std::string::npos ? s_.size() : end + 2;
            } else {
                return;
            }
        }
    }

    bool Eat(char c) {
        Space();
        if (i_ >= s_.size() || s_[i_] != c) return false;
        ++i_;
        return true;
    }

    bool Word(const char* w) {
        const size_t n = strlen(w);
        if (s_.compare(i_, n, w) != 0) return false;
        i_ += n;
        return true;
    }

    bool Hex4(unsigned& v) {
        if (i_ + 4 > s_.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return false;
        }
        return true;
    }

    static void Utf8(unsigned c, std::string& out) {
        if (c < 0x80) {
            out += (char)c;
        } else if (c < 0x800) {
            out += (char)(0xC0 | (c >> 6));
            out += (char)(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            out += (char)(0xE0 | (c >> 12));
            out += (char)(0x80 | ((c >> 6) & 0x3F));
            out += (char)(0x80 | (c & 0x3F));
        } else {
            out += (char)(0xF0 | (c >> 18));
            out += (char)(0x80 | ((c >> 12) & 0x3F));
            out += (char)(0x80 | ((c >> 6) & 0x3F));
            out += (char)(0x80 | (c & 0x3F));
        }
    }

    bool Str(std::string& out) {
        if (!Eat('"')) return false;
        while (i_ < s_.size()) {
            const char c = s_[i_++];
            if (c == '"') return true;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) return false;
            const char e = s_[i_++];
            switch (e) {
                case '"': case '\\': case '/': out += e; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned v = 0;
                    if (!Hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00) {
                        unsigned low = 0;
                        if (!Word("\\u") || !Hex4(low) || low < 0xDC00 || low > 0xDFFF) return false;
                        v = 0x10000 + ((v - 0xD800) << 10) + (low - 0xDC00);
                    }
                    Utf8(v, out);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool Value(Json& v, int depth) {
        if (depth > 32) return false;
        Space();
        if (i_ >= s_.size()) return false;
        const char c = s_[i_];
        if (c == '"') {
            v.type = Json::String;
            return Str(v.str);
        }
        if (c == '{') {
            ++i_;
            v.type = Json::Object;
            if (Eat('}')) return true;
            do {
                std::string key;
                if (!Str(key) || !Eat(':')) return false;
                v.keys.push_back(std::move(key));
                v.items.emplace_back();
                if (!Value(v.items.back(), depth + 1)) return false;
            } while (Eat(','));
            return Eat('}');
        }
        if (c == '[') {
            ++i_;
            v.type = Json::Array;
            if (Eat(']')) return true;
            do {
                v.items.emplace_back();
                if (!Value(v.items.back(), depth + 1)) return false;
            } while (Eat(','));
            return Eat(']');
        }
        if (Word("true") || Word("false")) {
            v.type = Json::Bool;
            return true;
        }
        if (Word("null")) return true;
        const size_t start = i_;
        while (i_ < s_.size() && strchr("+-.0123456789eE", s_[i_]) && s_[i_]) ++i_;
        v.type = Json::Number;
        return i_ > start;
    }

    const std::string& s_;
    size_t i_ = 0;
};

// ---------------------------------------------------------------- manifests

struct Manifest {
    std::string name, disableEnv, enableEnv;
    fs::path library;          // full path; only a file name when the manifest gives no folder
    bool libraryBare = false;  // a plain file name: Windows searches for it when the layer loads
    bool gazeExtension = false;
};

bool ReadManifest(const fs::path& file, Manifest& m, std::string& why) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        why = "cannot open the manifest";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    Json root;
    if (!Parser(text).Parse(root) || root.type != Json::Object) {
        why = "the manifest is not valid JSON";
        return false;
    }
    unsigned major = 0, minor = 0, patch = 0;
    const std::string* format = root.Text("file_format_version");
    if (!format || sscanf_s(format->c_str(), "%u.%u.%u", &major, &minor, &patch) != 3 || major != 1 || minor != 0 ||
        patch != 0) {
        why = "unsupported file_format_version";
        return false;
    }
    const Json* layer = root.Field("api_layer");
    const std::string* name = layer ? layer->Text("name") : nullptr;
    const std::string* api = layer ? layer->Text("api_version") : nullptr;
    const std::string* lib = layer ? layer->Text("library_path") : nullptr;
    const std::string* impl = layer ? layer->Text("implementation_version") : nullptr;
    const std::string* disable = layer ? layer->Text("disable_environment") : nullptr;
    if (!name || !api || !lib || !impl || !disable) {
        why = "the manifest misses a required field";
        return false;
    }
    if (sscanf_s(api->c_str(), "%u.%u", &major, &minor) != 2 || (major == 0 && minor == 0) || major > 1) {
        why = "unsupported api_version";
        return false;
    }
    char* end = nullptr;
    strtoul(impl->c_str(), &end, 10);
    if (*end != '\0') {
        why = "invalid implementation_version";
        return false;
    }
    m.name = *name;
    m.disableEnv = *disable;
    if (const std::string* enable = layer->Text("enable_environment")) m.enableEnv = *enable;
    m.libraryBare = lib->find_first_of("\\/") == std::string::npos;
    m.library = Widen(*lib);
    if (!m.libraryBare) {
        if (m.library.is_relative()) m.library = file.parent_path() / m.library;
        std::error_code ec;
        if (!fs::exists(m.library, ec)) {
            why = "its library " + Narrow(m.library.wstring()) + " does not exist";
            return false;
        }
    }
    if (const Json* exts = layer->Field("instance_extensions"))
        for (const Json& e : exts->items)
            if (const std::string* ext = e.Text("name")) m.gazeExtension |= *ext == kGazeExtension;
    return true;
}

// A 64-bit DLL the game's process can load.
bool IsX64Dll(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    char dos[0x40] = {};
    if (!f.read(dos, sizeof(dos)) || dos[0] != 'M' || dos[1] != 'Z') return false;
    int32_t pe = 0;
    memcpy(&pe, dos + 0x3C, sizeof(pe));
    char nt[6] = {};
    if (pe <= 0 || !f.seekg(pe) || !f.read(nt, sizeof(nt)) || memcmp(nt, "PE\0\0", 4) != 0) return false;
    uint16_t machine = 0;
    memcpy(&machine, nt + 4, sizeof(machine));
    return machine == IMAGE_FILE_MACHINE_AMD64;
}

// Would the loader get a working layer from this manifest?
bool Loadable(const Manifest& m) { return m.libraryBare || IsX64Dll(m.library); }

std::vector<fs::path> JsonFiles(const fs::path& dir) {
    std::vector<fs::path> files;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->path().extension() == L".json") files.push_back(it->path());
    return files;
}

std::vector<std::wstring> Split(const std::wstring& s, const wchar_t* separators) {
    std::vector<std::wstring> parts;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find_first_of(separators, start);
        if (end == std::wstring::npos) end = s.size();
        if (end > start) parts.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

std::wstring Lower(std::wstring s) {
    for (auto& c : s) c = (wchar_t)towlower(c);
    return s;
}

// A manifest for the same layer in this folder that would load as well, other than `except`.
bool HasLoadableCopy(const fs::path& dir, const std::string& name, const fs::path& except) {
    for (const fs::path& file : JsonFiles(dir)) {
        std::error_code ec;
        if (!except.empty() && fs::equivalent(file, except, ec)) continue;
        Manifest other;
        std::string why;
        if (ReadManifest(file, other, why) && other.name == name && Loadable(other)) return true;
    }
    return false;
}

}  // namespace

Plan MakePlan(const Input& in) {
    Plan plan;
    std::set<std::string> listed, used;
    for (const std::wstring& token : Split(in.setting, L";, \t")) {
        if (Lower(token) == L"off") {
            plan.notes.push_back("openxr_gaze_layers=off: no layer is moved");
            return plan;
        }
        listed.insert(Narrow(token));
    }
    const std::vector<std::wstring> searched = Split(in.layerPath, L";");
    std::vector<int> priority;  // per plan.move entry: 0 named in the setting, 1 VIVE's layer, 2 lists the extension
    for (const std::wstring& manifest : in.enabledManifests) {
        const fs::path file = manifest;
        Manifest m;
        std::string why;
        if (!ReadManifest(file, m, why)) {
            plan.notes.push_back(Narrow(manifest) + ": left in place, " + why);
            continue;
        }
        const std::wstring lowerPath = Lower(manifest);
        const bool byName = listed.count(m.name) > 0;
        const bool toolkit = lowerPath.find(L"xr_apilayer_mbucchia_toolkit") != std::wstring::npos;
        const bool vive = Lower(file.stem().wstring()) == kViveFacialManifest;
        if (m.name == in.ownLayer || (!byName && !vive && (toolkit || !m.gazeExtension))) continue;
        used.insert(m.name);
        const std::string label = m.name + " (" + Narrow(manifest) + ")";
        if (!m.enableEnv.empty() && !in.envSet(m.enableEnv)) {
            plan.notes.push_back(label + ": left in place, it is off (" + m.enableEnv + " is not set)");
            continue;
        }
        if (in.envSet(m.disableEnv)) {
            plan.notes.push_back(label + ": left in place, it is off (" + m.disableEnv + " is set)");
            continue;
        }
        bool duplicate = false;
        for (const Layer& l : plan.move) duplicate |= l.name == m.name;
        if (duplicate) continue;
        if (!Loadable(m)) {
            plan.notes.push_back(label + ": left in place, its library is not a 64-bit DLL");
            continue;
        }
        if (HasLoadableCopy(file.parent_path(), m.name, file)) {
            // The loader enables every manifest of a name it finds in its search folders.
            plan.notes.push_back(label + ": left in place, its folder has a second manifest of the layer");
            continue;
        }
        Layer layer;
        layer.name = m.name;
        layer.disableEnv = m.disableEnv;
        layer.manifest = manifest;
        layer.declaresGaze = m.gazeExtension;
        layer.reason = byName ? "named in openxr_gaze_layers"
                       : m.gazeExtension ? "its manifest lists XR_EXT_eye_gaze_interaction"
                                         : "VIVE's layer that provides the gaze";
        bool found = false;
        for (const std::wstring& dir : searched) found |= HasLoadableCopy(dir, m.name, {});
        if (!found) layer.folder = file.parent_path().wstring();
        plan.move.push_back(std::move(layer));
        priority.push_back(byName ? 0 : vive ? 1 : 2);
    }
    if (in.maxMoved > 0 && plan.move.size() > in.maxMoved) {
        // Keep the best maxMoved, in the loader's order.
        std::vector<size_t> rank(plan.move.size());
        for (size_t i = 0; i < rank.size(); ++i) rank[i] = i;
        std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return priority[a] < priority[b]; });
        std::vector<bool> keep(plan.move.size(), false);
        for (size_t i = 0; i < in.maxMoved; ++i) keep[rank[i]] = true;
        std::vector<Layer> kept;
        for (size_t i = 0; i < plan.move.size(); ++i) {
            if (keep[i]) {
                kept.push_back(std::move(plan.move[i]));
                continue;
            }
            plan.notes.push_back(plan.move[i].name + " (" + Narrow(plan.move[i].manifest) +
                                 "): left in place, OpenComposite's OpenXR loader reads only " +
                                 std::to_string(in.maxMoved) + " moved layer(s) from XR_ENABLE_API_LAYERS");
        }
        plan.move = std::move(kept);
    }
    // A folder shared by two moved layers goes on the path once.
    std::vector<std::wstring> folders;
    for (Layer& l : plan.move) {
        if (l.folder.empty()) continue;
        if (std::find(folders.begin(), folders.end(), l.folder) != folders.end()) l.folder.clear();
        else folders.push_back(l.folder);
    }
    for (const std::string& name : listed)
        if (!used.count(name)) plan.notes.push_back("openxr_gaze_layers names " + name + ", not an enabled implicit layer");
    return plan;
}

}  // namespace qvfr::xrorder

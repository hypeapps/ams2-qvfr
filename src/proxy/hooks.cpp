#include "hooks.h"

#include <windows.h>

#include <MinHook.h>
#include <bcrypt.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "sysinfo.h"

namespace qvfr::hooks {

namespace {

struct AnchorDef {
    const char* name;
    uint32_t rva;
    const char* desc;
    const char* pattern;
};

const AnchorDef kAnchors[] = {
#define QVFR_ANCHOR_EXE(file, sha, size)
#define QVFR_ANCHOR(id, rva, desc, pattern) {#id, rva, desc, pattern},
#include "generated/anchors_ams2avx.inc"
#undef QVFR_ANCHOR
#undef QVFR_ANCHOR_EXE
};
static_assert(sizeof(kAnchors) / sizeof(kAnchors[0]) == (size_t)Anchor::Count);

struct ExeDef {
    const char* file;
    const char* sha256;
    uint64_t size;
};
const ExeDef kExe =
#define QVFR_ANCHOR_EXE(file, sha, size) {file, sha, size};
#define QVFR_ANCHOR(id, rva, desc, pattern)
#include "generated/anchors_ams2avx.inc"
#undef QVFR_ANCHOR
#undef QVFR_ANCHOR_EXE

std::once_flag g_once;
bool g_mhOk = false;
bool g_engine = false;
uintptr_t g_base = 0;
std::mutex g_m;
std::vector<void*> g_targets;

// "48 8B ?? 05" -> bytes and mask.
bool ParsePattern(const char* text, std::vector<uint8_t>& bytes, std::vector<uint8_t>& mask) {
    for (const char* p = text; *p;) {
        if (*p == ' ') {
            ++p;
            continue;
        }
        if (p[0] == '?' && p[1] == '?') {
            bytes.push_back(0);
            mask.push_back(0);
        } else {
            char hex[3] = {p[0], p[1], 0};
            char* end = nullptr;
            unsigned long v = strtoul(hex, &end, 16);
            if (end != hex + 2) return false;
            bytes.push_back((uint8_t)v);
            mask.push_back(1);
        }
        p += 2;
    }
    return !bytes.empty();
}

bool CheckAnchor(const AnchorDef& a) {
    std::vector<uint8_t> bytes, mask;
    if (!ParsePattern(a.pattern, bytes, mask)) {
        Log("Hooks: anchor %s has a bad pattern", a.name);
        return false;
    }
    const uint8_t* p = reinterpret_cast<const uint8_t*>(g_base + a.rva);
    if (!Readable(p, bytes.size())) {
        Log("Hooks: anchor %s (RVA 0x%X) is not readable", a.name, a.rva);
        return false;
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (mask[i] && p[i] != bytes[i]) {
            Log("Hooks: anchor %s (RVA 0x%X) does not match at +%zu: 0x%02X instead of 0x%02X", a.name, a.rva, i, p[i],
                bytes[i]);
            return false;
        }
    }
    return true;
}

void IdentifyExe() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    fs::path path = exe;
    std::string file = Narrow(path.filename().wstring());
    std::error_code ec;
    uint64_t size = fs::file_size(path, ec);
    if (_stricmp(file.c_str(), kExe.file) != 0 || ec || size != kExe.size) {
        Log("Hooks: %s (%llu B) is not the supported %s (%llu B); engine hooks off", file.c_str(),
            (unsigned long long)size, kExe.file, (unsigned long long)kExe.size);
        return;
    }
    double t0 = NowMs();
    std::string sha = sysinfo::HashFile(path, BCRYPT_SHA256_ALGORITHM);
    if (sha != kExe.sha256) {
        Log("Hooks: %s SHA256 %s differs from the supported build %s (game update?); engine hooks off", file.c_str(),
            sha.c_str(), kExe.sha256);
        return;
    }
    int bad = 0;
    for (const AnchorDef& a : kAnchors)
        if (!CheckAnchor(a)) ++bad;
    if (bad) {
        Log("Hooks: %d of %zu anchors do not match in memory; engine hooks off", bad, std::size(kAnchors));
        return;
    }
    g_engine = true;
    Log("Hooks: %s build matches (SHA256 checked in %.0f ms), all %zu anchors match, engine hooks available", file.c_str(),
        NowMs() - t0, std::size(kAnchors));
}

}  // namespace

bool Readable(const void* p, size_t n) {
    if (!p || n == 0) return false;
    uintptr_t a = (uintptr_t)p, end = a + n;
    if (end < a) return false;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery((const void*)a, &mbi, sizeof(mbi))) return false;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) || mbi.Protect == 0) return false;
        const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (!(mbi.Protect & readable)) return false;
        a = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;
    }
    return true;
}

bool Init() {
    std::call_once(g_once, [] {
        g_base = (uintptr_t)GetModuleHandleW(nullptr);
        MH_STATUS st = MH_Initialize();
        if (st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED) {
            Log("Hooks: MinHook initialization failed (%s); no hooks", MH_StatusToString(st));
            return;
        }
        g_mhOk = true;
        IdentifyExe();
    });
    return g_mhOk;
}

bool EngineAvailable() { return g_engine; }
uintptr_t ExeBase() { return g_base; }

uintptr_t Address(Anchor a) {
    if (!g_engine || (int)a < 0 || a >= Anchor::Count) return 0;
    return g_base + kAnchors[(int)a].rva;
}

const char* Name(Anchor a) { return (int)a >= 0 && a < Anchor::Count ? kAnchors[(int)a].name : "?"; }

bool CreateAt(void* target, void* detour, void** original, const char* name) {
    if (!g_mhOk || !target) return false;
    std::lock_guard lock(g_m);
    MH_STATUS st = MH_CreateHook(target, detour, original);
    if (st == MH_OK) st = MH_EnableHook(target);
    if (st != MH_OK) {
        Log("Hooks: cannot hook %s at %p (%s)", name, target, MH_StatusToString(st));
        return false;
    }
    g_targets.push_back(target);
    return true;
}

bool Create(Anchor a, void* detour, void** original) {
    uintptr_t addr = Address(a);
    if (!addr) return false;
    return CreateAt((void*)addr, detour, original, Name(a));
}

void DisableAll() {
    if (!g_mhOk) return;
    std::lock_guard lock(g_m);
    if (g_targets.empty()) return;
    MH_STATUS st = MH_DisableHook(MH_ALL_HOOKS);
    Log("Hooks: %zu hooks disabled (%s)", g_targets.size(), MH_StatusToString(st));
}

void EnableAll() {
    if (!g_mhOk) return;
    std::lock_guard lock(g_m);
    if (g_targets.empty()) return;
    MH_STATUS st = MH_EnableHook(MH_ALL_HOOKS);
    Log("Hooks: %zu hooks enabled again (%s)", g_targets.size(), MH_StatusToString(st));
}

}  // namespace qvfr::hooks

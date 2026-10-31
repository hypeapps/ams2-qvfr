#include "callers.h"

#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <mutex>
#include <string>

#include "common.h"

namespace qvfr::callers {

namespace {

constexpr int kMaxSites = 32;
constexpr int kMaxStacks = 32;      // unique stacks per method; further ones are only counted
constexpr int kMaxCapture = 62;     // skip + capture must stay below 63 for RtlCaptureStackBackTrace
constexpr int kProxyFrames = 8;     // room for our own frames (wrapper, Note, Guarded, lambda) above the depth

struct Stack {
    uint64_t hash;
    uint64_t hits, lastHits;
};

// Fixed tables: after the first hit of a stack, Note only hashes and counts (no allocation).
struct SiteState {
    const char* name;
    int count;
    Stack stacks[kMaxStacks];
    uint64_t otherHits, lastOtherHits;  // stacks beyond kMaxStacks
};

std::mutex g_mutex;
SiteState g_sites[kMaxSites];
int g_siteCount = 0;

// The proxy module: its frames are skipped, so a stack starts at the game's call.
struct Range {
    uintptr_t begin = 0, end = 0;
};

Range SelfRange() {
    Range r;
    HMODULE self = nullptr;
    MODULEINFO mi{};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&SelfRange), &self) &&
        GetModuleInformation(GetCurrentProcess(), self, &mi, sizeof(mi))) {
        r.begin = reinterpret_cast<uintptr_t>(mi.lpBaseOfDll);
        r.end = r.begin + mi.SizeOfImage;
    }
    return r;
}

// "AMS2AVX.exe+0x1A2B3C" (return address relative to the module base), or the bare address outside any module.
std::string FormatFrame(void* addr) {
    char buf[64];
    HMODULE mod = nullptr;
    wchar_t path[MAX_PATH];
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(addr), &mod) &&
        GetModuleFileNameW(mod, path, MAX_PATH)) {
        const wchar_t* name = wcsrchr(path, L'\\');
        snprintf(buf, sizeof(buf), "+0x%llX",
                 (unsigned long long)(reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(mod)));
        return Narrow(name ? name + 1 : path) + buf;
    }
    snprintf(buf, sizeof(buf), "0x%llX", (unsigned long long)reinterpret_cast<uintptr_t>(addr));
    return buf;
}

}  // namespace

void Note(Site& site) {
    if (!Cfg().traceCallers) return;
    Guarded("caller trace", [&] {
        static const Range self = SelfRange();
        const int depth = Cfg().traceDepth;
        void* frames[kMaxCapture];
        USHORT n = RtlCaptureStackBackTrace(1, (DWORD)std::min(kMaxCapture - 1, depth + kProxyFrames), frames, nullptr);
        int kept = 0;
        uint64_t hash = 14695981039346656037ull;  // FNV-1a over the kept addresses
        for (USHORT i = 0; i < n && kept < depth; ++i) {
            uintptr_t a = reinterpret_cast<uintptr_t>(frames[i]);
            if (a >= self.begin && a < self.end) continue;
            frames[kept++] = frames[i];
            hash = (hash ^ a) * 1099511628211ull;
        }

        std::lock_guard lock(g_mutex);
        if (site.id < 0) {
            if (g_siteCount == kMaxSites) return;
            site.id = g_siteCount++;
            g_sites[site.id].name = site.name;
        }
        SiteState& s = g_sites[site.id];
        for (int i = 0; i < s.count; ++i) {
            if (s.stacks[i].hash == hash) {
                ++s.stacks[i].hits;
                return;
            }
        }
        if (s.count == kMaxStacks) {
            if (s.otherHits++ == 0)
                Log("Caller stacks of %s: more than %d unique, further ones are only counted", s.name, kMaxStacks);
            return;
        }
        Stack& st = s.stacks[s.count++];
        st.hash = hash;
        st.hits = 1;
        std::string line;
        for (int i = 0; i < kept; ++i) line += (i ? " <- " : "") + FormatFrame(frames[i]);
        Log("Caller stack %s #%d (thread %lu, %d frames): %s", s.name, s.count, GetCurrentThreadId(), kept,
            line.c_str());
    });
}

void LogStats(bool totals) {
    if (!Cfg().traceCallers) return;
    std::lock_guard lock(g_mutex);
    for (int i = 0; i < g_siteCount; ++i) {
        SiteState& s = g_sites[i];
        std::string line;
        auto add = [&](const char* label, uint64_t& hits, uint64_t& last) {
            uint64_t d = totals ? hits : hits - last;
            last = hits;
            if (!d) return;
            char buf[64];
            snprintf(buf, sizeof(buf), "%s %llu", label, (unsigned long long)d);
            line += (line.empty() ? "" : ", ") + std::string(buf);
        };
        for (int j = 0; j < s.count; ++j) {
            char label[16];
            snprintf(label, sizeof(label), "#%d", j + 1);
            add(label, s.stacks[j].hits, s.stacks[j].lastHits);
        }
        add("other", s.otherHits, s.lastOtherHits);
        if (!line.empty()) Log("Caller stacks %s%s: %s", s.name, totals ? " (session total)" : "", line.c_str());
    }
}

}  // namespace qvfr::callers

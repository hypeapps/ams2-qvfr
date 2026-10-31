// Engine hooks: identifies the game exe, checks the anchors (RVA + expected bytes) and installs MinHook hooks.
// Only the reference build of AMS2AVX.exe (SHA256 in generated/anchors_ams2avx.inc) is supported: on any other exe
// the engine anchors stay unavailable and nothing in the game is hooked. Hooks at other addresses (D3D11) work anywhere.
#pragma once

#include <cstdint>

namespace qvfr::hooks {

enum class Anchor : int {
#define QVFR_ANCHOR_EXE(file, sha, size)
#define QVFR_ANCHOR(id, rva, desc, pattern) id,
#include "generated/anchors_ams2avx.inc"
#undef QVFR_ANCHOR
#undef QVFR_ANCHOR_EXE
    Count
};

// Once per process (later calls return the first result): MinHook, exe identification and the anchor check.
// Returns false when MinHook cannot start; EngineAvailable() tells whether the engine anchors matched.
bool Init();
bool EngineAvailable();
uintptr_t ExeBase();
uintptr_t Address(Anchor a);  // exe base + RVA; 0 when the engine anchors are not available
const char* Name(Anchor a);

// Creates and enables a hook. The trampoline goes to *original. Hooks live until the process exits.
bool Create(Anchor a, void* detour, void** original);
bool CreateAt(void* target, void* detour, void** original, const char* name);

void DisableAll();  // VR_Shutdown: detours stop; trampolines stay, a thread may still be inside one
void EnableAll();   // a later VR_Init

// Memory range check with VirtualQuery (committed and readable). Its cost grows with the size of the region that
// holds p, about 1 ms per GB, so it is only for small regions (the exe image), never for game heap objects.
bool Readable(const void* p, size_t n);
// Copies n bytes from game memory under SEH; false on an access violation (or a guard page). Cheap: use this for
// game objects.
bool TryRead(const void* p, void* out, size_t n);

}  // namespace qvfr::hooks

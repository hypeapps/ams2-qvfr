// --xr-gaze-layers: which implicit OpenXR layers move below the mod's layer (src/proxy/xr_layer_order.cpp), on manifests
// and fake DLLs written to a temporary folder. Needs no registry, loader or runtime.

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "xr_layer_order.h"
#include "xr_smoke.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

// Only the headers the check reads: "MZ", the offset of the PE header, "PE\0\0" and the machine.
void WriteDll(const fs::path& file, uint16_t machine) {
    std::string b(0x46, '\0');
    b[0] = 'M';
    b[1] = 'Z';
    const int32_t pe = 0x40;
    memcpy(&b[0x3C], &pe, sizeof(pe));
    memcpy(&b[0x40], "PE\0\0", 4);
    memcpy(&b[0x44], &machine, sizeof(machine));
    std::ofstream(file, std::ios::binary) << b;
}

struct Spec {
    std::string name;
    std::string dll = "layer.dll";
    bool gaze = false;
    std::string extra;  // more fields of api_layer
};

void WriteManifest(const fs::path& file, const Spec& s) {
    std::ofstream f(file, std::ios::binary);
    f << "{\n  // a comment, which the loader's JSON reader accepts\n  \"file_format_version\": \"1.0.0\",\n  \"api_layer\": {\n"
      << "    \"name\": \"" << s.name << "\",\n    \"library_path\": \".\\\\" << s.dll << "\",\n"
      << "    \"api_version\": \"1.0\",\n    \"implementation_version\": \"1\",\n    \"description\": \"test\",\n"
      << "    \"disable_environment\": \"DISABLE_" << s.name << "\",\n"
      << "    \"instance_extensions\": [" << (s.gaze ? "{\"name\": \"XR_EXT_eye_gaze_interaction\", \"extension_version\": 2}" : "")
      << "]" << s.extra << "\n  }\n}\n";
}

constexpr uint16_t kX64 = IMAGE_FILE_MACHINE_AMD64, kX86 = IMAGE_FILE_MACHINE_I386;

}  // namespace

int RunXrGazeLayers(const fs::path& home) {
    printf("OpenXR gaze layers moved below the mod's layer\n");
    const fs::path root = home / L"gaze_layers";
    std::error_code ec;
    fs::remove_all(root, ec);

    std::vector<std::wstring> manifests;
    // One layer in its own folder: a 64-bit DLL and the manifest.
    auto add = [&](const wchar_t* folder, const wchar_t* file, const Spec& s, uint16_t machine = kX64) {
        const fs::path dir = root / folder;
        fs::create_directories(dir, ec);
        WriteDll(dir / s.dll, machine);
        WriteManifest(dir / file, s);
        manifests.push_back((dir / file).wstring());
        return dir;
    };

    add(L"quad", L"quad.json", {"XR_APILAYER_quad_views"});
    const fs::path vive = add(L"vive", L"ViveOpenXRFacialTracking.json", {"XR_APILAYER_vive_facial"});
    const fs::path gaze = add(L"gaze", L"openxr-api-layer.json", {"XR_APILAYER_gaze", "layer.dll", true});
    // The 32-bit twin next to it (mbucchia's installers ship one) is harmless: the 64-bit process cannot load it.
    WriteDll(gaze / L"layer32.dll", kX86);
    WriteManifest(gaze / L"openxr-api-layer-32.json", {"XR_APILAYER_gaze", "layer32.dll", true});
    const fs::path listed = add(L"listed", L"listed.json", {"XR_APILAYER_listed"});
    add(L"toolkit\\XR_APILAYER_MBUCCHIA_toolkit", L"manifest.json", {"XR_APILAYER_toolkit", "layer.dll", true});
    add(L"own", L"own.json", {"XR_APILAYER_AMS2QVFR_eye_gaze", "layer.dll", true});
    add(L"needs_env", L"m.json",
        {"XR_APILAYER_needs_env", "layer.dll", true, ",\n    \"enable_environment\": \"ENABLE_NOT_SET\""});
    add(L"switched_off", L"m.json", {"XR_APILAYER_switched_off", "layer.dll", true});
    add(L"x86", L"m.json", {"XR_APILAYER_x86", "layer.dll", true}, kX86);
    const fs::path twin = add(L"twin", L"a.json", {"XR_APILAYER_twin", "layer.dll", true});
    WriteManifest(twin / L"b.json", {"XR_APILAYER_twin", "layer.dll", true});
    // A manifest the loader would refuse as an explicit layer must not reach XR_ENABLE_API_LAYERS.
    const fs::path nolib = add(L"nolib", L"m.json", {"XR_APILAYER_nolib", "missing.dll", true});
    fs::remove(nolib / L"missing.dll", ec);
    const fs::path broken = add(L"broken", L"m.json", {"XR_APILAYER_broken", "layer.dll", true});
    std::ofstream(broken / L"m.json", std::ios::binary) << "{ not json";
    const fs::path onPath = add(L"on_path", L"m.json", {"XR_APILAYER_on_path", "layer.dll", true});
    // The player already points the loader at a copy of this layer.
    const fs::path playerDir = root / L"player";
    fs::create_directories(playerDir, ec);
    WriteDll(playerDir / L"layer.dll", kX64);
    WriteManifest(playerDir / L"copy.json", {"XR_APILAYER_on_path", "layer.dll", true});
    // Two layers of one folder need the folder on the path only once.
    const fs::path shared = add(L"shared", L"one.json", {"XR_APILAYER_shared_one", "layer.dll", true});
    WriteManifest(shared / L"two.json", {"XR_APILAYER_shared_two", "layer.dll", true});
    manifests.push_back((shared / L"two.json").wstring());

    qvfr::xrorder::Input in;
    in.enabledManifests = manifests;
    in.setting = L"XR_APILAYER_listed;XR_APILAYER_absent";
    in.layerPath = playerDir.wstring();
    in.ownLayer = "XR_APILAYER_AMS2QVFR_eye_gaze";
    in.envSet = [](const std::string& name) { return name == "DISABLE_XR_APILAYER_switched_off"; };

    const qvfr::xrorder::Plan plan = qvfr::xrorder::MakePlan(in);
    std::vector<std::string> names;
    for (const auto& l : plan.move) {
        names.push_back(l.name);
        printf("  moved %s: %s\n", l.name.c_str(), l.reason.c_str());
    }
    for (const auto& n : plan.notes) printf("  note: %s\n", n.c_str());
    const std::vector<std::string> expected = {"XR_APILAYER_vive_facial", "XR_APILAYER_gaze",     "XR_APILAYER_listed",
                                               "XR_APILAYER_on_path",     "XR_APILAYER_shared_one", "XR_APILAYER_shared_two"};
    Check(names == expected, "the gaze layers move, in the loader's order, and no other layer does");

    auto find = [&](const char* name) -> const qvfr::xrorder::Layer* {
        for (const auto& l : plan.move)
            if (l.name == name) return &l;
        return nullptr;
    };
    auto noted = [&](const char* text) {
        for (const auto& n : plan.notes)
            if (n.find(text) != std::string::npos) return true;
        return false;
    };
    if (names == expected) {
        Check(find("XR_APILAYER_vive_facial")->folder == vive.wstring() && !find("XR_APILAYER_vive_facial")->declaresGaze,
              "VIVE's layer is recognized by its manifest file, without the extension listed");
        Check(find("XR_APILAYER_gaze")->folder == gaze.wstring() && find("XR_APILAYER_gaze")->declaresGaze,
              "a layer that lists the gaze extension moves, its 32-bit twin does not stop it");
        Check(find("XR_APILAYER_listed")->folder == listed.wstring(), "a layer named in openxr_gaze_layers moves");
        Check(find("XR_APILAYER_listed")->disableEnv == "DISABLE_XR_APILAYER_listed", "the plan has the disable variable");
        Check(find("XR_APILAYER_on_path")->folder.empty(), "no folder added for a layer the loader already finds");
        Check(find("XR_APILAYER_shared_one")->folder == shared.wstring() && find("XR_APILAYER_shared_two")->folder.empty(),
              "a folder shared by two layers is added once");
    }
    Check(noted("XR_APILAYER_needs_env") && noted("XR_APILAYER_switched_off"), "layers that are off stay out, with a note");
    Check(noted("XR_APILAYER_x86"), "a layer without a 64-bit DLL stays in place");
    Check(noted("XR_APILAYER_twin"), "a layer with two loadable manifests in its folder stays in place");
    Check(noted("missing.dll"), "a manifest whose library is missing stays in place");
    Check(noted("not valid JSON"), "a broken manifest stays in place");
    Check(noted("names XR_APILAYER_absent"), "a name in the setting that matches nothing is reported");

    // OpenComposite's loader takes only one more name: the best layer stays, in priority over the loader's order.
    in.maxMoved = 1;
    const qvfr::xrorder::Plan one = qvfr::xrorder::MakePlan(in);
    Check(one.move.size() == 1 && one.move[0].name == "XR_APILAYER_listed" && one.move[0].folder == listed.wstring(),
          "with one place a layer named in openxr_gaze_layers wins");
    in.setting.clear();
    const qvfr::xrorder::Plan vivePlan = qvfr::xrorder::MakePlan(in);
    bool limitNoted = false;
    for (const auto& n : vivePlan.notes) limitNoted |= n.find("XR_APILAYER_gaze") != std::string::npos &&
                                                       n.find("reads only 1 moved layer") != std::string::npos;
    Check(vivePlan.move.size() == 1 && vivePlan.move[0].name == "XR_APILAYER_vive_facial" && limitNoted,
          "without one, VIVE's layer wins over a layer that only lists the extension, and the rest is noted");
    // Even when the other layer comes first in the loader's order (the case of issue #15 the other way round).
    in.enabledManifests = {(gaze / L"openxr-api-layer.json").wstring(), (vive / L"ViveOpenXRFacialTracking.json").wstring()};
    const qvfr::xrorder::Plan swapped = qvfr::xrorder::MakePlan(in);
    Check(swapped.move.size() == 1 && swapped.move[0].name == "XR_APILAYER_vive_facial", "VIVE's layer wins in any order");
    in.maxMoved = 0;

    in.setting = L"OFF";
    const qvfr::xrorder::Plan off = qvfr::xrorder::MakePlan(in);
    Check(off.move.empty(), "openxr_gaze_layers=off moves nothing");

    fs::remove_all(root, ec);
    printf("%s (%d failures)\n", g_failures ? "OpenXR gaze layers test FAILED" : "OpenXR gaze layers test OK", g_failures);
    return g_failures ? 1 : 0;
}

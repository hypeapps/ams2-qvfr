// Smoke tests of the OpenXR path (M8.1) without a headset, a runtime or OpenComposite.
//   RunXrLayer:   the proxy DLL as an OpenXR API layer between a fake loader (this file) and a fake runtime: the
//                 negotiation, the added extension, the gaze set added to attach and sync, the layer's own sync when the
//                 "application" stops syncing, the gaze direction from xrLocateSpace, two sessions in a row.
//   RunOpenXrPath: [runtime] path=openxr with the Valve DLL standing in for OpenComposite: the backend path, the
//                 process environment for the loader (layer path and name; the runtime untouched) and the shipped
//                 layer manifest.

#include "xr_smoke.h"

#include <windows.h>

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- fake runtime

template <class H>
H MakeHandle(uintptr_t v) {
    return reinterpret_cast<H>(v);
}

struct Fake {
    std::vector<std::string> instanceExts;
    std::map<std::string, XrPath> paths;
    std::map<XrPath, std::string> names;
    std::vector<std::string> actionSets, actions;
    std::vector<std::string> suggestedProfiles;
    uint64_t nextHandle = 0x1000;
    XrSession session = XR_NULL_HANDLE;
    bool attached = false;
    uint32_t attachSets = 0;
    bool attachHadOurs = false;
    int syncs = 0, syncsWithOurs = 0;
    bool lastSyncHadOurs = false;
    int locates = 0, waits = 0, endFrames = 0, spaces = 0;
    XrActionSet ourSet = XR_NULL_HANDLE;  // the set named "ams2qvfr-gaze"
} g_f;

XrResult XRAPI_CALL FEnumerateExt(const char*, uint32_t cap, uint32_t* count, XrExtensionProperties* props) {
    static const char* exts[] = {"XR_KHR_D3D11_enable", XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME};
    *count = 2;
    if (cap == 0) return XR_SUCCESS;
    if (cap < 2) return XR_ERROR_SIZE_INSUFFICIENT;
    for (int i = 0; i < 2; ++i) {
        strcpy_s(props[i].extensionName, exts[i]);
        props[i].extensionVersion = 1;
    }
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateApiLayerInstance(const XrInstanceCreateInfo* info, const XrApiLayerCreateInfo*,
                                            XrInstance* instance) {
    g_f.instanceExts.clear();
    for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) g_f.instanceExts.push_back(info->enabledExtensionNames[i]);
    *instance = MakeHandle<XrInstance>(g_f.nextHandle++);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FDestroyInstance(XrInstance) { return XR_SUCCESS; }

XrResult XRAPI_CALL FGetInstanceProperties(XrInstance, XrInstanceProperties* p) {
    strcpy_s(p->runtimeName, "Fake runtime");
    p->runtimeVersion = XR_MAKE_VERSION(1, 2, 3);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FGetSystem(XrInstance, const XrSystemGetInfo*, XrSystemId* id) {
    *id = 7;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FGetSystemProperties(XrInstance, XrSystemId, XrSystemProperties* p) {
    strcpy_s(p->systemName, "Fake HMD");
    for (auto* n = static_cast<XrBaseOutStructure*>(p->next); n; n = n->next)
        if (n->type == XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT)
            reinterpret_cast<XrSystemEyeGazeInteractionPropertiesEXT*>(n)->supportsEyeGazeInteraction = XR_TRUE;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateSession(XrInstance, const XrSessionCreateInfo*, XrSession* s) {
    *s = g_f.session = MakeHandle<XrSession>(g_f.nextHandle++);
    g_f.attached = false;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FDestroySession(XrSession) {
    g_f.session = XR_NULL_HANDLE;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FBeginSession(XrSession, const XrSessionBeginInfo*) { return XR_SUCCESS; }
XrResult XRAPI_CALL FEndSession(XrSession) { return XR_SUCCESS; }
XrResult XRAPI_CALL FRequestExitSession(XrSession) { return XR_SUCCESS; }

XrResult XRAPI_CALL FStringToPath(XrInstance, const char* s, XrPath* p) {
    auto it = g_f.paths.find(s);
    if (it == g_f.paths.end()) {
        XrPath np = g_f.paths.size() + 1;
        it = g_f.paths.emplace(s, np).first;
        g_f.names[np] = s;
    }
    *p = it->second;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FPathToString(XrInstance, XrPath p, uint32_t cap, uint32_t* n, char* buf) {
    const std::string& s = g_f.names[p];
    *n = (uint32_t)s.size() + 1;
    if (cap >= *n) strcpy_s(buf, cap, s.c_str());
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateActionSet(XrInstance, const XrActionSetCreateInfo* i, XrActionSet* s) {
    if (g_f.attached) return XR_ERROR_ACTIONSETS_ALREADY_ATTACHED;
    *s = MakeHandle<XrActionSet>(g_f.nextHandle++);
    g_f.actionSets.push_back(i->actionSetName);
    if (!strcmp(i->actionSetName, "ams2qvfr-gaze")) g_f.ourSet = *s;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateAction(XrActionSet, const XrActionCreateInfo* i, XrAction* a) {
    *a = MakeHandle<XrAction>(g_f.nextHandle++);
    g_f.actions.push_back(i->actionName);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FSuggest(XrInstance, const XrInteractionProfileSuggestedBinding* b) {
    if (g_f.attached) return XR_ERROR_ACTIONSETS_ALREADY_ATTACHED;
    g_f.suggestedProfiles.push_back(g_f.names[b->interactionProfile]);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FAttach(XrSession, const XrSessionActionSetsAttachInfo* i) {
    if (g_f.attached) return XR_ERROR_ACTIONSETS_ALREADY_ATTACHED;
    g_f.attached = true;
    g_f.attachSets = i->countActionSets;
    g_f.attachHadOurs = false;
    for (uint32_t k = 0; k < i->countActionSets; ++k) g_f.attachHadOurs |= i->actionSets[k] == g_f.ourSet;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FSync(XrSession, const XrActionsSyncInfo* i) {
    if (!g_f.attached) return XR_ERROR_ACTIONSET_NOT_ATTACHED;
    ++g_f.syncs;
    g_f.lastSyncHadOurs = false;
    for (uint32_t k = 0; k < i->countActiveActionSets; ++k)
        g_f.lastSyncHadOurs |= i->activeActionSets[k].actionSet == g_f.ourSet;
    g_f.syncsWithOurs += g_f.lastSyncHadOurs;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateActionSpace(XrSession, const XrActionSpaceCreateInfo*, XrSpace* s) {
    *s = MakeHandle<XrSpace>(g_f.nextHandle++);
    ++g_f.spaces;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FCreateReferenceSpace(XrSession, const XrReferenceSpaceCreateInfo*, XrSpace* s) {
    *s = MakeHandle<XrSpace>(g_f.nextHandle++);
    ++g_f.spaces;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FDestroySpace(XrSpace) {
    --g_f.spaces;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FGetActionStatePose(XrSession, const XrActionStateGetInfo*, XrActionStatePose* st) {
    st->isActive = g_f.lastSyncHadOurs ? XR_TRUE : XR_FALSE;
    return XR_SUCCESS;
}

// The gaze 10 degrees to the right: a rotation of -10 degrees about +y turns (0, 0, -1) into (sin 10, 0, -cos 10).
XrResult XRAPI_CALL FLocateSpace(XrSpace, XrSpace, XrTime, XrSpaceLocation* loc) {
    ++g_f.locates;
    const double half = -10.0 * 3.14159265358979 / 180.0 / 2.0;
    loc->pose.orientation = {0.0f, (float)std::sin(half), 0.0f, (float)std::cos(half)};
    loc->pose.position = {0.03f, 0.0f, 0.0f};
    loc->locationFlags = XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FWaitFrame(XrSession, const XrFrameWaitInfo*, XrFrameState* st) {
    ++g_f.waits;
    st->predictedDisplayTime = 1000000000LL + g_f.waits * 13888889LL;
    st->predictedDisplayPeriod = 13888889LL;
    st->shouldRender = XR_TRUE;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FEndFrame(XrSession, const XrFrameEndInfo*) {
    ++g_f.endFrames;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL FPollEvent(XrInstance, XrEventDataBuffer*) { return XR_EVENT_UNAVAILABLE; }

XrResult XRAPI_CALL FGipa(XrInstance, const char* name, PFN_xrVoidFunction* fn) {
    static const std::map<std::string, PFN_xrVoidFunction> table = {
        {"xrEnumerateInstanceExtensionProperties", (PFN_xrVoidFunction)FEnumerateExt},
        {"xrDestroyInstance", (PFN_xrVoidFunction)FDestroyInstance},
        {"xrGetInstanceProperties", (PFN_xrVoidFunction)FGetInstanceProperties},
        {"xrGetSystem", (PFN_xrVoidFunction)FGetSystem},
        {"xrGetSystemProperties", (PFN_xrVoidFunction)FGetSystemProperties},
        {"xrCreateSession", (PFN_xrVoidFunction)FCreateSession},
        {"xrDestroySession", (PFN_xrVoidFunction)FDestroySession},
        {"xrBeginSession", (PFN_xrVoidFunction)FBeginSession},
        {"xrEndSession", (PFN_xrVoidFunction)FEndSession},
        {"xrRequestExitSession", (PFN_xrVoidFunction)FRequestExitSession},
        {"xrStringToPath", (PFN_xrVoidFunction)FStringToPath},
        {"xrPathToString", (PFN_xrVoidFunction)FPathToString},
        {"xrCreateActionSet", (PFN_xrVoidFunction)FCreateActionSet},
        {"xrCreateAction", (PFN_xrVoidFunction)FCreateAction},
        {"xrSuggestInteractionProfileBindings", (PFN_xrVoidFunction)FSuggest},
        {"xrAttachSessionActionSets", (PFN_xrVoidFunction)FAttach},
        {"xrSyncActions", (PFN_xrVoidFunction)FSync},
        {"xrCreateActionSpace", (PFN_xrVoidFunction)FCreateActionSpace},
        {"xrCreateReferenceSpace", (PFN_xrVoidFunction)FCreateReferenceSpace},
        {"xrDestroySpace", (PFN_xrVoidFunction)FDestroySpace},
        {"xrGetActionStatePose", (PFN_xrVoidFunction)FGetActionStatePose},
        {"xrLocateSpace", (PFN_xrVoidFunction)FLocateSpace},
        {"xrWaitFrame", (PFN_xrVoidFunction)FWaitFrame},
        {"xrEndFrame", (PFN_xrVoidFunction)FEndFrame},
        {"xrPollEvent", (PFN_xrVoidFunction)FPollEvent},
    };
    auto it = table.find(name);
    if (it == table.end()) {
        *fn = nullptr;
        return XR_ERROR_FUNCTION_UNSUPPORTED;
    }
    *fn = it->second;
    return XR_SUCCESS;
}

// ---------------------------------------------------------------- helpers

int g_failures = 0;

void Check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::string ReadNewestLog(const fs::path& dir) {
    fs::path newest;
    fs::file_time_type t{};
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == L".log" && (newest.empty() || e.last_write_time() > t)) {
            newest = e.path();
            t = e.last_write_time();
        }
    std::ifstream f(newest, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <class T>
T Fn(PFN_xrGetInstanceProcAddr gipa, XrInstance inst, const char* name) {
    PFN_xrVoidFunction f = nullptr;
    gipa(inst, name, &f);
    return reinterpret_cast<T>(f);
}

std::wstring Env(const wchar_t* name) {
    wchar_t buf[4096];
    DWORD n = GetEnvironmentVariableW(name, buf, 4096);
    return n > 0 && n < 4096 ? std::wstring(buf, n) : std::wstring();
}

}  // namespace

int RunXrLayer(const fs::path& dll, const fs::path& home) {
    printf("OpenXR layer with a fake loader and runtime (%ls)\n", dll.c_str());
    HMODULE m = LoadLibraryW(dll.c_str());
    if (!m) {
        printf("Cannot load the proxy (error %lu)\n", GetLastError());
        return 1;
    }
    auto negotiate = (PFN_xrNegotiateLoaderApiLayerInterface)GetProcAddress(m, "xrNegotiateLoaderApiLayerInterface");
    Check(negotiate != nullptr, "the proxy exports xrNegotiateLoaderApiLayerInterface");
    if (!negotiate) return 1;

    XrNegotiateLoaderInfo li{XR_LOADER_INTERFACE_STRUCT_LOADER_INFO, XR_LOADER_INFO_STRUCT_VERSION,
                             sizeof(XrNegotiateLoaderInfo)};
    li.minInterfaceVersion = 1;
    li.maxInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    li.minApiVersion = XR_MAKE_VERSION(1, 0, 0);
    li.maxApiVersion = XR_MAKE_VERSION(1, 0x3ff, 0xfff);
    XrNegotiateApiLayerRequest req{XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST, XR_API_LAYER_INFO_STRUCT_VERSION,
                                   sizeof(XrNegotiateApiLayerRequest)};
    Check(negotiate(&li, "XR_APILAYER_other", &req) != XR_SUCCESS, "negotiation refuses another layer name");
    XrResult r = negotiate(&li, "XR_APILAYER_AMS2QVFR_eye_gaze", &req);
    Check(r == XR_SUCCESS && req.getInstanceProcAddr && req.createApiLayerInstance, "negotiation with our name");
    if (r != XR_SUCCESS) return 1;
    PFN_xrGetInstanceProcAddr gipa = req.getInstanceProcAddr;

    XrApiLayerNextInfo next{XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO, XR_API_LAYER_NEXT_INFO_STRUCT_VERSION,
                            sizeof(XrApiLayerNextInfo)};
    strcpy_s(next.layerName, "fake runtime");
    next.nextGetInstanceProcAddr = FGipa;
    next.nextCreateApiLayerInstance = FCreateApiLayerInstance;
    XrApiLayerCreateInfo lci{XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO, XR_API_LAYER_CREATE_INFO_STRUCT_VERSION,
                             sizeof(XrApiLayerCreateInfo)};
    lci.nextInfo = &next;
    const char* appExts[] = {"XR_KHR_D3D11_enable"};
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    strcpy_s(ci.applicationInfo.applicationName, "smoke (as OpenComposite)");
    ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    ci.enabledExtensionCount = 1;
    ci.enabledExtensionNames = appExts;
    XrInstance inst = XR_NULL_HANDLE;
    r = req.createApiLayerInstance(&ci, &lci, &inst);
    Check(r == XR_SUCCESS && inst != XR_NULL_HANDLE, "xrCreateApiLayerInstance through the layer");
    bool gazeExt = false;
    for (auto& e : g_f.instanceExts) gazeExt |= e == XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME;
    Check(g_f.instanceExts.size() == 2 && gazeExt, "the runtime got the application's extension plus eye gaze");
    Check(g_f.actionSets.empty() && g_f.suggestedProfiles.empty(),
          "no action set or binding of the layer at instance creation (they wait for the first attach)");

    auto createSession = Fn<PFN_xrCreateSession>(gipa, inst, "xrCreateSession");
    auto destroySession = Fn<PFN_xrDestroySession>(gipa, inst, "xrDestroySession");
    auto attach = Fn<PFN_xrAttachSessionActionSets>(gipa, inst, "xrAttachSessionActionSets");
    auto sync = Fn<PFN_xrSyncActions>(gipa, inst, "xrSyncActions");
    auto waitFrame = Fn<PFN_xrWaitFrame>(gipa, inst, "xrWaitFrame");
    auto endFrame = Fn<PFN_xrEndFrame>(gipa, inst, "xrEndFrame");
    auto getSystem = Fn<PFN_xrGetSystem>(gipa, inst, "xrGetSystem");
    auto createActionSet = Fn<PFN_xrCreateActionSet>(gipa, inst, "xrCreateActionSet");
    auto destroyInstance = Fn<PFN_xrDestroyInstance>(gipa, inst, "xrDestroyInstance");
    auto statePose = Fn<PFN_xrGetActionStatePose>(gipa, inst, "xrGetActionStatePose");
    Check(waitFrame && (void*)waitFrame != (void*)FWaitFrame && (void*)sync != (void*)FSync,
          "xrWaitFrame and xrSyncActions are the layer's hooks");
    Check((void*)statePose == (void*)FGetActionStatePose, "a function the layer does not hook goes straight to the runtime");

    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XrSystemId sys = 0;
    getSystem(inst, &sgi, &sys);

    // The "application" (OpenComposite): its own action set, a session, attach, then frames.
    XrActionSetCreateInfo asi{XR_TYPE_ACTION_SET_CREATE_INFO};
    strcpy_s(asi.actionSetName, "opencomposite-internal-info-set");
    strcpy_s(asi.localizedActionSetName, "OC");
    XrActionSet appSet = XR_NULL_HANDLE;
    createActionSet(inst, &asi, &appSet);
    auto suggest = Fn<PFN_xrSuggestInteractionProfileBindings>(gipa, inst, "xrSuggestInteractionProfileBindings");
    XrPath hpProfile = XR_NULL_PATH, xClick = XR_NULL_PATH;
    FStringToPath(inst, "/interaction_profiles/hp/mixed_reality_controller", &hpProfile);
    FStringToPath(inst, "/user/hand/left/input/x/click", &xClick);
    XrActionSuggestedBinding appBinding{MakeHandle<XrAction>(0x99), xClick};
    XrInteractionProfileSuggestedBinding appSuggest{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    appSuggest.interactionProfile = hpProfile;
    appSuggest.countSuggestedBindings = 1;
    appSuggest.suggestedBindings = &appBinding;
    Check(suggest(inst, &appSuggest) == XR_SUCCESS && g_f.suggestedProfiles.size() == 1,
          "the application's suggestion passes through before any of the layer's");

    for (int sessionNo = 1; sessionNo <= 2; ++sessionNo) {
        printf(" session %d\n", sessionNo);
        XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
        sci.systemId = sys;
        XrSession s = XR_NULL_HANDLE;
        createSession(inst, &sci, &s);
        XrSessionActionSetsAttachInfo ai{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        ai.countActionSets = 1;
        ai.actionSets = &appSet;
        r = attach(s, &ai);
        Check(r == XR_SUCCESS && g_f.attachSets == 2 && g_f.attachHadOurs, "attach: the runtime got the gaze set as well");
        if (sessionNo == 1)
            Check(g_f.actionSets.size() == 2 && g_f.actions.size() == 1 && g_f.suggestedProfiles.size() == 2 &&
                      g_f.suggestedProfiles[1] == "/interaction_profiles/ext/eye_gaze_interaction",
                  "gaze set, action and eye gaze binding created in the first attach, after the application's");
        else
            Check(g_f.actionSets.size() == 2 && g_f.suggestedProfiles.size() == 2,
                  "the second session reuses the instance's gaze set and binding");

        const int syncs0 = g_f.syncs, withOurs0 = g_f.syncsWithOurs, locates0 = g_f.locates;
        XrActiveActionSet active{appSet, XR_NULL_PATH};
        XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
        si.countActiveActionSets = 1;
        si.activeActionSets = &active;
        XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                     {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
        for (auto& v : views) {
            v.fov = {-0.9f, 0.9f, 0.8f, -0.8f};
            v.subImage.imageRect.extent = {3024, 2536};
        }
        XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        proj.viewCount = 2;
        proj.views = views;
        XrCompositionLayerQuad quadLayer{XR_TYPE_COMPOSITION_LAYER_QUAD};
        quadLayer.size = {1.0f, 0.5f};
        const XrCompositionLayerBaseHeader* layers[2] = {(const XrCompositionLayerBaseHeader*)&proj,
                                                         (const XrCompositionLayerBaseHeader*)&quadLayer};
        XrFrameEndInfo fe{XR_TYPE_FRAME_END_INFO};
        fe.layerCount = 2;
        fe.layers = layers;
        for (int i = 0; i < 5; ++i) {
            XrFrameState st{XR_TYPE_FRAME_STATE};
            sync(s, &si);
            waitFrame(s, nullptr, &st);
            endFrame(s, &fe);
        }
        Check(g_f.syncs - syncs0 == 5 && g_f.syncsWithOurs - withOurs0 == 5,
              "the application's 5 xrSyncActions carried the gaze set, no extra sync");
        Check(g_f.locates - locates0 == 5, "the gaze was located after each of the 5 xrWaitFrame");

        // The application stops syncing: after 50 ms the layer syncs by itself (and keeps the application's set).
        Sleep(80);
        const int syncs1 = g_f.syncs;
        for (int i = 0; i < 3; ++i) {
            XrFrameState st{XR_TYPE_FRAME_STATE};
            waitFrame(s, nullptr, &st);
            Sleep(60);
        }
        Check(g_f.syncs - syncs1 == 3 && g_f.lastSyncHadOurs, "without the application's sync the layer synced each frame");
        destroySession(s);
        Check(g_f.spaces == 0, "both spaces of the session destroyed with it");
    }
    destroyInstance(inst);

    const std::string log = ReadNewestLog(home / L"logs");
    Check(log.find("first gaze, direction (0.1736, 0.0000, -0.9848)") != std::string::npos,
          "the log has the gaze 10 degrees right: direction (0.1736, 0.0000, -0.9848)");
    Check(log.find("eye gaze interaction supported yes") != std::string::npos, "the log has the eye gaze system property");
    Check(log.find("2 layers: projection2 quad") != std::string::npos, "the log has the frame layer layout");
    Check(log.find("the layer syncs itself") != std::string::npos, "the log names the layer's own sync");
    printf("%s (%d failures)\n", g_failures ? "OpenXR layer test FAILED" : "OpenXR layer test OK", g_failures);
    return g_failures ? 1 : 0;
}

int RunOpenXrPath(const fs::path& dll, const fs::path& home) {
    printf("[runtime] path=openxr with the Valve DLL as the backend (%ls)\n", dll.c_str());
    if (FILE* f = _wfopen((home / L"qvfr.ini").c_str(), L"wb")) {
        // Relative to the proxy's folder; the Valve DLL stands in for OpenComposite.
        fputs("[runtime]\npath=openxr\nopencomposite=qvfr\\openvr\\openvr_api.dll\nopenxr_layer=1\n[proxy]\nwrap=1\n[log]\nwarning_box=0\n", f);
        fclose(f);
    }
    SetEnvironmentVariableW(L"XR_ENABLE_API_LAYERS", L"XR_APILAYER_player_own");
    HMODULE m = LoadLibraryW(dll.c_str());
    if (!m) {
        printf("Cannot load the proxy (error %lu)\n", GetLastError());
        return 1;
    }
    using PresentFn = bool(VR_CALLTYPE*)();
    auto present = (PresentFn)GetProcAddress(m, "VR_IsHmdPresent");
    if (!present) return 1;
    const bool hmd = present();  // the first export call loads the settings and the backend
    printf("  VR_IsHmdPresent = %d (through the backend)\n", hmd);

    // The layer manifest is a shipped file in x64\qvfr\ (here proxy\qvfr\), not one the proxy writes.
    const fs::path layerDir = dll.parent_path() / L"qvfr";
    const std::wstring layerPath = Env(L"XR_API_LAYER_PATH"), enable = Env(L"XR_ENABLE_API_LAYERS"),
                       runtime = Env(L"XR_RUNTIME_JSON");
    printf("  XR_API_LAYER_PATH=%ls\n  XR_ENABLE_API_LAYERS=%ls\n  XR_RUNTIME_JSON=%ls\n", layerPath.c_str(),
           enable.c_str(), runtime.c_str());
    Check(layerPath == layerDir.wstring(), "XR_API_LAYER_PATH is x64\\qvfr (the shipped manifest's folder)");
    Check(enable == L"XR_APILAYER_AMS2QVFR_eye_gaze;XR_APILAYER_player_own",
          "XR_ENABLE_API_LAYERS: our layer first, the player's kept");
    Check(runtime.empty(), "XR_RUNTIME_JSON untouched (the mod never chooses the OpenXR runtime)");

    std::ifstream mf(layerDir / L"XR_APILAYER_AMS2QVFR_eye_gaze.json", std::ios::binary);
    std::stringstream ss;
    ss << mf.rdbuf();
    const std::string manifest = ss.str();
    // The loader resolves a relative library_path against the manifest's folder.
    const std::string key = "\"library_path\": \"";
    const size_t k = manifest.find(key);
    std::string lib;
    for (size_t i = k == std::string::npos ? manifest.size() : k + key.size(); i < manifest.size() && manifest[i] != '"';
         ++i) {
        if (manifest[i] == '\\' && i + 1 < manifest.size()) ++i;
        lib += manifest[i];
    }
    std::error_code ec;
    const fs::path resolved = fs::weakly_canonical(layerDir / lib, ec);
    printf("  manifest library_path \"%s\" -> %ls\n", lib.c_str(), resolved.c_str());
    Check(manifest.find("\"name\": \"XR_APILAYER_AMS2QVFR_eye_gaze\"") != std::string::npos &&
              fs::equivalent(resolved, dll, ec),
          "the manifest names the layer and its relative library_path is the proxy DLL");

    const std::string log = ReadNewestLog(home / L"logs");
    Check(log.find("Backend: ") != std::string::npos && log.find("qvfr\\openvr\\openvr_api.dll") != std::string::npos &&
              log.find("official Valve 0.9.12") != std::string::npos,
          "the log names the backend from opencomposite= (relative to the proxy)");
    Check(log.find("Runtime: OpenXR implicit layer") != std::string::npos ||
              log.find("OpenXR implicit layers in HKLM: none") != std::string::npos,
          "the log lists the OpenXR implicit layers");
    printf("%s (%d failures)\n", g_failures ? "path=openxr test FAILED" : "path=openxr test OK", g_failures);
    return g_failures ? 1 : 0;
}

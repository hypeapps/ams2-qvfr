// OpenXR API layer in the proxy DLL (see xr_layer.h). The chained calls always go to the next layer or the runtime;
// our own work (extension, action set, gaze, logs) runs inside Guarded, so a fault there only turns the layer into
// a pass-through. Single instance and one session at a time, as OpenComposite creates them (it restarts the session
// on the first Submit and when the game attaches its inputs).

#include "xr_layer.h"

#include <windows.h>

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "diag.h"
#include "fps_hud.h"

namespace qvfr::xrlayer {

namespace {

struct Next {
    PFN_xrGetInstanceProcAddr gipa = nullptr;
    PFN_xrDestroyInstance destroyInstance = nullptr;
    PFN_xrGetInstanceProperties getInstanceProperties = nullptr;
    PFN_xrGetSystem getSystem = nullptr;
    PFN_xrGetSystemProperties getSystemProperties = nullptr;
    PFN_xrCreateSession createSession = nullptr;
    PFN_xrDestroySession destroySession = nullptr;
    PFN_xrBeginSession beginSession = nullptr;
    PFN_xrEndSession endSession = nullptr;
    PFN_xrRequestExitSession requestExitSession = nullptr;
    PFN_xrAttachSessionActionSets attach = nullptr;
    PFN_xrSyncActions sync = nullptr;
    PFN_xrSuggestInteractionProfileBindings suggest = nullptr;
    PFN_xrWaitFrame waitFrame = nullptr;
    PFN_xrEndFrame endFrame = nullptr;
    PFN_xrStringToPath stringToPath = nullptr;
    PFN_xrPathToString pathToString = nullptr;
    PFN_xrCreateActionSet createActionSet = nullptr;
    PFN_xrCreateAction createAction = nullptr;
    PFN_xrCreateActionSpace createActionSpace = nullptr;
    PFN_xrCreateReferenceSpace createReferenceSpace = nullptr;
    PFN_xrDestroySpace destroySpace = nullptr;
    PFN_xrLocateSpace locateSpace = nullptr;
    PFN_xrGetActionStatePose getActionStatePose = nullptr;
    PFN_xrPollEvent pollEvent = nullptr;
} g_next;

std::mutex g_m;  // everything below
std::atomic<bool> g_loaded{false};
XrInstance g_instance = XR_NULL_HANDLE;
bool g_extEnabled = false;          // XR_EXT_eye_gaze_interaction is on the instance
XrActionSet g_set = XR_NULL_HANDLE;
XrAction g_action = XR_NULL_HANDLE;
bool g_bindingsSuggested = false;
bool g_gazeTried = false;           // CreateGazeAction ran for this instance

struct Session {
    XrSession h = XR_NULL_HANDLE;
    int number = 0;                 // 1 = the first session of the process
    bool attached = false;          // our set is attached to it
    XrSpace gazeSpace = XR_NULL_HANDLE, viewSpace = XR_NULL_HANDLE;
    uint64_t waits = 0, endFrames = 0;
    uint64_t appSyncs = 0, ownSyncs = 0, gazeOk = 0, gazeInactive = 0, gazeInvalid = 0, locateFail = 0;
    uint64_t gazeFlipped = 0;       // located gazes whose -Z pointed backwards (mirrored in Z)
    double firstGazeMs = -1, lastSampleMs = -1;  // the first located gaze; the last gaze sample logged
    double lastAppSyncMs = -1;
    std::vector<XrActiveActionSet> appSets;  // the game's (OpenComposite's) last active sets, for our own sync
    std::string endFrameSig;                 // layer layout of the last logged xrEndFrame
    int endFrameLogs = 0;
    double statsMs = 0;
} g_s;
int g_sessions = 0;

std::mutex g_gm;  // the gaze result (read by the proxy on the game's render thread)
Gaze g_gaze;
std::atomic<int64_t> g_periodNs{0};  // predictedDisplayPeriod of the last xrWaitFrame
hud::Dispatch g_hudApi;
hud::Overlay g_hud;
bool g_shouldRender = false;

constexpr double kOwnSyncAfterMs = 50;  // no xrSyncActions from OpenComposite for this long: we sync ourselves
constexpr double kSampleForMs = 60000, kSampleEveryMs = 500;  // gaze samples in the log after the first gaze

const char* R(XrResult r) {
    switch (r) {
        case XR_SUCCESS: return "XR_SUCCESS";
        case XR_SESSION_LOSS_PENDING: return "XR_SESSION_LOSS_PENDING";
        case XR_ERROR_EXTENSION_NOT_PRESENT: return "XR_ERROR_EXTENSION_NOT_PRESENT";
        case XR_ERROR_ACTIONSETS_ALREADY_ATTACHED: return "XR_ERROR_ACTIONSETS_ALREADY_ATTACHED";
        case XR_ERROR_ACTIONSET_NOT_ATTACHED: return "XR_ERROR_ACTIONSET_NOT_ATTACHED";
        case XR_ERROR_PATH_UNSUPPORTED: return "XR_ERROR_PATH_UNSUPPORTED";
        case XR_ERROR_SESSION_NOT_RUNNING: return "XR_ERROR_SESSION_NOT_RUNNING";
        case XR_ERROR_SESSION_LOST: return "XR_ERROR_SESSION_LOST";
        case XR_ERROR_FUNCTION_UNSUPPORTED: return "XR_ERROR_FUNCTION_UNSUPPORTED";
        case XR_ERROR_VALIDATION_FAILURE: return "XR_ERROR_VALIDATION_FAILURE";
        case XR_ERROR_RUNTIME_FAILURE: return "XR_ERROR_RUNTIME_FAILURE";
        case XR_ERROR_HANDLE_INVALID: return "XR_ERROR_HANDLE_INVALID";
        default: return r >= 0 ? "success code" : "error";
    }
}

template <class T>
void Get(XrInstance inst, const char* name, T& fn) {
    PFN_xrVoidFunction f = nullptr;
    if (!g_next.gipa || XR_FAILED(g_next.gipa(inst, name, &f))) f = nullptr;
    fn = reinterpret_cast<T>(f);
    if (!fn) Log("OpenXR layer: next has no %s", name);
}

std::string PathString(XrPath p) {
    if (!g_next.pathToString || p == XR_NULL_PATH) return "(null)";
    char buf[XR_MAX_PATH_LENGTH] = {};
    uint32_t n = 0;
    if (XR_FAILED(g_next.pathToString(g_instance, p, sizeof(buf), &n, buf))) return "(?)";
    return buf;
}

bool RuntimeHasExtension(PFN_xrGetInstanceProcAddr gipa, const char* ext, std::string& list) {
    PFN_xrVoidFunction f = nullptr;
    if (XR_FAILED(gipa(XR_NULL_HANDLE, "xrEnumerateInstanceExtensionProperties", &f)) || !f) {
        list = "(xrEnumerateInstanceExtensionProperties unavailable)";
        return false;
    }
    auto enumerate = reinterpret_cast<PFN_xrEnumerateInstanceExtensionProperties>(f);
    uint32_t n = 0;
    if (XR_FAILED(enumerate(nullptr, 0, &n, nullptr))) return false;
    std::vector<XrExtensionProperties> props(n, {XR_TYPE_EXTENSION_PROPERTIES});
    if (XR_FAILED(enumerate(nullptr, n, &n, props.data()))) return false;
    bool found = false;
    for (auto& p : props) {
        list += list.empty() ? "" : " ";
        list += p.extensionName;
        found |= strcmp(p.extensionName, ext) == 0;
    }
    return found;
}

// Instance-level part of the gaze: our action set, the pose action and its binding. Runs inside OpenComposite's first
// xrAttachSessionActionSets, after all its own suggested bindings: done right after the instance was created, SteamVR
// then refused OpenComposite's next suggestion (hp/mixed_reality_controller, XR_ERROR_PATH_UNSUPPORTED) and
// OpenComposite aborted (smoke test A/B 2026-11-02).
void CreateGazeAction() {
    if (!g_next.createActionSet || !g_next.createAction || !g_next.stringToPath || !g_next.suggest) return;
    XrActionSetCreateInfo si{XR_TYPE_ACTION_SET_CREATE_INFO};
    strcpy_s(si.actionSetName, "ams2qvfr-gaze");
    strcpy_s(si.localizedActionSetName, "ams2-qvfr eye gaze");
    XrResult r = g_next.createActionSet(g_instance, &si, &g_set);
    if (XR_FAILED(r)) {
        Log("OpenXR layer: xrCreateActionSet = %d (%s); no gaze", (int)r, R(r));
        g_set = XR_NULL_HANDLE;
        return;
    }
    XrActionCreateInfo ai{XR_TYPE_ACTION_CREATE_INFO};
    strcpy_s(ai.actionName, "eye-gaze");
    strcpy_s(ai.localizedActionName, "Eye gaze");
    ai.actionType = XR_ACTION_TYPE_POSE_INPUT;
    r = g_next.createAction(g_set, &ai, &g_action);
    if (XR_FAILED(r)) {
        Log("OpenXR layer: xrCreateAction = %d (%s); no gaze", (int)r, R(r));
        g_action = XR_NULL_HANDLE;
        return;
    }
    XrPath profile = XR_NULL_PATH, pose = XR_NULL_PATH;
    g_next.stringToPath(g_instance, "/interaction_profiles/ext/eye_gaze_interaction", &profile);
    g_next.stringToPath(g_instance, "/user/eyes_ext/input/gaze_ext/pose", &pose);
    XrActionSuggestedBinding b{g_action, pose};
    XrInteractionProfileSuggestedBinding sb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    sb.interactionProfile = profile;
    sb.countSuggestedBindings = 1;
    sb.suggestedBindings = &b;
    r = g_next.suggest(g_instance, &sb);
    g_bindingsSuggested = XR_SUCCEEDED(r);
    Log("OpenXR layer: gaze action set created, binding /user/eyes_ext/input/gaze_ext/pose suggested: %d (%s)", (int)r,
        R(r));
}

void LogSystem(XrSystemId sys) {
    if (!g_next.getSystemProperties) return;
    XrSystemEyeGazeInteractionPropertiesEXT eye{XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT};
    XrSystemProperties p{XR_TYPE_SYSTEM_PROPERTIES};
    if (g_extEnabled) p.next = &eye;
    XrResult r = g_next.getSystemProperties(g_instance, sys, &p);
    Log("OpenXR layer: system \"%s\" (vendor 0x%X), max layers %u, max swapchain %u x %u, orientation/position tracking "
        "%d/%d, eye gaze interaction supported %s (%d, %s)",
        p.systemName, p.vendorId, p.graphicsProperties.maxLayerCount, p.graphicsProperties.maxSwapchainImageWidth,
        p.graphicsProperties.maxSwapchainImageHeight, p.trackingProperties.orientationTracking,
        p.trackingProperties.positionTracking, g_extEnabled ? (eye.supportsEyeGazeInteraction ? "yes" : "NO") : "(no extension)",
        (int)r, R(r));
    if (XR_FAILED(r)) return;
    char buf[320];
    snprintf(buf, sizeof(buf), "\"%s\" (OpenXR system, vendor 0x%X), eye tracking for applications: %s", p.systemName,
             p.vendorId, g_extEnabled ? (eye.supportsEyeGazeInteraction ? "yes" : "no") : "no (no extension)");
    diag::Set("Headset", buf);
}

void ResetSession() {
    g_hud.Stop();  // destroy its swapchain before the session/VIEW space disappear
    g_shouldRender = false;
    g_periodNs = 0;
    if (g_next.destroySpace) {
        if (g_s.gazeSpace) g_next.destroySpace(g_s.gazeSpace);
        if (g_s.viewSpace) g_next.destroySpace(g_s.viewSpace);
    }
    g_s = Session{};
    std::lock_guard lock(g_gm);
    g_gaze = Gaze{};
}

void LogSessionStats(const char* why) {
    const Session& s = g_s;
    Log("OpenXR layer: session %d %s: %llu xrWaitFrame, %llu xrEndFrame; xrSyncActions by OpenComposite %llu, by the "
        "layer %llu; gaze located %llu (mirrored in Z %llu), action inactive %llu, orientation invalid %llu, xrLocateSpace "
        "failed %llu",
        s.number, why, (unsigned long long)s.waits, (unsigned long long)s.endFrames, (unsigned long long)s.appSyncs,
        (unsigned long long)s.ownSyncs, (unsigned long long)s.gazeOk, (unsigned long long)s.gazeFlipped,
        (unsigned long long)s.gazeInactive, (unsigned long long)s.gazeInvalid, (unsigned long long)s.locateFail);
}

// After xrWaitFrame: our own sync when OpenComposite has not synced lately, then the gaze at the display time.
void UpdateGaze(XrSession session, XrTime time) {
    Session& s = g_s;
    if (session != s.h || !s.attached || !s.gazeSpace || !s.viewSpace) return;
    const double now = NowMs();
    if (s.lastAppSyncMs < 0 || now - s.lastAppSyncMs > kOwnSyncAfterMs) {
        std::vector<XrActiveActionSet> sets = s.appSets;  // keep OpenComposite's sets active as it last asked
        sets.push_back({g_set, XR_NULL_PATH});
        XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
        si.countActiveActionSets = (uint32_t)sets.size();
        si.activeActionSets = sets.data();
        XrResult r = g_next.sync(session, &si);
        if (s.ownSyncs++ == 0)
            Log("OpenXR layer: session %d: no xrSyncActions from OpenComposite for %.0f ms, the layer syncs itself "
                "(with %zu of OpenComposite's sets): %d (%s)",
                s.number, kOwnSyncAfterMs, s.appSets.size(), (int)r, R(r));
    }
    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = g_action;
    XrActionStatePose st{XR_TYPE_ACTION_STATE_POSE};
    XrResult r = g_next.getActionStatePose(session, &gi, &st);
    Gaze g;
    g.seq = s.waits;
    if (XR_FAILED(r) || !st.isActive) {
        ++s.gazeInactive;
    } else {
        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
        r = g_next.locateSpace(s.gazeSpace, s.viewSpace, time, &loc);
        if (XR_FAILED(r)) {
            ++s.locateFail;
        } else if (!(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
            ++s.gazeInvalid;
        } else {
            // Forward (0, 0, -1) rotated by the orientation: v' = v + 2 q.xyz x (q.xyz x v + w v).
            const XrQuaternionf& q = loc.pose.orientation;
            const double v[3] = {0, 0, -1};
            const double t[3] = {q.y * v[2] - q.z * v[1] + q.w * v[0], q.z * v[0] - q.x * v[2] + q.w * v[1],
                                 q.x * v[1] - q.y * v[0] + q.w * v[2]};
            const double d[3] = {v[0] + 2 * (q.y * t[2] - q.z * t[1]), v[1] + 2 * (q.z * t[0] - q.x * t[2]),
                                 v[2] + 2 * (q.x * t[1] - q.y * t[0])};
            const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (len > 1e-6) {
                // An eye cannot look backwards: -Z of the pose pointing behind the head means a runtime whose gaze
                // pose has +Z forward with right and up as usual (Pimax's SteamVR driver "aapvr"; checked in the
                // headset 2026-11-03: the fully reversed vector moved the focus left for a gaze to the right and down
                // for a gaze up). The gaze is then the pose's -Z mirrored in Z.
                const bool flipped = d[2] > 0;
                g.dir[0] = (float)(d[0] / len);
                g.dir[1] = (float)(d[1] / len);
                g.dir[2] = (float)((flipped ? -d[2] : d[2]) / len);
                g.ok = true;
                if (flipped && s.gazeFlipped++ == 0)
                    Log("OpenXR layer: session %d: the gaze pose's -Z points backwards (%.4f, %.4f, %.4f); the runtime "
                        "has +Z forward, the gaze is mirrored in Z",
                        s.number, d[0] / len, d[1] / len, d[2] / len);
                if (s.gazeOk++ == 0) {
                    s.firstGazeMs = now;
                    Log("OpenXR layer: session %d: first gaze, direction (%.4f, %.4f, %.4f), flags 0x%llX, position "
                        "(%.3f, %.3f, %.3f) m",
                        s.number, g.dir[0], g.dir[1], g.dir[2], (unsigned long long)loc.locationFlags,
                        loc.pose.position.x, loc.pose.position.y, loc.pose.position.z);
                }
                // A minute of samples, to see how a runtime reports the gaze (head-relative or not).
                if (now - s.firstGazeMs < kSampleForMs && (s.lastSampleMs < 0 || now - s.lastSampleMs >= kSampleEveryMs)) {
                    s.lastSampleMs = now;
                    constexpr double kRadToDeg = 57.29577951308232;
                    Log("OpenXR layer: gaze sample %.1f s: %.1f deg right, %.1f deg up%s, position (%.3f, %.3f, %.3f) m",
                        (now - s.firstGazeMs) / 1000, std::atan2(g.dir[0], -g.dir[2]) * kRadToDeg,
                        std::atan2(g.dir[1], -g.dir[2]) * kRadToDeg, flipped ? " (mirrored in Z)" : "", loc.pose.position.x,
                        loc.pose.position.y, loc.pose.position.z);
                }
            }
        }
    }
    {
        std::lock_guard lock(g_gm);
        g_gaze = g;
    }
    const int interval = Cfg().statsIntervalS;
    if (s.statsMs == 0) s.statsMs = now;
    if (interval > 0 && now - s.statsMs >= interval * 1000.0) {
        s.statsMs = now;
        LogSessionStats("so far");
    }
}

const char* LayerTypeName(XrStructureType t) {
    switch (t) {
        case XR_TYPE_COMPOSITION_LAYER_PROJECTION: return "projection";
        case XR_TYPE_COMPOSITION_LAYER_QUAD: return "quad";
        case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: return "cylinder";
        case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: return "equirect2";
        case XR_TYPE_COMPOSITION_LAYER_CUBE_KHR: return "cube";
        default: return "other";
    }
}

// The layer layout of a frame: logged for the first frames of a session and whenever it changes (a limited number).
void LogEndFrame(const XrFrameEndInfo* info) {
    Session& s = g_s;
    std::string sig;
    for (uint32_t i = 0; i < info->layerCount; ++i) {
        if (!info->layers[i]) continue;
        sig += LayerTypeName(info->layers[i]->type);
        sig += info->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION
                   ? std::to_string(reinterpret_cast<const XrCompositionLayerProjection*>(info->layers[i])->viewCount)
                   : "";
        sig += ' ';
    }
    const bool first = s.endFrames <= 3;
    if (!first && (sig == s.endFrameSig || s.endFrameLogs >= 20)) return;
    s.endFrameSig = sig;
    ++s.endFrameLogs;
    Log("OpenXR layer: session %d xrEndFrame %llu: blend mode %d, %u layers: %s", s.number,
        (unsigned long long)s.endFrames, (int)info->environmentBlendMode, info->layerCount, sig.c_str());
    for (uint32_t i = 0; i < info->layerCount; ++i) {
        const XrCompositionLayerBaseHeader* l = info->layers[i];
        if (!l) continue;
        if (l->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
            auto p = reinterpret_cast<const XrCompositionLayerProjection*>(l);
            for (uint32_t v = 0; v < p->viewCount; ++v) {
                const auto& pv = p->views[v];
                Log("  layer %u projection view %u: swapchain %p rect (%d, %d) %d x %d array %u, fov tan l %.4f r %.4f "
                    "u %.4f d %.4f, flags 0x%llX%s",
                    i, v, (void*)pv.subImage.swapchain, pv.subImage.imageRect.offset.x, pv.subImage.imageRect.offset.y,
                    pv.subImage.imageRect.extent.width, pv.subImage.imageRect.extent.height,
                    pv.subImage.imageArrayIndex, std::tan(pv.fov.angleLeft), std::tan(pv.fov.angleRight),
                    std::tan(pv.fov.angleUp), std::tan(pv.fov.angleDown), (unsigned long long)l->layerFlags,
                    pv.next ? " (with next)" : "");
            }
        } else if (l->type == XR_TYPE_COMPOSITION_LAYER_QUAD) {
            auto q = reinterpret_cast<const XrCompositionLayerQuad*>(l);
            Log("  layer %u quad: swapchain %p rect %d x %d, size %.3f x %.3f m, position (%.3f, %.3f, %.3f), eye %d, "
                "flags 0x%llX",
                i, (void*)q->subImage.swapchain, q->subImage.imageRect.extent.width, q->subImage.imageRect.extent.height,
                q->size.width, q->size.height, q->pose.position.x, q->pose.position.y, q->pose.position.z,
                (int)q->eyeVisibility, (unsigned long long)l->layerFlags);
        } else {
            Log("  layer %u type %d", i, (int)l->type);
        }
    }
}

// ---------------------------------------------------------------- hooks

XrResult XRAPI_CALL HookGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function);

XrResult XRAPI_CALL HookDestroyInstance(XrInstance instance) {
    Guarded("xr DestroyInstance", [&] {
        std::lock_guard lock(g_m);
        Log("OpenXR layer: xrDestroyInstance");
        if (g_s.h) LogSessionStats("at instance end");
        g_hud.Stop();
        g_s = Session{};
        g_set = XR_NULL_HANDLE;
        g_action = XR_NULL_HANDLE;
        g_instance = XR_NULL_HANDLE;
        std::lock_guard g(g_gm);
        g_gaze = Gaze{};
    });
    return g_next.destroyInstance(instance);
}

XrResult XRAPI_CALL HookGetSystem(XrInstance instance, const XrSystemGetInfo* info, XrSystemId* systemId) {
    XrResult r = g_next.getSystem(instance, info, systemId);
    Guarded("xr GetSystem", [&] {
        std::lock_guard lock(g_m);
        static int logs = 0;
        if (logs++ >= 4) return;
        Log("OpenXR layer: xrGetSystem(form factor %d) = %d (%s)", info ? (int)info->formFactor : -1, (int)r, R(r));
        if (XR_SUCCEEDED(r) && logs == 1) LogSystem(*systemId);
    });
    return r;
}

XrResult XRAPI_CALL HookCreateSession(XrInstance instance, const XrSessionCreateInfo* info, XrSession* session) {
    XrResult r = g_next.createSession(instance, info, session);
    Guarded("xr CreateSession", [&] {
        std::lock_guard lock(g_m);
        const XrBaseInStructure* gb = info ? static_cast<const XrBaseInStructure*>(info->next) : nullptr;
        Log("OpenXR layer: xrCreateSession = %d (%s), graphics binding type %d%s", (int)r, R(r), gb ? (int)gb->type : 0,
            gb && gb->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR ? " (D3D11)" : "");
        if (XR_FAILED(r)) return;
        if (g_s.h) {
            Log("OpenXR layer: session %d still open when a new one starts; its gaze state is dropped", g_s.number);
            ResetSession();
        }
        g_s.h = *session;
        g_s.number = ++g_sessions;
        if (g_next.createReferenceSpace) {
            XrReferenceSpaceCreateInfo ri{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
            ri.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
            ri.poseInReferenceSpace.orientation.w = 1;
            XrResult rr = g_next.createReferenceSpace(*session, &ri, &g_s.viewSpace);
            if (XR_FAILED(rr)) {
                g_s.viewSpace = XR_NULL_HANDLE;
                Log("OpenXR layer: VIEW reference space = %d (%s)", (int)rr, R(rr));
            }
        }
        ID3D11Device* device = nullptr;
        for (auto b = gb; b; b = b->next)
            if (b->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR)
                device = reinterpret_cast<const XrGraphicsBindingD3D11KHR*>(b)->device;
        XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES};
        if (info && g_next.getSystemProperties)
            g_next.getSystemProperties(instance, info->systemId, &props);
        g_hud.Start(*session, g_s.viewSpace, device, props.graphicsProperties.maxLayerCount, g_hudApi);
    });
    return r;
}

XrResult XRAPI_CALL HookDestroySession(XrSession session) {
    Guarded("xr DestroySession", [&] {
        std::lock_guard lock(g_m);
        if (session != g_s.h) return;
        LogSessionStats("ends (xrDestroySession)");
        ResetSession();
    });
    return g_next.destroySession(session);
}

XrResult XRAPI_CALL HookBeginSession(XrSession session, const XrSessionBeginInfo* info) {
    XrResult r = g_next.beginSession(session, info);
    Guarded("xr BeginSession", [&] {
        std::lock_guard lock(g_m);
        Log("OpenXR layer: session %d xrBeginSession(view configuration %d) = %d (%s)", session == g_s.h ? g_s.number : -1,
            info ? (int)info->primaryViewConfigurationType : -1, (int)r, R(r));
    });
    return r;
}

XrResult XRAPI_CALL HookEndSession(XrSession session) {
    XrResult r = g_next.endSession(session);
    Guarded("xr EndSession", [&] {
        std::lock_guard lock(g_m);
        Log("OpenXR layer: session %d xrEndSession = %d (%s)", session == g_s.h ? g_s.number : -1, (int)r, R(r));
    });
    return r;
}

XrResult XRAPI_CALL HookRequestExitSession(XrSession session) {
    XrResult r = g_next.requestExitSession(session);
    Guarded("xr RequestExitSession", [&] {
        std::lock_guard lock(g_m);
        Log("OpenXR layer: session %d xrRequestExitSession = %d (%s)", session == g_s.h ? g_s.number : -1, (int)r, R(r));
    });
    return r;
}

XrResult XRAPI_CALL HookSuggest(XrInstance instance, const XrInteractionProfileSuggestedBinding* info) {
    XrResult r = g_next.suggest(instance, info);
    Guarded("xr Suggest", [&] {
        std::lock_guard lock(g_m);
        static int logs = 0;
        if (!info || logs++ >= 40) return;
        std::string paths;
        for (uint32_t i = 0; i < info->countSuggestedBindings && i < 8; ++i)
            paths += " " + PathString(info->suggestedBindings[i].binding);
        Log("OpenXR layer: OpenComposite xrSuggestInteractionProfileBindings(%s, %u bindings:%s) = %d (%s)",
            PathString(info->interactionProfile).c_str(), info->countSuggestedBindings, paths.c_str(), (int)r, R(r));
    });
    return r;
}

XrResult XRAPI_CALL HookAttach(XrSession session, const XrSessionActionSetsAttachInfo* info) {
    bool done = false;
    XrResult r = XR_SUCCESS;
    Guarded("xr Attach", [&] {
        std::lock_guard lock(g_m);
        if (info && session == g_s.h && g_extEnabled && Cfg().gazeRead && !g_set && !g_gazeTried) {
            g_gazeTried = true;  // once per instance: after the first attach no binding can be suggested any more
            CreateGazeAction();
        }
        const bool ours = info && session == g_s.h && g_set && g_bindingsSuggested && !g_s.attached;
        Log("OpenXR layer: session %d OpenComposite xrAttachSessionActionSets(%u sets)%s", g_s.number,
            info ? info->countActionSets : 0, ours ? ", the gaze set added" : "");
        if (!ours) return;
        std::vector<XrActionSet> sets(info->actionSets, info->actionSets + info->countActionSets);
        sets.push_back(g_set);
        XrSessionActionSetsAttachInfo ai = *info;
        ai.countActionSets = (uint32_t)sets.size();
        ai.actionSets = sets.data();
        r = g_next.attach(session, &ai);
        if (XR_FAILED(r)) {
            Log("OpenXR layer: attach with the gaze set = %d (%s); retried without it, no gaze in this session", (int)r,
                R(r));
            return;  // done stays false: OpenComposite's own call below
        }
        done = true;
        g_s.attached = true;
        XrActionSpaceCreateInfo si{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        si.action = g_action;
        si.poseInActionSpace.orientation.w = 1;
        XrResult rs = g_next.createActionSpace(session, &si, &g_s.gazeSpace);
        if (XR_FAILED(rs)) g_s.gazeSpace = XR_NULL_HANDLE;
        Log("OpenXR layer: session %d: gaze set attached, action space = %d (%s), VIEW space %s", g_s.number, (int)rs,
            R(rs), g_s.viewSpace ? "ok" : "MISSING");
        diag::Set("OpenXR mod layer", XR_SUCCEEDED(rs) && g_s.viewSpace ? "loaded, gaze action set attached to the session"
                                                                         : "loaded, gaze action set attached, NO gaze space");
    });
    if (done) return r;
    r = g_next.attach(session, info);
    if (XR_FAILED(r)) Guarded("xr Attach (log)", [&] { Log("OpenXR layer: xrAttachSessionActionSets = %d (%s)", (int)r, R(r)); });
    return r;
}

XrResult XRAPI_CALL HookSync(XrSession session, const XrActionsSyncInfo* info) {
    bool done = false;
    XrResult r = XR_SUCCESS;
    Guarded("xr Sync", [&] {
        std::lock_guard lock(g_m);
        Session& s = g_s;
        if (!info || session != s.h || !s.attached) return;
        s.appSets.assign(info->activeActionSets, info->activeActionSets + info->countActiveActionSets);
        std::vector<XrActiveActionSet> sets = s.appSets;
        sets.push_back({g_set, XR_NULL_PATH});
        XrActionsSyncInfo si = *info;
        si.countActiveActionSets = (uint32_t)sets.size();
        si.activeActionSets = sets.data();
        r = g_next.sync(session, &si);
        done = true;
        if (s.appSyncs++ == 0)
            Log("OpenXR layer: session %d: first xrSyncActions from OpenComposite (%u sets), the gaze set added: %d (%s)",
                s.number, info->countActiveActionSets, (int)r, R(r));
        s.lastAppSyncMs = NowMs();
    });
    return done ? r : g_next.sync(session, info);
}

XrResult XRAPI_CALL HookWaitFrame(XrSession session, const XrFrameWaitInfo* info, XrFrameState* state) {
    XrResult r = g_next.waitFrame(session, info, state);
    if (XR_SUCCEEDED(r) && state) g_periodNs = state->predictedDisplayPeriod;
    if (XR_SUCCEEDED(r) && state)
        Guarded("xr WaitFrame", [&] {
            std::lock_guard lock(g_m);
            if (session != g_s.h) return;
            g_shouldRender = state->shouldRender == XR_TRUE;
            if (g_s.waits++ == 0)
                Log("OpenXR layer: session %d: first xrWaitFrame, period %.3f ms, shouldRender %d", g_s.number,
                    state->predictedDisplayPeriod / 1e6, state->shouldRender);
            UpdateGaze(session, state->predictedDisplayTime);
        });
    return r;
}

XrResult XRAPI_CALL HookEndFrame(XrSession session, const XrFrameEndInfo* info) {
    XrFrameEndInfo withHud{};
    XrCompositionLayerQuad hudQuad{};
    std::vector<const XrCompositionLayerBaseHeader*> hudLayers;
    bool added = false, shouldRender = false;
    Guarded("xr EndFrame", [&] {
        std::lock_guard lock(g_m);
        if (session != g_s.h || !info) return;
        shouldRender = g_shouldRender;
        ++g_s.endFrames;
        LogEndFrame(info);
        // A HUD problem must not trip the proxy's global diagnostic kill switch or interrupt the scene.
        try {
            added = g_hud.Prepare(info, withHud, hudQuad, hudLayers, NowMs(), g_periodNs.load() / 1e6, shouldRender);
        } catch (...) { g_hud.Disable("exception while preparing overlay"); }
    });
    const XrResult r = g_next.endFrame(session, added ? &withHud : info);  // exactly one downstream EndFrame
    {
        std::lock_guard lock(g_m);
        if (session == g_s.h) {
            try { g_hud.Ended(info, r, NowMs(), added, shouldRender); }
            catch (...) { g_hud.Disable("exception while sampling frames"); }
        }
    }
    return r;
}

// Session states and interaction profile changes: OpenComposite attaches its action sets only once the session is
// FOCUSED (and AMS2 never loads inputs), so the states explain a missing gaze.
XrResult XRAPI_CALL HookPollEvent(XrInstance instance, XrEventDataBuffer* ev) {
    XrResult r = g_next.pollEvent(instance, ev);
    if (r == XR_SUCCESS && ev)
        Guarded("xr PollEvent", [&] {
            std::lock_guard lock(g_m);
            static int logs = 0;
            if (logs >= 60) return;
            if (ev->type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                ++logs;
                auto* e = reinterpret_cast<const XrEventDataSessionStateChanged*>(ev);
                static const char* const names[] = {"UNKNOWN", "IDLE", "READY", "SYNCHRONIZED", "VISIBLE",
                                                    "FOCUSED", "STOPPING", "LOSS_PENDING", "EXITING"};
                const int st = (int)e->state;
                Log("OpenXR layer: session %d state %s%s", e->session == g_s.h ? g_s.number : -1,
                    st >= 0 && st <= 8 ? names[st] : "?",
                    st == XR_SESSION_STATE_FOCUSED && !g_s.attached ? " (OpenComposite may attach its action sets now)"
                                                                    : "");
            } else if (ev->type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
                ++logs;
                Log("OpenXR layer: interaction profile changed (session %d)", g_s.number);
            }
        });
    return r;
}

struct HookEntry {
    const char* name;
    PFN_xrVoidFunction fn;
    void* next;  // the next function must exist for the hook to be returned
};

HookEntry* Hooks(size_t& n) {
    static HookEntry hooks[] = {
        {"xrGetInstanceProcAddr", (PFN_xrVoidFunction)HookGetInstanceProcAddr, &g_next.gipa},
        {"xrDestroyInstance", (PFN_xrVoidFunction)HookDestroyInstance, &g_next.destroyInstance},
        {"xrGetSystem", (PFN_xrVoidFunction)HookGetSystem, &g_next.getSystem},
        {"xrCreateSession", (PFN_xrVoidFunction)HookCreateSession, &g_next.createSession},
        {"xrDestroySession", (PFN_xrVoidFunction)HookDestroySession, &g_next.destroySession},
        {"xrBeginSession", (PFN_xrVoidFunction)HookBeginSession, &g_next.beginSession},
        {"xrEndSession", (PFN_xrVoidFunction)HookEndSession, &g_next.endSession},
        {"xrRequestExitSession", (PFN_xrVoidFunction)HookRequestExitSession, &g_next.requestExitSession},
        {"xrSuggestInteractionProfileBindings", (PFN_xrVoidFunction)HookSuggest, &g_next.suggest},
        {"xrAttachSessionActionSets", (PFN_xrVoidFunction)HookAttach, &g_next.attach},
        {"xrSyncActions", (PFN_xrVoidFunction)HookSync, &g_next.sync},
        {"xrWaitFrame", (PFN_xrVoidFunction)HookWaitFrame, &g_next.waitFrame},
        {"xrEndFrame", (PFN_xrVoidFunction)HookEndFrame, &g_next.endFrame},
        {"xrPollEvent", (PFN_xrVoidFunction)HookPollEvent, &g_next.pollEvent},
    };
    n = sizeof(hooks) / sizeof(hooks[0]);
    return hooks;
}

XrResult XRAPI_CALL HookGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) {
    if (!g_next.gipa) return XR_ERROR_FUNCTION_UNSUPPORTED;  // no instance was created through the layer yet
    XrResult r = g_next.gipa(instance, name, function);
    if (XR_FAILED(r) || !name || !function || !*function || g_diagDisabled.load()) return r;
    size_t n = 0;
    HookEntry* hooks = Hooks(n);
    for (size_t i = 0; i < n; ++i)
        if (strcmp(name, hooks[i].name) == 0 && *static_cast<void**>(hooks[i].next)) {
            *function = hooks[i].fn;
            break;
        }
    return r;
}

void ResolveNext(XrInstance inst) {
    Get(inst, "xrEnumerateSwapchainFormats", g_hudApi.formats);
    Get(inst, "xrCreateSwapchain", g_hudApi.create);
    Get(inst, "xrDestroySwapchain", g_hudApi.destroy);
    Get(inst, "xrEnumerateSwapchainImages", g_hudApi.images);
    Get(inst, "xrAcquireSwapchainImage", g_hudApi.acquire);
    Get(inst, "xrWaitSwapchainImage", g_hudApi.wait);
    Get(inst, "xrReleaseSwapchainImage", g_hudApi.release);
    Get(inst, "xrDestroyInstance", g_next.destroyInstance);
    Get(inst, "xrGetInstanceProperties", g_next.getInstanceProperties);
    Get(inst, "xrGetSystem", g_next.getSystem);
    Get(inst, "xrGetSystemProperties", g_next.getSystemProperties);
    Get(inst, "xrCreateSession", g_next.createSession);
    Get(inst, "xrDestroySession", g_next.destroySession);
    Get(inst, "xrBeginSession", g_next.beginSession);
    Get(inst, "xrEndSession", g_next.endSession);
    Get(inst, "xrRequestExitSession", g_next.requestExitSession);
    Get(inst, "xrAttachSessionActionSets", g_next.attach);
    Get(inst, "xrSyncActions", g_next.sync);
    Get(inst, "xrSuggestInteractionProfileBindings", g_next.suggest);
    Get(inst, "xrWaitFrame", g_next.waitFrame);
    Get(inst, "xrEndFrame", g_next.endFrame);
    Get(inst, "xrStringToPath", g_next.stringToPath);
    Get(inst, "xrPathToString", g_next.pathToString);
    Get(inst, "xrCreateActionSet", g_next.createActionSet);
    Get(inst, "xrCreateAction", g_next.createAction);
    Get(inst, "xrCreateActionSpace", g_next.createActionSpace);
    Get(inst, "xrCreateReferenceSpace", g_next.createReferenceSpace);
    Get(inst, "xrDestroySpace", g_next.destroySpace);
    Get(inst, "xrLocateSpace", g_next.locateSpace);
    Get(inst, "xrGetActionStatePose", g_next.getActionStatePose);
    Get(inst, "xrPollEvent", g_next.pollEvent);
}

XrResult XRAPI_CALL CreateApiLayerInstance(const XrInstanceCreateInfo* info, const XrApiLayerCreateInfo* layerInfo,
                                           XrInstance* instance) {
    if (!info || !layerInfo || !layerInfo->nextInfo || !layerInfo->nextInfo->nextCreateApiLayerInstance ||
        !layerInfo->nextInfo->nextGetInstanceProcAddr)
        return XR_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(g_m);
    g_next = Next{};
    g_next.gipa = layerInfo->nextInfo->nextGetInstanceProcAddr;
    XrApiLayerCreateInfo chain = *layerInfo;
    chain.nextInfo = layerInfo->nextInfo->next;

    // OpenComposite's extensions plus XR_EXT_eye_gaze_interaction when the runtime has it.
    std::vector<const char*> exts;
    bool added = false, requested = false;
    Guarded("xr CreateInstance (extensions)", [&] {
        std::string runtimeExts, appExts;
        for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
            exts.push_back(info->enabledExtensionNames[i]);
            appExts += appExts.empty() ? "" : " ";
            appExts += info->enabledExtensionNames[i];
            requested |= strcmp(info->enabledExtensionNames[i], XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME) == 0;
        }
        const bool has = RuntimeHasExtension(g_next.gipa, XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME, runtimeExts);
        Log("OpenXR layer: xrCreateInstance by \"%s\" (engine \"%s\", API %u.%u.%u), next layer \"%s\"",
            info->applicationInfo.applicationName, info->applicationInfo.engineName,
            XR_VERSION_MAJOR(info->applicationInfo.apiVersion), XR_VERSION_MINOR(info->applicationInfo.apiVersion),
            XR_VERSION_PATCH(info->applicationInfo.apiVersion), layerInfo->nextInfo->layerName);
        Log("OpenXR layer: extensions requested by OpenComposite: %s", appExts.c_str());
        Log("OpenXR layer: extensions of the next layer/runtime: %s", runtimeExts.c_str());
        if (has && !requested) {
            exts.push_back(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);
            added = true;
        }
        if (!has) {
            Log("OpenXR layer: the runtime has no %s; no gaze", XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);
            diag::Report(diag::Level::Info, "xr-no-gaze-ext",
                         "The OpenXR runtime has no eye tracking for applications (XR_EXT_eye_gaze_interaction), so the "
                         "focus area stays straight ahead (FFR).");
        }
    });
    XrInstanceCreateInfo ci = *info;
    if (added) {
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.enabledExtensionNames = exts.data();
    }
    XrResult r = layerInfo->nextInfo->nextCreateApiLayerInstance(&ci, &chain, instance);
    if (XR_FAILED(r) && added) {
        Guarded("xr CreateInstance (log)", [&] {
            Log("OpenXR layer: xrCreateInstance with the gaze extension = %d (%s); retrying without it", (int)r, R(r));
        });
        added = false;
        r = layerInfo->nextInfo->nextCreateApiLayerInstance(info, &chain, instance);
    }
    if (XR_FAILED(r)) {
        Guarded("xr CreateInstance (log)", [&] { Log("OpenXR layer: xrCreateInstance = %d (%s)", (int)r, R(r)); });
        return r;
    }
    g_instance = *instance;
    g_extEnabled = added || requested;
    g_bindingsSuggested = false;
    g_gazeTried = false;
    g_set = XR_NULL_HANDLE;
    g_action = XR_NULL_HANDLE;
    ResolveNext(g_instance);
    Guarded("xr CreateInstance (after)", [&] {
        if (g_next.getInstanceProperties) {
            XrInstanceProperties p{XR_TYPE_INSTANCE_PROPERTIES};
            g_next.getInstanceProperties(g_instance, &p);
            Log("OpenXR layer: instance created, runtime \"%s\" %u.%u.%u, gaze extension %s", p.runtimeName,
                XR_VERSION_MAJOR(p.runtimeVersion), XR_VERSION_MINOR(p.runtimeVersion), XR_VERSION_PATCH(p.runtimeVersion),
                g_extEnabled ? "enabled" : "off");
            char buf[192];
            snprintf(buf, sizeof(buf), "\"%s\" %u.%u.%u", p.runtimeName, XR_VERSION_MAJOR(p.runtimeVersion),
                     XR_VERSION_MINOR(p.runtimeVersion), XR_VERSION_PATCH(p.runtimeVersion));
            diag::Set("OpenXR runtime", buf);
        }
    });
    return r;
}

}  // namespace

bool Loaded() { return g_loaded.load(); }

bool GazeAttached() {
    std::lock_guard lock(g_m);
    return g_s.attached && g_s.gazeSpace && g_s.viewSpace;
}

double DisplayPeriodMs() { return g_periodNs.load() / 1e6; }

bool ReadGaze(Gaze& out) {
    if (!g_loaded.load()) return false;
    std::lock_guard lock(g_gm);
    out = g_gaze;
    return out.seq != 0;
}

}  // namespace qvfr::xrlayer

// The loader's entry point (the manifest names no other function).
extern "C" __declspec(dllexport) XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo, const char* layerName, XrNegotiateApiLayerRequest* request) {
    using namespace qvfr;
    if (!loaderInfo || !request || loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
        request->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
        loaderInfo->minInterfaceVersion > XR_CURRENT_LOADER_API_LAYER_VERSION ||
        loaderInfo->maxInterfaceVersion < XR_CURRENT_LOADER_API_LAYER_VERSION ||
        (layerName && strcmp(layerName, xrlayer::kLayerName) != 0))
        return XR_ERROR_INITIALIZATION_FAILED;
    request->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    request->layerApiVersion = XR_CURRENT_API_VERSION;
    request->getInstanceProcAddr = xrlayer::HookGetInstanceProcAddr;
    request->createApiLayerInstance = xrlayer::CreateApiLayerInstance;
    xrlayer::g_loaded = true;
    Guarded("xr Negotiate (log)", [&] {
        diag::Set("OpenXR mod layer", "loaded by the OpenXR loader, gaze action set not attached yet");
        Log("OpenXR layer: negotiated with the loader (interface %u..%u, API %u.%u..%u.%u)", loaderInfo->minInterfaceVersion,
            loaderInfo->maxInterfaceVersion, XR_VERSION_MAJOR(loaderInfo->minApiVersion),
            XR_VERSION_MINOR(loaderInfo->minApiVersion), XR_VERSION_MAJOR(loaderInfo->maxApiVersion),
            XR_VERSION_MINOR(loaderInfo->maxApiVersion));
    });
    return XR_SUCCESS;
}

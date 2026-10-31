// Caller trace ([trace] callers=1): where in the game the key OpenVR calls come from.
// Every unique call stack of a method is logged once as module+RVA; hit counts go to the periodic stats.
#pragma once

namespace qvfr::callers {

// One per traced method, as a function-local static: static callers::Site site{"IVRSystem::GetProjectionMatrix"};
struct Site {
    const char* name;
    int id = -1;  // slot in the stack table, assigned on the first hit
};

void Note(Site& site);         // no-op unless callers=1; runs inside Guarded
void LogStats(bool totals);    // hits per stack since the previous stats, or session totals

}  // namespace qvfr::callers

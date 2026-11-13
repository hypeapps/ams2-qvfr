#pragma once

#include <algorithm>
#include <cmath>
#include <deque>

namespace qvfr::hud {

struct Settings {
    bool enabled = false, details = true, hotkey = true;
    float scale = 1, xDegrees = -18, yDegrees = 12;
    bool operator==(const Settings&) const = default;
};

inline float Bounded(float v, float fallback, float lo, float hi) {
    return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}
inline Settings Sanitize(Settings s) {
    s.scale = Bounded(s.scale, 1, 0.5f, 2);
    s.xDegrees = Bounded(s.xDegrees, -18, -30, 30);
    s.yDegrees = Bounded(s.yDegrees, 12, -30, 30);
    return s;
}

struct Reading {
    double fps = 0, frameMs = 0, runtimeHz = 0;
    bool valid = false;
};

// Count successful scene xrEndFrame calls, never eye submits or generated compositor frames.
// One-second sliding window; reset after a >2 s gap. Snapshot expires even if no new scene arrives.
class Meter {
    std::deque<double> times_;
public:
    void Reset() { times_.clear(); }
    void Submit(double now) {
        if (!std::isfinite(now)) return;
        if (!times_.empty() && now <= times_.back()) return;
        if (!times_.empty() && now - times_.back() > 2000) Reset();
        times_.push_back(now);
        while (times_.size() > 2 && times_[1] <= now - 1000) times_.pop_front();
        while (times_.size() > 2048) times_.pop_front();
    }
    Reading Read(double now, double periodMs) const {
        Reading r;
        if (std::isfinite(periodMs) && periodMs > 0) r.runtimeHz = 1000 / periodMs;
        if (times_.size() < 2 || now - times_.back() > 2000 || now < times_.back()) return r;
        const double span = times_.back() - times_.front();
        if (span < 200) return r;  // no misleading warm-up number from a handful of startup calls
        r.frameMs = span / (times_.size() - 1);
        r.fps = 1000 / r.frameMs;
        r.valid = true;
        return r;
    }
};

}  // namespace qvfr::hud

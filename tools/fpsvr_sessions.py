"""Summary of fpsVR sessions (%LOCALAPPDATA%\\fpsVR\\N.json).

For each session fpsVR stores frame time histograms (gputimes, cputimes) and sums (gputimessum,
cputimessum). The bucket width is derived from the condition that the histogram mean = sum / frame count,
so it does not depend on the fpsVR version. The last bucket means "everything above".

Usage: python -I tools/fpsvr_sessions.py [dir] [--app "Automobilista 2"]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def bin_width(hist, total_sum):
    n = sum(hist[:-1])
    if n == 0:
        return None
    # mean of the bucket centers: w * sum((i + 0.5) * c) / n  ==  total_sum / n_all
    weighted = sum((i + 0.5) * c for i, c in enumerate(hist[:-1]))
    n_all = sum(hist)
    return (total_sum / n_all) * n / weighted if weighted else None


def percentile(hist, w, p):
    n = sum(hist)
    target = p / 100.0 * n
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return (i + 1) * w if i < len(hist) - 1 else float("inf")
    return float("inf")


def share_above(hist, w, limit_ms):
    if limit_ms != limit_ms:  # NaN: the session has no refresh rate
        return float("nan")
    n = sum(hist)
    k = int(limit_ms / w)
    return 100.0 * sum(hist[k:]) / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=os.path.join(os.environ.get("LOCALAPPDATA", ""), "fpsVR"))
    ap.add_argument("--app", default=None)
    a = ap.parse_args()

    rows = []
    for f in Path(a.dir).glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if a.app and d.get("app") != a.app:
            continue
        rows.append((d.get("DateStart", ""), f.name, d))
    rows.sort()

    for start, name, d in rows:
        hz = d.get("hz") or 0
        budget = 1000.0 / hz if hz else float("nan")
        frames = d.get("frames", 0)
        used = d.get("framesused", 0)
        print(f"== {name}  {d.get('app')}  {start[:16]} -> {d.get('DateEnd', '')[11:16]}  SteamVR {d.get('SteamVR')}")
        print(f"   HMD {d.get('hmd')} {hz:.0f} Hz (budget {budget:.2f} ms), render {d.get('maxX')}x{d.get('maxY')}"
              f" (base {d.get('baseX')}x{d.get('baseY')}), time {d.get('stime', 0) / 60:.1f} min")
        print(f"   frames {frames}, used in stats {used}, dropped {d.get('dropped')} "
              f"({100.0 * d.get('dropped', 0) / max(frames, 1):.2f}%), reprojected {d.get('reprojected')} "
              f"({100.0 * d.get('reprojected', 0) / max(frames, 1):.2f}%)")
        for key, label in (("gputimes", "GPU"), ("cputimes", "CPU")):
            h = d.get(key) or []
            s = d.get(key + "sum")
            if not h or not s:
                continue
            w = bin_width(h, s)
            n = sum(h)
            if not w:
                continue
            print(f"   {label}: avg {s / n:5.2f} ms | p50 {percentile(h, w, 50):5.2f} | p90 {percentile(h, w, 90):5.2f} |"
                  f" p99 {percentile(h, w, 99):5.2f} ms | > budget {share_above(h, w, budget):5.1f}% of frames"
                  f"  (bucket {w:.3f} ms, {len(h)} buckets)")
        print(f"   GPU temp avg/max {d.get('GPU_Tavg', 0):.0f}/{d.get('GPU_Tmax', 0):.0f} C,"
              f" CPU {d.get('CPU_Tavg', 0):.0f}/{d.get('CPU_Tmax', 0):.0f} C")


if __name__ == "__main__":
    main()

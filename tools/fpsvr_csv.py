"""Stats from the fpsVR frame log (Documents\\fpsVR\\CSV\\Frametimes#Raw#*.csv).

One row per frame: SteamVR Time [s], FPS, GPU frametime [ms], CPU frametime [ms], GPU Usage [%],
CPU Usage [%], CPU0..CPUn [%]. Times have a 0.1 ms resolution.

The log is split in two parts: the first --still seconds (standing on the grid) and the rest (driving).

Usage: python -I tools/fpsvr_csv.py [file.csv ...] [--still 20] [--hz 72]
Without a file it takes the newest log from Documents\\fpsVR\\CSV.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[k]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [l for l in f if l.strip() and not l.startswith("sep=")]
    header_idx = next(i for i, l in enumerate(lines) if l.lstrip().startswith('"SteamVR Time"'))
    reader = csv.reader(lines[header_idx:], skipinitialspace=True)
    cols = next(reader)
    rows = []
    for r in reader:
        try:
            rows.append({c: float(x) for c, x in zip(cols, r)})
        except ValueError:
            continue
    return cols, rows


def summarize(label, rows, budget):
    if not rows:
        print(f"  {label}: no frames")
        return
    t0, t1 = rows[0]["SteamVR Time"], rows[-1]["SteamVR Time"]
    gpu = [r["GPU frametime"] for r in rows]
    cpu = [r["CPU frametime"] for r in rows]
    fps = [r["FPS"] for r in rows if r["FPS"] > 0]
    over_gpu = 100.0 * sum(g > budget for g in gpu) / len(gpu)
    over_cpu = 100.0 * sum(c > budget for c in cpu) / len(cpu)
    # gaps between frames longer than 1.5 budgets = a frame skipped by the application
    gaps = sum(1 for a, b in zip(rows, rows[1:]) if (b["SteamVR Time"] - a["SteamVR Time"]) * 1000 > 1.5 * budget)
    print(f"  {label}: {t1 - t0:6.1f} s, {len(rows)} frames, FPS avg {sum(fps) / max(len(fps), 1):5.1f}, gaps > 1.5 frames: {gaps}")
    print(f"     GPU ms: avg {sum(gpu) / len(gpu):5.2f} | p50 {pct(gpu, 50):5.1f} | p90 {pct(gpu, 90):5.1f} | p99 {pct(gpu, 99):5.1f}"
          f" | max {max(gpu):5.1f} | > {budget:.1f} ms: {over_gpu:4.1f}%")
    print(f"     CPU ms: avg {sum(cpu) / len(cpu):5.2f} | p50 {pct(cpu, 50):5.1f} | p90 {pct(cpu, 90):5.1f} | p99 {pct(cpu, 99):5.1f}"
          f" | max {max(cpu):5.1f} | > {budget:.1f} ms: {over_cpu:4.1f}%")
    gu = [r["GPU Usage"] for r in rows]
    cu = [r["CPU Usage"] for r in rows]
    print(f"     usage: GPU avg {sum(gu) / len(gu):4.0f}%, CPU avg {sum(cu) / len(cu):4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--still", type=float, default=20.0, help="length of the standstill at the start of the log [s]")
    ap.add_argument("--hz", type=float, default=72.0)
    a = ap.parse_args()
    files = a.files
    if not files:
        d = Path(os.environ["USERPROFILE"]) / "Documents" / "fpsVR" / "CSV"
        files = [str(max(d.glob("*.csv"), key=lambda p: p.stat().st_mtime))]
    budget = 1000.0 / a.hz
    for f in files:
        cols, rows = load(f)
        print(f"== {Path(f).name}  ({len(rows)} frames, budget {budget:.2f} ms at {a.hz:.0f} Hz)")
        if not rows:
            continue
        t0 = rows[0]["SteamVR Time"]
        still = [r for r in rows if r["SteamVR Time"] - t0 < a.still]
        drive = [r for r in rows if r["SteamVR Time"] - t0 >= a.still]
        summarize(f"standstill (first {a.still:.0f} s)", still, budget)
        summarize("driving (rest)", drive, budget)
        summarize("whole log", rows, budget)


if __name__ == "__main__":
    main()

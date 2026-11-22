"""Compare the image quality of proxy shots between runs (builds, settings).

Each run is a proxy log (x64\\qvfr\\logs\\qvfr_*.log) of a benchmark sequence with [quad] shot_bench_s > 0. The shots
of a run are the folders x64\\qvfr\\shots\\<stamp>\\ that its log names ("Shot <stamp>: ..."); each one gets the label
of the sequence step it was taken in. Shots with the same run and step are repeats of one image.

Metrics per view image (outer_l, outer_r, focus_l, focus_r; 8-bit display-encoded, a border of 4% left out):
  colours:   mean and percentiles of luma (0..255), L* standard deviation (contrast), mean CIELAB chroma,
             Hasler-Suesstrunk colourfulness
  sharpness: mean Sobel magnitude of L*, high-frequency energy (std of L* minus its 3 x 3 mean), overshoot
             (how far L* leaves the 5 x 5 min..max of its smoothed image, averaged over the strongest 10% edges:
             sharpening halos raise it)
  noise:     high-frequency energy in the flattest 30% of pixels; between repeats, mean abs delta L* (all and flat)

Usage:
  python tools/shot_compare.py --run 050=path/to/qvfr_A.log --run a2=path/to/qvfr_B.log --out report_dir
Optional --shots DIR when the shots are not in ..\\shots next to the log's folder. Needs numpy and Pillow.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS = ["focus_l", "focus_r", "outer_l", "outer_r"]
BORDER = 0.04
CROP = 320


def parse_log(log: Path):
    """[(stamp, step label)] in log order."""
    step = "no sequence"
    out = []
    step_re = re.compile(r'Benchmark sequence: step \d+/\d+ "([^"]*)"')
    shot_re = re.compile(r"\] Shot (\d{8}_\d{6}): \d+ of \d+ images read back")
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = step_re.search(line)
            if m:
                step = m.group(1)
                continue
            m = shot_re.search(line)
            if m:
                out.append((m.group(1), step))
    return out


def load(path: Path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    by, bx = int(h * BORDER), int(w * BORDER)
    return a[by : h - by, bx : w - bx]


def to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], dtype=np.float32
    )
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return L, a, b


def box(x, r):
    """Mean over (2r+1)^2 with edge padding."""
    p = np.pad(x, r, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def extreme(x, r, fn):
    p = np.pad(x, r, mode="edge")
    h, w = x.shape
    rows = p[:, :w].copy()
    for i in range(1, 2 * r + 1):
        rows = fn(rows, p[:, i : i + w])
    out = rows[:h].copy()
    for i in range(1, 2 * r + 1):
        out = fn(out, rows[i : i + h])
    return out


def sobel(L):
    p = np.pad(L, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy) / 8.0


def metrics(rgb):
    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * 255.0
    L, a, b = to_lab(rgb)
    g = sobel(L)
    hf = L - box(L, 1)
    smooth = box(box(L, 1), 1)
    hi = extreme(smooth, 2, np.maximum)
    lo = extreme(smooth, 2, np.minimum)
    over = np.maximum(L - hi, 0) + np.maximum(lo - L, 0)
    edges = g >= np.percentile(g, 90)
    flat = g <= np.percentile(g, 30)
    R, G, B = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    rg, yb = R - G, 0.5 * (R + G) - B
    colourful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    p = np.percentile(luma, [1, 5, 50, 95, 99])
    return {
        "luma mean": float(luma.mean()),
        "luma p1": float(p[0]),
        "luma p5": float(p[1]),
        "luma p50": float(p[2]),
        "luma p95": float(p[3]),
        "luma p99": float(p[4]),
        "L* std (contrast)": float(L.std()),
        "chroma mean": float(np.hypot(a, b).mean()),
        "colourfulness": float(colourful),
        "gradient mean": float(g.mean()),
        "high-freq std": float(hf.std()),
        "overshoot at edges": float(over[edges].mean()),
        "noise flat (hf std)": float(hf[flat].std()),
    }, L, flat


def temporal(Ls, flats):
    if len(Ls) < 2:
        return {}
    d_all, d_flat = [], []
    for i in range(1, len(Ls)):
        if Ls[i].shape != Ls[i - 1].shape:
            continue
        d = np.abs(Ls[i] - Ls[i - 1])
        d_all.append(d.mean())
        d_flat.append(d[flats[i] & flats[i - 1]].mean())
    if not d_all:
        return {}
    return {"repeat mean abs dL* (all)": float(np.mean(d_all)), "repeat mean abs dL* (flat)": float(np.mean(d_flat))}


def crop(path: Path, size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size, w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=path to the proxy log")
    ap.add_argument("--shots", help="shots root (default: ..\\shots next to the log's folder)")
    ap.add_argument("--out", required=True, help="folder for report.md and the crops")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = {}  # (run, step) -> [shot folders]
    for spec in args.run:
        name, _, log = spec.partition("=")
        log = Path(log)
        root = Path(args.shots) if args.shots else log.parent.parent / "shots"
        for stamp, step in parse_log(log):
            d = root / stamp
            if all((d / f"{v}.png").exists() for v in VIEWS):
                groups.setdefault((name, step), []).append(d)
            else:
                print(f"{name}: shot {stamp} ({step}) incomplete or missing in {root}", file=sys.stderr)
    if not groups:
        sys.exit("no complete shots found")

    report = ["# Shot comparison", ""]
    for (name, step), dirs in groups.items():
        report.append(f"- `{name}` / `{step}`: {len(dirs)} shots: " + ", ".join(d.name for d in dirs))
    keys = list(groups)
    for view in VIEWS:
        table = {}
        for key in keys:
            per = []
            Ls, flats = [], []
            for d in groups[key]:
                m, L, flat = metrics(load(d / f"{view}.png"))
                per.append(m)
                Ls.append(L)
                flats.append(flat)
            avg = {k: (np.mean([p[k] for p in per]), np.std([p[k] for p in per])) for k in per[0]}
            for k, v in temporal(Ls, flats).items():
                avg[k] = (v, 0.0)
            table[key] = avg
            print(f"{view} {key[0]}/{key[1]}: done", file=sys.stderr)
        names = [k for k in table[keys[0]]]
        for key in keys[1:]:
            names += [k for k in table[key] if k not in names]
        report += ["", f"## {view}", "", "| metric | " + " | ".join(f"{r} / {s}" for r, s in keys) + " |",
                   "|---" * (len(keys) + 1) + "|"]
        for n in names:
            cells = []
            for key in keys:
                v = table[key].get(n)
                cells.append("" if v is None else (f"{v[0]:.3f} ± {v[1]:.3f}" if v[1] else f"{v[0]:.3f}"))
            report.append(f"| {n} | " + " | ".join(cells) + " |")
        # Side by side centre crops of the first shot of each group, 2x nearest.
        tiles = [crop(groups[key][0] / f"{view}.png", CROP) for key in keys]
        s = tiles[0].size[0]
        sheet = Image.new("RGB", (s * len(tiles), s))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((s, s)), (i * s, 0))
        sheet = sheet.resize((sheet.size[0] * 2, sheet.size[1] * 2), Image.NEAREST)
        sheet.save(out / f"crop_{view}.png")
        report += ["", f"Crop `crop_{view}.png`, left to right: " + ", ".join(f"{r} / {s}" for r, s in keys)]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()

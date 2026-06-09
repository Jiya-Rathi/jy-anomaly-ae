"""
lcfs_sol_distance_distribution.py

Computes the minimum distance from each image's LCFS pixels to the
D-shaped SOL inner boundary, across all four classes. Plots the
resulting distributions to empirically determine the band threshold.

Classes: healthy | broken_lcfs | bad_black_core | bad_nonconverged

Usage:
    python lcfs_sol_distance_distribution.py \
        --base   /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/ \
        --sol    /path/to/small_sol_RZ.txt \
        --outdir /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/SOL_extraction/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from PIL import Image


# ── Configuration ──────────────────────────────────────────────────────────────
PLOT_CROP    = (72, 28, 300, 500)
R_MIN, R_MAX = 0.5, 3.2
Z_MIN, Z_MAX = -2.5, 2.5
LCFS_COLOR   = np.array([250, 5, 128])
LCFS_TOL     = 15

CLASSES = ["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"]
COLORS  = {
    "healthy":          "#1f77b4",   # blue
    "broken_lcfs":      "#ff7f0e",   # orange
    "bad_black_core":   "#d62728",   # red
    "bad_nonconverged": "#9467bd",   # purple
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def pixels_to_physical(rows, cols):
    x0, y0, x1, y1 = PLOT_CROP
    R = R_MIN + (cols - x0) / (x1 - x0) * (R_MAX - R_MIN)
    Z = Z_MAX - (rows - y0) / (y1 - y0) * (Z_MAX - Z_MIN)
    return R, Z


def min_dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    """Minimum distance from any point in pts to the polyline."""
    min_d = np.inf
    for i in range(len(poly) - 1):
        p1, p2   = poly[i], poly[i + 1]
        seg      = p2 - p1
        seg_len2 = np.dot(seg, seg)
        if seg_len2 == 0:
            d = np.linalg.norm(pts - p1, axis=1)
        else:
            t    = np.clip(((pts - p1) @ seg) / seg_len2, 0, 1)
            proj = p1 + t[:, None] * seg
            d    = np.linalg.norm(pts - proj, axis=1)
        min_d = min(min_d, d.min())
    return float(min_d)


def process_folder(folder: Path, sol_inner: np.ndarray) -> dict:
    """
    Returns dict with:
      distances : list of per-image min LCFS-to-SOL distances
      no_lcfs   : count of images where LCFS color was not found
      filenames : list of filenames (parallel to distances)
    """
    images    = sorted(folder.glob("*.png"))
    distances = []
    filenames = []
    no_lcfs   = 0

    print(f"\n  Processing {folder.name}/  ({len(images)} images)")
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"    {i}/{len(images)} ...", flush=True)

        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.all(np.abs(img.astype(int) - LCFS_COLOR) <= LCFS_TOL, axis=2)

        if mask.sum() == 0:
            no_lcfs += 1
            continue

        rows, cols   = np.where(mask)
        R, Z         = pixels_to_physical(rows, cols)
        pts          = np.column_stack([R, Z])
        min_d        = min_dist_to_polyline(pts, sol_inner)

        distances.append(min_d)
        filenames.append(img_path.name)

    print(f"    Done. {len(distances)} with LCFS, {no_lcfs} without LCFS.")
    return {"distances": distances, "no_lcfs": no_lcfs, "filenames": filenames}


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(results: dict, outdir: str, band_candidates: list):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        "LCFS Minimum Distance to SOL Inner Boundary (D-shape)\n"
        "Distribution across all four classes",
        fontsize=13, y=0.98
    )

    # ── Panel 1: KDE ──────────────────────────────────────────────────────────
    ax1 = axes[0]
    for cls in CLASSES:
        dists = results[cls]["distances"]
        if len(dists) < 2:
            continue
        d = np.array(dists)
        kde = gaussian_kde(d, bw_method=0.15)
        x   = np.linspace(0, d.max() * 1.1, 500)
        ax1.plot(x, kde(x), color=COLORS[cls], linewidth=2,
                 label=f"{cls}  (n={len(d)}, "
                       f"median={np.median(d):.3f}, "
                       f"min={d.min():.3f})")
        ax1.axvline(np.median(d), color=COLORS[cls],
                    linewidth=1, linestyle=":", alpha=0.7)

    for band, label in band_candidates:
        ax1.axvline(band, color="black", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"band={band} a₀  ({label})")

    ax1.set_xlabel("Min distance LCFS → SOL inner boundary (a₀)", fontsize=11)
    ax1.set_ylabel("Density (KDE)", fontsize=11)
    ax1.set_title("KDE — dotted vertical lines = class medians")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # ── Panel 2: Histogram (normalised, overlapping) ───────────────────────────
    ax2 = axes[1]
    all_dists = np.concatenate([results[c]["distances"] for c in CLASSES
                                 if results[c]["distances"]])
    bins = np.linspace(0, np.percentile(all_dists, 99), 60)

    for cls in CLASSES:
        dists = results[cls]["distances"]
        if not dists:
            continue
        ax2.hist(np.array(dists), bins=bins, density=True,
                 color=COLORS[cls], alpha=0.35, label=cls, edgecolor="none")

    for band, label in band_candidates:
        ax2.axvline(band, color="black", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"band={band} ({label})")

    ax2.set_xlabel("Min distance LCFS → SOL inner boundary (a₀)", fontsize=11)
    ax2.set_ylabel("Density (histogram, normalised)", fontsize=11)
    ax2.set_title("Histogram overlay — choose threshold where broken_lcfs "
                  "separates from healthy")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "lcfs_sol_distance_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out}")


def print_summary(results: dict):
    print("\n── Distance summary (a₀) ─────────────────────────────────────────")
    print(f"{'Class':<20} {'n':>5} {'no_lcfs':>8} "
          f"{'min':>8} {'p5':>8} {'median':>8} {'p95':>8} {'max':>8}")
    print("─" * 75)
    for cls in CLASSES:
        d = np.array(results[cls]["distances"])
        nl = results[cls]["no_lcfs"]
        if len(d) == 0:
            print(f"{cls:<20} {'0':>5} {nl:>8}")
            continue
        print(f"{cls:<20} {len(d):>5} {nl:>8} "
              f"{d.min():>8.4f} "
              f"{np.percentile(d,5):>8.4f} "
              f"{np.median(d):>8.4f} "
              f"{np.percentile(d,95):>8.4f} "
              f"{d.max():>8.4f}")
    print("─" * 75)


def save_csv(results: dict, outdir: str):
    """Save per-image distances to CSV for further analysis."""
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "lcfs_sol_distances.csv")
    with open(out, "w") as f:
        f.write("filename,class,min_dist_a0\n")
        for cls in CLASSES:
            for fname, dist in zip(results[cls]["filenames"],
                                   results[cls]["distances"]):
                f.write(f"{fname},{cls},{dist:.6f}\n")
    print(f"CSV saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",   required=True,
                        help="Base folder containing class subfolders")
    parser.add_argument("--sol",    required=True,
                        help="D-shape inner SOL boundary (small_sol_RZ.txt)")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for figure + CSV")
    parser.add_argument("--bands",  nargs="+", type=float,
                        default=[0.02, 0.05, 0.10],
                        help="Candidate band thresholds to mark on plot")
    args = parser.parse_args()

    sol_inner = np.loadtxt(args.sol)
    print(f"D-shape SOL: {len(sol_inner)} points loaded")

    base = Path(args.base)
    results = {}
    for cls in CLASSES:
        folder = base / cls
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            results[cls] = {"distances": [], "no_lcfs": 0, "filenames": []}
            continue
        results[cls] = process_folder(folder, sol_inner)

    print_summary(results)
    save_csv(results, args.outdir)

    # Band candidates to show on plot (value, label)
    band_candidates = [(b, f"{b} a₀") for b in args.bands]
    plot_distributions(results, args.outdir, band_candidates)

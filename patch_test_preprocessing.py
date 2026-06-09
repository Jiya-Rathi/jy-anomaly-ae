"""
patch_test_preprocessing.py  (v2 — convex hull interior + colour output)

Visual sanity check for the proposed LCFS-SOL-whitecore preprocessing.
Randomly samples N images from each class, applies the full preprocessing
pipeline, and saves side-by-side figures (original vs processed).

Processed image is RGB:
    LCFS line pixels  → cyan   (0, 255, 255)
    SOL boundary      → yellow (255, 255, 0)
    White core pixels → white  (255, 255, 255)
    Everything else   → black  (0, 0, 0)

Key fix vs v1:
    The LCFS magenta line is often rendered as many fragmented pixel groups
    (anti-aliasing, IDL rendering artefacts).  The previous largest-connected-
    component + angle-sort approach produced a broken polygon that mis-labelled
    the interior.  v2 takes the CONVEX HULL of ALL detected LCFS pixels, which
    is always a proper closed polygon regardless of fragmentation level, and
    reliably encloses the plasma interior.

Usage:
    python patch_test_preprocessing.py \\
        --manifest-dir  /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests/ \\
        --images-root   /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/ \\
        --sol           /mnt/beegfs/mantis/jrathi/outer_vacuum_boundary_spline_RZ.txt \\
        --outdir        ./patch_test_outputs/ \\
        --n-per-class   5 \\
        --seed          42
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.draw import line as draw_line


# ── Constants ──────────────────────────────────────────────────────────────────

PLOT_CROP    = (72, 28, 300, 500)
R_MIN, R_MAX = 0.5, 3.2
Z_MIN, Z_MAX = -2.5, 2.5

LCFS_COLOR   = np.array([250, 5, 128])   # #FA0580
LCFS_TOL     = 15

# White core: pixels with ALL channels >= this value, inside the LCFS polygon
# 255 = pure white only; lower values catch near-white colormap pixels too
WHITE_THRESHOLD = 240

# Processed image colours (RGB)
COLOR_BG   = np.array([0,   0,   0  ], dtype=np.uint8)   # background
COLOR_SOL  = np.array([255, 255, 0  ], dtype=np.uint8)   # SOL   → yellow
COLOR_LCFS = np.array([0,   255, 255], dtype=np.uint8)   # LCFS  → cyan
COLOR_CORE = np.array([255, 255, 255], dtype=np.uint8)   # core  → white

CLASS_ORDER  = ["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"]
CLASS_COLORS = {
    "healthy":          "#1f77b4",
    "broken_lcfs":      "#ff7f0e",
    "bad_black_core":   "#d62728",
    "bad_nonconverged": "#9467bd",
}


# ── Coordinate helpers (CROPPED image space) ───────────────────────────────────

def _crop_hw():
    x0, y0, x1, y1 = PLOT_CROP
    return y1 - y0, x1 - x0


def crop_pix_to_phys(rows, cols):
    H, W = _crop_hw()
    R = R_MIN + np.asarray(cols, float) / W * (R_MAX - R_MIN)
    Z = Z_MAX - np.asarray(rows, float) / H * (Z_MAX - Z_MIN)
    return R, Z


def phys_to_crop_pix(R, Z):
    H, W = _crop_hw()
    cols = (np.asarray(R, float) - R_MIN) / (R_MAX - R_MIN) * W
    rows = (Z_MAX - np.asarray(Z, float)) / (Z_MAX - Z_MIN) * H
    return rows.astype(int), cols.astype(int)


# ── LCFS polygon — convex hull approach ────────────────────────────────────────

def build_lcfs_polygon(cropped_rgb):
    """
    Detect LCFS pixels and build an interior polygon using the convex hull
    of ALL detected LCFS pixels.

    Using a convex hull instead of angle-sorting the largest connected
    component avoids the broken-polygon problem caused by the LCFS being
    rendered as many fragmented pixel groups (artefact of IDL anti-aliasing).
    The convex hull is always a valid closed polygon that reliably encloses
    the plasma interior.

    Returns
    -------
    polygon   : np.ndarray [K, 2] in (R, Z), or None
    lcfs_mask : np.ndarray [H, W] bool  (all detected LCFS pixels, for drawing)
    note      : str
    """
    lcfs_mask = np.all(
        np.abs(cropped_rgb.astype(int) - LCFS_COLOR) <= LCFS_TOL, axis=2
    )

    n_px = lcfs_mask.sum()
    if n_px < 3:
        return None, lcfs_mask, "WARNING: fewer than 3 LCFS pixels found"

    rows, cols = np.where(lcfs_mask)
    R, Z       = crop_pix_to_phys(rows, cols)
    points     = np.column_stack([R, Z])

    # Count fragments for the console note
    _, n_components = ndimage.label(lcfs_mask)

    try:
        hull    = ConvexHull(points)
        polygon = points[hull.vertices]   # CCW ordered hull vertices
        note    = (f"{n_components} LCFS fragments, {n_px} px → "
                   f"convex hull ({len(hull.vertices)} pts)")
    except Exception as exc:
        return None, lcfs_mask, f"ConvexHull failed: {exc}"

    return polygon, lcfs_mask, note


# ── Inside-LCFS mask ───────────────────────────────────────────────────────────

def get_inside_lcfs_mask(polygon, H, W):
    all_rows, all_cols = np.mgrid[0:H, 0:W]
    R_all, Z_all       = crop_pix_to_phys(all_rows.ravel(), all_cols.ravel())
    pts                = np.column_stack([R_all, Z_all])
    inside             = MplPath(polygon).contains_points(pts).reshape(H, W)
    return inside


# ── SOL rasterization ──────────────────────────────────────────────────────────

def rasterize_sol(sol_coords, H, W):
    sol_rows, sol_cols = phys_to_crop_pix(sol_coords[:, 0], sol_coords[:, 1])
    mask = np.zeros((H, W), dtype=bool)
    for i in range(len(sol_rows) - 1):
        r0, c0 = int(sol_rows[i]),     int(sol_cols[i])
        r1, c1 = int(sol_rows[i + 1]), int(sol_cols[i + 1])
        rr, cc = draw_line(r0, c0, r1, c1)
        valid  = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        mask[rr[valid], cc[valid]] = True
    return mask


# ── Main preprocessing ─────────────────────────────────────────────────────────

def preprocess_image(image_path, sol_coords):
    """
    Full preprocessing pipeline for one image.

    Returns
    -------
    cropped   : np.ndarray [H, W, 3] uint8   original cropped RGB
    processed : np.ndarray [H, W, 3] uint8   colour-coded processed image
    info      : dict
    """
    x0, y0, x1, y1 = PLOT_CROP
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    cropped = img_rgb[y0:y1, x0:x1].copy()
    H, W    = cropped.shape[:2]

    polygon, lcfs_mask, note = build_lcfs_polygon(cropped)

    # Start with all-black RGB canvas
    processed = np.zeros((H, W, 3), dtype=np.uint8)

    if polygon is None:
        return cropped, processed, {
            "note": note, "n_lcfs_px": 0,
            "n_white_core": 0, "n_sol_px": 0,
        }

    # Layer 1 — SOL boundary (yellow)
    sol_mask = rasterize_sol(sol_coords, H, W)
    processed[sol_mask] = COLOR_SOL

    # Layer 2 — LCFS line (cyan) — drawn on top of SOL
    processed[lcfs_mask] = COLOR_LCFS

    # Layer 3 — white core pixels inside LCFS (white) — drawn on top
    inside_mask     = get_inside_lcfs_mask(polygon, H, W)
    white_mask      = np.all(cropped >= WHITE_THRESHOLD, axis=2)
    white_core_mask = white_mask & inside_mask
    processed[white_core_mask] = COLOR_CORE

    info = {
        "note":         note,
        "n_lcfs_px":    int(lcfs_mask.sum()),
        "n_white_core": int(white_core_mask.sum()),
        "n_sol_px":     int(sol_mask.sum()),
    }
    return cropped, processed, info


# ── Manifest loading ───────────────────────────────────────────────────────────

def load_all_manifests(manifest_dir):
    manifest_dir = Path(manifest_dir)
    by_class     = defaultdict(list)
    for filename in ["train.txt", "val.txt", "selection.txt", "test.txt"]:
        mf_path = manifest_dir / filename
        if not mf_path.exists():
            continue
        with open(mf_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split("\t")
                if len(parts) == 2:
                    by_class[parts[1]].append(parts[0])
    return by_class


# ── Figures ────────────────────────────────────────────────────────────────────

def _colour_legend():
    return [
        mpatches.Patch(color="#00ffff", label="LCFS line"),
        mpatches.Patch(color="#ffff00", label="SOL boundary"),
        mpatches.Patch(color="white",   label="White core"),
        mpatches.Patch(color="black",   label="Background",
                       linewidth=0.5, edgecolor="gray"),
    ]


def make_class_figure(class_name, results, outdir):
    n     = len(results)
    color = CLASS_COLORS.get(class_name, "gray")

    fig, axes = plt.subplots(
        2, n,
        figsize=(3.5 * n, 9),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Patch test — class: {class_name}   ({n} random samples)",
        fontsize=13, fontweight="bold", color=color, y=0.99,
    )
    axes[0, 0].set_ylabel("Original\n(cropped)", fontsize=10, labelpad=6)
    axes[1, 0].set_ylabel("Processed\n(colour)",  fontsize=10, labelpad=6)

    for col, (cropped, processed, info, shot) in enumerate(results):
        axes[0, col].imshow(cropped)
        axes[0, col].set_title(shot, fontsize=6, pad=3)
        axes[0, col].axis("off")

        axes[1, col].imshow(processed)            # RGB — no cmap
        axes[1, col].axis("off")

        lines = [f"LCFS: {info['n_lcfs_px']} px",
                 f"Core: {info['n_white_core']} px"]
        if info["note"]:
            lines.insert(0, f"⚠ {info['note']}")
        axes[1, col].set_xlabel("\n".join(lines), fontsize=5.5, labelpad=5)

    fig.legend(handles=_colour_legend(), loc="lower center",
               ncol=4, fontsize=8, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.005))

    out_path = outdir / f"patch_test_{class_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def make_summary_figure(all_results, outdir):
    n_cls  = len(CLASS_ORDER)
    fig, axes = plt.subplots(
        n_cls, 2,
        figsize=(7, 4.5 * n_cls),
        gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    )
    axes[0, 0].set_title("Original (cropped)", fontsize=11, pad=8)
    axes[0, 1].set_title("Processed (colour)", fontsize=11, pad=8)

    for i, cls in enumerate(CLASS_ORDER):
        color   = CLASS_COLORS.get(cls, "gray")
        ax_orig = axes[i, 0]
        ax_proc = axes[i, 1]

        if cls not in all_results or not all_results[cls]:
            for ax in (ax_orig, ax_proc):
                ax.axis("off")
                ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                        transform=ax.transAxes)
            ax_orig.set_ylabel(cls, fontsize=11, color=color, fontweight="bold")
            continue

        cropped, processed, info, shot = all_results[cls][0]
        ax_orig.imshow(cropped)
        ax_orig.set_ylabel(cls, fontsize=11, color=color,
                           fontweight="bold", labelpad=6)
        ax_orig.set_title(shot, fontsize=7)
        ax_orig.axis("off")

        ax_proc.imshow(processed)
        ax_proc.axis("off")
        ax_proc.set_title(f"core={info['n_white_core']} px", fontsize=7)

    fig.legend(handles=_colour_legend(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.005))

    out_path = outdir / "patch_test_SUMMARY.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Summary: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--manifest-dir",  required=True)
    parser.add_argument("--images-root",   required=True)
    parser.add_argument("--sol",           required=True)
    parser.add_argument("--outdir",        required=True)
    parser.add_argument("--n-per-class",   type=int, default=5)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sol_coords = np.loadtxt(args.sol)
    print(f"SOL: {len(sol_coords)} points\n")

    by_class = load_all_manifests(Path(args.manifest_dir))
    print("Pool per class:")
    for cls in CLASS_ORDER:
        print(f"  {cls:22s}: {len(by_class.get(cls, []))} images")
    print()

    all_results = {}

    for cls in CLASS_ORDER:
        pool = by_class.get(cls, [])
        if not pool:
            print(f"[SKIP] {cls}\n")
            continue

        n       = min(args.n_per_class, len(pool))
        samples = random.sample(pool, n)

        print(f"── {cls}  ({n} samples) ──────────────────────────")
        class_results = []

        for rel_path in samples:
            shot = Path(rel_path).stem
            try:
                cropped, processed, info = preprocess_image(
                    str(Path(args.images_root) / rel_path), sol_coords
                )
                class_results.append((cropped, processed, info, shot))
                print(f"  {shot}: LCFS={info['n_lcfs_px']:4d}px  "
                      f"core={info['n_white_core']:4d}px  "
                      f"SOL={info['n_sol_px']:3d}px  | {info['note']}")
            except Exception as exc:
                print(f"  [ERROR] {shot}: {exc}")

        all_results[cls] = class_results
        print()
        if class_results:
            make_class_figure(cls, class_results, outdir)

    print("Generating summary...")
    make_summary_figure(all_results, outdir)
    print(f"\nDone → {outdir}")


if __name__ == "__main__":
    main()

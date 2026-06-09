"""
show_color_mask.py

Shows only pixels matching a target hex color as white, everything else black.
Can run on a single image or batch-process an entire folder.

Usage — single image:
    python show_color_mask.py \
        --image /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/broken_lcfs/jy_186881_3000.png \
        --outdir /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/SOL_extraction/

Usage — batch (whole folder):
    python show_color_mask.py \
        --folder /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/broken_lcfs/ \
        --outdir  /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/SOL_extraction/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ── Colors ─────────────────────────────────────────────────────────────────────
LCFS_HEX = "FA0580"   # fixed


def hex_to_rgb(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)])


def build_mask(img_rgb: np.ndarray, target: np.ndarray, tol: int) -> np.ndarray:
    return np.all(np.abs(img_rgb.astype(int) - target) <= tol, axis=2)


# ── Per-image processing ───────────────────────────────────────────────────────

def process_image(image_path: str, outdir: str, sol_hex: str, tol: int) -> dict:
    img         = np.array(Image.open(image_path).convert("RGB"))
    sol_target  = hex_to_rgb(sol_hex)
    lcfs_target = hex_to_rgb(LCFS_HEX)

    sol_mask  = build_mask(img, sol_target,  tol)
    lcfs_mask = build_mask(img, lcfs_target, tol)
    overlap   = sol_mask & lcfs_mask

    stem = Path(image_path).stem
    print(f"{stem}  |  SOL={sol_mask.sum()} px  "
          f"LCFS={lcfs_mask.sum()} px  overlap={overlap.sum()} px  "
          f"{'⚠ BROKEN' if overlap.sum() > 0 else '✓ OK'}")

    # ── 4-panel figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"{stem}\n"
        f"SOL=#{sol_hex.upper()} ({sol_mask.sum()} px)   "
        f"LCFS=#{LCFS_HEX} ({lcfs_mask.sum()} px)   "
        f"tol=±{tol}",
        fontsize=11
    )

    axes[0].imshow(img)
    axes[0].set_title("Original")

    axes[1].imshow(sol_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"SOL mask  #{sol_hex.upper()}\n{sol_mask.sum()} pixels")

    axes[2].imshow(lcfs_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"LCFS mask  #{LCFS_HEX}\n{lcfs_mask.sum()} pixels")

    # Overlay: SOL=yellow, LCFS=magenta, overlap=red
    overlay_rgb = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    overlay_rgb[sol_mask]   = [200, 200,   0]   # yellow  = SOL
    overlay_rgb[lcfs_mask]  = [255,   0, 200]   # magenta = LCFS
    overlay_rgb[overlap]    = [255,   0,   0]   # red     = overlap

    axes[3].imshow(img, alpha=0.35)
    axes[3].imshow(overlay_rgb, alpha=0.65)
    status = f"⚠ OVERLAP  {overlap.sum()} px" if overlap.sum() > 0 else "✓ No overlap"
    axes[3].set_title(f"Overlay\n{status}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{stem}_SOL_{sol_hex.upper()}_tol{tol}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {out_path}")

    return {
        "file":       stem,
        "sol_px":     int(sol_mask.sum()),
        "lcfs_px":    int(lcfs_mask.sum()),
        "overlap_px": int(overlap.sum()),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Single image path")
    group.add_argument("--folder", help="Folder of PNGs to batch-process")

    parser.add_argument("--outdir", required=True,
                        help="Directory to save output figures")
    parser.add_argument("--hex",  default="909000",
                        help="SOL hex color (default: 909000).  Also try ABBF1C.")
    parser.add_argument("--tol",  type=int, default=10,
                        help="Tolerance ± per channel (default: 10).  Try 0, 15, 20.")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, args.outdir, args.hex, args.tol)

    elif args.folder:
        images = sorted(Path(args.folder).glob("*.png"))
        print(f"Found {len(images)} PNGs in {args.folder}\n")
        results = []
        for img_path in images:
            results.append(process_image(str(img_path), args.outdir, args.hex, args.tol))

        # ── Summary ────────────────────────────────────────────────────────────
        zero_sol      = [r for r in results if r["sol_px"]     == 0]
        zero_lcfs     = [r for r in results if r["lcfs_px"]    == 0]
        overlap_cases = [r for r in results if r["overlap_px"] >  0]

        print("\n── Summary ───────────────────────────────────────────")
        print(f"Total images processed : {len(results)}")
        print(f"SOL  color not found   : {len(zero_sol)}   "
              f"(try --tol 20 or --hex ABBF1C)")
        print(f"LCFS color not found   : {len(zero_lcfs)}  "
              f"(try --tol 20 or --hex FA0580)")
        print(f"Overlap detected       : {len(overlap_cases)}")
        if overlap_cases:
            print("\n  Overlap cases:")
            for r in overlap_cases:
                print(f"    {r['file']}  "
                      f"overlap={r['overlap_px']} px")
        print("──────────────────────────────────────────────────────")

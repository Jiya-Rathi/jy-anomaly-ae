"""
extract_boundaries.py
─────────────────────
Converts jy (toroidal current density) simulation PNGs into clean
boundary images:

    LCFS pixels  →  solid WHITE
    SOL pixels   →  RED dotted line
    everything   →  BLACK

Core logic: nested for-loop over every pixel with three independent
R / G / B range checks per boundary type.

LCFS colour confirmed via colour picker on original (uncompressed) images:
    #FA0580  →  R=250, G=5, B=128

Usage
─────
    # Single image
    python extract_boundaries.py path/to/jy_XXXXXX_3000.png

    # Explicit output path
    python extract_boundaries.py path/to/jy_in.png  path/to/jy_out.png

    # Batch-process a whole folder
    python extract_boundaries.py path/to/broken_lcfs/

    # Inspect rare colours in a crop (helps find the SOL line colour)
    python extract_boundaries.py --probe path/to/jy_XXXXXX_3000.png

Dependencies:  pip install Pillow numpy scipy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values
# ═══════════════════════════════════════════════════════════════════════════

# Crop box (PIL: left, top, right, bottom) — from AE_research_record.txt
CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM = 72, 28, 300, 500

# ── LCFS colour filter ─────────────────────────────────────────────────────
# Confirmed via colour picker on original images: #FA0580 = R250 G5 B128
# Tolerance of ±20 per channel handles sub-pixel anti-aliasing on the line.
LCFS_R_MIN, LCFS_R_MAX = 230, 255   # red:   high
LCFS_G_MIN, LCFS_G_MAX =   0,  25   # green: very low
LCFS_B_MIN, LCFS_B_MAX = 108, 148   # blue:  mid

# ── SOL colour filter ──────────────────────────────────────────────────────
# Set these once you've confirmed the SOL line colour with --probe.
# While they are None the script falls back to auto edge-detection.
SOL_R_MIN, SOL_R_MAX = None, None
SOL_G_MIN, SOL_G_MAX = None, None
SOL_B_MIN, SOL_B_MAX = None, None

# ── Dotted-line pattern (SOL) ──────────────────────────────────────────────
DASH_LEN = 6    # pixels ON  per dash
GAP_LEN  = 4    # pixels OFF per gap

# ── Auto-SOL fallback (used only when SOL_*_MIN/MAX are all None) ──────────
AUTO_SOL_BRIGHTNESS_MIN = 60   # R+G+B above this → inside plasma region
AUTO_SOL_BORDER_MARGIN  = 8    # ignore this many px from crop edge (removes axis ticks)
AUTO_SOL_MIN_COMPONENT  = 15   # drop connected blobs smaller than this (removes speckle)

# ── Output colours ─────────────────────────────────────────────────────────
OUT_LCFS = (255, 255, 255)   # white
OUT_SOL  = (220,  30,  30)   # red
OUT_BG   = (  0,   0,   0)   # black


# ═══════════════════════════════════════════════════════════════════════════
#  PIXEL FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def is_lcfs(r: int, g: int, b: int) -> bool:
    """Three independent channel-range checks for the LCFS colour (#FA0580)."""
    return (LCFS_R_MIN <= r <= LCFS_R_MAX and
            LCFS_G_MIN <= g <= LCFS_G_MAX and
            LCFS_B_MIN <= b <= LCFS_B_MAX)


def is_sol(r: int, g: int, b: int) -> bool:
    """Three independent channel-range checks for the SOL line colour."""
    return (SOL_R_MIN <= r <= SOL_R_MAX and
            SOL_G_MIN <= g <= SOL_G_MAX and
            SOL_B_MIN <= b <= SOL_B_MAX)


# ═══════════════════════════════════════════════════════════════════════════
#  AUTO-SOL FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def auto_sol_mask(arr: np.ndarray, lcfs_mask: np.ndarray) -> np.ndarray:
    """
    Detect SOL as the 1-px outer boundary of the colourmap region
    (where the plasma colours transition to the black background).
    Used only when explicit SOL colour ranges are not set.
    """
    H, W = arr.shape[:2]
    brightness = arr.astype(np.int32).sum(axis=2)

    plasma  = brightness >= AUTO_SOL_BRIGHTNESS_MIN
    struct  = ndimage.generate_binary_structure(2, 2)
    eroded  = ndimage.binary_erosion(plasma, structure=struct, iterations=1)
    boundary = plasma & ~eroded

    # Mask out crop border (axis ticks / labels)
    m = AUTO_SOL_BORDER_MARGIN
    border = np.zeros((H, W), bool)
    border[:m, :] = border[-m:, :] = border[:, :m] = border[:, -m:] = True
    boundary &= ~border
    boundary &= ~lcfs_mask

    # Drop isolated speckle blobs
    labelled, num = ndimage.label(boundary)
    sol = np.zeros((H, W), bool)
    for lbl in range(1, num + 1):
        comp = labelled == lbl
        if comp.sum() >= AUTO_SOL_MIN_COMPONENT:
            sol |= comp

    return sol


# ═══════════════════════════════════════════════════════════════════════════
#  DASH PATTERN
# ═══════════════════════════════════════════════════════════════════════════

def apply_dash(mask: np.ndarray) -> np.ndarray:
    """
    Row-major sort of mask pixels, then apply DASH_LEN on / GAP_LEN off.
    Returns a new boolean mask with the dash pattern applied.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return mask

    order = np.lexsort((cols, rows))          # sort: top→bottom, left→right
    rows, cols = rows[order], cols[order]

    period = DASH_LEN + GAP_LEN
    dashed = np.zeros_like(mask)
    for idx in range(len(rows)):
        if (idx % period) < DASH_LEN:
            dashed[rows[idx], cols[idx]] = True

    return dashed


# ═══════════════════════════════════════════════════════════════════════════
#  PROBE MODE  — find the SOL line colour
# ═══════════════════════════════════════════════════════════════════════════

def probe(input_path: Path) -> None:
    """
    Print a frequency table of rare colours in the crop.
    Thin overlay lines appear a small number of times compared to the
    colourmap gradient — look for anything non-rainbow in the output.
    """
    from collections import Counter

    img  = Image.open(input_path).convert("RGB")
    W_r, H_r = img.size
    arr  = np.array(img.crop((
        min(CROP_LEFT, W_r), min(CROP_TOP, H_r),
        min(CROP_RIGHT, W_r), min(CROP_BOTTOM, H_r)
    )))

    print(f"\nProbe : {input_path.name}  (crop {arr.shape[1]}x{arr.shape[0]})")
    print("─" * 55)

    flat   = [tuple(arr[r, c])
              for r in range(arr.shape[0])
              for c in range(arr.shape[1])]
    counts = Counter(flat)

    rare = sorted(
        [(cnt, col) for col, cnt in counts.items() if 1 <= cnt <= 300],
        reverse=True
    )

    print(f"{'Count':>6}   {'R':>3} {'G':>3} {'B':>3}   hex")
    print("─" * 40)
    for cnt, (r, g, b) in rare[:80]:
        print(f"{cnt:6d}    {r:3d} {g:3d} {b:3d}   #{r:02X}{g:02X}{b:02X}")

    if not rare:
        print("No rare colours found — overlay contours may be absent in this image.")

    common = counts.most_common(5)
    print(f"\nTop-5 most common (gradient / background reference):")
    for (r, g, b), cnt in common:
        print(f"  {cnt:7d}  R={r:3d} G={g:3d} B={b:3d}  #{r:02X}{g:02X}{b:02X}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_image(input_path: Path, output_path: Path) -> None:
    print(f"\n{'─'*55}")
    print(f"Input : {input_path.name}")

    # Load & crop
    img = Image.open(input_path).convert("RGB")
    W_r, H_r = img.size

    left   = min(CROP_LEFT,   W_r)
    top    = min(CROP_TOP,    H_r)
    right  = min(CROP_RIGHT,  W_r)
    bottom = min(CROP_BOTTOM, H_r)

    arr  = np.array(img.crop((left, top, right, bottom)))
    H, W = arr.shape[:2]
    print(f"  Raw {W_r}x{H_r}  →  Crop {W}x{H}")

    # ── Step 1: nested for-loop — LCFS detection ───────────────────────────
    lcfs_mask = np.zeros((H, W), dtype=bool)

    for row in range(H):
        for col in range(W):
            r = int(arr[row, col, 0])
            g = int(arr[row, col, 1])
            b = int(arr[row, col, 2])
            if is_lcfs(r, g, b):
                lcfs_mask[row, col] = True

    n_lcfs = lcfs_mask.sum()
    print(f"  LCFS : {n_lcfs} pixels  "
          f"(R {LCFS_R_MIN}–{LCFS_R_MAX}, "
          f"G {LCFS_G_MIN}–{LCFS_G_MAX}, "
          f"B {LCFS_B_MIN}–{LCFS_B_MAX})")
    if n_lcfs == 0:
        print("  ⚠  No LCFS pixels found. Adjust the R/G/B ranges above if needed.")

    # ── Step 2: SOL detection ──────────────────────────────────────────────
    sol_using_colour = not any(
        v is None for v in (SOL_R_MIN, SOL_R_MAX,
                            SOL_G_MIN, SOL_G_MAX,
                            SOL_B_MIN, SOL_B_MAX)
    )

    if sol_using_colour:
        # Nested for-loop with explicit colour filter — same pattern as LCFS
        sol_mask = np.zeros((H, W), dtype=bool)
        for row in range(H):
            for col in range(W):
                if lcfs_mask[row, col]:
                    continue
                r = int(arr[row, col, 0])
                g = int(arr[row, col, 1])
                b = int(arr[row, col, 2])
                if is_sol(r, g, b):
                    sol_mask[row, col] = True
        mode = f"R {SOL_R_MIN}–{SOL_R_MAX}, G {SOL_G_MIN}–{SOL_G_MAX}, B {SOL_B_MIN}–{SOL_B_MAX}"
    else:
        # Auto fallback: morphological outer plasma boundary
        sol_mask = auto_sol_mask(arr, lcfs_mask)
        mode = "auto (outer plasma edge — set SOL ranges once colour is confirmed)"

    n_sol = sol_mask.sum()
    print(f"  SOL  : {n_sol} pixels  ({mode})")
    if n_sol == 0:
        print("  ⚠  No SOL pixels found. Run --probe to identify the line colour.")

    # ── Step 3: dash pattern ───────────────────────────────────────────────
    sol_dashed = apply_dash(sol_mask)

    # ── Step 4: compose output (black canvas, SOL first, LCFS on top) ──────
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[sol_dashed] = OUT_SOL
    out[lcfs_mask]  = OUT_LCFS

    Image.fromarray(out, mode="RGB").save(output_path)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH
# ═══════════════════════════════════════════════════════════════════════════

def batch(input_dir: Path, output_dir: Path) -> None:
    pngs = sorted(input_dir.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch: {len(pngs)} images  →  {output_dir}")
    for png in pngs:
        out = output_dir / (png.stem + "_boundaries.png")
        try:
            process_image(png, out)
        except Exception as exc:
            print(f"  ERROR {png.name}: {exc}")
    print(f"\nDone. {len(pngs)} images processed.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LCFS (white) and SOL (red dotted) boundaries from jy PNGs."
    )
    parser.add_argument("input", type=Path,
                        help="PNG file or folder of PNGs.")
    parser.add_argument("output", type=Path, nargs="?",
                        help="Output PNG or folder (optional, auto-named if omitted).")
    parser.add_argument("--probe", action="store_true",
                        help="Print colour frequency table to identify the SOL colour.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist.")
        sys.exit(1)

    if args.probe:
        target = sorted(args.input.glob("*.png"))[0] if args.input.is_dir() else args.input
        probe(target)
        return

    if args.input.is_dir():
        out_dir = args.output or args.input.with_name(args.input.name + "_boundaries")
        batch(args.input, out_dir)
    else:
        out_file = args.output or args.input.with_name(args.input.stem + "_boundaries.png")
        process_image(args.input, out_file)


if __name__ == "__main__":
    main()

"""
ae_lib/lcfs_masking.py

Standalone preprocessing module: converts a raw cropped RGB jy-field image
into a multi-level grayscale image that encodes the three diagnostic
structures for anomaly detection:

    Background   →   0   (black)
    SOL boundary →  85   (~1/3 white)
    LCFS line    → 170   (~2/3 white)
    White core   → 255   (white)

These fixed discrete values mean the AE always sees:
    white core at 1.0, LCFS at ~0.667, SOL at ~0.333, background at 0
after the fixed /255 normalization applied in data.py.
Per-image min-max normalization is deliberately NOT used so that the
absence of white core (bad_black_core) is a true zero, not a rescaled signal.

Public API:
    load_sol_coords(path)              → np.ndarray [M, 2]
    apply_lcfs_masking(cropped_rgb,
                       sol_coords)     → np.ndarray [H, W] uint8

Internal helpers (not imported by data.py):
    _build_lcfs_polygon
    _get_inside_lcfs_mask
    _rasterize_sol

Physical coordinate constants MUST match the PLOT_CROP, R_MIN/MAX, Z_MIN/MAX
used when the jy IDL screenshots were generated. These are the same values
used in lcfs_sol_coord_overlap.py.
"""

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.draw import line as draw_line


# ── Physical coordinate constants ──────────────────────────────────────────────
# These must match the PLOT_CROP and axis limits calibrated for the jy images.
# DO NOT change without re-calibrating with lcfs_sol_coord_overlap.py.

_PLOT_CROP    = (72, 28, 300, 500)      # (left, upper, right, lower) in full image
_R_MIN, _R_MAX = 0.5, 3.2              # physical R range [a₀]
_Z_MIN, _Z_MAX = -2.5, 2.5            # physical Z range [a₀]

_LCFS_COLOR   = np.array([250, 5, 128])  # #FA0580
_LCFS_TOL     = 15                        # ± per channel tolerance

_WHITE_THRESHOLD = 240                    # pixels with ALL channels >= this
                                          # (inside LCFS) are treated as white core

# Output gray levels — discrete, fixed across all images
GRAY_BG   =   0    # background
GRAY_SOL  =  85    # SOL boundary   (~1/3)
GRAY_LCFS = 170    # LCFS line      (~2/3)
GRAY_CORE = 255    # white core     (1.0)


# ── Crop dimensions ────────────────────────────────────────────────────────────

def _crop_hw() -> tuple:
    """Height and width of the cropped image region."""
    x0, y0, x1, y1 = _PLOT_CROP
    return y1 - y0, x1 - x0          # (H, W)


# ── Coordinate transforms  (CROPPED image ↔ physical space) ───────────────────

def _crop_pix_to_phys(rows, cols):
    """
    (row, col) in CROPPED image pixel space  →  (R, Z) in physical space.
    Top-left of crop  =  (R_MIN, Z_MAX).
    Bottom-right      =  (R_MAX, Z_MIN).
    """
    H, W = _crop_hw()
    R = _R_MIN + np.asarray(cols, float) / W * (_R_MAX - _R_MIN)
    Z = _Z_MAX - np.asarray(rows, float) / H * (_Z_MAX - _Z_MIN)
    return R, Z


def _phys_to_crop_pix(R, Z):
    """
    (R, Z) in physical space  →  (row, col) in CROPPED image pixel space.
    Returns integer arrays.
    """
    H, W = _crop_hw()
    cols = (np.asarray(R, float) - _R_MIN) / (_R_MAX - _R_MIN) * W
    rows = (_Z_MAX - np.asarray(Z, float)) / (_Z_MAX - _Z_MIN) * H
    return rows.astype(int), cols.astype(int)


# ── LCFS polygon extraction ────────────────────────────────────────────────────

def _build_lcfs_polygon(cropped_rgb: np.ndarray):
    """
    Detect LCFS pixels (color ≈ #FA0580) in the cropped RGB image and build
    a convex-hull polygon in physical (R, Z) space.

    Using a convex hull of ALL detected LCFS pixels is robust to the heavy
    fragmentation caused by IDL anti-aliasing (86-255 disconnected components
    observed in practice). The convex hull is always a valid closed polygon
    that reliably encloses the plasma interior regardless of fragmentation.

    Parameters
    ----------
    cropped_rgb : np.ndarray  [H, W, 3]  uint8

    Returns
    -------
    polygon   : np.ndarray [K, 2] columns = (R, Z), or None if not found
    lcfs_mask : np.ndarray [H, W] bool  — ALL detected LCFS pixels (for drawing)
    note      : str  — empty if normal, warning message if something is unusual
    """
    lcfs_mask = np.all(
        np.abs(cropped_rgb.astype(int) - _LCFS_COLOR) <= _LCFS_TOL,
        axis=2,
    )

    n_px = int(lcfs_mask.sum())
    if n_px < 3:
        return None, lcfs_mask, f"WARNING: only {n_px} LCFS pixels found"

    rows, cols = np.where(lcfs_mask)
    R, Z       = _crop_pix_to_phys(rows, cols)
    points     = np.column_stack([R, Z])

    # Count fragments for diagnostic reporting
    _, n_components = ndimage.label(lcfs_mask)

    try:
        hull    = ConvexHull(points)
        polygon = points[hull.vertices]          # CCW-ordered hull vertices
        note    = (
            f"{n_components} LCFS fragments, {n_px} px → "
            f"convex hull ({len(hull.vertices)} pts)"
        )
    except Exception as exc:
        return None, lcfs_mask, f"ConvexHull failed: {exc}"

    return polygon, lcfs_mask, note


# ── Inside-LCFS mask ───────────────────────────────────────────────────────────

def _get_inside_lcfs_mask(polygon: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Return bool [H, W] — True where a pixel's physical (R, Z) falls inside
    the LCFS polygon.

    Uses matplotlib Path with the even-odd winding rule, which correctly
    handles non-convex (and convex hull) polygons.
    """
    all_rows, all_cols = np.mgrid[0:H, 0:W]
    R_all, Z_all       = _crop_pix_to_phys(all_rows.ravel(), all_cols.ravel())
    pts                = np.column_stack([R_all, Z_all])
    inside             = MplPath(polygon).contains_points(pts).reshape(H, W)
    return inside


# ── SOL boundary rasterization ─────────────────────────────────────────────────

def _rasterize_sol(sol_coords: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Project the SOL polygon boundary (physical R, Z coords) into cropped
    pixel space and rasterize as a 1-pixel-wide line using Bresenham's
    algorithm (via skimage.draw.line).

    Parameters
    ----------
    sol_coords : np.ndarray [M, 2]  columns = (R, Z)
    H, W       : cropped image dimensions

    Returns
    -------
    mask : np.ndarray [H, W] bool
    """
    sol_rows, sol_cols = _phys_to_crop_pix(sol_coords[:, 0], sol_coords[:, 1])
    mask = np.zeros((H, W), dtype=bool)

    for i in range(len(sol_rows) - 1):
        r0, c0 = int(sol_rows[i]),     int(sol_cols[i])
        r1, c1 = int(sol_rows[i + 1]), int(sol_cols[i + 1])
        rr, cc = draw_line(r0, c0, r1, c1)
        valid  = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        mask[rr[valid], cc[valid]] = True

    return mask


# ── Public API ─────────────────────────────────────────────────────────────────

def load_sol_coords(path: str) -> np.ndarray:
    """
    Load the SOL boundary from a two-column text file.

    Parameters
    ----------
    path : str  Path to outer_vacuum_boundary_spline_RZ.txt

    Returns
    -------
    sol_coords : np.ndarray [M, 2]  columns = (R, Z) in physical units [a₀]
    """
    sol_coords = np.loadtxt(path)
    if sol_coords.ndim != 2 or sol_coords.shape[1] != 2:
        raise ValueError(
            f"SOL file must have exactly 2 columns (R, Z). "
            f"Got shape {sol_coords.shape}: {path}"
        )
    return sol_coords


def apply_lcfs_masking(cropped_rgb: np.ndarray,
                       sol_coords: np.ndarray) -> np.ndarray:
    """
    Convert a cropped RGB jy-field image into a multi-level grayscale image
    that encodes three diagnostic structures on a black background.

    Gray levels:
        GRAY_BG   =   0  →  background (everything outside structures)
        GRAY_SOL  =  85  →  SOL outer vacuum boundary (yellow in patch test)
        GRAY_LCFS = 170  →  LCFS magenta line (cyan in patch test)
        GRAY_CORE = 255  →  white core pixels inside LCFS (white in patch test)

    Layers are painted in order SOL → LCFS → core so that the LCFS and
    white core always appear on top when they overlap with the SOL line.

    If no LCFS pixels are detected (degenerate image), returns an image
    with only the SOL boundary rendered at GRAY_SOL.

    Parameters
    ----------
    cropped_rgb : np.ndarray [H, W, 3] uint8
        Raw image already cropped to the PLOT_CROP region.
        Must NOT be resized yet (coordinate math uses full crop dimensions).
    sol_coords  : np.ndarray [M, 2]
        SOL boundary in physical (R, Z) coords, from load_sol_coords().

    Returns
    -------
    processed : np.ndarray [H, W] uint8
        Multi-level grayscale image with values in {0, 85, 170, 255}.
    """
    H, W = cropped_rgb.shape[:2]

    # Start with all-black canvas
    processed = np.full((H, W), GRAY_BG, dtype=np.uint8)

    # ── Layer 1: SOL boundary ──────────────────────────────────────────────────
    sol_mask = _rasterize_sol(sol_coords, H, W)
    processed[sol_mask] = GRAY_SOL

    # ── Build LCFS polygon (needed for layers 2 and 3) ────────────────────────
    polygon, lcfs_mask, note = _build_lcfs_polygon(cropped_rgb)

    if polygon is None:
        # No LCFS found — return with only SOL boundary rendered.
        # data.py will log the warning via the note field if needed.
        return processed

    # ── Layer 2: LCFS line ────────────────────────────────────────────────────
    # Draw ALL detected LCFS pixels (full lcfs_mask, not just hull vertices).
    # This preserves the actual rendered line shape — important for broken_lcfs
    # where the distorted LCFS morphology is the primary anomaly signal.
    processed[lcfs_mask] = GRAY_LCFS

    # ── Layer 3: white core pixels inside LCFS ────────────────────────────────
    inside_mask     = _get_inside_lcfs_mask(polygon, H, W)
    white_mask      = np.all(cropped_rgb >= _WHITE_THRESHOLD, axis=2)
    white_core_mask = white_mask & inside_mask
    processed[white_core_mask] = GRAY_CORE

    return processed

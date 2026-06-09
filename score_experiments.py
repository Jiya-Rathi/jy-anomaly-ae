"""
score_experiments.py

Compares anomaly-scoring functions on a TRAINED checkpoint WITHOUT retraining.

For each image in the selection set, computes six anomaly scores:

    1. mse                — plain per-pixel MSE over the whole image (baseline,
                            reproduces the training-time evaluation)
    2. mse_interior       — MSE computed ONLY inside the LCFS polygon
                            (tests the "core signal is diluted" hypothesis)
    3. laplacian          — MSE on Laplacian-filtered images (edge emphasis)
    4. laplacian_interior — Laplacian MSE inside the LCFS polygon only
    5. ssim               — 1 - mean(SSIM map) over the whole image
    6. ssim_interior      — 1 - mean(SSIM map) inside the LCFS polygon only

Then computes one-vs-healthy AUC per anomaly class for each scoring function,
and prints a comparison table + saves a bar chart and CSV.

The key question this answers:
    Can ANY scoring function recover bad_black_core above the 0.90 floor
    on the existing masked-image checkpoint, making a single model viable?

Interior masks are recomputed from the raw images using the same LCFS polygon
logic as preprocessing (ae_lib.lcfs_masking), then resized to image_size with
nearest-neighbour interpolation so they align with the resized model input.

Usage:
    HIP_VISIBLE_DEVICES=0 python score_experiments.py \\
        --checkpoint lcfs_single_outputs/checkpoints/manual_best.pt \\
        --config     configs/trial_142_lcfs.yaml \\
        --gpu        0 \\
        --outdir     ./score_experiments_outputs/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score


# ── Constants ──────────────────────────────────────────────────────────────────

ANOMALY_CLASSES = ["broken_lcfs", "bad_black_core", "bad_nonconverged"]
CLASS_COLORS = {
    "broken_lcfs":      "#ff7f0e",
    "bad_black_core":   "#d62728",
    "bad_nonconverged": "#9467bd",
}
SCORE_ORDER = [
    "mse", "mse_interior",
    "laplacian", "laplacian_interior",
    "ssim", "ssim_interior",
]


# ── Interior-mask computation (from raw images) ───────────────────────────────

def compute_interior_masks(paths, images_root, sol_coords, image_size):
    """
    For each image, recompute the LCFS interior mask and resize to image_size.

    Returns a torch-ready bool array [N, image_size, image_size].
    Falls back to an all-True mask for images where no LCFS is detected
    (so they still get a score; flagged in the returned warning count).
    """
    from ae_lib.lcfs_masking import (
        _PLOT_CROP, _build_lcfs_polygon, _get_inside_lcfs_mask,
    )

    images_root = Path(images_root)
    masks = np.zeros((len(paths), image_size, image_size), dtype=bool)
    n_fallback = 0

    x0, y0, x1, y1 = _PLOT_CROP

    for i, rel_path in enumerate(paths):
        img_rgb = np.array(Image.open(images_root / rel_path).convert("RGB"))
        cropped = img_rgb[y0:y1, x0:x1]
        H, W    = cropped.shape[:2]

        polygon, _, _ = _build_lcfs_polygon(cropped)
        if polygon is None:
            masks[i] = True       # fallback: score whole image
            n_fallback += 1
            continue

        inside = _get_inside_lcfs_mask(polygon, H, W)        # [H, W] bool
        # Resize to image_size with nearest neighbour (preserves binary mask)
        mask_img = Image.fromarray((inside.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)
        masks[i] = np.array(mask_img) > 127

    if n_fallback:
        print(f"  [warn] {n_fallback} image(s) had no LCFS — scored whole image")

    return masks


# ── Filter banks ───────────────────────────────────────────────────────────────

def _laplacian_kernel(device):
    """3x3 discrete Laplacian (edge / second-derivative) kernel."""
    k = torch.tensor([[0.,  1., 0.],
                      [1., -4., 1.],
                      [0.,  1., 0.]], device=device)
    return k.view(1, 1, 3, 3)


def _gaussian_window(size, sigma, device):
    """2D Gaussian window for SSIM, normalised to sum 1. Shape [1,1,size,size]."""
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d    = g1d / g1d.sum()
    win    = g1d[:, None] * g1d[None, :]
    return win.view(1, 1, size, size)


# ── Score functions (all operate on a batch + optional interior mask) ─────────

def _reconstruct(model, batch):
    """Run the model forward pass, returning the reconstruction tensor."""
    out = model(batch)
    # Some AE implementations return a tuple (recon, latent); handle both
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _masked_mean(err, mask):
    """
    Mean of err over interior pixels.
    err  : [B, H, W]
    mask : [B, H, W] bool  (None -> mean over all pixels)
    """
    if mask is None:
        return err.mean(dim=(1, 2))
    m = mask.to(err.dtype)
    n = m.sum(dim=(1, 2)).clamp(min=1.0)
    return (err * m).sum(dim=(1, 2)) / n


def score_mse(model, batch, mask=None):
    recon = _reconstruct(model, batch)
    err   = (batch - recon) ** 2          # [B, 1, H, W]
    return _masked_mean(err.squeeze(1), mask)


def score_laplacian(model, batch, lap_kernel, mask=None):
    recon   = _reconstruct(model, batch)
    inp_lap = F.conv2d(F.pad(batch, (1, 1, 1, 1), mode="reflect"), lap_kernel)
    rec_lap = F.conv2d(F.pad(recon, (1, 1, 1, 1), mode="reflect"), lap_kernel)
    err     = (inp_lap - rec_lap) ** 2     # [B, 1, H, W]
    return _masked_mean(err.squeeze(1), mask)


def score_ssim(model, batch, window, mask=None, C1=0.01**2, C2=0.03**2):
    """Anomaly score = 1 - mean(SSIM map). Higher = more anomalous."""
    recon = _reconstruct(model, batch)
    pad   = window.shape[-1] // 2

    mu_x = F.conv2d(batch, window, padding=pad)
    mu_y = F.conv2d(recon, window, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(batch * batch, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(recon * recon, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(batch * recon, window, padding=pad) - mu_xy

    ssim_map = (((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
                ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
    dissim = (1.0 - ssim_map).clamp(min=0.0)    # [B, 1, H, W]
    return _masked_mean(dissim.squeeze(1), mask)


# ── Score the full selection set with all six functions ───────────────────────

def compute_all_scores(model, images, interior_masks, batch_size, device):
    """
    Returns a dict: score_name -> np.ndarray [N].
    """
    lap_kernel = _laplacian_kernel(device)
    ssim_win   = _gaussian_window(11, 1.5, device)

    n = images.shape[0]
    out = {name: np.zeros(n, dtype=np.float64) for name in SCORE_ORDER}

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = images[start:end]
            mask  = interior_masks[start:end]    # [B, H, W] bool tensor

            out["mse"][start:end]            = score_mse(model, batch).cpu().numpy()
            out["mse_interior"][start:end]   = score_mse(model, batch, mask).cpu().numpy()
            out["laplacian"][start:end]      = score_laplacian(model, batch, lap_kernel).cpu().numpy()
            out["laplacian_interior"][start:end] = score_laplacian(model, batch, lap_kernel, mask).cpu().numpy()
            out["ssim"][start:end]           = score_ssim(model, batch, ssim_win).cpu().numpy()
            out["ssim_interior"][start:end]  = score_ssim(model, batch, ssim_win, mask).cpu().numpy()

    return out


# ── Per-class AUC ──────────────────────────────────────────────────────────────

def per_class_auc(scores, classes):
    """One-vs-healthy AUC for each anomaly class."""
    classes = np.asarray(classes)
    healthy = classes == "healthy"
    result  = {}
    for cls in ANOMALY_CLASSES:
        anom = classes == cls
        if anom.sum() == 0 or healthy.sum() == 0:
            result[cls] = float("nan")
            continue
        y_true  = np.concatenate([np.zeros(healthy.sum()), np.ones(anom.sum())])
        y_score = np.concatenate([scores[healthy], scores[anom]])
        result[cls] = roc_auc_score(y_true, y_score)
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--outdir",     default="./score_experiments_outputs/")
    args = parser.parse_args()

    from ae_lib.config     import Config
    from ae_lib.data       import load_split
    from ae_lib.model      import Autoencoder
    from ae_lib.lcfs_masking import load_sol_coords

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load config + model ────────────────────────────────────────────────────
    cfg    = Config.from_yaml(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model  = Autoencoder(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    # ── Load selection set ─────────────────────────────────────────────────────
    sel_path = Path(cfg.manifests_dir) / "selection.txt"
    print(f"Loading selection set: {sel_path}")
    sel = load_split(sel_path, cfg, device)
    print(f"n_selection = {len(sel)}")

    # ── Compute interior masks from raw images ────────────────────────────────
    print("Computing LCFS interior masks from raw images...")
    sol_coords = load_sol_coords(cfg.sol_file_path)
    masks_np   = compute_interior_masks(
        sel.paths, cfg.images_root, sol_coords, cfg.image_size
    )
    interior_masks = torch.from_numpy(masks_np).to(device)
    print(f"Interior masks: {masks_np.shape}, "
          f"mean interior fraction = {masks_np.mean():.3f}")

    # ── Compute all six scores ──────────────────────────────────────────────────
    print("Scoring with all six functions...")
    all_scores = compute_all_scores(
        model, sel.images, interior_masks, cfg.batch_size, device
    )

    # ── Per-class AUC for each scoring function ────────────────────────────────
    print("\n" + "=" * 74)
    print(f"{'scoring function':22s} {'broken_lcfs':>13s} "
          f"{'bad_black_core':>15s} {'bad_nonconv':>13s}")
    print("-" * 74)

    auc_table = {}
    for name in SCORE_ORDER:
        aucs = per_class_auc(all_scores[name], sel.classes)
        auc_table[name] = aucs
        # Flag bad_black_core values that clear the 0.90 floor
        bc_flag = " *" if aucs["bad_black_core"] >= 0.90 else ""
        print(f"{name:22s} {aucs['broken_lcfs']:>13.4f} "
              f"{aucs['bad_black_core']:>15.4f}{bc_flag:2s} "
              f"{aucs['bad_nonconverged']:>11.4f}")
    print("=" * 74)
    print("  * = bad_black_core clears the 0.90 hard floor")

    # ── Save CSVs ───────────────────────────────────────────────────────────────
    # Per-image scores
    df_scores = pd.DataFrame({
        "path":  sel.paths,
        "class": sel.classes,
        "shot":  sel.shots,
    })
    for name in SCORE_ORDER:
        df_scores[name] = all_scores[name]
    df_scores.to_csv(outdir / "all_scores_per_image.csv", index=False)

    # AUC summary
    df_auc = pd.DataFrame(auc_table).T
    df_auc.index.name = "score_function"
    df_auc.to_csv(outdir / "auc_comparison.csv")
    print(f"\nSaved: {outdir / 'all_scores_per_image.csv'}")
    print(f"Saved: {outdir / 'auc_comparison.csv'}")

    # ── Bar chart ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    x      = np.arange(len(SCORE_ORDER))
    width  = 0.26

    for i, cls in enumerate(ANOMALY_CLASSES):
        vals = [auc_table[name][cls] for name in SCORE_ORDER]
        ax.bar(x + (i - 1) * width, vals, width,
               label=cls, color=CLASS_COLORS[cls], alpha=0.85)

    ax.axhline(0.90, color="black", linestyle="--", linewidth=1,
               label="0.90 floor")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_ORDER, rotation=20, ha="right")
    ax.set_ylabel("One-vs-healthy AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Anomaly-scoring function comparison (same checkpoint, no retraining)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir / 'auc_comparison.png'}")

    # ── Verdict ──────────────────────────────────────────────────────────────────
    print("\n── Verdict ─────────────────────────────────────────")
    best_bc = max(SCORE_ORDER, key=lambda n: auc_table[n]["bad_black_core"])
    best_bc_auc = auc_table[best_bc]["bad_black_core"]
    print(f"Best bad_black_core AUC: {best_bc_auc:.4f}  (via '{best_bc}')")
    if best_bc_auc >= 0.90:
        print("→ A scoring change recovers bad_black_core. Single masked model "
              "may be viable; check this function's broken_lcfs AUC too.")
    else:
        print("→ No scoring function recovers bad_black_core above 0.90. "
              "This supports the two-model ensemble approach for the paper.")


if __name__ == "__main__":
    main()

"""
score_variants.py

Evaluate three anomaly-score formulations against trial 142's checkpoint
WITHOUT retraining. Same model, same data, three different ways of
turning (input, reconstruction) into a scalar anomaly score:

    1. mse        : per-image mean squared error  (the current/default)
    2. max_err    : per-image MAX squared pixel error
                    motivation: broken_lcfs is a *localized* edge defect, so
                    averaging dilutes the signal; the max picks up the worst pixel
    3. ssim       : 1 - SSIM(input, reconstruction)
                    motivation: structural similarity penalizes pattern mismatch,
                    not just intensity diffs; should be more sensitive to
                    edge-shape distortions than per-pixel MSE

Each score gets per-class AUC on the selection set, reported alongside the
existing pipeline's MSE result for direct comparison.

Usage:
    python score_variants.py \
        --checkpoint study_outputs/checkpoints/trial_142_best.pt \
        --config     study_outputs/configs/trial_142.yaml \
        --gpu        0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Same imports the rest of the project uses
from ae_lib.config import Config
from ae_lib.data import load_split
from ae_lib.model import Autoencoder


# -----------------------------------------------------------------------------
# Score functions
# Each returns a 1-D numpy array of length N (one score per image)
# -----------------------------------------------------------------------------

@torch.no_grad()
def score_mse(model, images: torch.Tensor, batch_size: int) -> np.ndarray:
    """Per-image mean squared error. Matches reconstruction_error() in model.py."""
    out = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        x_hat = model(batch)
        per_image = ((batch - x_hat) ** 2).mean(dim=(1, 2, 3))
        out.append(per_image.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_max_err(model, images: torch.Tensor, batch_size: int) -> np.ndarray:
    """Per-image max squared pixel error.

    For broken_lcfs (a localized edge defect) the *worst* pixel may carry more
    signal than the average. If lcfs AUC under max_err is meaningfully higher
    than under MSE, the score function is the bottleneck, not the model.
    """
    out = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        x_hat = model(batch)
        sq_err = (batch - x_hat) ** 2          # [B, 1, H, W]
        per_image = sq_err.amax(dim=(1, 2, 3)) # max over all non-batch dims
        out.append(per_image.cpu().numpy())
    return np.concatenate(out)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian, then outer-product to 2-D, normalized to sum=1."""
    half = (window_size - 1) / 2.0
    x = torch.arange(window_size, device=device, dtype=dtype) - half
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win_2d = g.unsqueeze(0) * g.unsqueeze(1)   # [W, W]
    return win_2d.unsqueeze(0).unsqueeze(0)    # [1, 1, W, W]


@torch.no_grad()
def score_ssim(
    model,
    images:       torch.Tensor,
    batch_size:   int,
    window_size:  int  = 11,
    sigma:        float = 1.5,
) -> np.ndarray:
    """Per-image SSIM-based anomaly score = 1 - mean(SSIM_map).

    Standard single-channel SSIM (Wang 2004). Inputs are in [0, 1] after
    HPF -> data range = 1.0. C1 = (0.01*1)^2, C2 = (0.03*1)^2 are the
    canonical stabilizers from the original paper.
    """
    device = images.device
    dtype  = images.dtype
    win = _gaussian_window(window_size, sigma, device, dtype)
    pad = window_size // 2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    out = []
    for i in range(0, images.shape[0], batch_size):
        x     = images[i : i + batch_size]
        x_hat = model(x)

        mu_x   = F.conv2d(x,     win, padding=pad)
        mu_y   = F.conv2d(x_hat, win, padding=pad)
        mu_xx  = mu_x * mu_x
        mu_yy  = mu_y * mu_y
        mu_xy  = mu_x * mu_y

        sigma_xx = F.conv2d(x * x,     win, padding=pad) - mu_xx
        sigma_yy = F.conv2d(x_hat * x_hat, win, padding=pad) - mu_yy
        sigma_xy = F.conv2d(x * x_hat, win, padding=pad) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
        ssim_map = num / den                       # [B, 1, H, W]
        per_image_ssim = ssim_map.mean(dim=(1, 2, 3))
        per_image_score = 1.0 - per_image_ssim     # higher = more anomalous
        out.append(per_image_score.cpu().numpy())
    return np.concatenate(out)


# -----------------------------------------------------------------------------
# Per-class AUC (matches evaluation.py)
# -----------------------------------------------------------------------------

ANOMALY_CLASSES = ["broken_lcfs", "bad_black_core", "bad_nonconverged"]


def per_class_auc(scores: np.ndarray, classes: List[str]) -> Dict[str, float]:
    """One-vs-healthy AUC per anomaly class.

    Note: AUC is invariant under monotone rescaling, so we don't need to
    normalize the scores -- raw values give the same AUC as normalized.
    """
    classes_arr  = np.asarray(classes)
    healthy_mask = classes_arr == "healthy"
    out: Dict[str, float] = {}
    for cls in ANOMALY_CLASSES:
        anomaly_mask = classes_arr == cls
        if anomaly_mask.sum() == 0 or healthy_mask.sum() == 0:
            out[cls] = float("nan")
            continue
        y_true  = np.concatenate([
            np.zeros(healthy_mask.sum(), dtype=int),
            np.ones (anomaly_mask.sum(), dtype=int),
        ])
        y_score = np.concatenate([
            scores[healthy_mask],
            scores[anomaly_mask],
        ])
        out[cls] = float(roc_auc_score(y_true, y_score))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trial_<N>_best.pt")
    p.add_argument("--config",     type=Path, required=True,
                   help="Path to trial_<N>.yaml snapshot")
    p.add_argument("--gpu",        type=int,  default=0)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        sys.exit(f"config not found: {args.config}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load config + model + checkpoint
    cfg = Config.from_yaml(args.config)
    model = Autoencoder(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded model. {model.parameter_count():,} parameters.")

    # --- Load selection set (same path the trainer uses)
    print("Loading selection set...")
    sel = load_split(Path(cfg.manifests_dir) / "selection.txt", cfg, device)
    print(f"  {len(sel)} images, classes: "
          f"{ {c: sum(1 for x in sel.classes if x == c) for c in set(sel.classes)} }")

    # --- Score it three ways
    print("\nScoring (mse)...")
    s_mse  = score_mse    (model, sel.images, cfg.batch_size)
    print("Scoring (max_err)...")
    s_max  = score_max_err(model, sel.images, cfg.batch_size)
    print("Scoring (ssim)...")
    s_ssim = score_ssim   (model, sel.images, cfg.batch_size)

    # --- AUCs
    auc_mse  = per_class_auc(s_mse,  sel.classes)
    auc_max  = per_class_auc(s_max,  sel.classes)
    auc_ssim = per_class_auc(s_ssim, sel.classes)

    # --- Print side-by-side
    print()
    print("=" * 72)
    print(f"{'class':<22}{'mse':>12}{'max_err':>14}{'1-ssim':>14}")
    print("-" * 72)
    for cls in ANOMALY_CLASSES:
        print(f"{cls:<22}"
              f"{auc_mse [cls]:>12.4f}"
              f"{auc_max [cls]:>14.4f}"
              f"{auc_ssim[cls]:>14.4f}")
    print("=" * 72)
    print()
    print("Reading guide: if max_err or 1-ssim's broken_lcfs AUC is")
    print("meaningfully above mse's (say +0.10), the AE is fine and the")
    print("MSE score function is what's limiting lcfs detection.")
    print("If all three are within ~0.02 of each other, the bottleneck is")
    print("the model itself, not the score function.")


if __name__ == "__main__":
    main()

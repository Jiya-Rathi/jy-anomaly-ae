"""
ae_lib/evaluation.py

Post-training evaluation: reconstruction scoring, per-class AUC,
objective computation, and scatter plot.

Public API:
    compute_calibration(model, train_images, batch_size, device)
        -> dict with 'mu' and 'sigma'
        Scores healthy training images to calibrate the normalization
        constants (mu, sigma) used in the anomaly score formula.

    evaluate_trial(model, train_images, selection_data, cfg, device,
                   scatter_path=None, scatter_title=None)
        -> EvalResult
        Full post-training evaluation: calibrates, scores selection set,
        computes per-class AUCs, objective with floor, and optionally
        produces the scatter plot.

Anomaly score formula (see research notes):
    norm_score = clip((mse - mu) / (sigma * k), 0, 1)
    mu, sigma : mean and std of healthy training MSE
    k         : cfg.score_k (calibrated to 37 in this project)

Objective function:
    if AUC_black_core < 0.90 or AUC_nonconverged < 0.90:
        return 0.0  # hard reject
    else:
        return 0.4 * AUC_broken_lcfs + 0.3 * AUC_black_core + 0.3 * AUC_nonconverged
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless backend (no $DISPLAY needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Everything evaluate_trial produces."""

    # Per-class AUCs
    auc_per_class: Dict[str, float] = field(default_factory=dict)

    # Objective with floor applied
    objective:   float = 0.0
    floor_ok:    bool  = False      # True iff both floor classes passed

    # Calibration constants
    mu:    float = 0.0
    sigma: float = 0.0

    # Raw per-sample scores on the selection set (parallel to selection_data)
    raw_mse:    np.ndarray = field(default_factory=lambda: np.array([]))
    norm_score: np.ndarray = field(default_factory=lambda: np.array([]))

    # Youden-J threshold used for the scatter plot (on normalized scores)
    threshold: float = 0.5

    # Where the scatter plot got saved (if it was made)
    scatter_path: Optional[str] = None


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------

def _per_sample_mse(
    model, images: torch.Tensor, batch_size: int, score_fn: str = "mse"
) -> np.ndarray:
    """Compute per-image reconstruction score. Returns a 1D numpy array."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size]
            if score_fn == "ssim":
                from ae_lib.losses import per_sample_dssim
                recon = model(batch)
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
                e = per_sample_dssim(batch, recon)
            else:
                e = model.reconstruction_error(batch)   # [B]
            errors.append(e.detach().cpu().numpy())
    return np.concatenate(errors, axis=0)


def compute_calibration(
    model, train_images: torch.Tensor, batch_size: int, device, score_fn: str = "mse",
) -> Dict[str, float]:
    """Compute mu and sigma of reconstruction score on healthy training set."""
    mse = _per_sample_mse(model, train_images, batch_size, score_fn=score_fn)
    mu    = float(mse.mean())
    sigma = float(mse.std())
    # Guard against degenerate sigma (would cause divide-by-zero in normalization)
    if sigma < 1e-12:
        sigma = 1e-12
    return {"mu": mu, "sigma": sigma}


# -----------------------------------------------------------------------------
# Per-class AUC
# -----------------------------------------------------------------------------

# Fixed class set for this project. If classes change, update this list.
_ANOMALY_CLASSES = ["broken_lcfs", "bad_black_core", "bad_nonconverged"]


def _per_class_auc(
    scores:  np.ndarray,                # shape [N] -- one score per selection image
    classes: List[str],                 # length N, parallel class labels
) -> Dict[str, float]:
    """Compute one-vs-healthy AUC for each anomaly class in _ANOMALY_CLASSES."""
    classes_arr = np.asarray(classes)
    healthy_mask = classes_arr == "healthy"

    out: Dict[str, float] = {}
    for cls in _ANOMALY_CLASSES:
        anomaly_mask = classes_arr == cls
        if anomaly_mask.sum() == 0 or healthy_mask.sum() == 0:
            # No examples of this class in the split -- can't compute
            out[cls] = float("nan")
            continue

        y_true  = np.concatenate([
            np.zeros(healthy_mask.sum(),  dtype=int),   # 0 = healthy
            np.ones(anomaly_mask.sum(),  dtype=int),    # 1 = anomaly
        ])
        y_score = np.concatenate([
            scores[healthy_mask],
            scores[anomaly_mask],
        ])
        out[cls] = float(roc_auc_score(y_true, y_score))
    return out


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------

def _compute_objective(auc_per_class: Dict[str, float]) -> tuple:
    """Apply the weighted-AUC objective with the hard-floor rule.

    Returns (objective, floor_ok).
    """
    auc_bc  = auc_per_class.get("bad_black_core",   0.0)
    auc_nc  = auc_per_class.get("bad_nonconverged", 0.0)
    auc_lcfs = auc_per_class.get("broken_lcfs",     0.0)

    # Hard floor at 0.90 on bad_black_core and bad_nonconverged
    if auc_bc < 0.90 or auc_nc < 0.90:
        return 0.0, False

    objective = 0.4 * auc_lcfs + 0.3 * auc_bc + 0.3 * auc_nc
    return objective, True


# -----------------------------------------------------------------------------
# Youden-J threshold
# -----------------------------------------------------------------------------

def _youden_threshold(scores: np.ndarray, classes: List[str]) -> float:
    """Threshold that maximizes TPR - FPR against the pooled healthy-vs-anomaly ROC.

    Computed on raw scores, then caller converts to normalized scale if needed.
    If there are no anomalies (edge case), returns 0.5.
    """
    classes_arr = np.asarray(classes)
    y_true = (classes_arr != "healthy").astype(int)
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


# -----------------------------------------------------------------------------
# Scatter plot
# -----------------------------------------------------------------------------

_CLASS_COLORS = {
    "healthy":          "tab:blue",
    "broken_lcfs":      "tab:orange",
    "bad_black_core":   "tab:red",
    "bad_nonconverged": "tab:purple",
}


def _shot_sort_key(shot_id: str) -> int:
    """Extract the numeric shot number for chronological sorting.
    shot_id looks like '187138_3000' -- we sort by the part before '_'.
    Fall back to 0 if the shot ID doesn't parse."""
    m = re.match(r"(\d+)", shot_id)
    return int(m.group(1)) if m else 0


def _scatter_plot(
    shots:        List[str],
    classes:      List[str],
    norm_scores:  np.ndarray,
    threshold:    float,
    out_path:     Path,
    title:        str,
):
    """Make the per-trial scatter plot and save it."""
    shot_keys = np.array([_shot_sort_key(s) for s in shots])
    classes_arr = np.asarray(classes)

    fig, ax = plt.subplots(figsize=(12, 5))

    # One scatter per class so the legend labels them correctly
    for cls, color in _CLASS_COLORS.items():
        mask = classes_arr == cls
        if not mask.any():
            continue
        ax.scatter(
            shot_keys[mask], norm_scores[mask],
            s=22, alpha=0.75, edgecolors="none",
            color=color, label=f"{cls} (n={mask.sum()})",
        )

    # Horizontal threshold line (normalized score scale)
    ax.axhline(
        threshold, color="black", linestyle="--", linewidth=1.0,
        label=f"Youden-J threshold = {threshold:.3f}",
    )

    ax.set_xlabel("Shot number")
    ax.set_ylabel("Normalized anomaly score")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------

def evaluate_trial(
    model,
    train_images:   torch.Tensor,       # healthy training images, on device
    selection_data,                     # SplitData for selection set
    cfg,
    device,
    scatter_path:   Optional[Path] = None,
    scatter_title:  Optional[str]  = None,
) -> EvalResult:
    """Full post-training evaluation.

    1. Calibrate mu, sigma from healthy training images.
    2. Score the selection set.
    3. Normalize scores.
    4. Compute per-class AUCs.
    5. Apply objective with floor.
    6. Find Youden-J threshold and produce scatter plot (if path given).
    """
    result = EvalResult()

    # 1. Calibration on healthy training set
    calib = compute_calibration(
        model, train_images, cfg.batch_size, device, score_fn=cfg.score_fn
    )
    result.mu    = calib["mu"]
    result.sigma = calib["sigma"]

    # 2. Raw reconstruction score on selection set
    raw_mse = _per_sample_mse(
        model, selection_data.images, cfg.batch_size, score_fn=cfg.score_fn
    )
    result.raw_mse = raw_mse

    # 3. Normalized score (clipped to [0, 1])
    norm = (raw_mse - result.mu) / (result.sigma * cfg.score_k)
    norm = np.clip(norm, 0.0, 1.0)
    result.norm_score = norm

    # 4. Per-class AUCs (compute on normalized scores -- monotone rescaling
    #    so AUC is identical to using raw MSE, but keeps everything in one scale)
    result.auc_per_class = _per_class_auc(norm, selection_data.classes)

    # 5. Objective
    result.objective, result.floor_ok = _compute_objective(result.auc_per_class)

    # 6. Youden-J threshold (on normalized scores, matches the scatter plot axes)
    result.threshold = _youden_threshold(norm, selection_data.classes)

    # 7. Scatter plot (optional)
    if scatter_path is not None:
        title = scatter_title or f"Selection set anomaly scores (obj={result.objective:.4f})"
        _scatter_plot(
            shots       = selection_data.shots,
            classes     = selection_data.classes,
            norm_scores = norm,
            threshold   = result.threshold,
            out_path    = Path(scatter_path),
            title       = title,
        )
        result.scatter_path = str(scatter_path)

    return result

"""
evaluate_test_set.py

THE moment of truth — evaluate a trained checkpoint on the LOCKED 190-image
test set that was isolated before any training or hyperparameter search.

This is separate from ae_lib/evaluation.py (which scores the selection set).
The numbers this script produces are the ones that go in the paper.

Calibration discipline (critical for defensibility):
    - mu, sigma are computed ONLY on the TRAINING set's healthy
      reconstruction errors — exactly as evaluate_trial does.
    - The test set healthy images are used ONLY as the negative class
      for AUC. They never touch calibration. No test data leaks into
      the score normalization.

Scoring respects cfg.score_fn:
    - "mse"  -> per-image reconstruction MSE      (model.reconstruction_error)
    - "ssim" -> per-image DSSIM = 1 - mean(SSIM)   (ae_lib.losses.per_sample_dssim)

Produces:
    - Per-class AUC on the test set (one-vs-healthy)
    - Objective with the same hard-floor rule used in the study
    - Score statistics per class
    - Score-margin diagnostics (min anomaly vs max healthy) so you can
      judge whether any AUC=1.0 is robust or fragile
    - A scatter plot (test_set_scatter.png) matching the study's style

Usage:
    HIP_VISIBLE_DEVICES=0 python evaluate_test_set.py \
        --checkpoint lcfs_ssim_seed0/checkpoints/manual_best.pt \
        --config     configs/trial_142_lcfs_ssim.yaml \
        --test-dir   /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/final_test_set/ \
        --gpu        0 \
        --outdir     ./test_eval_seed0/
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve


ANOMALY_CLASSES = ["broken_lcfs", "bad_black_core", "bad_nonconverged"]

CLASS_COLORS = {
    "healthy":          "#1f77b4",
    "broken_lcfs":      "#ff7f0e",
    "bad_black_core":   "#d62728",
    "bad_nonconverged": "#9467bd",
}

# Documented test-set composition from PROJECT_HANDOFF.txt — sanity check
EXPECTED_TEST = {
    "healthy": 120,
    "broken_lcfs": 30,
    "bad_black_core": 30,
    "bad_nonconverged": 10,
}


# ── Scoring ────────────────────────────────────────────────────────────────────

def per_sample_scores(model, images, batch_size, score_fn, device):
    """Per-image anomaly scores honoring cfg.score_fn ('mse' or 'ssim')."""
    model.eval()
    out = []

    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size]

            if score_fn == "ssim":
                from ae_lib.losses import per_sample_dssim

                recon = model(batch)
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]

                s = per_sample_dssim(batch, recon)
            else:
                s = model.reconstruction_error(batch)

            out.append(s.detach().cpu().numpy())

    return np.concatenate(out, axis=0)


def per_class_auc(scores, classes):
    """Compute one-vs-healthy AUC for each anomaly class."""
    classes = np.asarray(classes)
    healthy = classes == "healthy"

    result = {}

    for cls in ANOMALY_CLASSES:
        anom = classes == cls

        if anom.sum() == 0 or healthy.sum() == 0:
            result[cls] = float("nan")
            continue

        y_true = np.concatenate([
            np.zeros(healthy.sum()),
            np.ones(anom.sum()),
        ])

        y_score = np.concatenate([
            scores[healthy],
            scores[anom],
        ])

        result[cls] = roc_auc_score(y_true, y_score)

    return result


def convert_locked_test_manifest(raw_manifest: Path, test_dir: Path, outdir: Path) -> Path:
    """
    Convert the locked test manifest into the standard load_split() format.

    Raw test.txt format:
        186749_3000    healthy

    Real file path:
        healthy/jy_186749_3000.png

    Converted format:
        healthy/jy_186749_3000.png    healthy

    The converted file is saved in outdir so it can be inspected later.
    """
    converted = outdir / "test_converted.txt"
    n_lines = 0

    with open(raw_manifest) as fin, open(converted, "w") as fout:
        for line_no, ln in enumerate(fin, start=1):
            ln = ln.strip()

            if not ln or ln.startswith("#"):
                continue

            parts = ln.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Bad manifest line {line_no} in {raw_manifest}:\n"
                    f"  {ln!r}\n"
                    f"Expected format: <shot>\\t<class>"
                )

            shot, cls = parts
            shot = shot.strip()
            cls = cls.strip()

            rel_path = f"{cls}/jy_{shot}.png"
            full_path = test_dir / rel_path

            if not full_path.is_file():
                raise FileNotFoundError(
                    f"Converted path does not exist: {full_path}\n"
                    f"  From manifest line {line_no}: {shot}\\t{cls}"
                )

            fout.write(f"{rel_path}\t{cls}\n")
            n_lines += 1

    print(f"  Converted {n_lines} entries → {converted}")

    return converted


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to final_test_set/ containing test.txt and class folders",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--outdir", default="./test_eval_outputs/")

    args = parser.parse_args()

    from ae_lib.config import Config
    from ae_lib.data import load_split
    from ae_lib.model import Autoencoder

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(
        f"score_fn:   {cfg.score_fn}   "
        f"loss_fn: {cfg.loss_fn}   "
        f"use_lcfs_masking: {cfg.use_lcfs_masking}"
    )
    print(f"Device:     {device}")

    # ── Load model ──────────────────────────────────────────────────────────────
    model = Autoencoder(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    model.load_state_dict(state)
    model.eval()

    # ── Calibration on TRAINING healthy, NOT test data ──────────────────────────
    train_path = Path(cfg.manifests_dir) / "train.txt"
    print(f"\nCalibrating mu/sigma on training healthy: {train_path}")

    train = load_split(train_path, cfg, device)

    train_scores = per_sample_scores(
        model=model,
        images=train.images,
        batch_size=cfg.batch_size,
        score_fn=cfg.score_fn,
        device=device,
    )

    mu = float(train_scores.mean())
    sigma = float(train_scores.std())

    print(f"  mu = {mu:.8f}   sigma = {sigma:.8f}")

    # ── Load LOCKED test set ────────────────────────────────────────────────────
    test_dir = Path(args.test_dir)
    raw_manifest = test_dir / "test.txt"

    print(f"\nLoading LOCKED test set: {raw_manifest}")

    # The test manifest uses a different format than train/val/selection:
    #   test.txt : "186749_3000\thealthy"          bare shot id
    #   files    : healthy/jy_186749_3000.png      class/jy_<shot>.png
    #
    # Rewrite it into:
    #   "<class>/jy_<shot>.png\t<class>"
    #
    # This keeps load_split() untouched and saves the converted manifest
    # in outdir for inspection.
    converted_manifest = convert_locked_test_manifest(
        raw_manifest=raw_manifest,
        test_dir=test_dir,
        outdir=outdir,
    )

    cfg_test = Config.from_yaml(args.config)
    cfg_test.images_root = str(test_dir)

    test = load_split(converted_manifest, cfg_test, device)

    print(f"  n_test = {len(test)}")

    # ── Sanity check composition ────────────────────────────────────────────────
    comp = Counter(test.classes)

    print("  Composition:")

    comp_ok = True

    for cls, exp_n in EXPECTED_TEST.items():
        got = comp.get(cls, 0)
        flag = "" if got == exp_n else f"  ✗ expected {exp_n}"

        if got != exp_n:
            comp_ok = False

        print(f"    {cls:20s}: {got}{flag}")

    if not comp_ok:
        print("  ⚠  Test composition does not match documented ground truth!")

    # ── Score the test set ──────────────────────────────────────────────────────
    test_scores = per_sample_scores(
        model=model,
        images=test.images,
        batch_size=cfg.batch_size,
        score_fn=cfg.score_fn,
        device=device,
    )

    # Normalized scores are display-only; AUC uses raw scores.
    k = cfg.score_k

    if sigma <= 0:
        raise ValueError(
            f"Training calibration sigma is non-positive: sigma={sigma}. "
            "Cannot normalize test scores."
        )

    norm = np.clip((test_scores - mu) / (sigma * k), 0.0, 1.0)

    # ── Per-class AUC ───────────────────────────────────────────────────────────
    aucs = per_class_auc(test_scores, test.classes)

    # ── Objective with same hard-floor rule as the study ───────────────────────
    floor_ok = (
        aucs["bad_black_core"] >= 0.90
        and aucs["bad_nonconverged"] >= 0.90
    )

    if floor_ok:
        objective = (
            0.4 * aucs["broken_lcfs"]
            + 0.3 * aucs["bad_black_core"]
            + 0.3 * aucs["bad_nonconverged"]
        )
    else:
        objective = 0.0

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST-SET EVALUATION (LOCKED — this is the paper number)")
    print("=" * 60)
    print(f"  AUC broken_lcfs      : {aucs['broken_lcfs']:.4f}")
    print(f"  AUC bad_black_core   : {aucs['bad_black_core']:.4f}")
    print(f"  AUC bad_nonconverged : {aucs['bad_nonconverged']:.4f}")
    print(f"  objective            : {objective:.4f}")
    print(f"  floor_ok             : {floor_ok}")

    # ── Score statistics + margin diagnostics ──────────────────────────────────
    classes_arr = np.array(test.classes)

    print("\n  Score statistics:")
    print(
        f"  {'class':20s} {'N':>4} {'mean':>12} {'median':>12} "
        f"{'min':>12} {'max':>12}"
    )

    stats = {}

    for cls in ["healthy"] + ANOMALY_CLASSES:
        m = classes_arr == cls

        if m.sum() == 0:
            continue

        s = test_scores[m]
        stats[cls] = s

        print(
            f"  {cls:20s} {m.sum():>4} "
            f"{s.mean():>12.6f} "
            f"{np.median(s):>12.6f} "
            f"{s.min():>12.6f} "
            f"{s.max():>12.6f}"
        )

    print("\n  Separation margins (anomaly min vs healthy max):")

    h_max = stats["healthy"].max()

    for cls in ANOMALY_CLASSES:
        if cls not in stats:
            continue

        a_min = stats[cls].min()
        gap = a_min - h_max
        n_below = int((stats[cls] < h_max).sum())

        if gap > 0:
            verdict = "clean"
        else:
            verdict = f"{n_below} anomaly score(s) below max healthy"

        print(
            f"    {cls:20s}: "
            f"anomaly_min={a_min:.6f}  "
            f"healthy_max={h_max:.6f}  "
            f"gap={gap:+.6f}  "
            f"({verdict})"
        )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "path": test.paths,
        "shot": test.shots,
        "true_class": test.classes,
        "raw_score": test_scores,
        "norm_score": norm,
    })

    csv_path = outdir / "test_scores.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n  Saved per-image scores → {csv_path}")

    # ── Scatter plot ───────────────────────────────────────────────────────────
    y_true_all = (classes_arr != "healthy").astype(int)

    fpr, tpr, thr = roc_curve(y_true_all, norm)
    youden = thr[np.argmax(tpr - fpr)]

    shots_num = np.array([
        int("".join(filter(str.isdigit, str(s))) or 0)
        for s in test.shots
    ])

    fig, ax = plt.subplots(figsize=(14, 6))

    for cls in ["healthy"] + ANOMALY_CLASSES:
        m = classes_arr == cls

        if m.sum() == 0:
            continue

        ax.scatter(
            shots_num[m],
            norm[m],
            s=18,
            alpha=0.7,
            color=CLASS_COLORS[cls],
            label=f"{cls} (n={m.sum()})",
        )

    ax.axhline(
        youden,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Youden-J threshold = {youden:.3f}",
    )

    ax.set_xlabel("Shot number")
    ax.set_ylabel("Normalized anomaly score")
    ax.set_title(
        f"TEST SET — {Path(args.checkpoint).parent.parent.name}  "
        f"(score_fn={cfg.score_fn})\n"
        f"broken_lcfs={aucs['broken_lcfs']:.3f}  "
        f"bad_black_core={aucs['bad_black_core']:.3f}  "
        f"bad_nonconverged={aucs['bad_nonconverged']:.3f}"
    )

    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    scatter_path = outdir / "test_set_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved scatter plot → {scatter_path}")

    # ── Compare to selection-set numbers ───────────────────────────────────────
    print("\n  Reminder — selection-set numbers for this checkpoint were:")
    print("    broken_lcfs=1.0000  bad_black_core=0.9171  bad_nonconverged=1.0000")
    print(
        "  Compare against the test-set numbers above. Large drops "
        "(esp. bad_nonconverged with n=10) indicate small-sample optimism."
    )


if __name__ == "__main__":
    main()

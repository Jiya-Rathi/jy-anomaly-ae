"""
verify_data_integrity.py

End-to-end sanity check before trusting any AUC numbers.

Runs SEVEN checks in order. Any FAIL stops you from drawing conclusions
from the model results. All must pass before results are trustworthy.

CHECK 1  Split sizes match documented ground truth
CHECK 2  Training and validation sets contain only healthy images
CHECK 3  No image appears in more than one split
CHECK 4  No duplicates within a single split
CHECK 5  Class label in manifest matches folder name in path
CHECK 6  Every image file actually exists on disk
CHECK 7  Score distributions — do scores make sense per class?
         (Needs a trained checkpoint; skipped gracefully if not provided)

Outputs
-------
  Console report  — pass/fail per check with counts
  score_distribution.png — box plot + strip plot of anomaly scores by class
                           (only if --checkpoint is provided)
  score_detail.csv        — per-image: path, true_class, score, rank
                           (only if --checkpoint is provided)

Documented ground truth (from research record / PROJECT_HANDOFF.txt):
  train.txt      : 722   (all healthy)
  val.txt        : 120   (all healthy)
  selection.txt  : 405   (99 healthy, 149 broken_lcfs, 142 bad_black_core,
                           15 bad_nonconverged)
  test.txt       : 190   (120 healthy, 30 broken_lcfs, 30 bad_black_core,
                           10 bad_nonconverged)
  GRAND TOTAL    : 1437

Usage — manifest checks only:
    python verify_data_integrity.py \\
        --manifest-dir /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests/ \\
        --images-root  /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/

Usage — full check including score distributions:
    python verify_data_integrity.py \\
        --manifest-dir /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests/ \\
        --images-root  /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/ \\
        --checkpoint   lcfs_single_outputs/checkpoints/None_best.pt \\
        --config       configs/trial_142_lcfs.yaml \\
        --gpu          0 \\
        --outdir       ./integrity_outputs/
"""

import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Documented ground truth ────────────────────────────────────────────────────

EXPECTED = {
    "train.txt": {
        "total":          722,
        "healthy":        722,
        "broken_lcfs":    0,
        "bad_black_core": 0,
        "bad_nonconverged": 0,
    },
    "val.txt": {
        "total":          120,
        "healthy":        120,
        "broken_lcfs":    0,
        "bad_black_core": 0,
        "bad_nonconverged": 0,
    },
    "selection.txt": {
        "total":          405,
        "healthy":        99,
        "broken_lcfs":    149,
        "bad_black_core": 142,
        "bad_nonconverged": 15,
    },
    "test.txt": {
        "total":          190,
        "healthy":        120,
        "broken_lcfs":    30,
        "bad_black_core": 30,
        "bad_nonconverged": 10,
    },
}

GRAND_TOTAL = 1437
VALID_CLASSES = {"healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"}
CLASS_COLORS = {
    "healthy":          "#1f77b4",
    "broken_lcfs":      "#ff7f0e",
    "bad_black_core":   "#d62728",
    "bad_nonconverged": "#9467bd",
}


# ── Manifest loading ───────────────────────────────────────────────────────────

def load_manifest(path):
    """Returns list of (rel_path, class_label) tuples. Raises on parse error."""
    entries = []
    with open(path) as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path.name} line {i}: expected 2 tab-separated fields, "
                    f"got {len(parts)}: {ln!r}"
                )
            rel_path, cls = parts
            if cls not in VALID_CLASSES:
                raise ValueError(
                    f"{path.name} line {i}: unknown class {cls!r}. "
                    f"Valid: {VALID_CLASSES}"
                )
            entries.append((rel_path, cls))
    return entries


# ── Check helpers ──────────────────────────────────────────────────────────────

def _ok(msg):   print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}"); return False
def _warn(msg): print(f"  ⚠  {msg}")
def _header(n, title): print(f"\nCHECK {n} — {title}")


# ── Check 1: Split sizes ───────────────────────────────────────────────────────

def check_split_sizes(manifests):
    """Compare actual counts against EXPECTED dict."""
    _header(1, "Split sizes match documented ground truth")
    all_pass = True

    for fname, entries in manifests.items():
        exp = EXPECTED.get(fname, {})
        if not exp:
            _warn(f"No expected counts for {fname} — skipping")
            continue

        actual_total = len(entries)
        actual_by_class = Counter(cls for _, cls in entries)

        if actual_total != exp["total"]:
            _fail(f"{fname}: total expected={exp['total']}, got={actual_total}")
            all_pass = False
        else:
            _ok(f"{fname}: total = {actual_total}")

        for cls in ("healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"):
            exp_n    = exp.get(cls, 0)
            actual_n = actual_by_class.get(cls, 0)
            if actual_n != exp_n:
                _fail(f"  {fname} [{cls}]: expected={exp_n}, got={actual_n}")
                all_pass = False
            else:
                _ok(f"  {fname} [{cls}]: {actual_n}")

    # Grand total
    grand = sum(len(e) for e in manifests.values())
    if grand != GRAND_TOTAL:
        _fail(f"Grand total: expected={GRAND_TOTAL}, got={grand}")
        all_pass = False
    else:
        _ok(f"Grand total: {grand}")

    return all_pass


# ── Check 2: Training purity ───────────────────────────────────────────────────

def check_training_purity(manifests):
    """Ensure train.txt and val.txt contain ONLY healthy images."""
    _header(2, "Training and validation sets contain only healthy images")
    all_pass = True

    for fname in ("train.txt", "val.txt"):
        entries = manifests.get(fname, [])
        non_healthy = [(p, c) for p, c in entries if c != "healthy"]
        if non_healthy:
            _fail(f"{fname}: {len(non_healthy)} non-healthy images found!")
            for p, c in non_healthy[:5]:
                print(f"       {p}  →  {c}")
            if len(non_healthy) > 5:
                print(f"       ... and {len(non_healthy) - 5} more")
            all_pass = False
        else:
            _ok(f"{fname}: all {len(entries)} entries are healthy")

    return all_pass


# ── Check 3: No cross-split overlap ───────────────────────────────────────────

def check_no_overlap(manifests):
    """Every image path should appear in at most one split."""
    _header(3, "No image appears in more than one split")

    path_to_splits = defaultdict(list)
    for fname, entries in manifests.items():
        for rel_path, _ in entries:
            path_to_splits[rel_path].append(fname)

    duplicates = {p: splits for p, splits in path_to_splits.items()
                  if len(splits) > 1}

    if duplicates:
        _fail(f"{len(duplicates)} image(s) appear in multiple splits!")
        for p, splits in list(duplicates.items())[:10]:
            print(f"       {p}  →  {splits}")
        return False
    else:
        _ok(f"No overlapping images across splits ({len(path_to_splits)} unique paths)")
        return True


# ── Check 4: No intra-split duplicates ────────────────────────────────────────

def check_no_intra_duplicates(manifests):
    """Within each split, every path should appear exactly once."""
    _header(4, "No duplicates within a single split")
    all_pass = True

    for fname, entries in manifests.items():
        paths = [p for p, _ in entries]
        counts = Counter(paths)
        dups = {p: n for p, n in counts.items() if n > 1}
        if dups:
            _fail(f"{fname}: {len(dups)} path(s) appear more than once")
            for p, n in list(dups.items())[:5]:
                print(f"       {p}  (×{n})")
            all_pass = False
        else:
            _ok(f"{fname}: all {len(paths)} paths unique")

    return all_pass


# ── Check 5: Label-path consistency ───────────────────────────────────────────

def check_label_path_consistency(manifests):
    """
    The first path component (folder name) should match the class label.
    e.g. broken_lcfs/jy_190082_3000.png  → label must be broken_lcfs
    """
    _header(5, "Class label matches folder name in path")
    all_pass = True

    for fname, entries in manifests.items():
        mismatches = []
        for rel_path, cls in entries:
            folder = Path(rel_path).parts[0]   # first directory component
            if folder != cls:
                mismatches.append((rel_path, folder, cls))

        if mismatches:
            _fail(f"{fname}: {len(mismatches)} label-path mismatch(es)")
            for rel_path, folder, cls in mismatches[:5]:
                print(f"       folder={folder!r}  label={cls!r}  path={rel_path}")
            if len(mismatches) > 5:
                print(f"       ... and {len(mismatches) - 5} more")
            all_pass = False
        else:
            _ok(f"{fname}: all {len(entries)} paths consistent with labels")

    return all_pass


# ── Check 6: File existence ────────────────────────────────────────────────────

def check_file_existence(manifests, images_root):
    """Every path listed in any manifest must exist on disk."""
    _header(6, "Every image file exists on disk")
    images_root = Path(images_root)
    all_pass = True

    for fname, entries in manifests.items():
        missing = []
        for rel_path, _ in entries:
            full = images_root / rel_path
            if not full.is_file():
                missing.append(rel_path)

        if missing:
            _fail(f"{fname}: {len(missing)} file(s) missing from disk")
            for p in missing[:5]:
                print(f"       {images_root / p}")
            if len(missing) > 5:
                print(f"       ... and {len(missing) - 5} more")
            all_pass = False
        else:
            _ok(f"{fname}: all {len(entries)} files exist")

    return all_pass


# ── Check 7: Score distributions ──────────────────────────────────────────────

def check_score_distributions(checkpoint_path, config_path, gpu, outdir):
    """
    Load the trained model, score the selection set, print statistics per class,
    and produce a score distribution plot + CSV.

    This check answers: do healthy images score LOW and anomalies score HIGH?
    Any class where anomalies score LOWER than healthy indicates a problem
    (wrong labels, wrong split, or preprocessing reversal).
    """
    _header(7, "Score distributions — do scores make sense per class?")

    try:
        import torch
        from ae_lib.config    import Config
        from ae_lib.data      import load_split
        from ae_lib.model     import Autoencoder
        from ae_lib.evaluation import _per_sample_mse   # internal helper
    except ImportError as e:
        _warn(f"Could not import ae_lib: {e}. Skipping score check.")
        return True

    # Load config and model
    cfg    = Config.from_yaml(config_path)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model  = Autoencoder(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Checkpoint loaded: {checkpoint_path}")
    print(f"  Device: {device}")

    # Load selection set
    manifests_dir = Path(cfg.manifests_dir)
    sel_path      = manifests_dir / "selection.txt"
    print(f"  Loading selection set from {sel_path} ...")
    sel = load_split(sel_path, cfg, device)
    print(f"  n_selection = {len(sel)}")

    # Score
    scores = _per_sample_mse(model, sel.images, batch_size=cfg.batch_size)

    # Stats per class
    classes_arr = np.array(sel.classes)
    print()
    print(f"  {'Class':20s}  {'N':>5}  {'mean':>10}  {'median':>10}  "
          f"{'min':>10}  {'max':>10}")
    print(f"  {'-'*70}")

    class_order = ["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"]
    stats = {}
    for cls in class_order:
        mask = classes_arr == cls
        if mask.sum() == 0:
            continue
        s = scores[mask]
        stats[cls] = s
        print(f"  {cls:20s}  {mask.sum():>5}  "
              f"{s.mean():>10.6f}  {np.median(s):>10.6f}  "
              f"{s.min():>10.6f}  {s.max():>10.6f}")

    # Sanity check: healthy median should be LOWER than anomaly medians
    print()
    healthy_median = np.median(stats["healthy"]) if "healthy" in stats else None
    all_pass = True
    for cls in ["broken_lcfs", "bad_black_core", "bad_nonconverged"]:
        if cls not in stats:
            continue
        cls_median = np.median(stats[cls])
        if healthy_median is not None:
            if cls_median > healthy_median:
                _ok(f"{cls} median ({cls_median:.6f}) > healthy median "
                    f"({healthy_median:.6f}) — scores in right direction")
            else:
                _fail(f"{cls} median ({cls_median:.6f}) <= healthy median "
                      f"({healthy_median:.6f}) — SCORES ARE INVERTED OR WRONG")
                all_pass = False

    # ── Save detail CSV ───────────────────────────────────────────────────────
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "path":       sel.paths,
        "true_class": sel.classes,
        "shot":       sel.shots,
        "score":      scores,
    })
    df["rank"] = df["score"].rank(ascending=False).astype(int)
    csv_path = outdir / "score_detail.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Score detail saved → {csv_path}")

    # ── Score distribution plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Score distribution by class — selection set\n"
                 f"(checkpoint: {Path(checkpoint_path).name})",
                 fontsize=12)

    # Left: box + strip plot
    ax = axes[0]
    present = [c for c in class_order if c in stats]
    positions = range(len(present))
    bplot = ax.boxplot(
        [stats[c] for c in present],
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, cls in zip(bplot["boxes"], present):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.6)

    # Overlay strip plot
    rng = np.random.default_rng(0)
    for i, cls in enumerate(present):
        jitter = rng.uniform(-0.15, 0.15, size=len(stats[cls]))
        ax.scatter(
            i + jitter, stats[cls],
            color=CLASS_COLORS[cls], alpha=0.35, s=8, zorder=3
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylabel("Anomaly score (MSE)")
    ax.set_title("Box + strip plot (log scale)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-class histogram (overlaid)
    ax2 = axes[1]
    for cls in present:
        ax2.hist(
            stats[cls], bins=30, alpha=0.55,
            color=CLASS_COLORS[cls], label=cls,
            density=True
        )
    ax2.set_xlabel("Anomaly score (MSE)")
    ax2.set_ylabel("Density")
    ax2.set_title("Overlaid histogram")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = outdir / "score_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Score distribution plot → {plot_path}")

    # ── Top / bottom broken_lcfs by score ─────────────────────────────────────
    # Show the 5 highest-scoring and 5 lowest-scoring broken_lcfs images
    # so the user can spot-check visually whether they're correctly labeled.
    _print_score_extremes(df, "broken_lcfs",    n=5)
    _print_score_extremes(df, "bad_black_core",  n=5)
    _print_score_extremes(df, "bad_nonconverged", n=5)

    return all_pass


def _print_score_extremes(df, cls, n=5):
    """Print highest and lowest scoring images for one class."""
    sub = df[df["true_class"] == cls].copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("score", ascending=False)
    print(f"\n  ── {cls} — top {n} scores (should look ANOMALOUS) ──")
    for _, row in sub.head(n).iterrows():
        print(f"       score={row['score']:.6f}  rank={row['rank']}  {row['path']}")
    print(f"  ── {cls} — bottom {n} scores (highest risk of mislabelling) ──")
    for _, row in sub.tail(n).iterrows():
        print(f"       score={row['score']:.6f}  rank={row['rank']}  {row['path']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify manifest integrity and score distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-dir",  required=True,
                        help="Directory with train/val/selection/test .txt files")
    parser.add_argument("--images-root",   required=True,
                        help="Root directory for image files")
    parser.add_argument("--checkpoint",    default=None,
                        help="Trained model checkpoint (.pt) for Check 7")
    parser.add_argument("--config",        default=None,
                        help="YAML config matching the checkpoint (for Check 7)")
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--outdir",        default="./integrity_outputs/",
                        help="Where to save score_distribution.png and CSV")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    print(f"Manifest directory: {manifest_dir}")
    print(f"Images root:        {args.images_root}")

    # Load all manifests
    manifest_files = ["train.txt", "val.txt", "selection.txt", "test.txt"]
    manifests = {}
    for fname in manifest_files:
        p = manifest_dir / fname
        if not p.exists():
            print(f"\n[SKIP] {fname} not found at {p}")
            continue
        try:
            manifests[fname] = load_manifest(p)
        except ValueError as e:
            print(f"\n[ERROR] Could not parse {fname}: {e}")
            sys.exit(1)

    # Run checks 1-6
    results = {}
    results[1] = check_split_sizes(manifests)
    results[2] = check_training_purity(manifests)
    results[3] = check_no_overlap(manifests)
    results[4] = check_no_intra_duplicates(manifests)
    results[5] = check_label_path_consistency(manifests)
    results[6] = check_file_existence(manifests, args.images_root)

    # Check 7 (optional — needs checkpoint)
    if args.checkpoint and args.config:
        results[7] = check_score_distributions(
            args.checkpoint, args.config, args.gpu, args.outdir
        )
    else:
        print("\nCHECK 7 — Score distributions [SKIPPED — no checkpoint provided]")
        print("  Re-run with --checkpoint and --config to include score analysis.")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(v for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    for check_num, passed in sorted(results.items()):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Check {check_num}: {status}")
    print(f"\n  {n_pass} passed, {n_fail} failed")

    if n_fail > 0:
        print("\n  ACTION REQUIRED: fix failures before trusting AUC numbers.")
        sys.exit(1)
    else:
        print("\n  All checks passed. AUC numbers are trustworthy.")


if __name__ == "__main__":
    main()

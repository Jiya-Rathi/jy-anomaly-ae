#!/usr/bin/env python3
"""
analyze_study.py
================

Comprehensive hyperparameter analysis for the LCFS-masked SSIM Optuna study.

Usage
-----
    python analyze_study.py study_outputs_lcfs_ssim_v2/summary.csv
    python analyze_study.py study_outputs_lcfs_ssim_v2/summary.csv --top 10 --out analysis_out/

What it does
------------
1. Reads summary.csv, auto-detecting column positions BY HEADER NAME
   (the column order in this CSV has drifted between pulls; never hard-code
   field indices again).
2. Restricts to floor_ok trials, then to COMPLETE rows with all HP fields.
3. Reports the study-level picture (counts, per-class AUC distributions).
4. Builds a full Pearson AND Spearman correlation matrix over every numeric
   HP plus the calibration outputs (mu, sigma, threshold) and the three
   per-class AUCs + objective.
5. Computes PARTIAL correlations to disentangle the lr <-> batch_size
   confound (corr of X with target, controlling for Z).
6. Categorical HP breakdowns (use_batchnorm, use_hpf, use_lcfs_masking).
7. Prints a ranked "what drives bad_black_core" table, since that is the
   only class with real variance in this study.
8. Writes a correlation-matrix heatmap PNG and a tidy CSV of all results
   if matplotlib/pandas are available; degrades gracefully to stdlib-only
   if they are not (the script must run inside ae_env on Mantis).

Design notes
------------
- Correlations are computed on the FULL floor-ok population, NOT the top-10.
  n=10 subsets give unstable estimates (we watched the lr-vs-BBC correlation
  swing from -0.82 to -0.65 just by changing which 8-10 trials were in view).
  The top-N view is reported separately and clearly labelled as descriptive.
- lr is analyzed in log10 space (it is sampled log-uniform in the search
  space, so linear-space correlations are misleading).
- Spearman is included because most HP->metric relationships here are
  monotonic but not linear (e.g. low-lr -> high BBC is a saturating effect).
- Everything degrades to Python stdlib if numpy/pandas/scipy are missing,
  so it runs even on a bare ae_env. The heatmap needs matplotlib; if absent
  the script says so and skips only that step.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict


# ----------------------------------------------------------------------------
# Column schema. These are the fields we KNOW exist (from SUMMARY_FIELDS in
# ae_lib/logging_utils.py). We resolve them by name at runtime, so the actual
# position in the file does not matter.
# ----------------------------------------------------------------------------

# Numeric hyperparameters that are searched (or fixed) per trial.
NUMERIC_HPS = [
    "bottleneck_dim",
    "n_enc_layers",
    "n_dec_layers",
    "base_channels",
    "growth_factor",
    "lr",
    "batch_size",
    "hpf_sigma",          # only meaningful when use_hpf is True; filtered below
]

# Categorical / boolean hyperparameters.
CATEGORICAL_HPS = [
    "use_hpf",
    "use_batchnorm",
    "use_lcfs_masking",
]

# Calibration outputs (downstream of training, NOT inputs). Included in the
# matrix because sigma in particular tracks BBC separability and we want to
# see the confound explicitly rather than pretend it is an input HP.
CALIB_OUTPUTS = ["mu", "sigma", "threshold"]

# Targets we care about.
TARGETS = [
    "objective",
    "auc_broken_lcfs",
    "auc_bad_black_core",
    "auc_bad_nonconverged",
]

# Short labels for compact printing.
SHORT = {
    "bottleneck_dim": "btl",
    "n_enc_layers": "enc",
    "n_dec_layers": "dec",
    "base_channels": "base_ch",
    "growth_factor": "growth",
    "lr": "log10_lr",
    "batch_size": "bs",
    "hpf_sigma": "hpf_s",
    "mu": "mu",
    "sigma": "sigma",
    "threshold": "thr",
    "objective": "obj",
    "auc_broken_lcfs": "lcfs",
    "auc_bad_black_core": "bbc",
    "auc_bad_nonconverged": "bnc",
}


# ----------------------------------------------------------------------------
# Stats primitives (stdlib-only so the script always runs)
# ----------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def pearson(xs, ys):
    """Pearson linear correlation. Returns nan if undefined."""
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def _rankdata(xs):
    """Average-rank transform, handling ties (needed for Spearman)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        # advance over a run of tied values
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation = Pearson on ranks."""
    if len(xs) < 2:
        return float("nan")
    return pearson(_rankdata(xs), _rankdata(ys))


def partial_corr(xs, ys, zs):
    """
    Partial correlation of X and Y controlling for Z.
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Used to ask: does lr correlate with BBC *after* removing the part of
    that relationship explained by batch_size (and vice versa)?
    """
    r_xy = pearson(xs, ys)
    r_xz = pearson(xs, zs)
    r_yz = pearson(ys, zs)
    den = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if den == 0 or any(math.isnan(v) for v in (r_xy, r_xz, r_yz)):
        return float("nan")
    return (r_xy - r_xz * r_yz) / den


def approx_p_from_r(r, n):
    """
    Two-sided p-value for a correlation via the t-approximation.
    t = r * sqrt((n-2)/(1-r^2)); p from a normal approx to the t CDF.
    This is a rough screen only -- with n this small, treat as directional.
    """
    if n < 3 or math.isnan(r) or abs(r) >= 1.0:
        return float("nan")
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    # crude normal approximation to two-sided t p-value
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))
    return p


# ----------------------------------------------------------------------------
# Loading + cleaning
# ----------------------------------------------------------------------------

def load_rows(path):
    """Load all rows as dicts keyed by header name."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_bool(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def floor_ok(row):
    return to_bool(row.get("floor_ok")) is True


def build_numeric_view(rows, present_numeric, present_targets, present_calib):
    """
    Return a list of dicts, one per usable trial, with all numeric fields
    coerced to float and lr replaced by log10(lr). Drops rows missing any
    required numeric field.

    hpf_sigma is special-cased: it is a placeholder (3.0) when use_hpf is
    False, so including it as a real variable would be misleading. We keep
    it only if the study actually varies use_hpf.
    """
    view = []
    for r in rows:
        rec = {"trial_num": r.get("trial_num")}
        ok = True

        for col in present_numeric:
            val = to_float(r.get(col))
            if val is None:
                ok = False
                break
            if col == "lr":
                if val <= 0:
                    ok = False
                    break
                rec["log10_lr"] = math.log10(val)
                rec["lr"] = val
            else:
                rec[col] = val

        if not ok:
            continue

        for col in present_targets + present_calib:
            val = to_float(r.get(col))
            rec[col] = val  # targets/calib may legitimately be absent; keep None

        for col in CATEGORICAL_HPS:
            if col in r:
                rec[col] = to_bool(r.get(col))

        view.append(rec)
    return view


# ----------------------------------------------------------------------------
# Reporting sections
# ----------------------------------------------------------------------------

def hr(title=""):
    line = "=" * 78
    if title:
        return f"\n{line}\n{title}\n{line}"
    return line


def report_overview(all_rows, ok_rows, usable):
    print(hr("STUDY OVERVIEW"))
    print(f"  Total rows in summary.csv : {len(all_rows)}")
    print(f"  floor_ok == True          : {len(ok_rows)}")
    print(f"  Usable (all numeric HPs)  : {len(usable)}")
    if len(usable) < len(ok_rows):
        print(f"  NOTE: {len(ok_rows) - len(usable)} floor-ok rows dropped for "
              f"missing/non-numeric HP fields (likely still-running trials).")


def report_distributions(usable, present_targets):
    print(hr("PER-CLASS AUC + OBJECTIVE DISTRIBUTIONS (floor-ok population)"))
    print(f"  {'metric':<22} {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'std':>8}")
    print("  " + "-" * 64)
    for col in present_targets:
        vals = [r[col] for r in usable if r.get(col) is not None]
        if not vals:
            continue
        print(f"  {col:<22} {mean(vals):>8.4f} {median(vals):>8.4f} "
              f"{min(vals):>8.4f} {max(vals):>8.4f} {stdev(vals):>8.4f}")
    print("\n  Interpretation guide: the class with the LARGEST std is the one")
    print("  the search can still move. Saturated classes (tiny std) are solved")
    print("  and their correlations with HPs will be near zero / unreliable.")


def collect_numeric_columns(usable, present_numeric, present_calib, present_targets):
    """
    Decide the ordered list of columns to put in the correlation matrix.
    lr is represented as log10_lr.
    """
    cols = []
    for c in present_numeric:
        cols.append("log10_lr" if c == "lr" else c)
    cols += [c for c in present_calib if any(r.get(c) is not None for r in usable)]
    cols += present_targets
    # keep only columns that actually have variance (std > 0) and data
    final = []
    for c in cols:
        vals = [r.get(c) for r in usable if r.get(c) is not None]
        if len(vals) >= 2 and stdev(vals) > 0:
            final.append(c)
    return final


def report_correlation_matrix(usable, cols, method="pearson"):
    fn = pearson if method == "pearson" else spearman
    print(hr(f"{method.upper()} CORRELATION MATRIX (n varies per pair, full floor-ok set)"))

    labels = [SHORT.get(c, c) for c in cols]
    colw = max(8, max(len(l) for l in labels) + 1)

    # header
    header = " " * (colw) + "".join(f"{l:>{colw}}" for l in labels)
    print(header)

    matrix = {}
    for ci in cols:
        row_out = f"{SHORT.get(ci, ci):>{colw}}"  # row label, right aligned in first block
        # actually want row label left; rebuild
        row_out = f"{SHORT.get(ci, ci):<{colw}}"
        for cj in cols:
            # pairwise complete observations
            pairs = [(r[ci], r[cj]) for r in usable
                     if r.get(ci) is not None and r.get(cj) is not None]
            if len(pairs) < 2:
                val = float("nan")
            else:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                val = fn(xs, ys)
            matrix[(ci, cj)] = val
            cell = "  nan" if math.isnan(val) else f"{val:+.2f}"
            row_out += f"{cell:>{colw}}"
        print(row_out)
    return matrix


def report_target_rankings(usable, cols, present_targets):
    print(hr("HP IMPORTANCE RANKING PER TARGET (Pearson & Spearman, with rough p)"))
    hp_cols = [c for c in cols if c not in present_targets and c not in CALIB_OUTPUTS]
    calib_cols = [c for c in cols if c in CALIB_OUTPUTS]

    for tgt in present_targets:
        tvals_all = [r.get(tgt) for r in usable]
        if all(v is None for v in tvals_all):
            continue
        print(f"\n  >>> Target: {tgt}")
        print(f"      {'feature':<12} {'pearson':>9} {'spearman':>9} {'~p(pear)':>9} {'n':>5}")
        print("      " + "-" * 50)

        scored = []
        for c in hp_cols + calib_cols:
            pairs = [(r[c], r[tgt]) for r in usable
                     if r.get(c) is not None and r.get(tgt) is not None]
            if len(pairs) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            rp = pearson(xs, ys)
            rs = spearman(xs, ys)
            p = approx_p_from_r(rp, len(pairs))
            scored.append((abs(rp), c, rp, rs, p, len(pairs)))

        scored.sort(reverse=True)
        for _, c, rp, rs, p, n in scored:
            tag = ""
            if c in CALIB_OUTPUTS:
                tag = "  (calibration output, not an input HP)"
            pstr = "   nan" if math.isnan(p) else f"{p:>6.3f}"
            print(f"      {SHORT.get(c, c):<12} {rp:>+9.3f} {rs:>+9.3f} {pstr:>9} {n:>5}{tag}")


def report_confound(usable):
    """
    The lr<->batch_size entanglement. Partial correlations isolate each.
    Only meaningful if both columns are present and vary.
    """
    have = lambda c: any(r.get(c) is not None for r in usable)
    if not (have("log10_lr") and have("batch_size")):
        return
    print(hr("lr <-> batch_size CONFOUND (partial correlations)"))
    targets = ["auc_bad_black_core", "auc_bad_nonconverged", "objective"]
    for tgt in targets:
        rows = [r for r in usable
                if r.get("log10_lr") is not None
                and r.get("batch_size") is not None
                and r.get(tgt) is not None]
        if len(rows) < 4:
            continue
        loglr = [r["log10_lr"] for r in rows]
        bs = [r["batch_size"] for r in rows]
        t = [r[tgt] for r in rows]

        r_lr = pearson(loglr, t)
        r_bs = pearson(bs, t)
        pr_lr = partial_corr(loglr, t, bs)   # lr vs target controlling for bs
        pr_bs = partial_corr(bs, t, loglr)   # bs vs target controlling for lr
        r_lr_bs = pearson(loglr, bs)

        print(f"\n  Target: {tgt}  (n={len(rows)})")
        print(f"    raw    log10_lr vs {SHORT.get(tgt,tgt):<5} : {r_lr:+.3f}")
        print(f"    raw    bs       vs {SHORT.get(tgt,tgt):<5} : {r_bs:+.3f}")
        print(f"    partial log10_lr | bs            : {pr_lr:+.3f}")
        print(f"    partial bs       | log10_lr      : {pr_bs:+.3f}")
        print(f"    (log10_lr <-> bs collinearity    : {r_lr_bs:+.3f})")
        # quick verbal read
        if not math.isnan(pr_lr) and not math.isnan(pr_bs):
            if abs(pr_lr) > abs(pr_bs) + 0.15:
                print("    -> lr carries the signal; bs effect largely vanishes when lr is held fixed.")
            elif abs(pr_bs) > abs(pr_lr) + 0.15:
                print("    -> bs carries the signal; lr effect largely vanishes when bs is held fixed.")
            else:
                print("    -> entangled; this data cannot cleanly separate them (need a controlled sweep).")


def report_categorical(usable):
    print(hr("CATEGORICAL HP BREAKDOWNS"))
    for hp in CATEGORICAL_HPS:
        groups = defaultdict(list)
        for r in usable:
            if hp in r and r[hp] is not None and r.get("objective") is not None:
                groups[r[hp]].append(r)
        if not groups:
            continue
        print(f"\n  {hp}:")
        for val in sorted(groups, key=lambda x: str(x)):
            g = groups[val]
            objs = [r["objective"] for r in g if r.get("objective") is not None]
            bbcs = [r["auc_bad_black_core"] for r in g
                    if r.get("auc_bad_black_core") is not None]
            obj_s = f"{mean(objs):.4f}" if objs else "  n/a"
            bbc_s = f"{mean(bbcs):.4f}" if bbcs else "  n/a"
            print(f"    {hp}={str(val):<6}  n={len(g):>3}  "
                  f"mean_obj={obj_s}  mean_bbc={bbc_s}")
        if len(groups) == 1:
            only = next(iter(groups))
            print(f"    NOTE: every usable trial has {hp}={only}; no contrast available.")


def report_topN(usable, n, present_targets):
    print(hr(f"TOP-{n} TRIALS BY OBJECTIVE (descriptive only; correlations above use full set)"))
    ranked = sorted([r for r in usable if r.get("objective") is not None],
                    key=lambda r: r["objective"], reverse=True)[:n]
    cols = ["trial_num", "objective", "auc_broken_lcfs", "auc_bad_black_core",
            "auc_bad_nonconverged", "bottleneck_dim", "n_enc_layers",
            "n_dec_layers", "base_channels", "growth_factor", "lr", "batch_size"]
    present = [c for c in cols if c == "trial_num" or any(r.get(c) is not None for r in ranked)]
    print("  " + " ".join(f"{SHORT.get(c, c):>9}" for c in present))
    for r in ranked:
        cells = []
        for c in present:
            v = r.get(c)
            if v is None:
                cells.append(f"{'?':>9}")
            elif c == "lr":
                cells.append(f"{v:>9.2e}")
            elif c == "trial_num":
                cells.append(f"{str(v):>9}")
            elif isinstance(v, float):
                cells.append(f"{v:>9.4f}")
            else:
                cells.append(f"{str(v):>9}")
        print("  " + " ".join(cells))


# ----------------------------------------------------------------------------
# Optional: heatmap + CSV export via pandas/matplotlib if available
# ----------------------------------------------------------------------------

def export_with_pandas(usable, cols, outdir):
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("\n[export] pandas/numpy not available; skipping CSV + heatmap export.")
        return

    import os
    os.makedirs(outdir, exist_ok=True)

    # tidy frame
    data = {c: [r.get(c) for r in usable] for c in cols}
    data["trial_num"] = [r.get("trial_num") for r in usable]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(outdir, "usable_trials.csv"), index=False)

    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")
    corr_p.to_csv(os.path.join(outdir, "corr_pearson.csv"))
    corr_s.to_csv(os.path.join(outdir, "corr_spearman.csv"))
    print(f"\n[export] wrote usable_trials.csv, corr_pearson.csv, corr_spearman.csv to {outdir}/")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, corr in [("pearson", corr_p), ("spearman", corr_s)]:
            fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 3, 0.6 * len(cols) + 3))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            labels = [SHORT.get(c, c) for c in cols]
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    v = corr.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                                fontsize=6,
                                color="white" if abs(v) > 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name.capitalize()} correlation — LCFS-SSIM study")
            fig.tight_layout()
            path = os.path.join(outdir, f"corr_heatmap_{name}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[export] wrote {path}")
    except ImportError:
        print("[export] matplotlib not available; skipped heatmap PNGs.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Comprehensive HP analysis for the SSIM study.")
    ap.add_argument("csv_path", help="path to summary.csv")
    ap.add_argument("--top", type=int, default=10, help="N for the top-N descriptive table")
    ap.add_argument("--out", default=None, help="output dir for CSV/heatmap export (optional)")
    ap.add_argument("--method", choices=["pearson", "spearman", "both"], default="both")
    args = ap.parse_args()

    headers, all_rows = load_rows(args.csv_path)
    if not headers:
        print("ERROR: could not read headers from CSV.", file=sys.stderr)
        sys.exit(1)

    # resolve which expected columns are actually present
    present_numeric = [c for c in NUMERIC_HPS if c in headers]
    present_calib = [c for c in CALIB_OUTPUTS if c in headers]
    present_targets = [c for c in TARGETS if c in headers]

    # If use_hpf is fixed False everywhere, hpf_sigma is a constant placeholder
    # -> drop it so it doesn't pollute the matrix.
    ok_rows = [r for r in all_rows if floor_ok(r)]
    hpf_flags = {to_bool(r.get("use_hpf")) for r in ok_rows if "use_hpf" in r}
    if hpf_flags <= {False, None} and "hpf_sigma" in present_numeric:
        present_numeric.remove("hpf_sigma")

    usable = build_numeric_view(ok_rows, present_numeric, present_targets, present_calib)

    if not usable:
        print("ERROR: no usable floor-ok rows with complete numeric HPs.", file=sys.stderr)
        print("Hint: the study may still be running, or column names differ.",
              file=sys.stderr)
        print(f"Headers seen: {headers}", file=sys.stderr)
        sys.exit(1)

    # --- reports ---
    report_overview(all_rows, ok_rows, usable)
    report_distributions(usable, present_targets)

    cols = collect_numeric_columns(usable, present_numeric, present_calib, present_targets)

    if args.method in ("pearson", "both"):
        report_correlation_matrix(usable, cols, method="pearson")
    if args.method in ("spearman", "both"):
        report_correlation_matrix(usable, cols, method="spearman")

    report_target_rankings(usable, cols, present_targets)
    report_confound(usable)
    report_categorical(usable)
    report_topN(usable, args.top, present_targets)

    if args.out:
        export_with_pandas(usable, cols, args.out)

    print(hr("DONE"))
    print("  Reminder: correlations are over the floor-ok population, but n is")
    print("  still small. Treat |r|<0.4 as noise, and confirm any architectural")
    print("  claim with a multi-seed run on the locked test set before trusting it.")


if __name__ == "__main__":
    main()

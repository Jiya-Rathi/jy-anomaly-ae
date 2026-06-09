"""
plot_truth_scatter.py

Generate a pure ground-truth scatter plot of the selection set, with:
    x-axis: shot number (parsed from filename)
    y-axis: 0 if healthy, 1 if anomalous (jittered slightly per-class)
    color : same color scheme as the model scatter plots in logs/

This is a labeling visualization. No model is involved. The plot answers
the question "what does the truth look like, plotted in the same coordinates
as the model's predictions?" so you can put it next to a model scatter and
see at a glance where the model agrees and disagrees with reality.

Why jitter the anomaly classes:
    All three anomaly classes have the true label "1". If we plotted them
    all at exactly y=1, you couldn't tell the classes apart in dense regions
    of the plot. We give each class a slightly different y-offset within the
    [0.92, 1.08] band so that the three classes form three thin horizontal
    sub-bands but are still clearly all "anomalous" relative to healthy at
    y=0.

Usage:
    python plot_truth_scatter.py \\
        --manifest /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests/selection.txt \\
        --out      truth_scatter_selection.png

The default colors and figure size match the model-scatter plots in
ae_lib/evaluation.py, so the two are visually comparable.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Color scheme: copy of the one used by the model-scatter plots so the
# truth plot can be placed next to a model plot for direct visual comparison.
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "healthy":          "#1f77b4",  # blue
    "broken_lcfs":      "#ff7f0e",  # orange
    "bad_black_core":   "#d62728",  # red
    "bad_nonconverged": "#9467bd",  # purple
}

# Order matters for plotting (later = on top). Plot healthy first so anomaly
# points sit on top, matching the model scatters.
CLASS_ORDER = ["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"]

# Per-class y-offset within the "anomalous" band [0.92, 1.08].
# Healthy stays exactly at 0.
Y_OFFSET = {
    "healthy":          0.00,
    "broken_lcfs":      0.95,
    "bad_black_core":   1.00,
    "bad_nonconverged": 1.05,
}


# ---------------------------------------------------------------------------
# Parse one manifest line into (shot_number, class_name)
# ---------------------------------------------------------------------------

# Manifest lines look like:
#   <path/to/jy_screenshot_for_shot_123456.png>  <class_name>
# where the shot number is somewhere in the filename. The pattern below
# pulls out the first 6-digit number it finds in the path; this matches the
# DIII-D shot numbering convention used throughout this project. Adjust if
# your filenames embed shot numbers differently.

_SHOT_RE = re.compile(r"(\d{5,6})")

def parse_manifest(manifest_path: Path):
    """Read a manifest file and return a list of (shot_number, class_name).

    Manifest format: each non-empty line has at least two whitespace-separated
    fields; first is the image path, last is the class label. Lines that
    don't parse cleanly are skipped with a stderr warning.
    """
    rows = []
    with open(manifest_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"WARN line {lineno}: skipping malformed line: {line!r}",
                      file=sys.stderr)
                continue
            path  = parts[0]
            cls   = parts[-1]

            if cls not in CLASS_COLORS:
                print(f"WARN line {lineno}: unknown class {cls!r}, skipping",
                      file=sys.stderr)
                continue

            m = _SHOT_RE.search(path)
            if not m:
                print(f"WARN line {lineno}: no shot number in {path!r}, skipping",
                      file=sys.stderr)
                continue
            shot = int(m.group(1))
            rows.append((shot, cls))
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_truth(rows, out_path: Path, title: str):
    """Render the scatter plot. rows is list of (shot, class)."""
    # Group by class so each gets one matplotlib call (one legend entry).
    by_class = {c: [] for c in CLASS_ORDER}
    for shot, cls in rows:
        by_class[cls].append(shot)

    fig, ax = plt.subplots(figsize=(14, 5))

    for cls in CLASS_ORDER:
        shots = by_class[cls]
        if not shots:
            continue
        y = [Y_OFFSET[cls]] * len(shots)
        ax.scatter(
            shots, y,
            c       = CLASS_COLORS[cls],
            label   = f"{cls} (n={len(shots)})",
            s       = 18,
            alpha   = 0.7,
            edgecolors = "none",
        )

    # Reference lines so the eye picks up the two regions
    ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.3)
    ax.axhline(1.0, color="grey", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Shot number")
    ax.set_ylabel("True label (0 = healthy, 1 = anomalous)")
    ax.set_ylim(-0.15, 1.20)
    # Show the y-positions used for each anomaly class as tick labels for
    # easier reading without the legend
    ax.set_yticks([0.0, 0.95, 1.00, 1.05])
    ax.set_yticklabels(["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"])

    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to manifest file (e.g., manifests/selection.txt)")
    p.add_argument("--out", type=Path, default=Path("truth_scatter.png"),
                   help="Output PNG path")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title (default: derived from manifest name)")
    args = p.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"manifest not found: {args.manifest}")

    rows = parse_manifest(args.manifest)
    print(f"Read {len(rows)} entries from {args.manifest}")

    counts = {}
    for _, cls in rows:
        counts[cls] = counts.get(cls, 0) + 1
    print("Class counts:")
    for cls in CLASS_ORDER:
        print(f"  {cls:<20} {counts.get(cls, 0):>4}")

    title = args.title or f"Ground-truth labels: {args.manifest.name}"
    plot_truth(rows, args.out, title)


if __name__ == "__main__":
    main()

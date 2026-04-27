"""
split_manifests.py

Generates train/val/selection manifest files from the labeled image pool.

Reads images from:
    /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/<class>/

Writes three manifest files to:
    /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests/
        train.txt       722 healthy  (AE gradient updates)
        val.txt         120 healthy  (AE early-stopping validation)
        selection.txt   405 mixed    (Optuna objective scoring)

Each manifest line has the format:
    <relative_path>\t<class>
Example:
    healthy/jy_187138_3000.png\thealthy

Safety features:
    - Fixed random seed (reproducible)
    - Refuses to run if any manifest file already exists
    - Prints a plan and asks for confirmation before writing
    - Verifies that no image appears in more than one manifest
    - Verifies that no image in any manifest is also in final_test_set
"""

import os
import random
import sys
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SOURCE_ROOT    = Path("/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots")
TEST_ROOT      = Path("/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/final_test_set")
MANIFEST_ROOT  = Path("/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests")

# Healthy split: train / val / selection
#   Total healthy available: 941 (one duplicate was removed earlier)
N_HEALTHY_TRAIN = 722
N_HEALTHY_VAL   = 120
N_HEALTHY_SEL   = 99      # healthy images in the selection set

# Anomalies: ALL remaining anomaly images go into the selection set
#   broken_lcfs:      149
#   bad_black_core:   142
#   bad_nonconverged:  15

RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def list_images(folder: Path):
    """Return sorted list of .png files in folder."""
    if not folder.is_dir():
        print(f"ERROR: expected folder does not exist: {folder}")
        sys.exit(1)
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix == ".png")


def rel(path: Path) -> str:
    """Return path relative to SOURCE_ROOT, as a string."""
    return str(path.relative_to(SOURCE_ROOT))


def load_test_set_names() -> set:
    """Collect all image filenames that are in final_test_set (for the leak check)."""
    names = set()
    if not TEST_ROOT.is_dir():
        print(f"WARNING: {TEST_ROOT} does not exist -- cannot verify test-set isolation")
        return names
    for cls_dir in TEST_ROOT.iterdir():
        if cls_dir.is_dir():
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix == ".png":
                    names.add(f.name)
    return names


# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------

def preflight():
    # Source must exist
    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: source directory does not exist: {SOURCE_ROOT}")
        sys.exit(1)

    # No manifest file may already exist (prevents accidental overwrite)
    for name in ["train.txt", "val.txt", "selection.txt"]:
        f = MANIFEST_ROOT / name
        if f.exists():
            print(f"ERROR: {f} already exists.")
            print("       Refusing to run -- this prevents silently overwriting")
            print("       an existing split. Delete the manifest files first if")
            print("       you really want to regenerate.")
            sys.exit(1)

    # Count what's in each class folder
    print("Pre-flight check: images available per class")
    print("-" * 60)
    counts = {}
    for cls in ["healthy", "broken_lcfs", "bad_black_core", "bad_nonconverged"]:
        files = list_images(SOURCE_ROOT / cls)
        counts[cls] = files
        print(f"  {cls:20s}: {len(files):4d} images")
    print("-" * 60)

    # Sanity check: do we have enough healthy images for all three splits?
    need_healthy = N_HEALTHY_TRAIN + N_HEALTHY_VAL + N_HEALTHY_SEL
    have_healthy = len(counts["healthy"])
    if have_healthy < need_healthy:
        print(f"ERROR: need {need_healthy} healthy images but only have {have_healthy}")
        sys.exit(1)
    if have_healthy > need_healthy:
        print(f"NOTE: you have {have_healthy} healthy images, only {need_healthy} "
              f"will be used ({have_healthy - need_healthy} unused).")
    print()
    return counts


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  Manifest Split Script (train / val / selection)")
    print("=" * 60)
    print(f"  Source:    {SOURCE_ROOT}")
    print(f"  Test set:  {TEST_ROOT}")
    print(f"  Manifests: {MANIFEST_ROOT}")
    print(f"  Seed:      {RANDOM_SEED}")
    print("=" * 60)

    counts = preflight()

    # Seed once so every random draw below is reproducible
    random.seed(RANDOM_SEED)

    # --- Healthy split: shuffle once, then slice into train / val / selection
    healthy_shuffled = counts["healthy"][:]              # copy
    random.shuffle(healthy_shuffled)

    healthy_train = healthy_shuffled[:N_HEALTHY_TRAIN]
    healthy_val   = healthy_shuffled[N_HEALTHY_TRAIN : N_HEALTHY_TRAIN + N_HEALTHY_VAL]
    healthy_sel   = healthy_shuffled[N_HEALTHY_TRAIN + N_HEALTHY_VAL :
                                     N_HEALTHY_TRAIN + N_HEALTHY_VAL + N_HEALTHY_SEL]

    # --- Anomalies: all go to selection
    sel_broken_lcfs      = counts["broken_lcfs"][:]
    sel_bad_black_core   = counts["bad_black_core"][:]
    sel_bad_nonconverged = counts["bad_nonconverged"][:]

    # --- Show the plan
    print("Plan:")
    print(f"  train.txt:      {len(healthy_train):4d} healthy")
    print(f"  val.txt:        {len(healthy_val):4d} healthy")
    print(f"  selection.txt:  {len(healthy_sel):4d} healthy")
    print(f"                  {len(sel_broken_lcfs):4d} broken_lcfs")
    print(f"                  {len(sel_bad_black_core):4d} bad_black_core")
    print(f"                  {len(sel_bad_nonconverged):4d} bad_nonconverged")
    total_sel = (len(healthy_sel) + len(sel_broken_lcfs)
                 + len(sel_bad_black_core) + len(sel_bad_nonconverged))
    print(f"                  -----")
    print(f"                  {total_sel:4d} total selection")
    print()

    answer = input("Type 'yes' to write manifests, anything else to cancel: ").strip().lower()
    if answer != "yes":
        print("Cancelled. No manifests were written.")
        sys.exit(0)

    # --- Leak check: make sure no image we're about to list appears in test set
    test_names = load_test_set_names()
    print()
    print("Running leak check against final_test_set...")
    all_selected = (healthy_train + healthy_val + healthy_sel
                    + sel_broken_lcfs + sel_bad_black_core + sel_bad_nonconverged)
    leaks = [f for f in all_selected if f.name in test_names]
    if leaks:
        print(f"ERROR: found {len(leaks)} images that are also in final_test_set!")
        for f in leaks[:10]:
            print(f"  {f.name}")
        print("Aborting -- do NOT proceed. This indicates the folders are not")
        print("properly isolated. Check that final_test_set is OUTSIDE of")
        print("AEModel_jy_screenshots/ before re-running.")
        sys.exit(1)
    print(f"  No leaks found ({len(all_selected)} manifest entries checked).")

    # --- Duplicate check: make sure no image appears twice across manifests
    print("Running duplicate check across manifests...")
    seen = {}
    duplicates = []
    for bucket_name, bucket in [
        ("train", healthy_train),
        ("val", healthy_val),
        ("selection", healthy_sel + sel_broken_lcfs
                      + sel_bad_black_core + sel_bad_nonconverged),
    ]:
        for f in bucket:
            if f.name in seen:
                duplicates.append((f.name, seen[f.name], bucket_name))
            else:
                seen[f.name] = bucket_name
    if duplicates:
        print(f"ERROR: found {len(duplicates)} duplicate entries across manifests")
        for name, b1, b2 in duplicates[:10]:
            print(f"  {name} in both {b1} and {b2}")
        sys.exit(1)
    print(f"  No duplicates found.")

    # --- Write the manifests
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    def write_manifest(path: Path, entries: list, header_note: str):
        lines = [f"# {header_note}",
                 f"# generated: {datetime.now().isoformat()}",
                 f"# seed: {RANDOM_SEED}",
                 f"# count: {len(entries)}",
                 "# format: <relative_path>\\t<class>"]
        for f, cls in entries:
            lines.append(f"{rel(f)}\t{cls}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    print()
    print("Writing manifests...")

    train_entries = [(f, "healthy") for f in healthy_train]
    val_entries   = [(f, "healthy") for f in healthy_val]
    sel_entries = ([(f, "healthy")          for f in healthy_sel]
                 + [(f, "broken_lcfs")      for f in sel_broken_lcfs]
                 + [(f, "bad_black_core")   for f in sel_bad_black_core]
                 + [(f, "bad_nonconverged") for f in sel_bad_nonconverged])

    # Sort entries for human readability (doesn't affect training)
    train_entries.sort()
    val_entries.sort()
    sel_entries.sort()

    write_manifest(MANIFEST_ROOT / "train.txt",
                   train_entries,
                   "AE training set (healthy only, gradient updates)")
    write_manifest(MANIFEST_ROOT / "val.txt",
                   val_entries,
                   "AE validation set (healthy only, early stopping)")
    write_manifest(MANIFEST_ROOT / "selection.txt",
                   sel_entries,
                   "Optuna selection set (healthy + all anomaly classes)")

    # --- Final summary
    print()
    print("=" * 60)
    print("  Done. Manifests written:")
    print("=" * 60)
    for name in ["train.txt", "val.txt", "selection.txt"]:
        path = MANIFEST_ROOT / name
        with open(path) as fh:
            n_lines = sum(1 for ln in fh if not ln.startswith("#"))
        print(f"  {path}  ({n_lines} entries)")
    print()


if __name__ == "__main__":
    main()
"""
isolate_test_set.py

Isolates a held-out test set of 190 jy plot images from the labeled dataset.

Moves images from:
    /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/<class>/
To:
    /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/final_test_set/<class>/

Also writes:
    final_test_set/test.txt        -- list of shots, one per line, with class label
    final_test_set/isolation_log.txt -- full record of what was moved

Safety features:
    - Fixed random seed (reproducible)
    - Refuses to run if test.txt already exists
    - Asks for confirmation before moving anything
    - Prints before/after counts
"""

import os
import random
import shutil
import sys
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SOURCE_ROOT = Path("/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots")
TEST_ROOT   = Path("/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/final_test_set")

TEST_COUNTS = {
    "healthy":          120,
    "broken_lcfs":       30,
    "bad_black_core":    30,
    "bad_nonconverged":  10,
}

RANDOM_SEED = 42  # keep fixed so the selection is reproducible

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def list_images(folder: Path):
    """Return sorted list of .png files in folder."""
    if not folder.is_dir():
        print(f"ERROR: expected folder does not exist: {folder}")
        sys.exit(1)
    files = sorted(f for f in folder.iterdir()
                   if f.is_file() and f.suffix == ".png")
    return files


def shot_from_filename(filename: str) -> str:
    """Extract <shot>_3000 from jy_<shot>_3000.png."""
    stem = Path(filename).stem         # jy_187138_3000
    parts = stem.split("_", 1)         # ['jy', '187138_3000']
    if len(parts) != 2:
        return stem
    return parts[1]                    # '187138_3000'


# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

def preflight():
    # Source must exist
    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: source directory does not exist: {SOURCE_ROOT}")
        sys.exit(1)

    # test.txt must NOT already exist (prevents accidental re-run)
    test_txt = TEST_ROOT / "test.txt"
    if test_txt.exists():
        print(f"ERROR: {test_txt} already exists.")
        print("       Refusing to run -- this prevents corrupting an existing split.")
        print("       If you really want to regenerate, delete test.txt and")
        print("       move the test images back first.")
        sys.exit(1)

    # Make sure every class folder exists and has enough images
    print("Pre-flight check: counting available images per class")
    print("-" * 60)
    available = {}
    for cls, need in TEST_COUNTS.items():
        folder = SOURCE_ROOT / cls
        files = list_images(folder)
        available[cls] = files
        status = "OK" if len(files) >= need else "NOT ENOUGH"
        print(f"  {cls:20s}: {len(files):4d} available, {need:3d} needed  [{status}]")
        if len(files) < need:
            print(f"ERROR: class '{cls}' has only {len(files)} images but needs {need}")
            sys.exit(1)
    print("-" * 60)
    return available


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  Test Set Isolation Script")
    print("=" * 60)
    print(f"  Source: {SOURCE_ROOT}")
    print(f"  Dest:   {TEST_ROOT}")
    print(f"  Seed:   {RANDOM_SEED}")
    print("=" * 60)
    print()

    available = preflight()

    # Seed the random generator and pick files
    random.seed(RANDOM_SEED)
    selected = {}
    for cls, need in TEST_COUNTS.items():
        selected[cls] = random.sample(available[cls], need)

    # Show the plan and ask for confirmation
    print()
    print("Plan:")
    total = 0
    for cls, files in selected.items():
        print(f"  Move {len(files):3d} files from {cls}")
        total += len(files)
    print(f"  Total: {total} files")
    print()
    print(f"Destination folders will be created under:")
    print(f"  {TEST_ROOT}")
    print()

    answer = input("Type 'yes' to proceed, anything else to cancel: ").strip().lower()
    if answer != "yes":
        print("Cancelled. No files were moved.")
        sys.exit(0)

    # Create destination folders
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    for cls in TEST_COUNTS:
        (TEST_ROOT / cls).mkdir(exist_ok=True)

    # Move files and build the log
    test_txt_lines = []
    log_lines = []
    log_lines.append(f"Test set isolation log")
    log_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    log_lines.append(f"Seed: {RANDOM_SEED}")
    log_lines.append(f"Source: {SOURCE_ROOT}")
    log_lines.append(f"Dest:   {TEST_ROOT}")
    log_lines.append("")

    print()
    print("Moving files...")
    for cls, files in selected.items():
        log_lines.append(f"[{cls}]  {len(files)} files")
        for src in files:
            dst = TEST_ROOT / cls / src.name
            shutil.move(str(src), str(dst))
            shot = shot_from_filename(src.name)
            test_txt_lines.append(f"{shot}\t{cls}")
            log_lines.append(f"  {src.name} -> {dst}")
        log_lines.append("")
        print(f"  {cls:20s}: moved {len(files)} files")

    # Write test.txt (sorted by shot for readability)
    test_txt_lines.sort()
    test_txt = TEST_ROOT / "test.txt"
    with open(test_txt, "w") as f:
        f.write("# shot\tclass\n")
        f.write("\n".join(test_txt_lines) + "\n")

    # Write the log
    log_file = TEST_ROOT / "isolation_log.txt"
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # Final summary: show counts after the move
    print()
    print("=" * 60)
    print("  After move -- source folder counts:")
    print("=" * 60)
    for cls in TEST_COUNTS:
        remaining = len(list_images(SOURCE_ROOT / cls))
        moved     = TEST_COUNTS[cls]
        print(f"  {cls:20s}: {remaining:4d} remaining in source  ({moved} moved out)")
    print()
    print(f"  test.txt           -> {test_txt}")
    print(f"  isolation_log.txt  -> {log_file}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
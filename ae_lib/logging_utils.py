"""
ae_lib/logging_utils.py

Logging helpers for training runs.

Three things get produced per trial:
    logs/trial_<N>.log   human-readable text log
    logs/trial_<N>.csv   epoch-level metrics for post-hoc analysis
    (scatter plot is handled by evaluation.py)

Plus one study-level file shared across all trials:
    logs/summary.csv     one row per completed trial

Public API:
    make_trial_loggers(log_path, csv_path, to_stdout=False)
        -> (log_fn, csv_row_fn, close_fn)
        Returns two callbacks to pass into training.train_model,
        plus a close_fn to call when training is done.

    append_summary_row(summary_path, row_dict, fieldnames)
        Atomically append one row to the study-level summary CSV.
        Handles file creation + header + concurrency-safe locking.
"""

import csv
import fcntl
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple


# -----------------------------------------------------------------------------
# Per-trial loggers
# -----------------------------------------------------------------------------

def make_trial_loggers(
    log_path,
    csv_path,
    to_stdout: bool = False,
) -> Tuple[Callable[[str], None], Callable[[dict], None], Callable[[], None]]:
    """Create logger callbacks for one training run.

    Parameters
    ----------
    log_path : path to the text log file
    csv_path : path to the CSV metrics file
    to_stdout : if True, also print every log line to stdout (useful for CLI
                runs; leave False for parallel Optuna workers to avoid
                interleaved output).

    Returns
    -------
    (log_fn, csv_row_fn, close_fn)
        log_fn(str)        -- append one line (with timestamp) to the log.
        csv_row_fn(dict)   -- append one row to the CSV.
        close_fn()         -- flush and close file handles.
    """
    log_path = Path(log_path)
    csv_path = Path(csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Open both files up front; keep them open so writes are cheap.
    log_fh = open(log_path, "a", buffering=1)    # line buffered
    csv_fh = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_fh)

    # Track whether the CSV needs a header
    csv_needs_header = csv_path.stat().st_size == 0

    def log_fn(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        log_fh.write(line + "\n")
        log_fh.flush()
        if to_stdout:
            print(line, flush=True)

    def csv_row_fn(row: dict) -> None:
        nonlocal csv_needs_header
        if csv_needs_header:
            csv_writer.writerow(list(row.keys()))
            csv_needs_header = False
        csv_writer.writerow(list(row.values()))
        csv_fh.flush()

    def close_fn() -> None:
        try:
            log_fh.close()
        except Exception:
            pass
        try:
            csv_fh.close()
        except Exception:
            pass

    return log_fn, csv_row_fn, close_fn


# -----------------------------------------------------------------------------
# Study-level summary
# -----------------------------------------------------------------------------

def append_summary_row(
    summary_path,
    row: dict,
    fieldnames: List[str],
) -> None:
    """Append one row to the study-wide summary CSV, safely.

    Uses an advisory file lock (fcntl.flock) so multiple parallel Optuna
    workers can append without corrupting the file.

    Writes a header row the first time the file is created.
    If a row's dict is missing fields, they are written as empty strings.
    If a row has extra fields, they are silently dropped (fieldnames is the
    source of truth, not the dict).
    """
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode; create if missing.
    # Use a separate lock file so we hold the lock across header write too.
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        write_header = not summary_path.is_file() or summary_path.stat().st_size == 0
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            # Ensure all fieldnames are present in the row; missing -> ""
            clean = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean)
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fh.close()


# -----------------------------------------------------------------------------
# Default fieldnames for summary.csv
# -----------------------------------------------------------------------------
# This is the canonical column ordering for the per-trial summary row.
# train.py and tune.py should import this to keep columns consistent.

SUMMARY_FIELDS: List[str] = [
    "trial_num",
    "timestamp",
    "seed",
    "stop_reason",
    "total_epochs",
    "best_epoch",
    "best_val_loss",
    "objective",
    "floor_ok",
    "auc_broken_lcfs",
    "auc_bad_black_core",
    "auc_bad_nonconverged",
    "threshold",
    "mu",
    "sigma",
    # Hyperparameters (echoed for quick scanning)
    "use_hpf",
    "hpf_sigma",
    "bottleneck_dim",
    "n_enc_layers",
    "n_dec_layers",
    "base_channels",
    "growth_factor",
    "use_batchnorm",
    "lr",
    "batch_size",
    # Paths
    "checkpoint_path",
    "log_path",
    "csv_path",
    "scatter_path",
]

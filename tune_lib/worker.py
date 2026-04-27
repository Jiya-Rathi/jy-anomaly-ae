"""
tune_lib/worker.py

The single-worker entry point. One subprocess runs one instance of this.

Each worker:
    1. Uses whatever GPU CUDA_VISIBLE_DEVICES has restricted it to
       (always cuda:0 from inside the subprocess).
    2. Loads the shared SQLite-backed study.
    3. Runs study.optimize() for the requested number of trials.

Important: this module expects CUDA_VISIBLE_DEVICES to already be set
by the parent process (tune.py) BEFORE the subprocess was spawned.
Don't set it here -- by the time torch is imported, CUDA has already
picked which GPUs it sees.

Usage (called by tune.py as a subprocess):
    python -m tune_lib.worker \
        --worker-id  0 \
        --gpu-id     0 \
        --study-name ae_topology_v1 \
        --study-db   study_outputs/study.db \
        --output-dir study_outputs \
        --n-trials   25
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Imports below must happen AFTER CUDA_VISIBLE_DEVICES is set by the parent.
# If this module is imported directly (not as a subprocess), that's the
# caller's responsibility.

from tune_lib.objective import make_objective
from tune_lib.study     import create_or_load_study


# -----------------------------------------------------------------------------
# Worker logging
# -----------------------------------------------------------------------------

def _worker_log(worker_id: int, msg: str) -> None:
    """Prefix messages so parallel workers' output is distinguishable."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[worker {worker_id} | {ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run_worker(
    worker_id:  int,
    gpu_id:     int,
    study_name: str,
    study_db:   str,
    output_dir: str,
    n_trials:   int,
) -> int:
    """Run Optuna trials until n_trials have been completed, then exit.

    Returns 0 on clean exit, 1 on setup failure.
    """
    def log(msg: str) -> None:
        _worker_log(worker_id, msg)

    log(f"Starting with gpu_id={gpu_id}, n_trials={n_trials}")
    log(f"Study: name={study_name!r}, db={study_db}")
    log(f"Output dir: {output_dir}")

    # FIX 1: Defensively set HIP_VISIBLE_DEVICES inside the worker too.
    # The parent (tune.py) sets it when spawning, but if SLURM's process
    # management drops the env var during the chain, this is our backup.
    # MUST happen before `import torch`.
    import os
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    log(f"HIP_VISIBLE_DEVICES set to {gpu_id}")

    # Sanity-check CUDA visibility. From within the subprocess, we should
    # see exactly one GPU (the one HIP_VISIBLE_DEVICES restricted us to).
    import torch
    visible = torch.cuda.device_count()
    log(f"torch sees {visible} GPU(s); cuda_available={torch.cuda.is_available()}")

    # FIX 2: Hard fail if no GPU is visible. Better to crash this worker
    # immediately than silently fall back to CPU and waste hours.
    if not torch.cuda.is_available() or visible < 1:
        log(f"FATAL: no GPU visible to this worker. "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}")
        return 1
    if visible != 1:
        log(f"WARNING: expected exactly 1 GPU, saw {visible}. "
            f"Continuing but device selection may be wrong.")

    # FIX 3: Cap CPU thread usage. Without this, each PyTorch worker
    # tries to use all 192 cores; with 4 workers that's 768 threads
    # competing for 192 cores -- thrashing.
    # 4 threads/worker x 4 workers = 16 cores, matches --cpus-per-task=16.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    log(f"Capped torch threads: num_threads=4, interop=2")

    # Import train_one_model now (so any errors show up with a clear message).
    # Passing the function in as a dependency keeps tune_lib Optuna-only,
    # independent of train.py's import path.
    try:
        from train import train_one_model
    except Exception as e:
        log(f"FATAL: failed to import train.train_one_model: {e}")
        return 1

    # Load/create the shared study (workers other than the first will
    # find it already created and resume).
    try:
        study = create_or_load_study(
            study_name = study_name,
            study_db   = study_db,
            log        = log,
        )
    except Exception as e:
        log(f"FATAL: failed to create/load study: {e}")
        return 1

    # Build the objective closure for this worker's GPU
    objective = make_objective(
        gpu_id          = 0,                   # always cuda:0 from child's POV
        output_dir      = output_dir,
        train_one_model = train_one_model,
    )

    # Run trials. Optuna handles concurrency against the shared SQLite DB.
    # catch=() means unhandled exceptions during a trial are logged by
    # Optuna (trial marked FAILED) but don't crash the worker.
    log(f"Entering study.optimize for {n_trials} trials")
    try:
        study.optimize(
            objective,
            n_trials     = n_trials,
            catch        = (Exception,),     # failed trial != dead worker
            show_progress_bar = False,        # would clutter logs with many workers
        )
    except KeyboardInterrupt:
        log("KeyboardInterrupt -- stopping cleanly")
        return 0
    except Exception as e:
        log(f"FATAL: study.optimize raised: {e}")
        return 1

    # Final status
    all_trials = study.trials
    n_complete = sum(1 for t in all_trials if t.state.name == "COMPLETE")
    n_pruned   = sum(1 for t in all_trials if t.state.name == "PRUNED")
    n_failed   = sum(1 for t in all_trials if t.state.name == "FAIL")
    log(f"Worker done. Study totals: "
        f"complete={n_complete}, pruned={n_pruned}, failed={n_failed}, "
        f"all={len(all_trials)}")
    return 0


# -----------------------------------------------------------------------------
# CLI (called by tune.py as 'python -m tune_lib.worker --...')
# -----------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="One Optuna worker; spawned by tune.py as a subprocess.",
    )
    p.add_argument("--worker-id",  type=int, required=True,
                   help="Worker index (0..N-1), used for logging only.")
    p.add_argument("--gpu-id",     type=int, required=True,
                   help="GPU index as known to the parent "
                        "(logged only; child always uses cuda:0).")
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument("--study-db",   type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-trials",   type=int, required=True,
                   help="Number of trials this worker should run.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rc = run_worker(
        worker_id  = args.worker_id,
        gpu_id     = args.gpu_id,
        study_name = args.study_name,
        study_db   = args.study_db,
        output_dir = args.output_dir,
        n_trials   = args.n_trials,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

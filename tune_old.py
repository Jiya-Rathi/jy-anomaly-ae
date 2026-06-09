"""
tune.py

Top-level Optuna study orchestrator.

Creates a shared Optuna study (SQLite-backed, resumable) and spawns N
parallel worker subprocesses, each pinned to one GPU. Each worker runs
study.optimize() against the shared study until its share of trials is
complete.

Usage:
    python tune.py \\
        --study-name ae_topology_v1 \\
        --study-db   study_outputs/study.db \\
        --output-dir study_outputs \\
        --n-trials   100 \\
        --n-workers  4

Design (see research notes + design discussion):
    - Workers are SUBPROCESSES (not threads): one crash = one dead trial,
      not a dead study.
    - Each worker sees exactly one GPU via HIP_VISIBLE_DEVICES.
      (CUDA_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES also work in
      isolation on ROCm 6.3, but setting all three together can break
      GPU detection -- so we set just the one.)
    - The study is resumable: re-submitting this script with the same
      --study-db picks up where a crashed run left off.
    - No warm-start: Optuna's first 10 trials are pure random exploration.
      The old baseline config should be run separately via train.py for
      scientific comparison.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from tune_lib.study import create_or_load_study


# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print("=" * 70, flush=True)
    print(f"  {msg}",    flush=True)
    print("=" * 70, flush=True)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[tune {ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Worker spawning
# -----------------------------------------------------------------------------

def _spawn_worker(
    worker_id:  int,
    gpu_id:     int,
    study_name: str,
    study_db:   str,
    output_dir: str,
    n_trials:   int,
    log_dir:    Path,
) -> subprocess.Popen:
    """Spawn one worker subprocess with its GPU visibility restricted."""

    # Restrict GPU visibility via HIP_VISIBLE_DEVICES (AMD's native name).
    # Only the assigned GPU is visible; from the child's POV it's cuda:0.
    # Note: setting CUDA_VISIBLE_DEVICES + HIP_VISIBLE_DEVICES + ROCR_VISIBLE_DEVICES
    # together breaks GPU detection on ROCm 6.3 (verified empirically).
    # Any one of them works in isolation; we pick HIP for its AMD-native name.
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    # Defensively unset the others in case the user has them set in their shell.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)

    # Capture each worker's stdout/stderr to its own log file so the parent
    # log stays readable. Tail these if you want live progress per-worker.
    worker_log  = log_dir / f"worker_{worker_id}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tune_lib.worker",
        "--worker-id",  str(worker_id),
        "--gpu-id",     str(gpu_id),
        "--study-name", study_name,
        "--study-db",   study_db,
        "--output-dir", output_dir,
        "--n-trials",   str(n_trials),
    ]

    _log(f"Spawning worker {worker_id} on GPU {gpu_id}, "
         f"n_trials={n_trials}, log={worker_log}")

    # Line-buffered so `tail -f` on the worker log shows progress live.
    log_fh = open(worker_log, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    # Keep a reference to log_fh so we can close it on join
    proc._log_fh = log_fh          # type: ignore[attr-defined]
    proc._worker_id = worker_id    # type: ignore[attr-defined]
    return proc


def _wait_for_workers(procs: List[subprocess.Popen]) -> int:
    """Wait for all workers to exit. Returns number of non-zero exits."""
    failures = 0
    done = [False] * len(procs)

    while not all(done):
        for i, p in enumerate(procs):
            if done[i]:
                continue
            rc = p.poll()
            if rc is None:
                continue
            done[i] = True
            wid = getattr(p, "_worker_id", i)
            if rc == 0:
                _log(f"Worker {wid} finished cleanly")
            else:
                _log(f"Worker {wid} exited with code {rc}")
                failures += 1
            fh = getattr(p, "_log_fh", None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        if not all(done):
            time.sleep(2.0)

    return failures


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Launch the Optuna AE topology study.",
    )
    p.add_argument("--study-name", type=str, required=True,
                   help="Identifier for the study (e.g., 'ae_topology_v1').")
    p.add_argument("--study-db",   type=str, required=True,
                   help="Path to the SQLite file (created if missing).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Root for checkpoints/, logs/, configs/, summary.csv.")
    p.add_argument("--n-trials",   type=int, required=True,
                   help="Total number of trials for the whole study.")
    p.add_argument("--n-workers",  type=int, default=4,
                   help="Number of parallel workers (GPUs). Default: 4.")
    p.add_argument("--gpu-ids",    type=int, nargs="+", default=None,
                   help="Which GPU indices to use (default: 0..n_workers-1).")
    args = p.parse_args(argv)

    _banner("AE topology optimization study")
    _log(f"Study name : {args.study_name}")
    _log(f"Study DB   : {args.study_db}")
    _log(f"Output dir : {args.output_dir}")
    _log(f"Total trials: {args.n_trials}")
    _log(f"Workers    : {args.n_workers}")

    # Resolve GPU IDs
    if args.gpu_ids is None:
        gpu_ids = list(range(args.n_workers))
    else:
        if len(args.gpu_ids) != args.n_workers:
            _log(f"ERROR: --n-workers={args.n_workers} but "
                 f"got {len(args.gpu_ids)} GPU IDs")
            return 1
        gpu_ids = args.gpu_ids
    _log(f"GPU IDs    : {gpu_ids}")

    # Create (or load) the shared study in the parent process. This also
    # writes the initial state to the SQLite file so all workers can load it.
    output_dir = Path(args.output_dir)
    log_dir    = output_dir / "logs"
    _log("Creating/loading the study (parent process)...")
    study = create_or_load_study(
        study_name = args.study_name,
        study_db   = args.study_db,
        log        = _log,
    )

    # How many trials are already done?
    done_states = {"COMPLETE", "PRUNED", "FAIL"}
    n_done = sum(1 for t in study.trials if t.state.name in done_states)
    n_remaining = max(0, args.n_trials - n_done)
    _log(f"Trials already on disk: {n_done} "
         f"(target: {args.n_trials}, remaining: {n_remaining})")

    if n_remaining == 0:
        _log("Study already at target trial count -- nothing to do")
        return 0

    # Distribute remaining trials across workers.
    # If remaining=97 and workers=4: workers get 25, 24, 24, 24 (total 97).
    base   = n_remaining // args.n_workers
    extras = n_remaining %  args.n_workers
    per_worker = [base + (1 if i < extras else 0) for i in range(args.n_workers)]
    _log(f"Trials per worker: {per_worker}")

    # We don't need the study object in the parent anymore -- Optuna will
    # re-open it in each subprocess. Drop the handle so SQLite's connection
    # from the parent doesn't linger.
    del study

    # Spawn all workers
    procs: List[subprocess.Popen] = []
    for wid in range(args.n_workers):
        if per_worker[wid] == 0:
            continue                # nothing for this worker to do
        proc = _spawn_worker(
            worker_id  = wid,
            gpu_id     = gpu_ids[wid],
            study_name = args.study_name,
            study_db   = args.study_db,
            output_dir = args.output_dir,
            n_trials   = per_worker[wid],
            log_dir    = log_dir,
        )
        procs.append(proc)

    _log(f"All {len(procs)} workers spawned. Waiting for completion...")
    _log(f"(tail log_dir/worker_<N>.log for live per-worker progress)")

    # Wait for all workers to exit
    failures = _wait_for_workers(procs)

    # Final summary -- re-open the study to read final state
    study = create_or_load_study(
        study_name = args.study_name,
        study_db   = args.study_db,
        log        = lambda _msg: None,   # quiet re-open
    )
    all_trials  = study.trials
    n_complete  = sum(1 for t in all_trials if t.state.name == "COMPLETE")
    n_pruned    = sum(1 for t in all_trials if t.state.name == "PRUNED")
    n_failed    = sum(1 for t in all_trials if t.state.name == "FAIL")

    _banner("Study finished")
    _log(f"Total trials in DB : {len(all_trials)}")
    _log(f"  completed        : {n_complete}")
    _log(f"  pruned           : {n_pruned}")
    _log(f"  failed           : {n_failed}")
    _log(f"Worker non-zero exits: {failures}")

    if n_complete > 0:
        best = study.best_trial
        _log(f"Best trial: #{best.number} with objective={best.value:.4f}")
        _log(f"  params: {best.params}")
        _log(f"  attrs : "
             f"lcfs={best.user_attrs.get('auc_broken_lcfs', 'n/a')}, "
             f"bc={best.user_attrs.get('auc_bad_black_core', 'n/a')}, "
             f"nc={best.user_attrs.get('auc_bad_nonconverged', 'n/a')}")

    _log(f"Study DB   : {args.study_db}")
    _log(f"Summary CSV: {Path(args.output_dir) / 'summary.csv'}")
    _log(f"Dashboard  : optuna-dashboard sqlite:///{Path(args.study_db).resolve()}")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

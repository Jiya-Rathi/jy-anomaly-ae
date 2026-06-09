"""
tune_lib/worker.py

The single-worker entry point. One subprocess runs one instance of this.

Each worker:
    1. Uses whatever GPU HIP_VISIBLE_DEVICES has restricted it to
       (always cuda:0 from inside the subprocess).
    2. Loads the shared SQLite-backed study.
    3. Runs study.optimize() for the requested number of trials.

Important: this module expects the parent process (tune.py) to pass the desired
objective type and to set the GPU visibility environment before this subprocess
is spawned.
"""

import argparse
import sys
from datetime import datetime

from tune_lib.study import create_or_load_study


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
    worker_id: int,
    gpu_id: int,
    study_name: str,
    study_db: str,
    output_dir: str,
    n_trials: int,
    objective_type: str = "default",
) -> int:
    """Run Optuna trials until n_trials have been completed, then exit.

    Returns 0 on clean exit, 1 on setup failure.
    """
    def log(msg: str) -> None:
        _worker_log(worker_id, msg)

    log(
        f"Starting with gpu_id={gpu_id}, n_trials={n_trials}, "
        f"objective={objective_type!r}"
    )
    log(f"Study: name={study_name!r}, db={study_db}")
    log(f"Output dir: {output_dir}")

    # Defensively set HIP_VISIBLE_DEVICES inside the worker too.
    # This must happen before importing torch.
    import os
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    log(f"HIP_VISIBLE_DEVICES set to {gpu_id}")

    # Sanity-check CUDA visibility. From within the subprocess, we should
    # see exactly one GPU, the one HIP_VISIBLE_DEVICES restricted us to.
    import torch
    visible = torch.cuda.device_count()
    log(f"torch sees {visible} GPU(s); cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available() or visible < 1:
        log(
            "FATAL: no GPU visible to this worker. "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}"
        )
        return 1

    if visible != 1:
        log(
            f"WARNING: expected exactly 1 GPU, saw {visible}. "
            "Continuing but device selection may be wrong."
        )

    # Cap CPU thread usage to avoid oversubscription with multiple workers.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    log("Capped torch threads: num_threads=4, interop=2")

    try:
        from train import train_one_model
    except Exception as e:
        log(f"FATAL: failed to import train.train_one_model: {e}")
        return 1

    try:
        study = create_or_load_study(
            study_name=study_name,
            study_db=study_db,
            log=log,
        )
    except Exception as e:
        log(f"FATAL: failed to create/load study: {e}")
        return 1

    # Build the objective closure for this worker's GPU.
    try:
        if objective_type == "lcfs_soft":
            from tune_lib.objective_lcfs_soft import make_objective_lcfs_soft
            objective = make_objective_lcfs_soft(
                gpu_id=0,  # always cuda:0 from child's POV
                output_dir=output_dir,
                train_one_model=train_one_model,
            )
        elif objective_type == "lcfs":
            from tune_lib.objective_lcfs import make_objective_lcfs
            objective = make_objective_lcfs(
                gpu_id=0,  # always cuda:0 from child's POV
                output_dir=output_dir,
                train_one_model=train_one_model,
            )
        else:
            from tune_lib.objective import make_objective
            objective = make_objective(
                gpu_id=0,  # always cuda:0 from child's POV
                output_dir=output_dir,
                train_one_model=train_one_model,
            )
    except Exception as e:
        log(f"FATAL: failed to build {objective_type!r} objective: {e}")
        return 1

    log(f"Entering study.optimize for {n_trials} trials")
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(Exception,),
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        log("KeyboardInterrupt -- stopping cleanly")
        return 0
    except Exception as e:
        log(f"FATAL: study.optimize raised: {e}")
        return 1

    all_trials = study.trials
    n_complete = sum(1 for t in all_trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in all_trials if t.state.name == "PRUNED")
    n_failed = sum(1 for t in all_trials if t.state.name == "FAIL")

    log(
        "Worker done. Study totals: "
        f"complete={n_complete}, pruned={n_pruned}, failed={n_failed}, "
        f"all={len(all_trials)}"
    )
    return 0


# -----------------------------------------------------------------------------
# CLI, called by tune.py as 'python -m tune_lib.worker --...'
# -----------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="One Optuna worker; spawned by tune.py as a subprocess.",
    )
    p.add_argument("--worker-id", type=int, required=True,
                   help="Worker index (0..N-1), used for logging only.")
    p.add_argument("--gpu-id", type=int, required=True,
                   help="GPU index as known to the parent.")
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument("--study-db", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-trials", type=int, required=True,
                   help="Number of trials this worker should run.")
    p.add_argument("--objective", type=str, default="default",
                   choices=["default", "lcfs", "lcfs_soft"],
                   help="Which Optuna objective to use.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rc = run_worker(
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        study_name=args.study_name,
        study_db=args.study_db,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        objective_type=args.objective,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

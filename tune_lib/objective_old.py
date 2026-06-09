"""
tune_lib/objective.py

The Optuna objective function.

Each Optuna trial:
    1. Gets hyperparameters sampled from search_space.suggest_config()
    2. Trains an AE via train_one_model() with a pruning callback
    3. Evaluates the trained AE on the selection set
    4. Attaches rich metadata to the trial for dashboard display
    5. Returns the trial's objective value for Optuna to rank

Pruning flow:
    - After each epoch, the callback reports val_loss to Optuna.
    - Optuna's MedianPruner compares against the median val_loss at that
      epoch across all completed trials.
    - If the current trial is worse than median AND past warmup, the
      callback records that pruning was requested.
    - training.train_model sees the callback return True, stops training,
      saves a checkpoint, and returns.
    - Back in objective(), we check the flag and raise TrialPruned so
      Optuna correctly marks the trial state.

Return value:
    - On success: float objective (higher = better, range ~[0, 1]).
    - On prune: raises optuna.TrialPruned (Optuna handles this).
    - On failure: raises the original exception (Optuna marks as FAILED).
"""

from pathlib import Path
from typing import Callable

import optuna

from tune_lib.search_space import suggest_config


# -----------------------------------------------------------------------------
# Main objective
# -----------------------------------------------------------------------------

def make_objective(
    gpu_id:         int,
    output_dir:     str,
    train_one_model: Callable,
) -> Callable:
    """Build the Optuna objective function for one worker.

    We need a factory because each worker has its own gpu_id, but the
    objective function only takes `trial` as its argument (Optuna's
    required signature).

    Parameters
    ----------
    gpu_id : int
        Which GPU this worker uses. After CUDA_VISIBLE_DEVICES is set,
        this is always 0 from the child process's perspective.
    output_dir : str
        Root for checkpoints/, logs/, configs/, summary.csv.
    train_one_model : callable
        The train.train_one_model function. Passed in so this module
        doesn't depend on the top-level train.py import path.

    Returns
    -------
    objective : callable(trial) -> float
        What you pass to study.optimize().
    """
    output_dir = str(output_dir)

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        config = suggest_config(trial)

        # Flag closure: captures whether the pruner requested a stop.
        # Lists are used instead of booleans so the inner callback can mutate.
        pruning_requested = [False]

        def on_epoch_end(epoch: int, train_loss: float, val_loss: float) -> bool:
            """Called by training.train_model after each epoch.

            Reports val_loss (negated to match direction=maximize) to Optuna,
            then asks the pruner whether this trial should be stopped.
            """
            # Optuna maximizes the objective. val_loss is minimized. So we
            # report -val_loss as the intermediate "higher is better" metric.
            trial.report(-val_loss, step=epoch)
            if trial.should_prune():
                pruning_requested[0] = True
                return True        # training.train_model will stop cleanly
            return False

        # Run training + evaluation. Any unexpected exception (OOM, etc.)
        # propagates up; Optuna will mark the trial as FAILED.
        result = train_one_model(
            config_dict  = config,
            gpu_id       = gpu_id,
            trial_num    = trial.number,
            output_dir   = output_dir,
            on_epoch_end = on_epoch_end,
            to_stdout    = False,       # parallel workers -> file-only logs
        )

        # Attach rich metadata for optuna-dashboard / post-hoc analysis.
        # These are serializable (str/float/bool) -- Optuna rejects others.
        for cls, auc in result.auc_per_class.items():
            trial.set_user_attr(f"auc_{cls}", float(auc))
        trial.set_user_attr("objective",    float(result.objective))
        trial.set_user_attr("floor_ok",     bool(result.floor_ok))
        trial.set_user_attr("threshold",    float(result.threshold))
        trial.set_user_attr("mu",           float(result.mu))
        trial.set_user_attr("sigma",        float(result.sigma))
        if result.scatter_path is not None:
            trial.set_user_attr("scatter_path", str(result.scatter_path))

        # If the pruner fired, training already stopped -- but evaluation
        # still ran (gives us a scatter plot even for pruned trials).
        # Tell Optuna this was a prune, not a normal completion.
        if pruning_requested[0]:
            raise optuna.TrialPruned()

        return float(result.objective)

    return objective

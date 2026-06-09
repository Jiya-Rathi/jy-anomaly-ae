"""
tune_lib/objective_lcfs.py

Optuna objective for the LCFS-masking + SSIM study.

This mirrors tune_lib/objective.py, but uses suggest_config_lcfs() and applies
an LCFS-study-specific hard-floor rule:

    bad_black_core floor      = 0.89
    bad_nonconverged floor    = 0.90
    objective weights         = 0.4 / 0.3 / 0.3

The objective is recomputed here instead of trusting result.objective because
train_one_model/evaluation may still use the original 0.90 bad_black_core floor.
"""

from typing import Callable

import optuna

from tune_lib.search_space_lcfs import suggest_config_lcfs


BC_FLOOR = 0.89
NC_FLOOR = 0.90


def _lcfs_objective_from_aucs(auc_per_class: dict) -> tuple[float, bool]:
    """Return (objective, floor_ok) using the LCFS-study hard-floor rule."""
    auc_broken_lcfs = float(auc_per_class.get("broken_lcfs", 0.0))
    auc_bad_black_core = float(auc_per_class.get("bad_black_core", 0.0))
    auc_bad_nonconverged = float(auc_per_class.get("bad_nonconverged", 0.0))

    floor_ok = (
        auc_bad_black_core >= BC_FLOOR
        and auc_bad_nonconverged >= NC_FLOOR
    )

    if not floor_ok:
        return 0.0, False

    objective = (
        0.4 * auc_broken_lcfs
        + 0.3 * auc_bad_black_core
        + 0.3 * auc_bad_nonconverged
    )
    return float(objective), True


def make_objective_lcfs(
    gpu_id: int,
    output_dir: str,
    train_one_model: Callable,
) -> Callable:
    """Build the Optuna objective function for one LCFS-study worker."""
    output_dir = str(output_dir)

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config_lcfs(trial)
        config["trial_num"] = trial.number

        pruning_requested = [False]

        def on_epoch_end(epoch: int, train_loss: float, val_loss: float) -> bool:
            trial.report(-val_loss, step=epoch)
            if trial.should_prune():
                pruning_requested[0] = True
                return True
            return False

        result = train_one_model(
            config_dict=config,
            gpu_id=gpu_id,
            trial_num=trial.number,
            output_dir=output_dir,
            on_epoch_end=on_epoch_end,
            to_stdout=False,
        )

        objective_value, floor_ok = _lcfs_objective_from_aucs(result.auc_per_class)

        trial.set_user_attr(
            "auc_broken_lcfs",
            float(result.auc_per_class.get("broken_lcfs", 0.0)),
        )
        trial.set_user_attr(
            "auc_bad_black_core",
            float(result.auc_per_class.get("bad_black_core", 0.0)),
        )
        trial.set_user_attr(
            "auc_bad_nonconverged",
            float(result.auc_per_class.get("bad_nonconverged", 0.0)),
        )
        trial.set_user_attr("objective", float(objective_value))
        trial.set_user_attr("floor_ok", bool(floor_ok))
        trial.set_user_attr("bc_floor", float(BC_FLOOR))
        trial.set_user_attr("nc_floor", float(NC_FLOOR))

        # Preserve the useful diagnostics from the normal objective when present.
        if hasattr(result, "threshold"):
            trial.set_user_attr("threshold", float(result.threshold))
        if hasattr(result, "mu"):
            trial.set_user_attr("mu", float(result.mu))
        if hasattr(result, "sigma"):
            trial.set_user_attr("sigma", float(result.sigma))
        if getattr(result, "scatter_path", None) is not None:
            trial.set_user_attr("scatter_path", str(result.scatter_path))

        if pruning_requested[0]:
            raise optuna.TrialPruned()

        return float(objective_value)

    return objective

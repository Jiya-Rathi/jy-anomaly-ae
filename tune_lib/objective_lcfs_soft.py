"""
tune_lib/objective_lcfs_soft.py

Soft-floor + validity-guard objective for the v3 LCFS-masking + SSIM study.

Mirrors tune_lib/objective_lcfs.py, with two changes:

1. SOFT FLOOR (Defect 1)
   objective_lcfs.py returned 0.0 the instant bad_black_core < 0.89 or
   bad_nonconverged < 0.90, which destroyed TPE's gradient near the floor and
   threw away strong near-miss models (e.g. v2 trial 166: lcfs 1.000 /
   bc 0.882 / nc 0.996 -> 0.0). Here, each floored class instead gets a
   continuous penalty FACTOR in (0, 1]; the weighted AUC sum is multiplied by
   the product of the per-class factors. A model that misses bc by 0.01 ranks
   just below a floor-clearing model; one that misses by 0.10 is pushed far
   down but stays rankable, so TPE can climb toward the floor.

2. VALIDITY GUARD (Defect 2)
   The v2 broken_lcfs=0.412 trials all share best_epoch == 1: they peaked at
   the first epoch and never improved (an AE that never trained past random
   init), yet scored ~0.745 mid-pack because the mask-only output happens to
   score bc/nc high. These are INVALID runs, not weak models. We detect them
   via history.best_epoch (exposed on result by train_one_model) and mark the
   trial PRUNED so TPE does not model them as real observations.

floor_ok (hard 0.89/0.90) is still reported for final model SELECTION, but it
no longer gates the search objective. Always computed from result.auc_per_class
(never result.objective), matching objective_lcfs.py.
"""

from typing import Callable, Optional
import optuna
from tune_lib.search_space_lcfs import suggest_config_lcfs

# --- LCFS-study floors (same as objective_lcfs.py) -------------------------
BC_FLOOR = 0.89
NC_FLOOR = 0.90

WEIGHTS = {"broken_lcfs": 0.4, "bad_black_core": 0.3, "bad_nonconverged": 0.3}

# --- Soft-floor ramp -------------------------------------------------------
# Over [floor - RAMP_WIDTH, floor] the per-class factor rises linearly from
# RAMP_FLOOR_VALUE to 1.0. bc clusters in ~0.88-0.95 across v2, so a 0.05 ramp
# separates genuine near-misses (0.88-0.89) from real failures (<0.84) without
# becoming a hard cliff again.
RAMP_WIDTH = 0.05
RAMP_FLOOR_VALUE = 0.30   # factor exactly at (floor - RAMP_WIDTH)

# --- Validity guard --------------------------------------------------------
# best_epoch <= this means the AE peaked at init and never trained.
MIN_VALID_BEST_EPOCH = 2


def _class_factor(auc: float, floor: float) -> float:
    """Continuous penalty factor in (0, 1] for one floored class."""
    if auc >= floor:
        return 1.0
    ramp_bottom = floor - RAMP_WIDTH
    if auc >= ramp_bottom:
        frac = (auc - ramp_bottom) / RAMP_WIDTH
        return RAMP_FLOOR_VALUE + (1.0 - RAMP_FLOOR_VALUE) * frac
    return RAMP_FLOOR_VALUE * (auc / ramp_bottom) if ramp_bottom > 0 else 0.0


def _lcfs_objective_soft(auc_per_class: dict, best_epoch: Optional[int] = None):
    """Return (objective, floor_ok, valid).

    objective : float in [0, 1], continuous (no hard cliff at the floor)
    floor_ok  : True iff bc >= BC_FLOOR and nc >= NC_FLOOR (for selection)
    valid     : False iff the trial never trained (best_epoch < MIN_VALID_BEST_EPOCH)
    """
    auc_broken_lcfs      = float(auc_per_class.get("broken_lcfs", 0.0))
    auc_bad_black_core   = float(auc_per_class.get("bad_black_core", 0.0))
    auc_bad_nonconverged = float(auc_per_class.get("bad_nonconverged", 0.0))

    valid = True
    if best_epoch is not None and best_epoch < MIN_VALID_BEST_EPOCH:
        valid = False

    base = (
        WEIGHTS["broken_lcfs"]      * auc_broken_lcfs
        + WEIGHTS["bad_black_core"] * auc_bad_black_core
        + WEIGHTS["bad_nonconverged"] * auc_bad_nonconverged
    )
    factor = (
        _class_factor(auc_bad_black_core,   BC_FLOOR)
        * _class_factor(auc_bad_nonconverged, NC_FLOOR)
    )
    objective = base * factor
    floor_ok = (auc_bad_black_core >= BC_FLOOR) and (auc_bad_nonconverged >= NC_FLOOR)

    if not valid:
        objective = 0.0

    return float(objective), bool(floor_ok), bool(valid)


def make_objective_lcfs_soft(
    gpu_id: int,
    output_dir: str,
    train_one_model: Callable,
) -> Callable:
    """Build the Optuna objective function for one v3 worker."""
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

        # train_one_model attaches history.best_epoch onto result (see train.py).
        # Falls back to None if an older train.py is in use -> guard simply skips.
        best_epoch = getattr(result, "best_epoch", None)

        objective_value, floor_ok, valid = _lcfs_objective_soft(
            result.auc_per_class, best_epoch=best_epoch
        )

        trial.set_user_attr("auc_broken_lcfs",      float(result.auc_per_class.get("broken_lcfs", 0.0)))
        trial.set_user_attr("auc_bad_black_core",   float(result.auc_per_class.get("bad_black_core", 0.0)))
        trial.set_user_attr("auc_bad_nonconverged", float(result.auc_per_class.get("bad_nonconverged", 0.0)))
        trial.set_user_attr("objective", float(objective_value))
        trial.set_user_attr("floor_ok",  bool(floor_ok))
        trial.set_user_attr("valid",     bool(valid))
        trial.set_user_attr("bc_floor",  float(BC_FLOOR))
        trial.set_user_attr("nc_floor",  float(NC_FLOOR))
        if best_epoch is not None:
            trial.set_user_attr("best_epoch", int(best_epoch))
        if hasattr(result, "threshold"):
            trial.set_user_attr("threshold", float(result.threshold))
        if hasattr(result, "mu"):
            trial.set_user_attr("mu", float(result.mu))
        if hasattr(result, "sigma"):
            trial.set_user_attr("sigma", float(result.sigma))
        if getattr(result, "scatter_path", None) is not None:
            trial.set_user_attr("scatter_path", str(result.scatter_path))

        # Invalid (never-trained) trials: mark PRUNED so TPE doesn't treat them
        # as real observations. Distinct from a genuine low score.
        if not valid:
            raise optuna.TrialPruned()

        if pruning_requested[0]:
            raise optuna.TrialPruned()

        return float(objective_value)

    return objective

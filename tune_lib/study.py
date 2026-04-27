"""
tune_lib/study.py

Creates or loads the Optuna study used by all 4 parallel workers.

Two public functions:

    create_or_load_study(study_name, study_db, log=print) -> optuna.Study
        If study_db exists, loads and resumes it.
        Otherwise creates a fresh study with our chosen sampler + pruner.

    enqueue_baseline(study, baseline_config)
        If baseline_config is provided and the study hasn't seen trials yet,
        enqueue it so the first trial uses these known-good hyperparameters.

Design choices (see research notes):
    - Direction: maximize (bigger objective = better model).
    - Sampler: TPESampler with n_startup_trials=20.
        First 20 trials are random (pure exploration).
        Trials 21+ use the Parzen-estimator model to sample.
    - Pruner: MedianPruner with n_startup_trials=20, n_warmup_steps=50.
        No trial gets pruned before the study has 20 complete trials
        AND the trial has passed 50 epochs.
    - Storage: SQLite file on shared storage so workers can coordinate.
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# -----------------------------------------------------------------------------
# Constants (match the decisions in our design discussion)
# -----------------------------------------------------------------------------

_TPE_STARTUP_TRIALS    = 20    # first 20 trials are purely random (TPE warm-up).
                               # bumped from 10 for better initial space coverage
                               # in a 10-hyperparameter search.
_PRUNER_STARTUP_TRIALS = 20    # no pruning until 20 trials are complete (matches TPE).
_PRUNER_WARMUP_STEPS   = 50    # within a trial, no pruning until epoch 50
_SAMPLER_SEED          = None  # was 42 -- fixed seeds make parallel workers suggest
                               # duplicate configs during random startup. seed=None
                               # uses /dev/urandom for genuine diversity.


# -----------------------------------------------------------------------------
# Study creation / resume
# -----------------------------------------------------------------------------

def create_or_load_study(
    study_name: str,
    study_db: str,
    log:      Callable[[str], None] = print,
) -> optuna.Study:
    """Create a new study or resume an existing one.

    Parameters
    ----------
    study_name : str
        Human-readable identifier (e.g., 'ae_topology_v1').
    study_db : str or Path
        Path to the SQLite file. Created if missing.
    log : callable(str) -> None
        Where to print status messages.

    Returns
    -------
    optuna.Study
    """
    study_db = Path(study_db)
    study_db.parent.mkdir(parents=True, exist_ok=True)

    # Optuna expects an SQLAlchemy-style URL for SQLite:
    #   sqlite:///absolute/path/to/file.db
    storage_url = f"sqlite:///{study_db.resolve()}"

    sampler = TPESampler(
        n_startup_trials = _TPE_STARTUP_TRIALS,
        # seed=None (not _SAMPLER_SEED): with parallel workers, fixing the
        # seed causes all workers to get identical suggestions during the
        # random-startup phase (constant_liar doesn't apply to random
        # sampling). Letting the seed be non-deterministic gives each
        # worker different random startup picks. We still get reproducible
        # study-level outcomes (the *winning architecture* is what matters
        # scientifically, not bit-identical individual trials).
        seed             = None,
        constant_liar    = True,        # parallel-safe: pretend pending trials
                                        # have bad scores so workers get diverse
                                        # suggestions instead of duplicates
        multivariate     = True,        # consider hyperparameter correlations
                                        # (generally improves sample efficiency)
    )
    pruner = MedianPruner(
        n_startup_trials = _PRUNER_STARTUP_TRIALS,
        n_warmup_steps   = _PRUNER_WARMUP_STEPS,
        interval_steps   = 1,
    )

    db_existed = study_db.is_file() and study_db.stat().st_size > 0

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage_url,
        sampler        = sampler,
        pruner         = pruner,
        direction      = "maximize",
        load_if_exists = True,       # resume if this study_name already in the DB
    )

    if db_existed:
        n_done = sum(1 for t in study.trials
                     if t.state in (optuna.trial.TrialState.COMPLETE,
                                    optuna.trial.TrialState.PRUNED))
        log(f"Resuming study '{study_name}' from {study_db}")
        log(f"  trials already on disk: {len(study.trials)} "
            f"(completed or pruned: {n_done})")
    else:
        log(f"Creating new study '{study_name}' at {study_db}")

    log(f"  sampler : TPESampler(n_startup={_TPE_STARTUP_TRIALS}, "
        f"seed={_SAMPLER_SEED}, constant_liar=True, multivariate=True)")
    log(f"  pruner  : MedianPruner(n_startup={_PRUNER_STARTUP_TRIALS}, "
        f"n_warmup_steps={_PRUNER_WARMUP_STEPS})")

    return study


# -----------------------------------------------------------------------------
# Warm-start
# -----------------------------------------------------------------------------

# Fields the baseline config must specify to be a valid warm-start trial.
# These match the hyperparameters in search_space.suggest_config().
_WARMSTART_FIELDS = [
    "use_hpf", "hpf_sigma",
    "bottleneck_dim", "n_enc_layers", "n_dec_layers",
    "base_channels", "growth_factor", "use_batchnorm",
    "lr", "batch_size",
]


def enqueue_baseline(
    study:           optuna.Study,
    baseline_config: Optional[Dict],
    log:             Callable[[str], None] = print,
) -> bool:
    """Enqueue a known-good configuration as the next trial.

    Called once at study creation. Optuna's enqueue_trial mechanism
    schedules these hyperparameters for the next trial Optuna picks,
    which means they'll be tried before any sampler-generated trials.

    Returns True if a baseline was enqueued, False otherwise (e.g., no
    baseline provided, or study already has trials).
    """
    if baseline_config is None:
        log("No baseline config provided -- skipping warm-start")
        return False

    # Only enqueue if the study is empty. Re-enqueueing on a resumed study
    # would just repeat work.
    if len(study.trials) > 0:
        log(f"Study already has {len(study.trials)} trials "
            f"-- skipping warm-start")
        return False

    # Validate the baseline has every field search_space.suggest_config asks for.
    # Optuna's enqueue_trial is forgiving: missing fields will be sampled
    # normally. But for a warm-start to be meaningful, we want all fields
    # explicitly set.
    missing = [f for f in _WARMSTART_FIELDS if f not in baseline_config]
    if missing:
        log(f"WARNING: baseline_config is missing fields: {missing}")
        log("  those fields will be sampled by the TPE sampler instead")

    # Optuna represents booleans as {0, 1} in the sampler; translate on enqueue.
    enqueue = {}
    for k, v in baseline_config.items():
        if k not in _WARMSTART_FIELDS:
            continue
        if isinstance(v, bool):
            enqueue[k] = int(v)
        else:
            enqueue[k] = v

    study.enqueue_trial(enqueue)
    log(f"Enqueued baseline warm-start trial:")
    for k in _WARMSTART_FIELDS:
        if k in enqueue:
            log(f"  {k:16s} = {enqueue[k]}")
    return True

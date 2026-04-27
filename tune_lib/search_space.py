"""
tune_lib/search_space.py

The Optuna search space for the AE topology study.

One public function:
    suggest_config(trial) -> dict
        Given an Optuna trial, sample one value for each of the 10
        hyperparameters in the search space and return a config dict
        that train_one_model() can consume directly.

Search space (matches Config's validation ranges in ae_lib/config.py):
    use_hpf         : bool
    hpf_sigma       : float [1.0, 8.0]  (only sampled when use_hpf=True)
    bottleneck_dim  : int   [16, 256]
    n_enc_layers    : int   [2, 5]
    n_dec_layers    : int   [2, 5]
    base_channels   : int   [8, 64]
    growth_factor   : float [1.5, 3.0]
    use_batchnorm   : bool
    lr              : float [1e-5, 1e-2]  log-uniform
    batch_size      : int   [8, 64]       step 8

Notes:
    - The training schedule (max_epochs, min_epochs, patience) is NOT
      part of the search space. It's fixed across trials for comparability.
    - The seed is also fixed across trials so that a given architecture
      is reproducible; we may later vary seeds across runs of the study
      if we want to measure stochasticity.
"""

from typing import Dict


def suggest_config(trial) -> Dict:
    """Sample one config from the search space for this Optuna trial.

    Returns a dict with every field train_one_model() needs, except
    trial_num (that gets injected by the objective function).
    """

    # Binary choices (Optuna represents these as categorical int {0, 1})
    use_hpf       = bool(trial.suggest_categorical("use_hpf",       [0, 1]))
    use_batchnorm = bool(trial.suggest_categorical("use_batchnorm", [0, 1]))

    # HPF sigma is only sampled when HPF is on. When HPF is off we still
    # put a value in the config (Config requires one) but it will be ignored
    # by the data pipeline. Using a fixed dummy value keeps the config
    # schema stable; Optuna's history will show hpf_sigma as missing for
    # use_hpf=0 trials.
    if use_hpf:
        hpf_sigma = trial.suggest_float("hpf_sigma", 1.0, 8.0)
    else:
        hpf_sigma = 3.0          # placeholder, unused

    # Architecture
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 16, 256)
    n_enc_layers   = trial.suggest_int("n_enc_layers",    2,   5)
    n_dec_layers   = trial.suggest_int("n_dec_layers",    2,   5)
    base_channels  = trial.suggest_int("base_channels",   8,  64)
    growth_factor  = trial.suggest_float("growth_factor", 1.5, 3.0)

    # Optimization
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 64, step=8)

    # Build the config dict. All other fields (paths, max_epochs, etc.)
    # fall back to Config defaults.
    config = {
        "bottleneck_dim": bottleneck_dim,
        "n_enc_layers":   n_enc_layers,
        "n_dec_layers":   n_dec_layers,
        "base_channels":  base_channels,
        "growth_factor":  growth_factor,
        "use_batchnorm":  use_batchnorm,

        "use_hpf":   use_hpf,
        "hpf_sigma": hpf_sigma,

        "lr":         lr,
        "batch_size": batch_size,

        # Training schedule: fixed across trials for fair comparison
        "max_epochs": 300,
        "min_epochs": 30,
        "patience":   30,

        # Seed: fixed across trials so the same architecture gives the
        # same result. Reproducibility is Level 1 per design.
        "seed":          42,
        "deterministic": False,
    }

    return config

"""
tune_lib/search_space_lcfs.py

Optuna search space for the LCFS-masking + SSIM preprocessing study.

This is the sister module to search_space.py. The key differences are fixed
for the whole LCFS study:

    1. use_lcfs_masking is always True
    2. use_hpf is always False
    3. loss_fn is always "ssim"
    4. score_fn is always "ssim"

use_hpf and hpf_sigma are still returned because Config expects those fields.
They are fixed placeholders and are ignored by the data pipeline when
use_lcfs_masking=True.
"""

from typing import Dict


SOL_FILE_PATH = "/mnt/beegfs/mantis/jrathi/outer_vacuum_boundary_spline_RZ.txt"


def suggest_config_lcfs(trial) -> Dict:
    """Sample one Optuna config for the LCFS-masking + SSIM study."""
    use_batchnorm = bool(trial.suggest_categorical("use_batchnorm", [0, 1]))
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 16, 256)
    n_enc_layers = trial.suggest_int("n_enc_layers", 2, 5)
    n_dec_layers = trial.suggest_int("n_dec_layers", 2, 5)
    base_channels = trial.suggest_int("base_channels", 8, 64)
    growth_factor = trial.suggest_float("growth_factor", 1.5, 3.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 64, step=8)

    return {
        "bottleneck_dim": bottleneck_dim,
        "n_enc_layers": n_enc_layers,
        "n_dec_layers": n_dec_layers,
        "base_channels": base_channels,
        "growth_factor": growth_factor,
        "use_batchnorm": use_batchnorm,

        # HPF is meaningless on near-binary masked images.
        # Keep these keys because Config requires them.
        "use_hpf": False,
        "hpf_sigma": 3.0,

        # Masking + SSIM: fixed for the entire LCFS study.
        "use_lcfs_masking": True,
        "sol_file_path": SOL_FILE_PATH,
        "loss_fn": "ssim",
        "score_fn": "ssim",

        # Schedule: same as original study for fair comparison.
        "max_epochs": 300,
        "min_epochs": 30,
        "patience": 30,
        "seed": 42,
        "deterministic": False,

        # Optimizer hyperparameters.
        "lr": lr,
        "batch_size": batch_size,
    }

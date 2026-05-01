"""
train.py

Train one autoencoder, end-to-end. Two ways to use this:

1. As a library (imported by tune.py):
       from train import train_one_model
       result = train_one_model(
           config_dict = {...},
           gpu_id      = 0,
           trial_num   = 7,
           output_dir  = "/mnt/.../outputs",
           on_epoch_end= optuna_pruning_callback,
       )
       # result is an ae_lib.evaluation.EvalResult

2. From the command line (for CLI runs -- baseline warm-start,
   final deterministic retrain, debugging):
       python train.py --config configs/baseline.yaml \
                       --gpu 0 --trial-num 0 \
                       --output-dir /mnt/.../outputs

Either way, the five stages are the same:
    1. Resolve paths under output_dir.
    2. Seed everything for reproducibility.
    3. Load train/val/selection splits -- all preloaded to GPU.
    4. Build model, run the training loop.
    5. Evaluate on selection set, write summary row.

MODEL-CLASS SWITCH (added 2026-04-28):
    cfg.model_type selects between the existing CNN Autoencoder ("cnn",
    default) and the fully-connected MLPAutoencoder ("mlp"). The MLP
    variant was added per professor suggestion to test whether the
    convolutional inductive bias is what's limiting broken_lcfs detection.
    Both model classes expose forward() and reconstruction_error() with
    the same signatures, so training/evaluation/scoring downstream is
    unchanged.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ae_lib.config         import Config
from ae_lib.data           import load_split
from ae_lib.evaluation     import evaluate_trial
from ae_lib.logging_utils  import (
    make_trial_loggers,
    append_summary_row,
    SUMMARY_FIELDS,
)
from ae_lib.model          import Autoencoder
from ae_lib.training       import train_model


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def _set_seeds(seed: int, deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch seeds. If deterministic=True, also force
    deterministic algorithms (slower, bit-reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Level 2 determinism: slower, but same-hardware bit-reproducible.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------------
# Model build switch
# -----------------------------------------------------------------------------

def _build_model(cfg, device):
    """Construct the model selected by cfg.model_type and move to device.

    Returns (model, summary_str). summary_str is a short multi-line
    description suitable for log_fn(); for the CNN path it comes from
    Autoencoder.architecture_summary(), for the MLP path we build it
    here since MLPAutoencoder doesn't have that method.
    """
    model_type = getattr(cfg, "model_type", "cnn")

    if model_type == "mlp":
        # Imported lazily so the CNN path doesn't pay the cost on every run.
        from ae_lib.model_mlp import build_mlp_from_config
        model = build_mlp_from_config(cfg).to(device)
        summary = (
            "Model: MLPAutoencoder\n"
            f"  input_size     : {model.input_size}x{model.input_size} "
            f"(downsampled from {cfg.image_size})\n"
            f"  hidden_sizes   : {model.hidden_sizes}\n"
            f"  bottleneck_dim : {model.bottleneck_dim}\n"
            f"  parameters     : {model.parameter_count():,}"
        )
        return model, summary

    # Default path: existing CNN Autoencoder. Behavior unchanged.
    model = Autoencoder(cfg).to(device)
    return model, model.architecture_summary()


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

class _TrialPaths:
    """All output paths for one trial, grouped."""

    def __init__(self, output_dir: Path, trial_num):
        # trial_num may be an int or None (None for manual CLI runs)
        tag = f"trial_{trial_num}" if trial_num is not None else "manual"

        self.output_dir     = output_dir
        self.checkpoint     = output_dir / "checkpoints" / f"{tag}_best.pt"
        self.log            = output_dir / "logs"         / f"{tag}.log"
        self.csv            = output_dir / "logs"         / f"{tag}.csv"
        self.scatter        = output_dir / "logs"         / f"{tag}_scatter.png"
        self.config_snap    = output_dir / "configs"      / f"{tag}.yaml"
        self.summary        = output_dir / "summary.csv"

        for p in (self.checkpoint, self.log, self.config_snap, self.summary):
            p.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Core library entry point
# -----------------------------------------------------------------------------

def train_one_model(
    config_dict:    dict,
    gpu_id:         int,
    trial_num:      Optional[int],
    output_dir,
    on_epoch_end:   Optional[Callable[[int, float, float], bool]] = None,
    to_stdout:      bool = False,
):
    """Run one training + evaluation and return the EvalResult.

    Parameters
    ----------
    config_dict : dict
        Keyword args for Config. trial_num will be injected.
    gpu_id : int
        Which GPU to use (0..N-1).
    trial_num : int or None
        Optuna trial number, used in output filenames. None for manual runs.
    output_dir : str or Path
        Root directory for all outputs (checkpoints/, logs/, configs/).
    on_epoch_end : optional (epoch, train_loss, val_loss) -> bool
        Called after each epoch. If it returns True, training stops (pruned).
        Optuna integration goes here. Leave None for CLI runs.
    to_stdout : bool
        Echo log lines to stdout. True for CLI runs; False for parallel
        Optuna workers (interleaved output would be unreadable).

    Returns
    -------
    EvalResult
        The full evaluation result. On failures before eval (e.g., pruned
        very early), returns an EvalResult with objective=0.0 and
        floor_ok=False so the caller can still score the trial.
    """
    output_dir = Path(output_dir)
    paths      = _TrialPaths(output_dir, trial_num)

    # --- Build + validate Config
    config_dict = dict(config_dict)   # don't mutate the caller's dict
    config_dict["trial_num"] = trial_num
    cfg = Config.from_dict(config_dict)

    # Snapshot the final config for reproducibility
    cfg.save_yaml(paths.config_snap)

    # --- Set up loggers
    log_fn, csv_row_fn, close_loggers = make_trial_loggers(
        log_path  = paths.log,
        csv_path  = paths.csv,
        to_stdout = to_stdout,
    )

    try:
        # --- Seed and pick device
        _set_seeds(cfg.seed, deterministic=cfg.deterministic)
        device = torch.device(f"cuda:{gpu_id}"
                              if torch.cuda.is_available() else "cpu")

        # --- Header in the log
        log_fn("=" * 60)
        log_fn(f"Trial {trial_num} starting on {device}")
        log_fn(f"Timestamp: {datetime.now().isoformat()}")
        log_fn(cfg.summary_str())

        # --- Load splits
        t0 = time.time()
        log_fn("Loading train/val/selection splits...")
        manifests = Path(cfg.manifests_dir)
        train_data     = load_split(manifests / "train.txt",     cfg, device)
        val_data       = load_split(manifests / "val.txt",       cfg, device)
        selection_data = load_split(manifests / "selection.txt", cfg, device)
        log_fn(
            f"Splits loaded in {time.time() - t0:.1f}s: "
            f"n_train={len(train_data)}, n_val={len(val_data)}, "
            f"n_selection={len(selection_data)}"
        )

        # --- Build model (CNN or MLP, selected by cfg.model_type)
        model, model_summary = _build_model(cfg, device)
        log_fn(model_summary)

        # --- Train
        t0 = time.time()
        history = train_model(
            model         = model,
            train_images  = train_data.images,
            val_images    = val_data.images,
            cfg           = cfg,
            device        = device,
            checkpoint_path = paths.checkpoint,
            on_epoch_end  = on_epoch_end,
            log_fn        = log_fn,
            csv_row_fn    = csv_row_fn,
        )
        train_time = time.time() - t0

        # --- Reload best checkpoint before evaluation
        log_fn(f"Reloading best checkpoint (epoch {history.best_epoch}) for evaluation")
        ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])

        # --- Evaluate on selection set (with scatter plot)
        scatter_title = (
            f"Trial {trial_num} -- "
            f"val_loss={history.best_val_loss:.6f} @ epoch {history.best_epoch}"
        )
        t0 = time.time()
        result = evaluate_trial(
            model          = model,
            train_images   = train_data.images,
            selection_data = selection_data,
            cfg            = cfg,
            device         = device,
            scatter_path   = paths.scatter,
            scatter_title  = scatter_title,
        )
        eval_time = time.time() - t0

        # --- Log evaluation summary
        log_fn("-" * 60)
        log_fn(f"Evaluation:")
        for cls, auc in result.auc_per_class.items():
            log_fn(f"  AUC {cls:20s}: {auc:.4f}")
        log_fn(f"  objective      : {result.objective:.4f}")
        log_fn(f"  floor_ok       : {result.floor_ok}")
        log_fn(f"  threshold      : {result.threshold:.4f}")
        log_fn(f"  mu / sigma     : {result.mu:.6g} / {result.sigma:.6g}")
        log_fn(f"Train time={train_time:.1f}s, eval time={eval_time:.1f}s")

        # --- Append one row to summary.csv
        # NOTE: For model_type="mlp" runs, the CNN-architecture columns
        # (n_enc_layers, n_dec_layers, base_channels, growth_factor) carry
        # filler values from the YAML and are not meaningful. The
        # bottleneck_dim and use_batchnorm columns ARE meaningful for both.
        # If MLP runs become a regular thing, extend SUMMARY_FIELDS to
        # add model_type and the mlp_* columns.
        _append_summary(
            summary_path = paths.summary,
            trial_num    = trial_num,
            cfg          = cfg,
            history      = history,
            result       = result,
            paths        = paths,
        )

        return result

    finally:
        close_loggers()


# -----------------------------------------------------------------------------
# Summary row helper
# -----------------------------------------------------------------------------

def _append_summary(summary_path, trial_num, cfg, history, result, paths):
    """Build a summary row dict and append it atomically."""
    row = {
        "trial_num":          trial_num if trial_num is not None else "manual",
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "seed":               cfg.seed,
        "stop_reason":        history.stop_reason,
        "total_epochs":       history.total_epochs,
        "best_epoch":         history.best_epoch,
        "best_val_loss":      f"{history.best_val_loss:.8f}",
        "objective":          f"{result.objective:.6f}",
        "floor_ok":           result.floor_ok,
        "auc_broken_lcfs":    f"{result.auc_per_class.get('broken_lcfs', float('nan')):.6f}",
        "auc_bad_black_core": f"{result.auc_per_class.get('bad_black_core', float('nan')):.6f}",
        "auc_bad_nonconverged": f"{result.auc_per_class.get('bad_nonconverged', float('nan')):.6f}",
        "threshold":          f"{result.threshold:.6f}",
        "mu":                 f"{result.mu:.8g}",
        "sigma":              f"{result.sigma:.8g}",
        # Hyperparameters
        "use_hpf":            cfg.use_hpf,
        "hpf_sigma":          cfg.hpf_sigma,
        "bottleneck_dim":     cfg.bottleneck_dim,
        "n_enc_layers":       cfg.n_enc_layers,
        "n_dec_layers":       cfg.n_dec_layers,
        "base_channels":      cfg.base_channels,
        "growth_factor":      cfg.growth_factor,
        "use_batchnorm":      cfg.use_batchnorm,
        "lr":                 cfg.lr,
        "batch_size":         cfg.batch_size,
        # Paths
        "checkpoint_path":    str(paths.checkpoint),
        "log_path":           str(paths.log),
        "csv_path":           str(paths.csv),
        "scatter_path":       str(paths.scatter),
    }
    append_summary_row(summary_path, row, SUMMARY_FIELDS)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_cli_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train one AE from a YAML config. "
                    "For Optuna-driven training, use tune.py instead."
    )
    p.add_argument(
        "--config", required=True, type=str,
        help="Path to a YAML config file (see Config dataclass for fields).",
    )
    p.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use (default: 0).",
    )
    p.add_argument(
        "--trial-num", type=int, default=None,
        help="Trial number used in output filenames. "
             "Omit for a 'manual' tag.",
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to write checkpoints/, logs/, configs/, summary.csv.",
    )
    p.add_argument(
        "--deterministic", action="store_true",
        help="Enable Level 2 determinism (slower, bit-reproducible).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed in the YAML config.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Do not echo log lines to stdout (file only).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_cli_args(argv)

    # Load YAML -> dict so we can merge in CLI overrides
    cfg = Config.from_yaml(args.config)
    config_dict = cfg.to_dict()

    if args.deterministic:
        config_dict["deterministic"] = True
    if args.seed is not None:
        config_dict["seed"] = args.seed

    # Drop trial_num from the dict; we pass it explicitly (set by train_one_model)
    config_dict.pop("trial_num", None)

    result = train_one_model(
        config_dict = config_dict,
        gpu_id      = args.gpu,
        trial_num   = args.trial_num,
        output_dir  = args.output_dir,
        on_epoch_end= None,                 # no Optuna in CLI mode
        to_stdout   = not args.quiet,
    )

    # Final terminal summary for CLI runs
    print("=" * 60)
    print(f"Training complete.")
    print(f"Objective: {result.objective:.4f}  (floor_ok={result.floor_ok})")
    for cls, auc in result.auc_per_class.items():
        print(f"  AUC {cls:20s}: {auc:.4f}")
    print(f"Scatter plot: {result.scatter_path}")
    print("=" * 60)

    # Exit code: 0 if the trial cleared the floor, 1 otherwise.
    # Useful if you want to script "retrain until you get a good one".
    sys.exit(0 if result.floor_ok else 1)


if __name__ == "__main__":
    main()

"""
ae_lib/training.py

Training loop for one autoencoder.

Public API:
    train_model(model, train_data, val_data, cfg, device,
                checkpoint_path, on_epoch_end=None)
        Runs the full training loop:
            - Adam optimizer, lr from cfg.lr
            - Reconstruction loss = mean MSE over pixels
            - Early stopping on val_loss with patience=cfg.patience
            - Minimum cfg.min_epochs before stopping is allowed
            - Checkpoints best-val-loss model to checkpoint_path
            - Optional on_epoch_end callback for external pruning
              (Optuna's MedianPruner hooks in here)

        Returns a TrainingHistory with per-epoch train/val losses,
        best-epoch info, and stop reason.

Design notes:
    - All data is preloaded to GPU (see data.py). Batching is just
      tensor indexing -- no DataLoader, no workers.
    - Each Optuna trial starts from scratch, so we do NOT save
      optimizer state in checkpoints.
    - Early stopping triggers on "no improvement for `patience` epochs"
      but is blocked until `min_epochs` has been reached.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Training history container
# -----------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Full record of one training run."""
    train_loss:      List[float] = field(default_factory=list)
    val_loss:        List[float] = field(default_factory=list)
    epoch_time_sec:  List[float] = field(default_factory=list)
    best_epoch:      int   = -1
    best_val_loss:   float = float("inf")
    stop_reason:     str   = ""     # 'max_epochs', 'early_stop', 'pruned', 'error'
    total_epochs:    int   = 0      # how many epochs actually ran


# -----------------------------------------------------------------------------
# One-epoch helpers (train / eval)
# -----------------------------------------------------------------------------

def _train_one_epoch(model, images, indices, batch_size, optimizer, loss_fn) -> float:
    """Run one training epoch. Returns mean loss over all samples (per-pixel MSE)."""
    model.train()
    n = images.shape[0]
    total_loss = 0.0
    total_seen = 0

    for start in range(0, n, batch_size):
        end       = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batch     = images[batch_idx]                # [B, 1, H, W]

        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(batch)
        loss           = loss_fn(reconstruction, batch)
        loss.backward()
        optimizer.step()

        # loss is a scalar mean over all pixels in the batch;
        # weight by batch size so the running average is well-defined
        # even if the final batch is smaller than the others.
        b = end - start
        total_loss += loss.item() * b
        total_seen += b

    return total_loss / total_seen


def _eval_one_epoch(model, images, batch_size, loss_fn) -> float:
    """Run one evaluation pass (no gradients). Returns mean loss."""
    model.eval()
    n = images.shape[0]
    total_loss = 0.0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = images[start:end]
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            b = end - start
            total_loss += loss.item() * b
            total_seen += b

    return total_loss / total_seen


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def _save_checkpoint(path: Path, model: nn.Module, cfg, epoch: int, val_loss: float):
    """Save model weights + light metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "epoch":       epoch,
        "val_loss":    val_loss,
        "config_dict": cfg.to_dict(),
    }, path)


# -----------------------------------------------------------------------------
# Main training entry point
# -----------------------------------------------------------------------------

def train_model(
    model:           nn.Module,
    train_images:    torch.Tensor,           # [N, 1, H, W] on device
    val_images:      torch.Tensor,           # [M, 1, H, W] on device
    cfg,
    device:          torch.device,
    checkpoint_path,
    on_epoch_end:    Optional[Callable[[int, float, float], bool]] = None,
    log_fn:          Optional[Callable[[str], None]] = None,
    csv_row_fn:      Optional[Callable[[dict], None]] = None,
) -> TrainingHistory:
    """Train one AE with early stopping and best-checkpoint saving.

    Parameters
    ----------
    model : nn.Module
        The AE, already on `device`.
    train_images, val_images : torch.Tensor
        Preloaded training and validation images, already on `device`.
    cfg : Config
        Provides lr, batch_size, max_epochs, min_epochs, patience, seed.
    device : torch.device
        Where model and data live.
    checkpoint_path : str or Path
        Where to write the best-val-loss checkpoint.
    on_epoch_end : optional callable (epoch, train_loss, val_loss) -> bool
        If provided and returns True, training stops immediately (pruned).
        Optuna's MedianPruner integration goes through here.
    log_fn : optional callable (str) -> None
        Called with one-line human-readable status per epoch.
        Default: print to stdout.
    csv_row_fn : optional callable (dict) -> None
        Called with per-epoch metrics dict. Let the caller write CSV rows.
        Dict keys: epoch, train_loss, val_loss, lr, epoch_time_sec,
                   best_epoch, best_val_loss.

    Returns
    -------
    TrainingHistory
        Full per-epoch record plus summary fields.
    """
    checkpoint_path = Path(checkpoint_path)
    if log_fn is None:
        log_fn = print

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn   = nn.MSELoss(reduction="mean")

    # Index shuffle uses a deterministic generator tied to the seed.
    # This way, two trials with the same seed see the same shuffle order.
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)

    history       = TrainingHistory()
    patience_ctr  = 0
    best_val_loss = float("inf")
    best_epoch    = -1

    n_train = train_images.shape[0]

    log_fn(f"Starting training: n_train={n_train}, "
           f"n_val={val_images.shape[0]}, device={device}")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        # Shuffle once per epoch
        perm = torch.randperm(n_train, generator=g).to(device)

        train_loss = _train_one_epoch(
            model, train_images, perm, cfg.batch_size, optimizer, loss_fn,
        )
        val_loss = _eval_one_epoch(
            model, val_images, cfg.batch_size, loss_fn,
        )

        epoch_time = time.time() - t0

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.epoch_time_sec.append(epoch_time)
        history.total_epochs = epoch

        # Check for improvement (require at least 1e-6 better to count)
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            _save_checkpoint(checkpoint_path, model, cfg, epoch, val_loss)
        else:
            patience_ctr += 1

        history.best_val_loss = best_val_loss
        history.best_epoch    = best_epoch

        # Per-epoch log line
        marker = " *" if improved else ""
        log_fn(
            f"epoch {epoch:4d}/{cfg.max_epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"best={best_val_loss:.6f}@{best_epoch} | "
            f"patience={patience_ctr}/{cfg.patience} | "
            f"time={epoch_time:.2f}s{marker}"
        )

        # CSV row (if caller asked for it)
        if csv_row_fn is not None:
            csv_row_fn({
                "epoch":          epoch,
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "lr":             cfg.lr,
                "epoch_time_sec": epoch_time,
                "best_epoch":     best_epoch,
                "best_val_loss":  best_val_loss,
            })

        # External pruning hook (Optuna). Called every epoch so the study
        # can decide to kill slow starters -- but the callback itself
        # decides whether that's allowed (e.g., disable for first 10 trials).
        if on_epoch_end is not None:
            should_stop = on_epoch_end(epoch, train_loss, val_loss)
            if should_stop:
                history.stop_reason = "pruned"
                log_fn(f"  pruned at epoch {epoch}")
                break

        # Early stopping (only after min_epochs)
        if epoch >= cfg.min_epochs and patience_ctr >= cfg.patience:
            history.stop_reason = "early_stop"
            log_fn(f"  early stopping at epoch {epoch} "
                   f"(no improvement for {cfg.patience} epochs)")
            break
    else:
        history.stop_reason = "max_epochs"

    # Safety net: ensure at least one checkpoint exists. If no epoch improved
    # (which is essentially impossible but defensive), save the current state.
    if not checkpoint_path.is_file():
        log_fn(f"  warning: no checkpoint saved during training; "
               f"saving final state")
        _save_checkpoint(
            checkpoint_path, model, cfg,
            history.total_epochs, val_loss,
        )
        history.best_val_loss = val_loss
        history.best_epoch    = history.total_epochs

    log_fn(
        f"Training done: stop_reason={history.stop_reason}, "
        f"total_epochs={history.total_epochs}, "
        f"best_epoch={history.best_epoch}, best_val={history.best_val_loss:.6f}"
    )
    return history

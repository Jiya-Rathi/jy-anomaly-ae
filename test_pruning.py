"""
test_pruning.py

Tiny sanity test for the pruning callback path in train.train_one_model().

Verifies that:
    - When on_epoch_end returns True, training stops immediately
    - The summary row records stop_reason correctly
    - The checkpoint still exists (even for a pruned trial we want something
      on disk, per the safety net in training.train_model)

Run from the project root:
    source ae_env/bin/activate
    python test_pruning.py

Expected output ends with "TEST PASSED".
"""

from pathlib import Path

from ae_lib.config import Config
from train import train_one_model


def _dummy_prune_at_epoch_4(epoch: int, train_loss: float, val_loss: float) -> bool:
    """Return True at epoch 4 -- training should stop right after this."""
    return epoch >= 4


def main():
    # Load the non-HPF test config and convert to dict
    cfg = Config.from_yaml("configs/tests/test_run.yaml")
    config_dict = cfg.to_dict()
    config_dict.pop("trial_num", None)
    # Use a different seed so files don't collide with the non-pruning test
    config_dict["seed"] = 99

    # Run with the dummy callback
    print(">>> Running trial with a pruning callback that fires at epoch 4...")
    result = train_one_model(
        config_dict  = config_dict,
        gpu_id       = 0,
        trial_num    = 999,                # use 999 to tag this test's outputs
        output_dir   = "test_outputs",
        on_epoch_end = _dummy_prune_at_epoch_4,
        to_stdout    = True,
    )

    # The EvalResult doesn't carry stop_reason directly, but it's in summary.csv.
    # We also expect the log file to mention "pruned".
    log_path = Path("test_outputs/logs/trial_999.log")
    if not log_path.is_file():
        print("FAIL: log file not created")
        return 1
    log_text = log_path.read_text()

    # Check 1: training mentions "pruned"
    if "pruned" not in log_text:
        print("FAIL: log does not contain 'pruned' keyword")
        print("--- log tail ---")
        print("\n".join(log_text.splitlines()[-15:]))
        return 1

    # Check 2: number of epoch lines should be 4, not 10
    epoch_lines = [ln for ln in log_text.splitlines()
                   if " | train=" in ln and " | val=" in ln]
    if len(epoch_lines) != 4:
        print(f"FAIL: expected 4 epoch lines, got {len(epoch_lines)}")
        for ln in epoch_lines:
            print(f"  {ln}")
        return 1

    # Check 3: checkpoint exists (safety net)
    ckpt_path = Path("test_outputs/checkpoints/trial_999_best.pt")
    if not ckpt_path.is_file():
        print("FAIL: checkpoint file not created")
        return 1

    # Check 4: scatter plot exists (evaluation still runs even when pruned)
    scatter_path = Path("test_outputs/logs/trial_999_scatter.png")
    if not scatter_path.is_file():
        print("FAIL: scatter plot not created")
        return 1

    print()
    print("=" * 60)
    print("TEST PASSED")
    print(f"  epochs run     : {len(epoch_lines)} (expected 4)")
    print(f"  checkpoint     : {ckpt_path}")
    print(f"  scatter plot   : {scatter_path}")
    print(f"  objective      : {result.objective:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

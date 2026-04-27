"""
ae_lib/config.py

Hyperparameter configuration for the autoencoder.

One Config object holds every setting needed to train one AE:
    - Architecture (layers, channels, bottleneck)
    - Preprocessing (HPF on/off, sigma)
    - Training (learning rate, batch size, epochs, etc.)
    - Paths (manifests, outputs)

Two ways to create a Config:
    1. From a YAML file:    Config.from_yaml("baseline.yaml")
    2. From a dict:         Config.from_dict({"lr": 1e-3, ...})

The dict form is what Optuna will use. Both paths go through the same
validation so a bad value gets caught the same way.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import yaml


# -----------------------------------------------------------------------------
# Valid ranges for each hyperparameter
# -----------------------------------------------------------------------------
# These match the Optuna search space. If Optuna ever suggests a value outside
# these bounds (shouldn't happen, but defensive programming), validation catches
# it. When loading from YAML, this also catches typos in the config file.

_VALID_RANGES = {
    "bottleneck_dim":   (16, 256),
    "n_enc_layers":     (2, 5),
    "n_dec_layers":     (2, 5),
    "base_channels":    (8, 64),
    "growth_factor":    (1.5, 3.0),
    "lr":               (1e-5, 1e-2),
    "batch_size":       (8, 64),
    "hpf_sigma":        (1.0, 8.0),
}


# -----------------------------------------------------------------------------
# The Config dataclass
# -----------------------------------------------------------------------------

@dataclass
class Config:
    """All hyperparameters and settings for one AE training run."""

    # --- Architecture
    #
    # The AE has a flat-vector bottleneck (Option 2 design):
    #   encoder: n_enc_layers downsample blocks, each halving spatial size
    #            -> flatten -> dense layer -> [batch, bottleneck_dim]
    #   decoder: dense layer -> unflatten -> n_enc_layers upsample blocks
    #            with n_dec_layers conv layers per resolution
    #
    # bottleneck_dim is the literal size of the flat latent vector.
    # n_enc_layers controls total downsampling: image_size / 2**n_enc_layers
    #   must divide evenly, and must be >= 1 at the bottleneck.
    # n_dec_layers is independent from n_enc_layers (asymmetric AEs allowed).
    # base_channels and growth_factor together define channel counts:
    #   layer i has channels = base_channels * growth_factor**i, rounded.
    bottleneck_dim: int
    n_enc_layers:   int
    n_dec_layers:   int
    base_channels:  int
    growth_factor:  float
    use_batchnorm:  bool

    # --- Preprocessing
    use_hpf:   bool
    hpf_sigma: float           # ignored if use_hpf is False

    # --- Training
    lr:          float
    batch_size:  int
    max_epochs:  int = 300
    min_epochs:  int = 30
    patience:    int = 30      # early stopping patience on val loss

    # --- Reproducibility
    seed:          int  = 42
    deterministic: bool = False   # set True for final reference retrain

    # --- Data paths
    manifests_dir:   str = "/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/manifests"
    images_root:     str = "/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots"

    # --- Preprocessing constants (not hyperparameters; same across all trials)
    crop_box:   tuple = (72, 28, 300, 500)   # (left, top, right, bottom)
    image_size: int   = 256

    # --- Anomaly score calibration
    score_k: float = 37.0      # see research notes -- calibrated to saturate at max anomaly

    # --- Output
    output_dir:  str           = "outputs"
    trial_num:   Optional[int] = None   # set by tune.py, None for manual CLI runs

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a Config from a plain dict. Validates after construction."""
        cfg = cls(**d)
        cfg._validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "Config":
        """Load a Config from a YAML file. The YAML's top-level keys must match
        the dataclass field names exactly."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path) as f:
            d = yaml.safe_load(f)
        if not isinstance(d, dict):
            raise ValueError(f"config file {path} did not parse to a dict")
        return cls.from_dict(d)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self):
        """Sanity-check every field. Raises ValueError on any problem."""
        # Check range-bounded numeric fields
        for name, (lo, hi) in _VALID_RANGES.items():
            val = getattr(self, name)
            if val < lo or val > hi:
                raise ValueError(
                    f"{name}={val} is outside valid range [{lo}, {hi}]"
                )

        # Training schedule sanity
        if self.min_epochs > self.max_epochs:
            raise ValueError(
                f"min_epochs ({self.min_epochs}) must be <= max_epochs ({self.max_epochs})"
            )
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

        # Paths must exist
        mdir = Path(self.manifests_dir)
        if not mdir.is_dir():
            raise FileNotFoundError(f"manifests_dir not found: {mdir}")
        for needed in ("train.txt", "val.txt", "selection.txt"):
            if not (mdir / needed).is_file():
                raise FileNotFoundError(f"missing manifest: {mdir / needed}")

        iroot = Path(self.images_root)
        if not iroot.is_dir():
            raise FileNotFoundError(f"images_root not found: {iroot}")

        # Crop box sanity
        l, t, r, b = self.crop_box
        if r <= l or b <= t:
            raise ValueError(f"invalid crop_box: {self.crop_box}")

        # Image size
        if self.image_size < 32 or self.image_size > 1024:
            raise ValueError(f"image_size {self.image_size} looks wrong")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for logging)."""
        return asdict(self)

    def save_yaml(self, path):
        """Write this config to a YAML file (e.g., to save the winning trial)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def summary_str(self) -> str:
        """Short human-readable summary, one line per important field."""
        lines = [
            "--- Config summary ---",
            f"  bottleneck_dim  : {self.bottleneck_dim}",
            f"  n_enc_layers    : {self.n_enc_layers}",
            f"  n_dec_layers    : {self.n_dec_layers}",
            f"  base_channels   : {self.base_channels}",
            f"  growth_factor   : {self.growth_factor}",
            f"  use_batchnorm   : {self.use_batchnorm}",
            f"  use_hpf         : {self.use_hpf}",
            f"  hpf_sigma       : {self.hpf_sigma}",
            f"  lr              : {self.lr}",
            f"  batch_size      : {self.batch_size}",
            f"  max_epochs      : {self.max_epochs}",
            f"  patience        : {self.patience}",
            f"  seed            : {self.seed}",
            f"  deterministic   : {self.deterministic}",
            f"  trial_num       : {self.trial_num}",
        ]
        return "\n".join(lines)

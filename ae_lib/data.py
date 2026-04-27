"""
ae_lib/data.py

Loads images listed in the manifest files and prepares them for training.

Public API:
    load_split(manifest_path, cfg, device) -> SplitData
        Reads one manifest, loads and preprocesses all listed images,
        stacks them to a single GPU tensor. If cfg.use_hpf is True,
        applies HPF on GPU after preprocessing.

    SplitData
        A small container with fields:
            images    torch.Tensor  [N, 1, H, W]  on device
            classes   list[str]     length N
            shots     list[str]     length N (e.g., '187138_3000')
            paths     list[str]     length N, relative to images_root

Design decisions (see research notes):
    - Preprocessing on CPU (crop/resize/grayscale/normalize) using PIL.
      This happens once at startup; speed is not a bottleneck.
    - HPF on GPU as a single depthwise conv across the whole batch.
    - Per-image min-max normalization before HPF so HPF's (+0.5, clamp)
      step lands in the expected range.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Container returned by load_split
# -----------------------------------------------------------------------------

@dataclass
class SplitData:
    """All images from one manifest, preloaded to GPU, with metadata."""
    images:  torch.Tensor   # [N, 1, H, W], on device, float32 in [0, 1]
    classes: List[str]      # parallel list of class labels
    shots:   List[str]      # parallel list of shot identifiers
    paths:   List[str]      # parallel list of relative paths (for debugging)

    def __len__(self) -> int:
        return self.images.shape[0]


# -----------------------------------------------------------------------------
# Manifest parsing
# -----------------------------------------------------------------------------

def _read_manifest(path: Path) -> List[tuple]:
    """Read a manifest file. Returns list of (rel_path, class) tuples.

    Skips comment lines (starting with '#') and blank lines.
    """
    entries = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"manifest {path}: malformed line (expected '<path>\\t<class>'): {ln!r}"
                )
            entries.append((parts[0], parts[1]))
    return entries


def _shot_from_path(rel_path: str) -> str:
    """Extract '<shot>_3000' from 'healthy/jy_187138_3000.png'."""
    stem = Path(rel_path).stem               # 'jy_187138_3000'
    parts = stem.split("_", 1)               # ['jy', '187138_3000']
    return parts[1] if len(parts) == 2 else stem


# -----------------------------------------------------------------------------
# CPU preprocessing per image
# -----------------------------------------------------------------------------

def _preprocess_one(path: Path, crop_box: tuple, image_size: int) -> np.ndarray:
    """Load image from disk, crop/resize/grayscale/normalize.

    Returns a float32 numpy array of shape [1, H, W] with values in [0, 1].
    """
    img = Image.open(path)
    img = img.crop(crop_box)                         # (left, top, right, bottom)
    img = img.resize((image_size, image_size),
                     resample=Image.BILINEAR)
    img = img.convert("L")                            # grayscale, single channel

    arr = np.asarray(img, dtype=np.float32)           # [H, W]
    # Per-image min-max normalization to [0, 1]
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)                      # constant image -> zeros

    return arr[np.newaxis, :, :]                      # [1, H, W]


# -----------------------------------------------------------------------------
# HPF on GPU
# -----------------------------------------------------------------------------

def _gaussian_kernel_2d(sigma: float, device, dtype=torch.float32) -> torch.Tensor:
    """Build a 2D Gaussian kernel. Kernel size = 2*ceil(3*sigma)+1 (odd)."""
    radius = int(np.ceil(3.0 * sigma))
    size   = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d    = g1d / g1d.sum()
    kernel = g1d[:, None] * g1d[None, :]              # [size, size]
    return kernel.view(1, 1, size, size)              # [1, 1, size, size]


def _apply_hpf(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply high-pass filter to a batch of images on the same device.

    HPF = image - gaussian_blur(image) + 0.5, then clamped to [0, 1].

    images: [N, 1, H, W] float in [0, 1], on any device.
    Returns a new tensor of the same shape on the same device.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"expected [N, 1, H, W], got {tuple(images.shape)}")

    kernel = _gaussian_kernel_2d(sigma, images.device, images.dtype)
    radius = kernel.shape[-1] // 2

    # 'reflect' padding avoids dark-edge artefacts at the image boundary.
    blurred = F.conv2d(
        F.pad(images, (radius, radius, radius, radius), mode="reflect"),
        kernel,
    )
    hp = images - blurred + 0.5
    return hp.clamp_(0.0, 1.0)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def load_split(manifest_path, cfg, device) -> SplitData:
    """Load one manifest's worth of images, preprocess, stack, move to GPU.

    Parameters
    ----------
    manifest_path : str or Path
        Path to train.txt, val.txt, or selection.txt.
    cfg : Config
        Used for: crop_box, image_size, images_root, use_hpf, hpf_sigma.
    device : torch.device
        Where the output tensor should live (e.g. torch.device('cuda:0')).

    Returns
    -------
    SplitData
    """
    manifest_path = Path(manifest_path)
    entries       = _read_manifest(manifest_path)
    if len(entries) == 0:
        raise ValueError(f"manifest {manifest_path} has no entries")

    images_root = Path(cfg.images_root)

    # --- CPU preprocessing: accumulate arrays one at a time
    arrs    = []
    classes = []
    shots   = []
    paths   = []
    for rel, cls in entries:
        full = images_root / rel
        if not full.is_file():
            raise FileNotFoundError(
                f"image listed in manifest but not on disk: {full}"
            )
        arrs.append(_preprocess_one(full, cfg.crop_box, cfg.image_size))
        classes.append(cls)
        shots.append(_shot_from_path(rel))
        paths.append(rel)

    # --- Stack to a single tensor and move to GPU
    batch = np.stack(arrs, axis=0)                    # [N, 1, H, W]
    tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)

    # --- HPF on GPU (in-place-ish; replaces tensor)
    if cfg.use_hpf:
        tensor = _apply_hpf(tensor, cfg.hpf_sigma)

    return SplitData(images=tensor, classes=classes, shots=shots, paths=paths)

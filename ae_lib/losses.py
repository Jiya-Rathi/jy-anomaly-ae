"""
ae_lib/losses.py

Differentiable SSIM loss (for training) and per-sample DSSIM (for scoring).

Public API:
    SSIMLoss(window_size=11, sigma=1.5)
        nn.Module. forward(recon, target) -> scalar = 1 - mean(SSIM).
        Drop-in replacement for nn.MSELoss in the training loop.

    per_sample_dssim(x, y, window_size=11, sigma=1.5) -> Tensor [B]
        Per-image anomaly score = 1 - mean(SSIM over that image).
        Higher = more anomalous, same convention as reconstruction MSE.

    gaussian_window(size, sigma, channels, device, dtype)
        Builds the Gaussian weighting window used by both.

SSIM definition (Wang et al. 2004), computed per local window:
    SSIM = [(2 mu_x mu_y + C1)(2 sigma_xy + C2)] /
           [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]
with C1 = (0.01)^2, C2 = (0.03)^2 for images normalised to [0, 1].

Both training loss and anomaly score use the SAME SSIM machinery so that
"train with SSIM, score with SSIM" is internally consistent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default stability constants for data in [0, 1]
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def gaussian_window(size: int, sigma: float, channels: int,
                    device, dtype=torch.float32) -> torch.Tensor:
    """
    Build a normalised 2D Gaussian window for SSIM convolution.

    Returns a tensor of shape [channels, 1, size, size] suitable for use
    as a grouped depthwise conv kernel (groups=channels).
    """
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d    = g1d / g1d.sum()
    win2d  = g1d[:, None] * g1d[None, :]                 # [size, size]
    return win2d.view(1, 1, size, size).expand(channels, 1, size, size).contiguous()


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
              C1: float = _C1, C2: float = _C2) -> torch.Tensor:
    """
    Compute the per-pixel SSIM map between two batches.

    x, y    : [B, C, H, W] in [0, 1]
    window  : [C, 1, k, k] grouped Gaussian kernel
    Returns : [B, C, H, W] SSIM map (values in [-1, 1])
    """
    channels = x.shape[1]
    pad      = window.shape[-1] // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    ssim = (((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
            ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
    return ssim


class SSIMLoss(nn.Module):
    """
    SSIM training loss: 1 - mean(SSIM).  Minimising this maximises structural
    similarity between reconstruction and target.

    Caches the Gaussian window per (device, channels) so it is built once.
    Drop-in replacement for nn.MSELoss(reduction='mean') in training.py:
        loss_fn = SSIMLoss()
        loss    = loss_fn(reconstruction, batch)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma       = sigma
        self._window     = None      # lazily built on first forward

    def _get_window(self, ref: torch.Tensor) -> torch.Tensor:
        channels = ref.shape[1]
        if (self._window is None
                or self._window.device != ref.device
                or self._window.dtype  != ref.dtype
                or self._window.shape[0] != channels):
            self._window = gaussian_window(
                self.window_size, self.sigma, channels, ref.device, ref.dtype
            )
        return self._window

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self._get_window(recon)
        ssim   = _ssim_map(recon, target, window)
        return 1.0 - ssim.mean()


def per_sample_dssim(x: torch.Tensor, y: torch.Tensor,
                     window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Per-image DSSIM anomaly score = 1 - mean(SSIM over that image).

    x, y : [B, C, H, W] in [0, 1]
    Returns : [B] tensor; higher = more anomalous.
    """
    window = gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    ssim   = _ssim_map(x, y, window)                 # [B, C, H, W]
    return 1.0 - ssim.mean(dim=(1, 2, 3))            # [B]

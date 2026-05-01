"""
ae_lib/model_mlp.py

Fully-connected (MLP) autoencoder, drop-in alternative to the CNN
Autoencoder in ae_lib/model.py. Created in response to professor
suggestion that the convolutional inductive bias may be limiting
broken_lcfs detection.

KEY DESIGN CHOICES (and why)
============================

1. Optional input downsampling (input_size config field).
   At 256x256 = 65,536 input dims with only 722 training healthy images,
   even modest first-hidden-layer sizes give 100,000+ parameters per
   training example -- pure memorization regime. Downsampling to 64x64
   (4096 dims) brings this into a more reasonable range.

   The previous concern was that 128x128 destroyed broken_lcfs detail
   for CNN models. That observation was about *convolutional kernels*,
   which rely on local spatial structure. For an MLP, every pixel is
   its own feature, so the spatial-detail tradeoff is different. We
   re-test 64x64 here as a fresh experiment.

2. Hidden layer sizes: explicit list, not parameterized depth.
   Avoids reintroducing a hyperparameter search. Defaults chosen to
   keep total parameters around 30M.

3. BatchNorm + ReLU + optional dropout.
   Matches the activation/normalization choices of the CNN AE that
   trial 142 used, so the only thing changing vs trial 142 is the
   architecture class itself.

4. Sigmoid output.
   Images are normalized to [0, 1] by the data pipeline. Sigmoid
   bounds the reconstruction to the same range. (CNN AE does the
   same.)
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAutoencoder(nn.Module):
    """Fully-connected autoencoder for jy images.

    Args:
        input_size: spatial size H=W of the input images. If smaller than
            the data pipeline's image_size, the forward pass will downsample
            with adaptive_avg_pool2d before flattening.
        bottleneck_dim: size of the latent vector.
        hidden_sizes: list of hidden layer widths in the encoder. The
            decoder uses the reverse. e.g. [1024, 256] -> encoder is
            input -> 1024 -> 256 -> bottleneck, decoder is bottleneck ->
            256 -> 1024 -> output.
        use_batchnorm: whether to add BatchNorm1d between hidden layers.
        dropout: dropout probability in encoder hidden layers (0 = off).
            Decoder is dropout-free (don't want to inject noise into
            reconstruction).
    """

    def __init__(
        self,
        input_size:     int        = 64,
        bottleneck_dim: int        = 240,
        hidden_sizes:   List[int]  = (1024, 256),
        use_batchnorm:  bool       = True,
        dropout:        float      = 0.1,
    ):
        super().__init__()
        self.input_size     = input_size
        self.flat_dim       = input_size * input_size
        self.bottleneck_dim = bottleneck_dim
        self.hidden_sizes   = list(hidden_sizes)

        # ----- Encoder
        enc_layers: List[nn.Module] = []
        prev = self.flat_dim
        for h in self.hidden_sizes:
            enc_layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        # Final encoder layer to bottleneck, no activation
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ----- Decoder (mirror of encoder, no dropout)
        dec_layers: List[nn.Module] = []
        prev = bottleneck_dim
        for h in reversed(self.hidden_sizes):
            dec_layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(nn.ReLU(inplace=True))
            prev = h
        # Final reconstruction layer to flat image, sigmoid for [0,1] range
        dec_layers.append(nn.Linear(prev, self.flat_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    # -------------------------------------------------------------------------

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """If the data pipeline produces a larger image than input_size,
        adaptively downsample. No-op when sizes match."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.adaptive_avg_pool2d(x, (self.input_size, self.input_size))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward returns the reconstructed image AT THE MODEL'S WORKING
        RESOLUTION (input_size). If the caller passes a 256x256 image and
        input_size is 64, the output is 64x64.

        The reconstruction-error scoring code must therefore downsample the
        original image before computing the error. See reconstruction_error()
        below for the canonical pattern.
        """
        x_small = self._maybe_downsample(x)             # [B, 1, S, S]
        flat    = x_small.flatten(start_dim=1)          # [B, S*S]
        z       = self.encoder(flat)                    # [B, bottleneck]
        recon   = self.decoder(z)                       # [B, S*S]
        return recon.view(-1, 1, self.input_size, self.input_size)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image MSE between input and reconstruction, both at the
        model's working resolution.

        Returns shape [B] (one scalar per image).
        """
        x_small = self._maybe_downsample(x)
        recon   = self.forward(x)   # already downsamples internally
        return ((x_small - recon) ** 2).mean(dim=(1, 2, 3))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Construction helper that reads the project Config.
# Add a new field `model_type: str = "cnn"` to Config to switch between
# the existing CNN Autoencoder and this MLPAutoencoder.
# -----------------------------------------------------------------------------

def build_mlp_from_config(cfg) -> MLPAutoencoder:
    """Build an MLPAutoencoder from the project Config object.

    Reads optional fields that we add to Config for MLP support:
      - mlp_input_size   (default 64)
      - mlp_hidden_sizes (default [1024, 256])
      - mlp_dropout      (default 0.1)
    Reads the existing fields:
      - bottleneck_dim
      - use_batchnorm
    """
    return MLPAutoencoder(
        input_size     = getattr(cfg, "mlp_input_size", 64),
        bottleneck_dim = cfg.bottleneck_dim,
        hidden_sizes   = getattr(cfg, "mlp_hidden_sizes", [1024, 256]),
        use_batchnorm  = cfg.use_batchnorm,
        dropout        = getattr(cfg, "mlp_dropout", 0.1),
    )

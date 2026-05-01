"""
ae_lib/model.py

The convolutional autoencoder, fully parametrized by a Config.

Design (see research notes "Option 2 -- flat vector bottleneck"):

    Encoder
        Input: [N, 1, image_size, image_size]
        For i in 1..n_enc_layers:
            Conv(prev -> channels[i]) + [BN] + ReLU + MaxPool(2x2)
        Output: [N, channels[last], image_size / 2**n_enc_layers, ...]
        Flatten -> Dense(-> bottleneck_dim)

    Bottleneck: [N, bottleneck_dim]

    Decoder
        Dense(bottleneck_dim -> channels[last] * spatial * spatial)
        Unflatten
        For i in 1..n_enc_layers   (must match encoder count for spatial symmetry)
            Upsample(scale=2)
            Repeat n_dec_layers times:
                Conv(prev -> channels[i]) + [BN] + ReLU
        Final Conv(-> 1) + Sigmoid
        Output: [N, 1, image_size, image_size]

Note on asymmetry:
    n_enc_layers == number of downsample stages == number of upsample stages.
    n_dec_layers controls how many convs are inserted at EACH decoder
    resolution. n_enc_layers and n_dec_layers may differ.
"""

from typing import List

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _channel_schedule(base: int, growth: float, n_layers: int) -> List[int]:
    """Channel counts per encoder layer: base * growth**i, clamped to >=1.

    Example: base=16, growth=2.0, n_layers=4 -> [16, 32, 64, 128]
    """
    return [max(1, int(round(base * (growth ** i)))) for i in range(n_layers)]


def _conv_block(in_c: int, out_c: int, use_bn: bool) -> nn.Sequential:
    """Single conv + optional BN + ReLU block. Kernel 3x3, stride 1, pad 1."""
    layers: List[nn.Module] = [
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self,
                 image_size:     int,
                 n_enc_layers:   int,
                 channels:       List[int],
                 bottleneck_dim: int,
                 use_batchnorm:  bool):
        super().__init__()

        # Build a stack of (conv -> [BN] -> ReLU -> maxpool) blocks.
        # Input channel of layer i is either 1 (first) or channels[i-1].
        stages: List[nn.Module] = []
        prev_c = 1
        for i in range(n_enc_layers):
            stages.append(_conv_block(prev_c, channels[i], use_batchnorm))
            stages.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_c = channels[i]
        self.stages = nn.Sequential(*stages)

        # After n_enc_layers halvings, the spatial size is:
        self._spatial_after = image_size // (2 ** n_enc_layers)
        if self._spatial_after < 1:
            raise ValueError(
                f"n_enc_layers={n_enc_layers} with image_size={image_size} "
                f"leaves spatial size < 1; reduce n_enc_layers"
            )

        # Flatten + dense to flat bottleneck
        flat_size = channels[-1] * self._spatial_after * self._spatial_after
        self.flatten  = nn.Flatten()
        self.to_latent = nn.Linear(flat_size, bottleneck_dim)

        # Save these so the decoder can mirror them
        self.final_channels = channels[-1]
        self.final_spatial  = self._spatial_after
        self.flat_size      = flat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = self.flatten(x)
        x = self.to_latent(x)
        return x


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self,
                 n_enc_layers:   int,
                 n_dec_layers:   int,
                 channels:       List[int],        # same list encoder used
                 bottleneck_dim: int,
                 final_spatial:  int,
                 use_batchnorm:  bool,
                 upsample_mode:  str = "nearest"):
        super().__init__()

        # Dense layer to expand latent back to spatial feature map
        flat_size = channels[-1] * final_spatial * final_spatial
        self.from_latent = nn.Linear(bottleneck_dim, flat_size)
        self.unflatten   = nn.Unflatten(
            1, (channels[-1], final_spatial, final_spatial)
        )

        # We mirror the encoder's downsample stages with upsample stages.
        # At each resolution, insert n_dec_layers conv blocks.
        #
        # Channel targets during decoder (reverse of encoder, then down to 1):
        #   starting at channels[-1], then channels[-2], ..., channels[0]
        #
        # After the last upsample, we do a final 1x1 conv -> 1 channel + sigmoid.
        stages: List[nn.Module] = []
        for i in range(n_enc_layers):
            # Channel count entering this upsample stage
            # (must match stage_in_c logic below so shapes line up).
            if i == 0:
                up_c = channels[-1]
            else:
                up_c = channels[n_enc_layers - i]

            if upsample_mode == "convtranspose":
                # k=2, s=2, p=0  ->  output spatial = 2 * input spatial,
                # exact mirror of MaxPool2d(2,2). Preserves channel count;
                # the conv blocks that follow handle the channel reduction.
                stages.append(nn.ConvTranspose2d(
                    up_c, up_c, kernel_size=2, stride=2,
                ))
            else:
                stages.append(nn.Upsample(scale_factor=2, mode="nearest"))

            # Target channel count for THIS decoder stage
            # (mirrored from encoder: stage 0 of decoder ~ last encoder channel,
            #  stage n_enc_layers-1 of decoder ~ first encoder channel)
            enc_index_to_mirror = n_enc_layers - 1 - i
            stage_out_c = channels[enc_index_to_mirror]

            # Input channel for this stage's first conv:
            # - For the first decoder stage, input is channels[-1] (from unflatten)
            # - For later stages, input is the PREVIOUS stage's out channel,
            #   which is channels[n_enc_layers - 1 - (i-1)] = channels[n_enc_layers - i]
            if i == 0:
                stage_in_c = channels[-1]
            else:
                stage_in_c = channels[n_enc_layers - i]

            # n_dec_layers conv blocks at this resolution.
            # First conv changes channels from stage_in_c -> stage_out_c;
            # subsequent convs keep stage_out_c -> stage_out_c.
            for j in range(n_dec_layers):
                in_c  = stage_in_c if j == 0 else stage_out_c
                out_c = stage_out_c
                stages.append(_conv_block(in_c, out_c, use_batchnorm))

        self.stages = nn.Sequential(*stages)

        # Final projection to single-channel output + sigmoid
        self.out_conv   = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.unflatten(x)
        x = self.stages(x)
        x = self.out_conv(x)
        x = self.activation(x)
        return x


# -----------------------------------------------------------------------------
# Full Autoencoder
# -----------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """The full AE. Takes a Config, builds encoder + decoder to match."""

    def __init__(self, cfg):
        super().__init__()

        channels = _channel_schedule(
            cfg.base_channels, cfg.growth_factor, cfg.n_enc_layers
        )

        self.encoder = Encoder(
            image_size     = cfg.image_size,
            n_enc_layers   = cfg.n_enc_layers,
            channels       = channels,
            bottleneck_dim = cfg.bottleneck_dim,
            use_batchnorm  = cfg.use_batchnorm,
        )
        self.decoder = Decoder(
            n_enc_layers   = cfg.n_enc_layers,
            n_dec_layers   = cfg.n_dec_layers,
            channels       = channels,
            bottleneck_dim = cfg.bottleneck_dim,
            final_spatial  = self.encoder.final_spatial,
            use_batchnorm  = cfg.use_batchnorm,
            upsample_mode  = cfg.decoder_upsample,
        )

        self._channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return just the latent vector."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between x and its reconstruction.
        Returns a 1D tensor of length batch_size."""
        x_hat = self.forward(x)
        # MSE over all non-batch dims -> one number per sample
        return ((x - x_hat) ** 2).mean(dim=(1, 2, 3))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        lines = [
            "--- Architecture summary ---",
            f"  channels per enc layer : {self._channels}",
            f"  final spatial size     : "
            f"{self.encoder.final_spatial}x{self.encoder.final_spatial}",
            f"  flat size before latent: {self.encoder.flat_size}",
            f"  bottleneck_dim         : {self.encoder.to_latent.out_features}",
            f"  total parameters       : {self.parameter_count():,}",
        ]
        return "\n".join(lines)

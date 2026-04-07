import torch
import torch.nn as nn
from src.spectralnet.core.layers.remizov_shift import RemizovShiftLayer, ShiftMixerBlock
from src.spectralnet.core.layers.spectral_remizov_layer import (
    SpectralRemizovLayer, get_activation,
)
from src.spectralnet.core.layers.aggregation import ResolventAggregation


def physical_act(x: torch.Tensor) -> torch.Tensor:
    return x * torch.exp(-x.abs().pow(2))


# Allowed aggregation types - explicit list for validation at startup
_VALID_AGG_TYPES = {"resolvent", "mean", "last"}


class SpectralNetS(nn.Module):
    """
    SpectralNet-S v1.4

    Changes relative to v1.3:
      - Support for layer_type="shift_rich" (RMSB-R1).
        Instead of a single shared evolution_step, an nn.ModuleList of N
        independent ShiftMixerBlocks is used (Shift → BN → 1×1 expand → GELU →
        1×1 project → residual). Each block has its own parameters, which
        allows scaling the parameter budget to the level of the spectral baseline.
        Switching: model.evolution.layer_type=shift_rich,
                   model.evolution.n_blocks=<N>,
                   model.evolution.expansion=<E>.

    Changes relative to v1.2:
      - Support for three aggregation modes via model_cfg.aggregation.type:
          "resolvent" — R_φ(λ): weighted sum w_k ~ e^{-λkτ}τ  [baseline EXP-003/004]
          "mean"      — uniform average over the trajectory   [ablation C.3]
          "last"      — only the last state u_n               [ablation C.3]
        Switching via Hydra: model.aggregation.type="mean"
      - ResolventAggregation is always initialized (n_steps is needed for the loop).
        When agg_type != "resolvent", its weights do not participate in the forward pass,
        but are present in the model parameters — this is fair for the parameter counter.
      - Validation of agg_type in __init__ rather than in forward — a config error
        is detected before data loading and training starts.
    """

    def __init__(self, model_cfg):
        super().__init__()

        channels    = model_cfg.channels
        hidden      = getattr(model_cfg, "head_hidden", 128)
        num_classes = getattr(model_cfg, "num_classes", 10)
        layer_type  = getattr(model_cfg.evolution, "layer_type", "spectral")
        modes      = getattr(model_cfg.core, "spectral_modes", 16)
        tau        = model_cfg.evolution.tau
        n_steps    = model_cfg.evolution.n_steps

        # Read and validate aggregation type
        agg_type = getattr(model_cfg.aggregation, "type", "resolvent")
        if agg_type not in _VALID_AGG_TYPES:
            raise ValueError(
                f"aggregation.type='{agg_type}' is not supported. "
                f"Allowed values: {_VALID_AGG_TYPES}"
            )
        self.agg_type = agg_type

        # Activation: switchable via config (C.2 ablation)
        act_name = getattr(model_cfg, "activation", "physicalact")
        self.act = get_activation(act_name)

        # Stem: the only BN in the architecture
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.layer_type = layer_type
        rank = getattr(model_cfg.core, "rank", 0)
        shift_hidden = getattr(model_cfg.core, "shift_hidden", 0)

        if layer_type == "shift_rich":
            # RMSB-R1: independent ShiftMixerBlock per step; each block has its own
            # parameters so the total budget scales with n_blocks × expansion.
            n_blocks   = getattr(model_cfg.evolution, "n_blocks", n_steps)
            expansion  = getattr(model_cfg.evolution, "expansion", 16)
            self.evolution_steps = nn.ModuleList([
                ShiftMixerBlock(
                    channels=channels,
                    expansion=expansion,
                    tau=tau,
                    spatial_res=(32, 32),
                )
                for _ in range(n_blocks)
            ])
            # n_blocks used as n_steps so ResolventAggregation weights align with
            # trajectory length; also stored on self for forward.
            self._n_blocks = n_blocks
        elif layer_type == "shift":
            self.evolution_step = RemizovShiftLayer(
                channels=channels, spatial_res=(32, 32), tau=tau,
            )
        else:
            self.evolution_step = SpectralRemizovLayer(
                channels=channels, modes=modes,
                activation=get_activation(act_name),
                rank=rank,
                shift_hidden=shift_hidden,
                tau=tau,
            )

        # ResolventAggregation is always initialized:
        # - for agg_type="resolvent" — main aggregation path
        # - for "mean"/"last" — not used in forward, but n_steps
        #   is needed for the evolution loop; weights are fairly counted in param count
        _agg_n = getattr(self, "_n_blocks", n_steps)
        self.resolvent_agg = ResolventAggregation(
            n_steps=_agg_n,
            tau=tau,
            init_lambda=model_cfg.aggregation.init_lambda,
        )

        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.stem(x))

        trajectory = [u]
        if self.layer_type == "shift_rich":
            # Each block is independent; residual connection is inside the block.
            for block in self.evolution_steps:
                u = block(u)
                trajectory.append(u)
        else:
            for _ in range(self.resolvent_agg.n_steps):
                u = self.evolution_step(u)
                if self.layer_type == "shift":
                    u = self.act(u)
                trajectory.append(u)

        if self.agg_type == "resolvent":
            res = self.resolvent_agg(trajectory)
        elif self.agg_type == "mean":
            res = torch.stack(trajectory, dim=0).mean(dim=0)
        else:
            res = trajectory[-1]

        z = res.mean(dim=(-2, -1))
        z = self.act(self.fc1(z))
        return self.fc2(z)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class RemizovShiftLocal(nn.Module):
    """Spatially-varying Remizov shift (Theorems 4-6) with learned local coefficients.

    Generates a(x), b_x(x), b_y(x), c(x) from input features via a small conv
    network, then applies the 2D composed diffusion-drift Chernoff step:

        (S(τ)u)(x) = ¼[ u(x + s_x + d_x, y + d_y)
                      + u(x - s_x + d_x, y + d_y)
                      + u(x + d_x, y + s_y + d_y)
                      + u(x + d_x, y - s_y + d_y) ]
                   + τ·c(x)·u(x)

    where s(x) = √(a(x)·τ), d_x = b_x(x)·τ, d_y = b_y(x)·τ.

    Drift is composed INTO the diffusion shifts: when s=0, all four grids
    collapse to (x+d_x, y+d_y), giving pure advection without parasitic
    numerical diffusion.

    Key differences from the scalar RemizovShiftLayer:
      - Coefficients are spatially varying (B, C, H, W), not per-channel scalars.
      - Input-adaptive: coefficient maps depend on the current feature map.
      - Designed as a branch inside SpectralRemizovLayer (C.5 hybrid).
    """

    def __init__(self, channels: int, shift_hidden: int = 8, tau: float = 0.1):
        super().__init__()
        self.channels = channels
        self.tau = tau

        # Small conv network: input features → 4 coefficient maps per channel.
        # Output layout per channel: [a_raw, bx, by, c_raw].
        self.coeff_net = nn.Sequential(
            nn.Conv2d(channels, shift_hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(shift_hidden, channels * 4, 1, bias=True),
        )
        self._init_coefficients()

        # Learnable residual scaling; exp(-2.3) ≈ 0.1 at init → gentle start.
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.1)))

    def _init_coefficients(self):
        # First conv: small Kaiming init so gradients flow but output is modest.
        nn.init.kaiming_normal_(self.coeff_net[0].weight, nonlinearity='relu')
        self.coeff_net[0].weight.data *= 0.1

        # Output conv: zero weight → output is pure bias at init (input-independent).
        nn.init.zeros_(self.coeff_net[-1].weight)

        C = self.channels
        bias = self.coeff_net[-1].bias.data
        bias[:C] = -4.0        # a: softplus(-4) ≈ 0.018 → s ≈ 0.04 (< 1 pixel)
        bias[C:2*C] = 0.0      # bx: no drift
        bias[2*C:3*C] = 0.0    # by: no drift
        bias[3*C:] = 0.0       # c:  no potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        coeffs = self.coeff_net(x).view(B, C, 4, H, W)

        a = F.softplus(coeffs[:, :, 0])               # > 0 (diffusion)
        bx = coeffs[:, :, 1]                           # drift x
        by = coeffs[:, :, 2]                           # drift y
        c = torch.clamp(coeffs[:, :, 3], -5.0, 5.0)   # potential, stability clamp

        s = torch.sqrt(a * self.tau + 1e-8)            # (B, C, H, W)
        dx = bx * self.tau
        dy = by * self.tau

        x_flat = x.reshape(B * C, 1, H, W)

        # Base grid in normalized [-1, 1] coordinates.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij',
        )
        base = torch.stack((grid_x, grid_y), dim=-1)           # (H, W, 2)
        grid = base.unsqueeze(0).expand(B * C, -1, -1, -1)     # (B*C, H, W, 2)

        s_f  = s.reshape(B * C, H, W, 1)
        dx_f = dx.reshape(B * C, H, W, 1)
        dy_f = dy.reshape(B * C, H, W, 1)

        # Drift is composed into each diffusion grid so that when s→0
        # all four grids collapse to pure advection (x+dx, y+dy).
        grid_px = grid + torch.cat([s_f  + dx_f, dy_f],         dim=-1)
        grid_nx = grid + torch.cat([-s_f + dx_f, dy_f],         dim=-1)
        grid_py = grid + torch.cat([dx_f,        s_f  + dy_f],  dim=-1)
        grid_ny = grid + torch.cat([dx_f,        -s_f + dy_f],  dim=-1)

        def _sample(g: torch.Tensor) -> torch.Tensor:
            return F.grid_sample(
                x_flat, g, align_corners=True,
                mode='bilinear', padding_mode='border',
            )

        u_px = _sample(grid_px)
        u_nx = _sample(grid_nx)
        u_py = _sample(grid_py)
        u_ny = _sample(grid_ny)

        # 2D composed diffusion-drift Chernoff step: ¼ per direction.
        evolution = 0.25 * (u_px + u_nx + u_py + u_ny)
        evolution = evolution.view(B, C, H, W)

        shift_out = evolution + self.tau * c * x

        return torch.exp(self.log_scale) * shift_out


class RemizovShiftLayer(nn.Module):
    """
    Implements one Chernoff step according to Theorem 6 (Remizov) with composed drift:

      (S(τ)u)(x) = ¼[ u(x + s + dx, y + dy) + u(x - s + dx, y + dy)
                     + u(x + dx, y + s + dy) + u(x + dx, y - s + dy) ]
                 + τ·c·u(x)

    where s = √(a·τ), dx = bx·τ, dy = by·τ.

    Drift is embedded into each diffusion shift: when s→0 all four
    grids collapse into pure advection u(x+dx, y+dy) without
    parasitic numerical diffusion.

    Note on scaling:
        In the original Theorem 6, the diffusion shift is 2·sqrt(a·τ).
        Here we use sqrt(a·τ) — the coefficient 2 is absorbed into the learnable
        parameter a, which is equivalent to replacing a_theorem → 4·a_here.
        The convergence of the approximation is preserved. When comparing values of a
        with the literature, apply the conversion: a_theorem = a_here / 4.

    v1.3 Fixes (composed drift):
    - Drift is embedded into diffusion shifts (composition instead of interpolation).
    - Parasitic numerical diffusion under pure drift is eliminated.
    - theta_a is initialized at -4.0 → softplus(-4) ≈ 0.018.
    - theta_b is split into b_x and b_y independently.
    """

    def __init__(self, channels: int, spatial_res: tuple = (32, 32), tau: float = 0.02):
        super().__init__()
        self.channels = channels
        self.tau = tau
        h, w = spatial_res

        # a(x) > 0 via softplus; initialization -4 → a_0 ≈ 0.018
        # At tau=0.02: sqrt(a·τ) ≈ sqrt(0.018·0.02) ≈ 0.019 — small step
        self.theta_a = nn.Parameter(torch.full((1, channels, 1, 1), -4.0))

        # b_x, b_y — drift along each axis independently
        self.theta_bx = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.theta_by = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # c(x) — potential
        self.theta_c = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Base coordinate grid [-1, 1] × [-1, 1], shape (1, H, W, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        # grid[..., 0] = x-coordinate, grid[..., 1] = y-coordinate
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # (1, H, W, 2)
        self.register_buffer("base_grid", base)

    def _sample(self, x_flat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """grid_sample wrapper. x_flat: (B*C, 1, H, W), grid: (B*C, H, W, 2)."""
        return F.grid_sample(x_flat, grid, align_corners=True,
                             mode='bilinear', padding_mode='border')

    def forward(self, x: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B C H W"]:
        B, C, H, W = x.shape

        a  = F.softplus(self.theta_a)           # (1, C, 1, 1), > 0
        bx = self.theta_bx                       # (1, C, 1, 1)
        by = self.theta_by                       # (1, C, 1, 1)
        c  = self.theta_c                        # (1, C, 1, 1)

        # Scale of the diffusion shift along each axis.
        #
        # Note on rescaling relative to Theorem 6 (Remizov):
        #   Original formula: shift = 2·sqrt(a·τ)
        #   Here:             shift =   sqrt(a·τ)   (coefficient 2 is absorbed into a)
        #
        # The parameter a in this implementation is equivalent to 4·a from the original theorem.
        # This is a correct rescaling: the convergence of the approximation and the condition
        # F_φ(0)=I are not violated, only the interpretation of the diffusion scale changes.
        # When comparing with literature, note: a_here = 4 · a_theorem.
        s = torch.sqrt(a * self.tau + 1e-8)      # (1, C, 1, 1)

        # Drift along axes
        dx = bx * self.tau                       # (1, C, 1, 1)
        dy = by * self.tau                       # (1, C, 1, 1)

        # Flatten along channels: (B*C, 1, H, W)
        x_flat = x.view(B * C, 1, H, W)

        # Base grid, expanded for each (b, c)-block: (B*C, H, W, 2)
        grid = self.base_grid.expand(B, 1, H, W, 2) \
                              .expand(B, C, H, W, 2) \
                              .reshape(B * C, H, W, 2)

        s_bc  = s.expand(B, C, 1, 1).reshape(B * C, 1, 1, 1)
        dx_bc = dx.expand(B, C, 1, 1).reshape(B * C, 1, 1, 1)
        dy_bc = dy.expand(B, C, 1, 1).reshape(B * C, 1, 1, 1)

        # Drift composed into each diffusion grid: when s→0 all four grids
        # collapse to pure advection (x+dx, y+dy) without parasitic diffusion.
        grid_px = grid + torch.cat([s_bc  + dx_bc, dy_bc],          dim=-1)
        grid_nx = grid + torch.cat([-s_bc + dx_bc, dy_bc],          dim=-1)
        grid_py = grid + torch.cat([dx_bc,         s_bc  + dy_bc],  dim=-1)
        grid_ny = grid + torch.cat([dx_bc,         -s_bc + dy_bc],  dim=-1)

        u_px = self._sample(x_flat, grid_px)
        u_nx = self._sample(x_flat, grid_nx)
        u_py = self._sample(x_flat, grid_py)
        u_ny = self._sample(x_flat, grid_ny)

        # 2D composed diffusion-drift Chernoff step: ¼ per direction.
        evolution = 0.25 * (u_px + u_nx + u_py + u_ny)
        evolution = evolution.view(B, C, H, W)

        # Potential term τ·c·u
        return evolution + self.tau * c * x


class ShiftMixerBlock(nn.Module):
    """RMSB-R1 building block: RemizovShiftLayer → BN → 1×1 expand → GELU → 1×1 project → residual.

    Adds inter-channel mixing capacity to the shift operator while keeping the
    Remizov shift step as the primary spatial evolution kernel. Each block has
    independent parameters, so stacking N blocks scales the parameter budget by N.

    Capacity breakdown (channels=C, expansion=E):
        RemizovShiftLayer  :  4·C  learnable scalars  (≪ 1% of budget)
        BN                 :  2·C
        expand  1×1 Conv   :  C·(C·E)  (bias=False)
        project 1×1 Conv   :  (C·E)·C  (bias=False)
        Total per block    ≈  2·C²·E  (mixer dominates)

    Design notes:
      - Shift step keeps theorem-grounded spatial evolution; mixer provides
        the representational width needed to compete with SpectralRemizovLayer.
      - Residual skip ensures gradient flow through deep stacks.
      - BN placed after shift (before mixer) to normalise scale before expansion.
      - No skip-conv inside shift; the outer residual handles that.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 16,
        tau: float = 0.1,
        spatial_res: tuple = (32, 32),
    ):
        super().__init__()
        hidden = channels * expansion
        self.shift = RemizovShiftLayer(channels, spatial_res=spatial_res, tau=tau)
        self.bn = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

        # Small init for project so residual starts near identity.
        nn.init.zeros_(self.project.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PDE step is the main path; mixer is a residual correction on top of it.
        # At init (project.weight == 0) this reduces to pure RemizovShiftLayer,
        # so gradients flow immediately into theta_a / theta_bx / theta_by / theta_c.
        x_shifted = self.shift(x)
        out = self.bn(x_shifted)
        out = self.project(self.act(self.expand(out)))
        return x_shifted + out
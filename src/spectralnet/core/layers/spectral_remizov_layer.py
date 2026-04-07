import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


def physical_act(x: torch.Tensor) -> torch.Tensor:
    """PhysicalAct: f(x) = x * exp(-|x|²).

    We use |x|² instead of x*x — this is correct for complex tensors,
    where x*x yields a complex square (a²−b²−2abi), rather than the field power |x|²=(a²+b²).
    For real tensors, the result is identical.

    Note: This is a physically motivated saturating nonlinearity, not a literal
    discretization of pure Kerr nonlinearity. In the code, it is used as a convenient
    phenomenological transfer function that aligns well with the project's narrative.
    """
    return x * torch.exp(-x.abs().pow(2))


class PhysicalAct(nn.Module):
    """nn.Module wrapper around physical_act.

    Needed for two practical purposes:
    1. The activation can be safely passed to nn.Sequential / block constructors;
    2. get_activation() returns a uniform nn.Module type for all variants.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return physical_act(x)


def get_activation(name: str) -> nn.Module:
    """Activation factory by string name.

    Supported variants:
        "physicalact" — x * exp(-|x|²), physically motivated saturation
        "gelu"        — nn.GELU()
        "relu"        — nn.ReLU()

    Usage example in a model:
        self.act = get_activation(cfg.model.activation)
        ...
        out = self.act(x_spectral + x_skip)
    """
    name = name.lower()
    if name == "physicalact":
        return PhysicalAct()
    elif name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    else:
        raise ValueError(
            f"Unknown activation: {name!r}. "
            f"Supported: physicalact, gelu, relu"
        )


class SpectralRemizovLayer(nn.Module):
    """
    One evolution step F_φ(τ) in the spectral domain.

        û_{l+1}(k) = W_φ(k) · û_l(k)

    where W_φ(k) is a learnable spectral operator.

    Parameterization modes (rank):
        rank=0  — Fourier-diagonal: W(k) is an independent (C_out, C_in) matrix
                  on each frequency bin (m,n). No cross-frequency coupling.
        rank>0  — W = D + UV*: D is the diagonal part (like rank=0),
                  plus a low-rank cross-frequency mixer through R global
                  spectral features. UV* introduces controlled interaction
                  between frequency components without materializing the full
                  dense matrix.

    Architecture of one pass:
        FFT → mask W(k) → iFFT ─┐
                                  ├─ sum → activation
        skip (1×1 conv)        ──┘
        shift S(a,b,c) (opt.) ──┘

    C.5 hybrid mode (shift_hidden > 0):
        Adds a parallel spatial branch RemizovShiftLocal, implementing
        Remizov's Theorems 4-6 with local variable coefficients a(x), b(x), c(x).
        Allows modeling variable-coefficient generators L = a(x)Δ + b(x)·∇ + c(x)
        simultaneously with the spectral operator D + UV*.

    Note on C.4/C.5:
        lr_scale=0.02 for U/V ensures Var(UV*·x) ≈ Var(D·x) given
        C=128, M=8, rank≤8. Additional normalization 1/√(C·M·M)
        is NOT used — it suppresses the UV* contribution to ~1e-7 and zeros out
        the branch. The scale balance is controlled solely through lr_scale.
    """

    def __init__(
        self,
        channels: int,
        modes: int = 16,
        activation: nn.Module = None,
        rank: int = 0,
        shift_hidden: int = 0,
        tau: float = 0.1,
    ):
        """
        Args:
            channels:     number of channels (C_in = C_out).
            modes:        desired number of spectral modes M.
            activation:   nonlinearity after path summation. Defaults to PhysicalAct.
            rank:         rank of the cross-frequency mixer. 0 = pure Fourier-diagonal.
            shift_hidden: number of hidden channels for the coeff-generator of the shift branch.
                          0 = no shift (purely spectral mode).
            tau:          Chernoff step, passed to RemizovShiftLocal.
        """
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.rank = rank
        self.act = activation if activation is not None else PhysicalAct()

        # D: diagonal-in-frequency mask (C_out, C_in, M, M), complex via real/imag parts.
        #
        # We keep conservative initialization here. The diagonal branch is already well
        # studied and serves as the baseline / reference path.
        scale = 1.0 / (channels * channels)
        self.w_real = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes)
        )
        self.w_imag = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes)
        )

        # UV*: low-rank cross-frequency coupling.
        #
        # Critical fix for C.4:
        # Previously, lr_scale was 1e-3. Combined with the product U @ (V* x),
        # this made the initial UV* contribution too quiet relative to the diagonal path.
        #
        # We increase std to Xavier-like init levels (0.02), and then in
        # _apply_mask() we normalize the inner product by sqrt(C * M * M).
        # Such batch scaling gives UV* a chance to influence the loss from the very first epochs,
        # without creating an explosive activation scale.
        if rank > 0:
            lr_scale = 0.02
            self.u_real = nn.Parameter(
                lr_scale * torch.randn(channels, rank, modes, modes)
            )
            self.u_imag = nn.Parameter(
                lr_scale * torch.randn(channels, rank, modes, modes)
            )
            self.v_real = nn.Parameter(
                lr_scale * torch.randn(channels, rank, modes, modes)
            )
            self.v_imag = nn.Parameter(
                lr_scale * torch.randn(channels, rank, modes, modes)
            )

        # Local residual / skip path in the spatial domain.
        # It stabilizes training and gives the model a direct bypass route outside the spectral branch.
        self.w_skip = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # C.5: Remizov shift branch (variable-coefficient PDE operator).
        # shift_hidden=0 disables it (backward-compatible with C.4 and earlier).
        self.shift = None
        if shift_hidden > 0:
            from src.spectralnet.core.layers.remizov_shift import RemizovShiftLocal
            self.shift = RemizovShiftLocal(channels, shift_hidden, tau)

    def _apply_mask(
        self,
        x_ft: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the spectral operator W = D + UV* to the first m1 × m2 coefficients.

        Why only the top-left spectral block is applied:
            This is the current implementation of truncated spectral processing — we explicitly
            work only with the low-frequency subspace of dimension m1 × m2.

        Why UV* is computed via einsum rather than a materialized dense matrix:
            A full cross-frequency mixing matrix would be too large in memory.
            Factorization through an R-dimensional bottleneck provides a controllable compromise
            between expressiveness and cost.
        """
        B, C, H, W_ft = x_ft.shape

        m1 = min(self.modes, H)
        m2 = min(self.modes, W_ft)

        D = torch.complex(
            self.w_real[:, :, :m1, :m2],
            self.w_imag[:, :, :m1, :m2],
        )

        # Take only the active spectral block that we actually work with.
        x_top = x_ft[:, :, :m1, :m2]

        # D-part: per-frequency channel mixing.
        #
        # This is the baseline Fourier-diagonal operator: frequencies are NOT mixed with each other,
        # but within each frequency bin, channel mixing is allowed.
        out_top = torch.einsum('bimn,oimn->bomn', x_top, D)

        # UV*-part: low-rank cross-frequency coupling through R global spectral features.
        if self.rank > 0:
            U = torch.complex(
                self.u_real[:, :, :m1, :m2],
                self.u_imag[:, :, :m1, :m2],
            )
            V = torch.complex(
                self.v_real[:, :, :m1, :m2],
                self.v_imag[:, :, :m1, :m2],
            )

            # z[b, r] = Σ_{c,m,n} conj(V[c,r,m,n]) * x[b,c,m,n]
            #         → global R-dimensional spectral summary for each batch item.
            #
            # With lr_scale=0.02 for U/V and C*m1*m2≈8192, Var(z)≈3.2, and Var(U·z)
            # at the output is comparable to the D-path. Additional normalization is not needed —
            # it would suppress the UV* contribution down to ~1e-7.
            z = torch.einsum('bimn,irmn->br', x_top, V.conj())

            out_top = out_top + torch.einsum('ormn,br->bomn', U, z)

        # Leave the rest of the spectrum as zero: this corresponds to the current mode,
        # where the model actively works only in the selected truncated subspace.
        out_ft = torch.zeros(
            B,
            self.channels,
            H,
            W_ft,
            dtype=x_ft.dtype,
            device=x_ft.device,
        )
        out_ft[:, :, :m1, :m2] = out_top
        return out_ft

    def forward(self, x: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B C H W"]:
        # Skip path goes directly in the spatial domain and then sums with the spectral path.
        x_skip = self.w_skip(x)

        # Spectral path: FFT → truncated operator → inverse FFT.
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft_masked = self._apply_mask(x_ft)
        x_spectral = torch.fft.irfft2(
            x_ft_masked, s=(x.shape[-2], x.shape[-1]), norm='ortho'
        )

        out = x_spectral + x_skip

        # C.5: shift branch operates in spatial domain parallel to the spectral path.
        if self.shift is not None:
            out = out + self.shift(x)

        return self.act(out)

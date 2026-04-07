import torch
import torch.nn as nn
from jaxtyping import Float
from typing import List

class ResolventAggregation(nn.Module):
    """
    Resolvent aggregation of states: R_λ ≈ Σ w_k * u_k
    Implements a discrete analogue of the Laplace transform for the evolution trajectory.
    """
    def __init__(self, n_steps: int, tau: float, init_lambda: float = 1.0):
        super().__init__()
        # n_steps is kept for interface compatibility, but logic is tied to list length
        self.n_steps = n_steps
        self.tau = tau
        
        # The parameter lambda is learnable, but bounded from below (via softplus)
        # to guarantee convergence of the Laplace integral (λ > ω).
        self.raw_lambda = nn.Parameter(torch.tensor([init_lambda]))

    def get_weights(self, num_states: int, device: torch.device) -> Float[torch.Tensor, "num_states"]:
        # λ = softplus(raw_lambda) for numerical stability
        lam = torch.nn.functional.softplus(self.raw_lambda)
        
        # Weights w_k = exp(-λ * k * τ) * τ
        # Dynamically generate steps for the actual number of passed states (n_steps + 1)
        steps = torch.arange(num_states, device=device).float()
        weights = torch.exp(-lam * steps * self.tau) * self.tau
        
        # Weight normalization (optional, to preserve gradient scale)
        return weights / (weights.sum() + 1e-8)

    def forward(self, states: List[Float[torch.Tensor, "B C ..."]]) -> Float[torch.Tensor, "B C ..."]:
        """
        states: list of state tensors [u_0, u_1, ..., u_n]
        """
        num_states = len(states)
        weights = self.get_weights(num_states, states[0].device)
        
        # Weighted summation of the trajectory
        # Using stack + sum is more efficient for autograd
        stacked_states = torch.stack(states, dim=0) # [num_states, B, C, ...]
        
        # Expand weights for broadcasting to match tensor dimensions
        dims_to_add = len(states[0].shape)
        weighted_view = weights.view(-1, *([1] * dims_to_add))
        
        return (stacked_states * weighted_view).sum(dim=0)
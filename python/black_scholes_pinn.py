"""
Physics-Informed Neural Network (PINN) for the Black-Scholes PDE.

The Black-Scholes PDE:
    dV/dt + (1/2) * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0

Network: (S, t) -> V(S, t)
The PDE is enforced via the loss function using automatic differentiation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class BlackScholesPINN(nn.Module):
    """
    Physics-Informed Neural Network for solving the Black-Scholes PDE.

    The network takes (S, t) as input and outputs V(S, t), the option price.
    The Black-Scholes PDE is encoded into the loss function via automatic
    differentiation of the network output with respect to inputs.

    Args:
        hidden_layers: List of hidden layer sizes (default: [128, 128, 128, 128])
        activation: Activation function (default: Tanh for smooth second derivatives)
        S_min: Minimum spot price for normalization
        S_max: Maximum spot price for normalization
        T: Maximum time to maturity for normalization
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        S_min: float = 0.0,
        S_max: float = 200.0,
        T: float = 1.0,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]

        self.S_min = S_min
        self.S_max = S_max
        self.T = T

        # Choose activation function
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sin":
            act_fn = SinActivation()
        elif activation == "softplus":
            act_fn = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        input_dim = 2  # (S, t)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))  # Output: V(S, t)

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better training convergence."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize_inputs(
        self, S: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize inputs to [0, 1] range for numerical stability."""
        S_norm = (S - self.S_min) / (self.S_max - self.S_min)
        t_norm = t / self.T
        return S_norm, t_norm

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute V(S, t).

        Args:
            S: Spot price tensor, shape (N,) or (N, 1)
            t: Time tensor, shape (N,) or (N, 1)

        Returns:
            V: Option price tensor, shape (N, 1)
        """
        # Ensure correct shapes
        if S.dim() == 1:
            S = S.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Normalize
        S_norm, t_norm = self.normalize_inputs(S, t)

        # Concatenate inputs
        x = torch.cat([S_norm, t_norm], dim=1)

        # Forward through network
        V = self.network(x)

        # Scale output by S_max for better numerical range
        V = V * self.S_max

        return V

    def compute_pde_residual(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        sigma: float,
        r: float,
    ) -> torch.Tensor:
        """
        Compute the Black-Scholes PDE residual using automatic differentiation.

        PDE: dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0

        Args:
            S: Spot prices requiring grad, shape (N, 1)
            t: Times requiring grad, shape (N, 1)
            sigma: Volatility
            r: Risk-free rate

        Returns:
            residual: PDE residual, shape (N, 1)
        """
        V = self.forward(S, t)

        # First-order derivatives
        dV_dS = torch.autograd.grad(
            V, S,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )[0]

        dV_dt = torch.autograd.grad(
            V, t,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Second-order derivative
        d2V_dS2 = torch.autograd.grad(
            dV_dS, S,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Black-Scholes PDE residual
        residual = (
            dV_dt
            + 0.5 * sigma**2 * S**2 * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        return residual


class SinActivation(nn.Module):
    """Sinusoidal activation function for PINNs (from SIREN networks)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class BlackScholesPINNWithVol(nn.Module):
    """
    Extended PINN that takes volatility sigma as an additional input.
    This allows computing Vega via automatic differentiation.

    Network: (S, t, sigma) -> V(S, t, sigma)
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        S_min: float = 0.0,
        S_max: float = 200.0,
        T: float = 1.0,
        sigma_min: float = 0.05,
        sigma_max: float = 1.5,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]

        self.S_min = S_min
        self.S_max = S_max
        self.T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        layers = []
        input_dim = 3  # (S, t, sigma)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with volatility as input."""
        if S.dim() == 1:
            S = S.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)

        S_norm = (S - self.S_min) / (self.S_max - self.S_min)
        t_norm = t / self.T
        sigma_norm = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)

        x = torch.cat([S_norm, t_norm, sigma_norm], dim=1)
        V = self.network(x)
        V = V * self.S_max

        return V


def create_model(
    hidden_layers: Optional[List[int]] = None,
    S_max: float = 200.0,
    T: float = 1.0,
    device: str = "cpu",
    with_vol: bool = False,
) -> nn.Module:
    """
    Factory function to create a PINN model.

    Args:
        hidden_layers: Hidden layer sizes
        S_max: Maximum spot price
        T: Maximum maturity
        device: Device to place model on
        with_vol: Whether to include volatility as an input

    Returns:
        PINN model
    """
    if with_vol:
        model = BlackScholesPINNWithVol(
            hidden_layers=hidden_layers,
            S_max=S_max,
            T=T,
        )
    else:
        model = BlackScholesPINN(
            hidden_layers=hidden_layers,
            S_max=S_max,
            T=T,
        )

    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = create_model(device=device)
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    S = torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0], device=device)
    t = torch.zeros(5, device=device)

    V = model(S, t)
    print(f"\nTest forward pass:")
    print(f"  S = {S.tolist()}")
    print(f"  t = {t.tolist()}")
    print(f"  V = {V.squeeze().tolist()}")

    # Test PDE residual computation
    S_pde = torch.tensor([100.0], device=device, requires_grad=True)
    t_pde = torch.tensor([0.5], device=device, requires_grad=True)

    residual = model.compute_pde_residual(S_pde, t_pde, sigma=0.2, r=0.05)
    print(f"\nPDE residual at (S=100, t=0.5): {residual.item():.6f}")
    print("(Should approach 0 after training)")

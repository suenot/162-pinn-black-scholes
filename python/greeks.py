"""
Greeks computation via automatic differentiation for the Black-Scholes PINN.

Computes:
- Delta (dV/dS): sensitivity to spot price
- Gamma (d^2V/dS^2): convexity / hedging cost
- Theta (dV/dt): time decay
- Vega (dV/d_sigma): sensitivity to volatility (requires extended model)
- Rho (dV/dr): sensitivity to interest rate

Usage:
    python greeks.py --spot 100 --strike 100 --maturity 1.0
    python greeks.py --model pinn_bs_model.pt --spot 50000 --strike 50000 --maturity 0.25
"""

import argparse
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from black_scholes_pinn import BlackScholesPINN, create_model
from data_loader import black_scholes_greeks


def compute_delta(
    model: BlackScholesPINN,
    S: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Delta = dV/dS using automatic differentiation.

    Args:
        model: Trained PINN model
        S: Spot price (requires_grad=True)
        t: Time

    Returns:
        Delta tensor
    """
    S = S.detach().requires_grad_(True)
    if S.dim() == 1:
        S = S.unsqueeze(1)
    if t.dim() == 1:
        t = t.unsqueeze(1)

    V = model(S, t)

    delta = torch.autograd.grad(
        V, S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    return delta


def compute_gamma(
    model: BlackScholesPINN,
    S: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Gamma = d^2V/dS^2 using automatic differentiation.

    Args:
        model: Trained PINN model
        S: Spot price (requires_grad=True)
        t: Time

    Returns:
        Gamma tensor
    """
    S = S.detach().requires_grad_(True)
    if S.dim() == 1:
        S = S.unsqueeze(1)
    if t.dim() == 1:
        t = t.unsqueeze(1)

    V = model(S, t)

    delta = torch.autograd.grad(
        V, S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    gamma = torch.autograd.grad(
        delta, S,
        grad_outputs=torch.ones_like(delta),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gamma


def compute_theta(
    model: BlackScholesPINN,
    S: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Theta = dV/dt using automatic differentiation.

    Note: This is the sensitivity to calendar time t (not time-to-maturity tau).
    Theta w.r.t. time-to-maturity has opposite sign.

    Args:
        model: Trained PINN model
        S: Spot price
        t: Time (requires_grad=True)

    Returns:
        Theta tensor
    """
    t = t.detach().requires_grad_(True)
    if S.dim() == 1:
        S = S.unsqueeze(1)
    if t.dim() == 1:
        t = t.unsqueeze(1)

    V = model(S, t)

    theta = torch.autograd.grad(
        V, t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    return theta


def compute_greeks(
    model: BlackScholesPINN,
    S: torch.Tensor,
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute all Greeks (Delta, Gamma, Theta) in one pass.

    Args:
        model: Trained PINN model
        S: Spot price
        t: Time

    Returns:
        Dictionary with 'delta', 'gamma', 'theta', and 'price'
    """
    S = S.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)

    if S.dim() == 1:
        S = S.unsqueeze(1)
    if t.dim() == 1:
        t = t.unsqueeze(1)

    V = model(S, t)

    # Delta = dV/dS
    delta = torch.autograd.grad(
        V, S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gamma = d^2V/dS^2
    gamma = torch.autograd.grad(
        delta, S,
        grad_outputs=torch.ones_like(delta),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Theta = dV/dt
    theta = torch.autograd.grad(
        V, t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    return {
        "price": V.detach(),
        "delta": delta.detach(),
        "gamma": gamma.detach(),
        "theta": theta.detach(),
    }


def compute_greeks_surface(
    model: BlackScholesPINN,
    S_range: np.ndarray,
    t_range: np.ndarray,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Compute Greeks over a 2D grid of (S, t) values.

    Args:
        model: Trained PINN model
        S_range: Array of spot prices
        t_range: Array of times
        device: Torch device

    Returns:
        Dictionary with 2D arrays for price, delta, gamma, theta
    """
    S_grid, t_grid = np.meshgrid(S_range, t_range)
    S_flat = S_grid.flatten()
    t_flat = t_grid.flatten()

    S_tensor = torch.tensor(S_flat, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)

    greeks = compute_greeks(model, S_tensor, t_tensor)

    n_S = len(S_range)
    n_t = len(t_range)

    return {
        "S_grid": S_grid,
        "t_grid": t_grid,
        "price": greeks["price"].cpu().numpy().reshape(n_t, n_S),
        "delta": greeks["delta"].cpu().numpy().reshape(n_t, n_S),
        "gamma": greeks["gamma"].cpu().numpy().reshape(n_t, n_S),
        "theta": greeks["theta"].cpu().numpy().reshape(n_t, n_S),
    }


def compare_greeks_with_analytical(
    model: BlackScholesPINN,
    S_values: np.ndarray,
    t_value: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    device: str = "cpu",
):
    """
    Print a comparison of PINN Greeks vs analytical Greeks.
    """
    tau = T - t_value  # Time to maturity

    # Analytical Greeks
    analytical = black_scholes_greeks(S_values, K, tau, r, sigma, option_type)

    # PINN Greeks
    S_tensor = torch.tensor(S_values, dtype=torch.float32, device=device)
    t_tensor = torch.full_like(S_tensor, t_value)

    pinn_greeks = compute_greeks(model, S_tensor, t_tensor)

    pinn_delta = pinn_greeks["delta"].cpu().numpy().flatten()
    pinn_gamma = pinn_greeks["gamma"].cpu().numpy().flatten()
    pinn_theta = pinn_greeks["theta"].cpu().numpy().flatten()

    print("\n" + "=" * 80)
    print(f"Greeks Comparison: PINN vs Analytical (t={t_value}, K={K}, sigma={sigma})")
    print("=" * 80)

    # Delta
    print(f"\n{'S':>8} | {'Delta(BS)':>10} {'Delta(PINN)':>12} {'Error':>10} | "
          f"{'Gamma(BS)':>10} {'Gamma(PINN)':>12} {'Error':>10}")
    print("-" * 80)

    for i, s in enumerate(S_values):
        d_err = abs(pinn_delta[i] - analytical["delta"][i])
        g_err = abs(pinn_gamma[i] - analytical["gamma"][i])
        print(f"{s:8.1f} | {analytical['delta'][i]:10.4f} {pinn_delta[i]:12.4f} {d_err:10.6f} | "
              f"{analytical['gamma'][i]:10.6f} {pinn_gamma[i]:12.6f} {g_err:10.6f}")

    print(f"\nDelta MAE: {np.mean(np.abs(pinn_delta - analytical['delta'])):.6f}")
    print(f"Gamma MAE: {np.mean(np.abs(pinn_gamma - analytical['gamma'])):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compute Greeks via PINN autograd")
    parser.add_argument("--model", type=str, default="pinn_bs_model.pt",
                        help="Path to trained model")
    parser.add_argument("--spot", type=float, default=100.0, help="Spot price")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--option_type", type=str, default="call")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load or create model
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        params = checkpoint["params"]
        model = create_model(
            S_max=params["S_max"], T=params["T"], device=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {args.model}")
        K = params["K"]
        r = params["r"]
        sigma = params["sigma"]
        T = params["T"]
    except FileNotFoundError:
        print(f"Model file {args.model} not found. Using untrained model (results will be random).")
        model = create_model(S_max=args.spot * 2, T=args.maturity, device=device)
        K = args.strike
        r = args.r
        sigma = args.sigma
        T = args.maturity

    model.eval()

    # Compute Greeks at a single point
    S = torch.tensor([args.spot], dtype=torch.float32, device=device)
    t = torch.tensor([0.0], dtype=torch.float32, device=device)

    greeks = compute_greeks(model, S, t)

    print(f"\nGreeks at S={args.spot}, t=0.0:")
    print(f"  Price: {greeks['price'].item():.4f}")
    print(f"  Delta: {greeks['delta'].item():.4f}")
    print(f"  Gamma: {greeks['gamma'].item():.6f}")
    print(f"  Theta: {greeks['theta'].item():.4f}")

    # Compare with analytical
    S_range = np.linspace(args.spot * 0.7, args.spot * 1.3, 13)
    compare_greeks_with_analytical(
        model, S_range, t_value=0.0,
        K=K, r=r, sigma=sigma, T=T,
        option_type=args.option_type, device=device,
    )


if __name__ == "__main__":
    main()

"""
Visualization tools for the Black-Scholes PINN.

Creates plots for:
1. Option price surface V(S, t)
2. PDE residual heatmap
3. Greeks surfaces (Delta, Gamma, Theta)
4. Training loss curves
5. Comparison with analytical Black-Scholes
6. Error distribution

Usage:
    python visualize.py
    python visualize.py --model pinn_bs_model.pt --save_dir plots/
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List

from black_scholes_pinn import BlackScholesPINN, create_model
from data_loader import black_scholes_analytical
from greeks import compute_greeks_surface


def plot_option_surface(
    model: BlackScholesPINN,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    S_max: float = 200.0,
    n_S: int = 80,
    n_t: int = 50,
    device: str = "cpu",
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    """
    Plot the option price surface V(S, t) from the PINN.
    """
    model.eval()

    S_range = np.linspace(1.0, S_max, n_S)
    t_range = np.linspace(0.0, T - 0.01, n_t)
    S_grid, t_grid = np.meshgrid(S_range, t_range)

    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device)

    with torch.no_grad():
        V_flat = model(S_flat, t_flat).cpu().numpy().flatten()

    V_grid = V_flat.reshape(n_t, n_S)

    fig = plt.figure(figsize=(14, 6))

    # 3D surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(
        S_grid, t_grid, V_grid,
        cmap=cm.viridis, alpha=0.8,
        linewidth=0, antialiased=True,
    )
    ax1.set_xlabel("Spot Price S")
    ax1.set_ylabel("Time t")
    ax1.set_zlabel("Option Price V")
    ax1.set_title(f"PINN Option Surface {title_suffix}")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 2D heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.pcolormesh(S_grid, t_grid, V_grid, cmap=cm.viridis, shading="auto")
    ax2.set_xlabel("Spot Price S")
    ax2.set_ylabel("Time t")
    ax2.set_title(f"Option Price Heatmap {title_suffix}")
    ax2.axvline(x=K, color="r", linestyle="--", alpha=0.5, label=f"Strike K={K}")
    ax2.legend()
    fig.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison_with_analytical(
    model: BlackScholesPINN,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    S_max: float = 200.0,
    device: str = "cpu",
    save_path: Optional[str] = None,
):
    """
    Compare PINN prices with analytical Black-Scholes at several time slices.
    """
    model.eval()

    S_range = np.linspace(1.0, S_max, 200)
    time_slices = [0.0, 0.25, 0.5, 0.75, 0.95]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_slices)))

    for i, t_val in enumerate(time_slices):
        tau = T - t_val  # time to maturity
        bs_prices = black_scholes_analytical(S_range, K, tau, r, sigma, "call")

        S_tensor = torch.tensor(S_range, dtype=torch.float32, device=device)
        t_tensor = torch.full_like(S_tensor, t_val)

        with torch.no_grad():
            pinn_prices = model(S_tensor, t_tensor).cpu().numpy().flatten()

        # Prices
        axes[0].plot(S_range, bs_prices, "-", color=colors[i], alpha=0.6,
                     label=f"BS t={t_val:.2f}")
        axes[0].plot(S_range, pinn_prices, "--", color=colors[i], alpha=0.9,
                     label=f"PINN t={t_val:.2f}")

        # Error
        error = np.abs(pinn_prices - bs_prices)
        axes[1].plot(S_range, error, color=colors[i], label=f"t={t_val:.2f}")

    axes[0].set_xlabel("Spot Price S")
    axes[0].set_ylabel("Option Price")
    axes[0].set_title("PINN vs Analytical Black-Scholes")
    axes[0].axvline(x=K, color="gray", linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Spot Price S")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Pricing Error |PINN - BS|")
    axes[1].axvline(x=K, color="gray", linestyle=":", alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_pde_residual(
    model: BlackScholesPINN,
    sigma: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    S_max: float = 200.0,
    n_S: int = 80,
    n_t: int = 50,
    device: str = "cpu",
    save_path: Optional[str] = None,
):
    """
    Plot the PDE residual over the (S, t) domain to show how well the PDE is satisfied.
    """
    model.eval()

    S_range = np.linspace(1.0, S_max, n_S)
    t_range = np.linspace(0.01, T - 0.01, n_t)
    S_grid, t_grid = np.meshgrid(S_range, t_range)

    S_flat = torch.tensor(
        S_grid.flatten(), dtype=torch.float32, device=device
    ).unsqueeze(1).requires_grad_(True)
    t_flat = torch.tensor(
        t_grid.flatten(), dtype=torch.float32, device=device
    ).unsqueeze(1).requires_grad_(True)

    residual = model.compute_pde_residual(S_flat, t_flat, sigma, r)
    residual_grid = residual.detach().cpu().numpy().reshape(n_t, n_S)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute residual
    im1 = axes[0].pcolormesh(
        S_grid, t_grid, np.abs(residual_grid),
        cmap="hot", shading="auto",
    )
    axes[0].set_xlabel("Spot Price S")
    axes[0].set_ylabel("Time t")
    axes[0].set_title("PDE Residual |R(S, t)|")
    fig.colorbar(im1, ax=axes[0])

    # Log residual
    log_residual = np.log10(np.abs(residual_grid) + 1e-10)
    im2 = axes[1].pcolormesh(
        S_grid, t_grid, log_residual,
        cmap="RdYlGn_r", shading="auto",
    )
    axes[1].set_xlabel("Spot Price S")
    axes[1].set_ylabel("Time t")
    axes[1].set_title("Log10 PDE Residual")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_greeks_surfaces(
    model: BlackScholesPINN,
    K: float = 100.0,
    T: float = 1.0,
    S_max: float = 200.0,
    device: str = "cpu",
    save_path: Optional[str] = None,
):
    """
    Plot Greeks surfaces (Delta, Gamma, Theta) over (S, t).
    """
    S_range = np.linspace(1.0, S_max, 60)
    t_range = np.linspace(0.0, T - 0.01, 40)

    surfaces = compute_greeks_surface(model, S_range, t_range, device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Price
    im0 = axes[0, 0].pcolormesh(
        surfaces["S_grid"], surfaces["t_grid"], surfaces["price"],
        cmap="viridis", shading="auto",
    )
    axes[0, 0].set_title("Option Price V(S,t)")
    axes[0, 0].set_xlabel("S")
    axes[0, 0].set_ylabel("t")
    axes[0, 0].axvline(x=K, color="w", linestyle="--", alpha=0.5)
    fig.colorbar(im0, ax=axes[0, 0])

    # Delta
    im1 = axes[0, 1].pcolormesh(
        surfaces["S_grid"], surfaces["t_grid"], surfaces["delta"],
        cmap="RdBu_r", shading="auto",
    )
    axes[0, 1].set_title("Delta (dV/dS)")
    axes[0, 1].set_xlabel("S")
    axes[0, 1].set_ylabel("t")
    axes[0, 1].axvline(x=K, color="k", linestyle="--", alpha=0.5)
    fig.colorbar(im1, ax=axes[0, 1])

    # Gamma
    im2 = axes[1, 0].pcolormesh(
        surfaces["S_grid"], surfaces["t_grid"], surfaces["gamma"],
        cmap="YlOrRd", shading="auto",
    )
    axes[1, 0].set_title("Gamma (d^2V/dS^2)")
    axes[1, 0].set_xlabel("S")
    axes[1, 0].set_ylabel("t")
    axes[1, 0].axvline(x=K, color="k", linestyle="--", alpha=0.5)
    fig.colorbar(im2, ax=axes[1, 0])

    # Theta
    im3 = axes[1, 1].pcolormesh(
        surfaces["S_grid"], surfaces["t_grid"], surfaces["theta"],
        cmap="coolwarm", shading="auto",
    )
    axes[1, 1].set_title("Theta (dV/dt)")
    axes[1, 1].set_xlabel("S")
    axes[1, 1].set_ylabel("t")
    axes[1, 1].axvline(x=K, color="k", linestyle="--", alpha=0.5)
    fig.colorbar(im3, ax=axes[1, 1])

    plt.suptitle("Greeks Surfaces from PINN", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(len(history["loss_total"]))

    # Total loss
    axes[0].semilogy(epochs, history["loss_total"], "b-", alpha=0.7, label="Total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Component losses
    axes[1].semilogy(epochs, history["loss_pde"], "r-", alpha=0.7, label="PDE")
    axes[1].semilogy(epochs, history["loss_bc"], "g-", alpha=0.7, label="BC")
    axes[1].semilogy(epochs, history["loss_ic"], "b-", alpha=0.7, label="IC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Components")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Validation error
    if history.get("val_mae"):
        val_epochs = np.linspace(0, len(epochs) - 1, len(history["val_mae"]))
        axes[2].semilogy(val_epochs, history["val_mae"], "b-o", alpha=0.7, label="MAE")
        axes[2].semilogy(val_epochs, history["val_max"], "r-o", alpha=0.7, label="Max Error")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Error")
        axes[2].set_title("Validation Error")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_delta_hedging_error(
    model: BlackScholesPINN,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    device: str = "cpu",
    save_path: Optional[str] = None,
):
    """
    Plot Delta at different spot prices, comparing PINN vs analytical.
    Shows how the PINN Delta can be used for hedging.
    """
    model.eval()

    S_range = np.linspace(60, 140, 200)
    t_values = [0.0, 0.5, 0.9]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["blue", "green", "red"]

    for t_val, color in zip(t_values, colors):
        tau = T - t_val

        # Analytical Delta
        from data_loader import black_scholes_greeks
        analytical = black_scholes_greeks(S_range, K, tau, r, sigma, "call")

        # PINN Delta
        S_tensor = torch.tensor(S_range, dtype=torch.float32, device=device)
        t_tensor = torch.full_like(S_tensor, t_val)

        S_tensor = S_tensor.requires_grad_(True)
        if S_tensor.dim() == 1:
            S_t = S_tensor.unsqueeze(1)
        else:
            S_t = S_tensor

        t_t = t_tensor.unsqueeze(1) if t_tensor.dim() == 1 else t_tensor

        V = model(S_t, t_t)
        delta_pinn = torch.autograd.grad(
            V, S_tensor,
            grad_outputs=torch.ones_like(V),
            create_graph=False,
        )[0]

        delta_pinn = delta_pinn.detach().cpu().numpy()

        ax.plot(S_range, analytical["delta"], "-", color=color, alpha=0.6,
                label=f"BS t={t_val:.1f}")
        ax.plot(S_range, delta_pinn, "--", color=color, alpha=0.9,
                label=f"PINN t={t_val:.1f}")

    ax.axvline(x=K, color="gray", linestyle=":", alpha=0.5, label=f"Strike K={K}")
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Delta")
    ax.set_title("Delta: PINN vs Analytical")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize PINN results")
    parser.add_argument("--model", type=str, default="pinn_bs_model.pt")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        params = checkpoint["params"]
        model = create_model(S_max=params["S_max"], T=params["T"], device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = checkpoint.get("history", {})
        print(f"Loaded model from {args.model}")
    except FileNotFoundError:
        print(f"Model {args.model} not found. Training a quick model for demo...")
        from train import train_pinn
        from data_loader import generate_synthetic_data

        params = {"K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0, "S_max": 200.0}
        model = create_model(S_max=params["S_max"], T=params["T"], device=device)
        data = generate_synthetic_data(**params, device=device)
        history = train_pinn(model, data, num_epochs=3000, print_every=1000)

    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    S_max = params["S_max"]

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def sp(name):
        return os.path.join(save_dir, name) if save_dir else None

    # Generate all plots
    print("\nGenerating plots...")

    plot_option_surface(model, K, r, sigma, T, S_max, device=device, save_path=sp("option_surface.png"))
    plot_comparison_with_analytical(model, K, r, sigma, T, S_max, device=device, save_path=sp("bs_comparison.png"))
    plot_pde_residual(model, sigma, r, T, S_max, device=device, save_path=sp("pde_residual.png"))
    plot_greeks_surfaces(model, K, T, S_max, device=device, save_path=sp("greeks_surfaces.png"))

    if history:
        plot_training_history(history, save_path=sp("training_history.png"))

    plot_delta_hedging_error(model, K, r, sigma, T, device=device, save_path=sp("delta_comparison.png"))

    print("All plots generated.")


if __name__ == "__main__":
    main()

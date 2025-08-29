"""
Training loop for the Black-Scholes PINN.

Trains the neural network to satisfy:
1. The Black-Scholes PDE in the interior domain
2. Boundary conditions at S=0 and S=S_max
3. Terminal condition at t=T (option payoff)

Usage:
    python train.py --epochs 10000 --lr 1e-3
    python train.py --epochs 20000 --lr 5e-4 --option_type put
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from black_scholes_pinn import BlackScholesPINN, create_model
from data_loader import generate_synthetic_data, generate_put_data, black_scholes_analytical


def compute_pde_loss(
    model: BlackScholesPINN,
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
    r: float,
) -> torch.Tensor:
    """
    Compute the PDE residual loss.

    L_pde = mean(residual^2) where
    residual = dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V
    """
    S.requires_grad_(True)
    t.requires_grad_(True)

    residual = model.compute_pde_residual(S, t, sigma, r)
    loss = torch.mean(residual ** 2)

    return loss


def compute_boundary_loss(
    model: BlackScholesPINN,
    S_lower: torch.Tensor,
    t_lower: torch.Tensor,
    V_lower: torch.Tensor,
    S_upper: torch.Tensor,
    t_upper: torch.Tensor,
    V_upper: torch.Tensor,
) -> torch.Tensor:
    """
    Compute boundary condition loss at S=0 and S=S_max.
    """
    V_pred_lower = model(S_lower, t_lower)
    V_pred_upper = model(S_upper, t_upper)

    loss_lower = torch.mean((V_pred_lower - V_lower) ** 2)
    loss_upper = torch.mean((V_pred_upper - V_upper) ** 2)

    return loss_lower + loss_upper


def compute_terminal_loss(
    model: BlackScholesPINN,
    S_ic: torch.Tensor,
    t_ic: torch.Tensor,
    V_ic: torch.Tensor,
) -> torch.Tensor:
    """
    Compute terminal condition loss at t=T (payoff).
    """
    V_pred = model(S_ic, t_ic)
    loss = torch.mean((V_pred - V_ic) ** 2)
    return loss


def compute_validation_error(
    model: BlackScholesPINN,
    S_val: torch.Tensor,
    t_val: torch.Tensor,
    V_val: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute validation error against analytical solution.

    Returns:
        (mean_abs_error, max_abs_error)
    """
    with torch.no_grad():
        V_pred = model(S_val, t_val)
        abs_error = torch.abs(V_pred - V_val)
        mae = torch.mean(abs_error).item()
        max_err = torch.max(abs_error).item()

    return mae, max_err


def get_loss_weights(
    epoch: int,
    total_epochs: int,
    strategy: str = "adaptive",
) -> Tuple[float, float, float]:
    """
    Get loss component weights based on training phase.

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        strategy: 'fixed', 'phased', or 'adaptive'

    Returns:
        (lambda_pde, lambda_bc, lambda_ic)
    """
    if strategy == "fixed":
        return 1.0, 10.0, 10.0

    elif strategy == "phased":
        progress = epoch / total_epochs
        if progress < 0.1:
            return 1.0, 50.0, 50.0
        elif progress < 0.3:
            return 1.0, 20.0, 20.0
        elif progress < 0.6:
            return 1.0, 10.0, 10.0
        else:
            return 1.0, 5.0, 5.0

    elif strategy == "adaptive":
        # Start with strong BC/IC emphasis, gradually relax
        progress = epoch / total_epochs
        bc_weight = 50.0 * (1.0 - progress) + 5.0 * progress
        ic_weight = 50.0 * (1.0 - progress) + 5.0 * progress
        return 1.0, bc_weight, ic_weight

    else:
        return 1.0, 10.0, 10.0


def train_pinn(
    model: BlackScholesPINN,
    data: Dict[str, torch.Tensor],
    num_epochs: int = 10000,
    learning_rate: float = 1e-3,
    weight_strategy: str = "adaptive",
    resample_every: int = 1000,
    print_every: int = 500,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train the PINN to solve the Black-Scholes PDE.

    Args:
        model: PINN model
        data: Training data from generate_synthetic_data()
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_strategy: Loss weighting strategy
        resample_every: Re-sample collocation points every N epochs
        print_every: Print progress every N epochs
        save_path: Path to save the trained model

    Returns:
        Training history dictionary
    """
    device = next(model.parameters()).device
    sigma = data["sigma"]
    r = data["r"]
    K = data["K"]
    T = data["T"]
    S_max = data["S_max"]

    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1000, min_lr=1e-6
    )

    # Training history
    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_ic": [],
        "val_mae": [],
        "val_max": [],
        "learning_rate": [],
    }

    print("=" * 70)
    print("Training Black-Scholes PINN")
    print("=" * 70)
    print(f"  Parameters: K={K}, r={r}, sigma={sigma}, T={T}, S_max={S_max}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"  Weight strategy: {weight_strategy}")
    print(f"  Device: {device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        # Re-sample collocation points periodically for better coverage
        if epoch > 0 and epoch % resample_every == 0:
            new_data = generate_synthetic_data(
                K=K, r=r, sigma=sigma, T=T, S_max=S_max,
                device=str(device), seed=42 + epoch,
            )
            data["S_pde"] = new_data["S_pde"]
            data["t_pde"] = new_data["t_pde"]

        # Get loss weights
        lam_pde, lam_bc, lam_ic = get_loss_weights(epoch, num_epochs, weight_strategy)

        # Compute PDE loss
        loss_pde = compute_pde_loss(model, data["S_pde"], data["t_pde"], sigma, r)

        # Compute boundary loss
        loss_bc = compute_boundary_loss(
            model,
            data["S_bc_lower"], data["t_bc_lower"], data["V_bc_lower"],
            data["S_bc_upper"], data["t_bc_upper"], data["V_bc_upper"],
        )

        # Compute terminal condition loss
        loss_ic = compute_terminal_loss(model, data["S_ic"], data["t_ic"], data["V_ic"])

        # Total loss
        loss_total = lam_pde * loss_pde + lam_bc * loss_bc + lam_ic * loss_ic

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(loss_total)

        # Record history
        history["loss_total"].append(loss_total.item())
        history["loss_pde"].append(loss_pde.item())
        history["loss_bc"].append(loss_bc.item())
        history["loss_ic"].append(loss_ic.item())
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Validation
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            model.eval()
            if "S_val" in data:
                mae, max_err = compute_validation_error(
                    model, data["S_val"], data["t_val"], data["V_val"]
                )
            else:
                mae, max_err = 0.0, 0.0

            history["val_mae"].append(mae)
            history["val_max"].append(max_err)

            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:6d}/{num_epochs} | "
                f"Loss: {loss_total.item():.6f} "
                f"(PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, IC: {loss_ic.item():.6f}) | "
                f"MAE: {mae:.4f} MaxErr: {max_err:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    elapsed_total = time.time() - start_time
    print("=" * 70)
    print(f"Training complete in {elapsed_total:.1f}s")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    if history["val_mae"]:
        print(f"Final MAE: {history['val_mae'][-1]:.4f}")
        print(f"Final Max Error: {history['val_max'][-1]:.4f}")

    # Save model
    if save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": history,
            "params": {"K": K, "r": r, "sigma": sigma, "T": T, "S_max": S_max},
        }, save_path)
        print(f"Model saved to {save_path}")

    return history


def validate_against_analytical(
    model: BlackScholesPINN,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    option_type: str = "call",
):
    """Print a comparison table of PINN vs analytical prices."""
    device = next(model.parameters()).device
    model.eval()

    print("\n" + "=" * 65)
    print(f"Validation: PINN vs Analytical Black-Scholes ({option_type.upper()})")
    print("=" * 65)
    print(f"{'S':>8} {'t':>8} {'BS Price':>10} {'PINN':>10} {'Error':>10} {'Rel %':>8}")
    print("-" * 65)

    test_points = [
        (80, 0.0), (90, 0.0), (100, 0.0), (110, 0.0), (120, 0.0),
        (100, 0.25), (100, 0.50), (100, 0.75),
    ]

    errors = []
    for S_val, t_val in test_points:
        # Time-to-maturity for analytical
        tau = T - t_val
        bs_price = black_scholes_analytical(
            np.array([S_val]), K, tau, r, sigma, option_type
        )[0]

        # PINN prediction
        S_t = torch.tensor([S_val], dtype=torch.float32, device=device)
        t_t = torch.tensor([t_val], dtype=torch.float32, device=device)
        with torch.no_grad():
            pinn_price = model(S_t, t_t).item()

        error = abs(pinn_price - bs_price)
        rel_error = error / max(bs_price, 0.01) * 100
        errors.append(error)

        print(f"{S_val:8.1f} {t_val:8.2f} {bs_price:10.4f} {pinn_price:10.4f} "
              f"{error:10.4f} {rel_error:7.2f}%")

    print("-" * 65)
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Max Absolute Error:  {np.max(errors):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Black-Scholes PINN")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--T", type=float, default=1.0, help="Maturity (years)")
    parser.add_argument("--S_max", type=float, default=200.0, help="Max spot price")
    parser.add_argument("--option_type", type=str, default="call", choices=["call", "put"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128])
    parser.add_argument("--weight_strategy", type=str, default="adaptive",
                        choices=["fixed", "phased", "adaptive"])
    parser.add_argument("--save", type=str, default="pinn_bs_model.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Create model
    model = create_model(
        hidden_layers=args.hidden,
        S_max=args.S_max,
        T=args.T,
        device=device,
    )

    # Generate training data
    if args.option_type == "call":
        data = generate_synthetic_data(
            K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max,
            device=device,
        )
    else:
        data = generate_put_data(
            K=args.K, r=args.r, sigma=args.sigma, T=args.T, S_max=args.S_max,
            device=device,
        )

    # Train
    history = train_pinn(
        model=model,
        data=data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_strategy=args.weight_strategy,
        save_path=args.save,
    )

    # Validate
    validate_against_analytical(
        model, K=args.K, r=args.r, sigma=args.sigma, T=args.T,
        option_type=args.option_type,
    )

    return model, history


if __name__ == "__main__":
    model, history = main()

"""
Chapter 141: Physics-Informed Neural Networks for Black-Scholes
================================================================

This package implements a PINN-based solver for the Black-Scholes PDE,
with support for European option pricing, Greeks computation via automatic
differentiation, and application to both stock and crypto (Bybit) markets.

Modules:
    - black_scholes_pinn: PINN model architecture for the Black-Scholes PDE
    - train: Training loop with PDE residual loss
    - data_loader: Data fetching for stocks and crypto (Bybit API)
    - greeks: Greeks computation via autograd (Delta, Gamma, Theta, Vega)
    - visualize: Plotting option surfaces, PDE residuals, and comparisons
    - backtest: Backtesting options pricing strategies

Usage:
    from python.black_scholes_pinn import BlackScholesPINN
    from python.train import train_pinn
    from python.greeks import compute_greeks
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"

from .black_scholes_pinn import BlackScholesPINN
from .greeks import compute_greeks, compute_delta, compute_gamma, compute_theta
from .data_loader import (
    generate_synthetic_data,
    fetch_bybit_options,
    black_scholes_analytical,
)

"""
Backtest for the Black-Scholes PINN option pricing.

Strategies:
1. Delta-hedging using PINN Greeks
2. Mispricing detection: compare PINN prices to market prices
3. Volatility surface arbitrage detection

Supports both stock options and Bybit crypto options.

Usage:
    python backtest.py --symbol BTC --exchange bybit
    python backtest.py --symbol AAPL --exchange stock
"""

import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from black_scholes_pinn import BlackScholesPINN, create_model
from data_loader import (
    black_scholes_analytical,
    black_scholes_greeks,
    generate_stock_options_data,
    fetch_bybit_options,
    fetch_bybit_spot_price,
    parse_bybit_option_symbol,
)
from greeks import compute_greeks


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    symbol: str
    option_type: str
    strike: float
    maturity: float
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    pinn_price: float
    market_price: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_mispricing: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


# =============================================================================
# Delta Hedging Backtest
# =============================================================================

def backtest_delta_hedging(
    model: BlackScholesPINN,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    S0: float = 100.0,
    n_steps: int = 252,
    n_paths: int = 100,
    device: str = "cpu",
) -> Dict:
    """
    Backtest a delta-hedging strategy using PINN-computed Greeks.

    Simulates GBM paths, sells a call option, and hedges using PINN Delta.
    The PnL should be near zero for a well-calibrated PINN.

    Args:
        model: Trained PINN
        K: Strike
        r: Risk-free rate
        sigma: True volatility
        T: Time to maturity
        S0: Initial spot
        n_steps: Number of hedging intervals
        n_paths: Number of Monte Carlo paths
        device: Torch device

    Returns:
        Dictionary with hedging PnL statistics
    """
    model.eval()
    dt = T / n_steps

    # Simulate GBM paths
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths, n_steps))
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0

    for i in range(n_steps):
        S_paths[:, i + 1] = S_paths[:, i] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, i]
        )

    # Delta hedging PnL for each path
    hedge_pnls = []
    analytical_pnls = []

    for path_idx in range(n_paths):
        # Sell the call at t=0
        S_t = S_paths[path_idx, 0]
        option_price_0 = black_scholes_analytical(
            np.array([S_t]), K, T, r, sigma, "call"
        )[0]

        # Cash account: receive option premium
        cash = option_price_0
        shares_held = 0.0

        cash_analytical = option_price_0
        shares_analytical = 0.0

        for step in range(n_steps):
            t_val = step * dt
            S_t = S_paths[path_idx, step]
            tau = T - t_val

            # PINN Delta
            S_tensor = torch.tensor([S_t], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)

            S_tensor_grad = S_tensor.detach().requires_grad_(True)
            V = model(S_tensor_grad.unsqueeze(1), t_tensor.unsqueeze(1))
            delta_pinn = torch.autograd.grad(
                V, S_tensor_grad, grad_outputs=torch.ones_like(V)
            )[0].item()

            # Analytical Delta
            greeks_analytical = black_scholes_greeks(
                np.array([S_t]), K, tau, r, sigma, "call"
            )
            delta_bs = greeks_analytical["delta"][0]

            # Rebalance: adjust shares to match delta
            # PINN hedge
            shares_change = delta_pinn - shares_held
            cash -= shares_change * S_t  # buy/sell shares
            cash *= np.exp(r * dt)  # earn interest
            shares_held = delta_pinn

            # Analytical hedge
            shares_change_a = delta_bs - shares_analytical
            cash_analytical -= shares_change_a * S_t
            cash_analytical *= np.exp(r * dt)
            shares_analytical = delta_bs

        # At expiration
        S_T = S_paths[path_idx, -1]
        payoff = max(S_T - K, 0)

        # Liquidate: sell shares, pay payoff
        final_pnl = cash + shares_held * S_T - payoff
        hedge_pnls.append(final_pnl)

        final_pnl_a = cash_analytical + shares_analytical * S_T - payoff
        analytical_pnls.append(final_pnl_a)

    hedge_pnls = np.array(hedge_pnls)
    analytical_pnls = np.array(analytical_pnls)

    results = {
        "pinn_hedge": {
            "mean_pnl": np.mean(hedge_pnls),
            "std_pnl": np.std(hedge_pnls),
            "min_pnl": np.min(hedge_pnls),
            "max_pnl": np.max(hedge_pnls),
        },
        "analytical_hedge": {
            "mean_pnl": np.mean(analytical_pnls),
            "std_pnl": np.std(analytical_pnls),
            "min_pnl": np.min(analytical_pnls),
            "max_pnl": np.max(analytical_pnls),
        },
        "hedge_pnls": hedge_pnls,
        "analytical_pnls": analytical_pnls,
    }

    return results


# =============================================================================
# Mispricing Detection Backtest
# =============================================================================

def backtest_mispricing(
    model: BlackScholesPINN,
    options_data: List[Dict],
    sigma: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    threshold: float = 0.02,
    device: str = "cpu",
) -> BacktestResult:
    """
    Backtest a mispricing strategy: buy underpriced, sell overpriced options.

    Compares PINN theoretical price to market mid-price.
    If |PINN - market| / market > threshold, trade.

    Args:
        model: Trained PINN
        options_data: List of option data dicts (from data_loader)
        sigma: Implied volatility for PINN pricing
        r: Risk-free rate
        T: Maturity
        threshold: Mispricing threshold (fraction)
        device: Torch device

    Returns:
        BacktestResult with trades and statistics
    """
    model.eval()
    result = BacktestResult()
    equity = 0.0

    for opt in options_data:
        spot = opt["spot"]
        strike = opt["strike"]
        maturity = opt["maturity"]
        market_price = opt["mid_price"]
        opt_type = opt["option_type"]

        if market_price <= 0.01 or maturity <= 0.001:
            continue

        # Compute PINN price
        S_tensor = torch.tensor([spot], dtype=torch.float32, device=device)
        # Map maturity to time: t = T - maturity (time since start)
        t_val = T - maturity if T >= maturity else 0.0
        t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)

        with torch.no_grad():
            pinn_price = model(S_tensor, t_tensor).item()

        # Ensure non-negative
        pinn_price = max(pinn_price, 0.0)

        # Check mispricing
        mispricing = (pinn_price - market_price) / market_price

        if abs(mispricing) > threshold:
            # Buy if PINN says it's cheap, sell if PINN says it's expensive
            side = "buy" if mispricing > threshold else "sell"

            # Simulate PnL: at expiry, the option converges to intrinsic value
            # For simplicity, PnL = direction * (fair_value - market_price)
            direction = 1.0 if side == "buy" else -1.0
            # Use analytical BS as "true" value for PnL calculation
            bs_price = black_scholes_analytical(
                np.array([spot]), strike, maturity, r, sigma, opt_type
            )[0]
            trade_pnl = direction * (bs_price - market_price)

            trade = Trade(
                timestamp=len(result.trades),
                symbol=opt.get("ticker", "UNKNOWN"),
                option_type=opt_type,
                strike=strike,
                maturity=maturity,
                side=side,
                price=market_price,
                quantity=1.0,
                pinn_price=pinn_price,
                market_price=market_price,
                pnl=trade_pnl,
            )

            result.trades.append(trade)
            equity += trade_pnl
            result.equity_curve.append(equity)

    # Compute statistics
    result.num_trades = len(result.trades)
    if result.num_trades > 0:
        pnls = [t.pnl for t in result.trades]
        result.total_pnl = sum(pnls)
        result.win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        result.avg_mispricing = np.mean([
            abs(t.pinn_price - t.market_price) / t.market_price
            for t in result.trades
        ])

        if len(pnls) > 1 and np.std(pnls) > 0:
            result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)

        # Max drawdown
        if result.equity_curve:
            peak = result.equity_curve[0]
            max_dd = 0.0
            for eq in result.equity_curve:
                peak = max(peak, eq)
                dd = peak - eq
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

    return result


# =============================================================================
# Crypto Options Backtest (Bybit)
# =============================================================================

def backtest_crypto_options(
    model: BlackScholesPINN,
    base_coin: str = "BTC",
    r: float = 0.05,
    device: str = "cpu",
    threshold: float = 0.05,
) -> Optional[BacktestResult]:
    """
    Backtest using live Bybit options data.

    Fetches current options, compares PINN prices to market marks,
    and identifies potential mispricings.

    Args:
        model: Trained PINN
        base_coin: BTC, ETH, SOL
        r: Risk-free rate
        device: Torch device
        threshold: Mispricing threshold

    Returns:
        BacktestResult or None if data fetch fails
    """
    # Fetch current data
    spot = fetch_bybit_spot_price(f"{base_coin}USDT")
    if spot is None:
        print(f"Could not fetch {base_coin} spot price.")
        return None

    tickers = fetch_bybit_options(base_coin)
    if not tickers:
        print("Could not fetch options data.")
        return None

    print(f"\n{base_coin} Spot Price: ${spot:,.2f}")
    print(f"Options fetched: {len(tickers)}")

    result = BacktestResult()
    equity = 0.0

    for ticker in tickers:
        parsed = parse_bybit_option_symbol(ticker["symbol"])
        if not parsed:
            continue

        try:
            mark_price = float(ticker.get("markPrice", 0))
            mark_iv = float(ticker.get("markIv", 0))
            strike = parsed["strike"]
            opt_type = parsed["option_type"]

            if mark_price <= 0 or mark_iv <= 0:
                continue

            # Use a simplified maturity estimate (real code would parse expiry dates)
            # For demonstration, estimate from the symbol
            maturity_days = 30.0  # Default estimate
            T_years = maturity_days / 365.0

            if T_years <= 0.001:
                continue

            # Scale spot to normalized range for PINN
            # The PINN was trained on [0, S_max], so we scale
            S_norm = spot / strike * 100.0  # Normalize relative to strike
            K_norm = 100.0

            S_tensor = torch.tensor([S_norm], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)

            with torch.no_grad():
                pinn_price = model(S_tensor, t_tensor).item()

            # Scale PINN price back
            pinn_price_real = max(pinn_price * strike / 100.0, 0.0)

            # Compare with market mark
            mispricing = (pinn_price_real - mark_price) / mark_price if mark_price > 0 else 0

            if abs(mispricing) > threshold:
                side = "buy" if mispricing < -threshold else "sell"
                direction = 1.0 if side == "buy" else -1.0

                # Simplified PnL estimate
                trade_pnl = direction * abs(pinn_price_real - mark_price) * 0.5

                trade = Trade(
                    timestamp=len(result.trades),
                    symbol=ticker["symbol"],
                    option_type=opt_type,
                    strike=strike,
                    maturity=T_years,
                    side=side,
                    price=mark_price,
                    quantity=1.0,
                    pinn_price=pinn_price_real,
                    market_price=mark_price,
                    pnl=trade_pnl,
                )

                result.trades.append(trade)
                equity += trade_pnl
                result.equity_curve.append(equity)

        except (ValueError, KeyError):
            continue

    # Statistics
    result.num_trades = len(result.trades)
    if result.num_trades > 0:
        pnls = [t.pnl for t in result.trades]
        result.total_pnl = sum(pnls)
        result.win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0

    return result


# =============================================================================
# Reporting
# =============================================================================

def print_backtest_report(result: BacktestResult, title: str = "Backtest"):
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Total Trades:    {result.num_trades}")
    print(f"  Total PnL:       ${result.total_pnl:,.2f}")
    print(f"  Win Rate:        {result.win_rate:.1%}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:    ${result.max_drawdown:,.2f}")
    print(f"  Avg Mispricing:  {result.avg_mispricing:.2%}")

    if result.trades:
        print(f"\n  Sample Trades:")
        print(f"  {'Symbol':<20} {'Type':<6} {'Side':<6} {'Strike':>10} "
              f"{'Market':>10} {'PINN':>10} {'PnL':>10}")
        print("  " + "-" * 82)
        for t in result.trades[:10]:
            print(f"  {t.symbol:<20} {t.option_type:<6} {t.side:<6} "
                  f"{t.strike:>10.2f} {t.market_price:>10.4f} "
                  f"{t.pinn_price:>10.4f} {t.pnl:>10.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backtest PINN options pricing")
    parser.add_argument("--model", type=str, default="pinn_bs_model.pt")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--exchange", type=str, default="stock",
                        choices=["stock", "bybit"])
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        params = checkpoint["params"]
        model = create_model(S_max=params["S_max"], T=params["T"], device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {args.model}")
    except FileNotFoundError:
        print(f"Model {args.model} not found. Training a quick model...")
        from train import train_pinn
        from data_loader import generate_synthetic_data

        params = {"K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0, "S_max": 200.0}
        model = create_model(S_max=params["S_max"], T=params["T"], device=device)
        data = generate_synthetic_data(**params, device=device)
        train_pinn(model, data, num_epochs=3000, print_every=1000)

    model.eval()
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    if args.exchange == "bybit":
        # Crypto options backtest
        result = backtest_crypto_options(
            model, base_coin=args.symbol, r=r,
            device=device, threshold=args.threshold,
        )
        if result:
            print_backtest_report(result, f"Bybit {args.symbol} Options")
    else:
        # Stock options backtest
        print("\n--- Delta Hedging Backtest ---")
        hedge_results = backtest_delta_hedging(
            model, K=K, r=r, sigma=sigma, T=T,
            S0=K, n_paths=200, device=device,
        )

        print(f"\nPINN Delta Hedging:")
        print(f"  Mean PnL:  {hedge_results['pinn_hedge']['mean_pnl']:.4f}")
        print(f"  Std PnL:   {hedge_results['pinn_hedge']['std_pnl']:.4f}")
        print(f"  Min PnL:   {hedge_results['pinn_hedge']['min_pnl']:.4f}")
        print(f"  Max PnL:   {hedge_results['pinn_hedge']['max_pnl']:.4f}")

        print(f"\nAnalytical Delta Hedging:")
        print(f"  Mean PnL:  {hedge_results['analytical_hedge']['mean_pnl']:.4f}")
        print(f"  Std PnL:   {hedge_results['analytical_hedge']['std_pnl']:.4f}")

        # Mispricing backtest
        print("\n--- Mispricing Detection ---")
        stock_opts = generate_stock_options_data(
            args.symbol, spot=K, sigma=sigma, r=r,
        )
        result = backtest_mispricing(
            model, stock_opts, sigma=sigma, r=r, T=T,
            threshold=args.threshold, device=device,
        )
        print_backtest_report(result, f"{args.symbol} Mispricing Detection")


if __name__ == "__main__":
    main()

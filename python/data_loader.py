"""
Data loader for Black-Scholes PINN.

Provides:
1. Synthetic data generation (collocation points, boundary/terminal conditions)
2. Analytical Black-Scholes solution for validation
3. Bybit crypto options data fetching
4. Stock options data (simulated)
"""

import numpy as np
import torch
import requests
from typing import Tuple, Dict, List, Optional
from scipy.stats import norm


# =============================================================================
# Analytical Black-Scholes Solution (for validation)
# =============================================================================

def black_scholes_analytical(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> np.ndarray:
    """
    Analytical Black-Scholes formula for European options.

    Args:
        S: Spot prices (array)
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'

    Returns:
        Option prices (array)
    """
    S = np.asarray(S, dtype=np.float64)

    # Handle edge cases
    if T <= 0:
        if option_type == "call":
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    return price


def black_scholes_greeks(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict[str, np.ndarray]:
    """
    Compute analytical Greeks for European options.

    Returns:
        Dictionary with Delta, Gamma, Theta, Vega, Rho
    """
    S = np.asarray(S, dtype=np.float64)

    if T <= 0:
        delta = np.where(S > K, 1.0, 0.0) if option_type == "call" else np.where(S < K, -1.0, 0.0)
        return {"delta": delta, "gamma": np.zeros_like(S), "theta": np.zeros_like(S),
                "vega": np.zeros_like(S), "rho": np.zeros_like(S)}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Common terms
    n_d1 = norm.pdf(d1)  # Standard normal PDF

    # Gamma (same for call and put)
    gamma = n_d1 / (S * sigma * np.sqrt(T))

    # Vega (same for call and put), per 1% vol change
    vega = S * n_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * n_d1 * sigma / (2.0 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (
            -S * n_d1 * sigma / (2.0 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# =============================================================================
# Synthetic Data Generation (Collocation Points)
# =============================================================================

def generate_synthetic_data(
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    S_max: float = 200.0,
    N_pde: int = 10000,
    N_bc: int = 1000,
    N_ic: int = 2000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic training data for the PINN.

    Creates collocation points for:
    1. Interior PDE points (Latin Hypercube sampling)
    2. Boundary condition points (S=0 and S=S_max)
    3. Terminal condition points (t=T)

    Args:
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        S_max: Maximum spot price
        N_pde: Number of PDE collocation points
        N_bc: Number of boundary points (per boundary)
        N_ic: Number of terminal condition points
        device: Torch device
        seed: Random seed

    Returns:
        Dictionary with all training data tensors
    """
    np.random.seed(seed)

    # ----- PDE Collocation Points (interior domain) -----
    # Use denser sampling near the strike (where option value changes rapidly)
    S_pde_uniform = np.random.uniform(0.01, S_max, N_pde // 2)
    S_pde_strike = K + np.random.normal(0, K * 0.3, N_pde // 2)
    S_pde_strike = np.clip(S_pde_strike, 0.01, S_max)
    S_pde = np.concatenate([S_pde_uniform, S_pde_strike])
    t_pde = np.random.uniform(0.0, T - 1e-6, N_pde)

    # ----- Boundary Conditions -----
    # Lower boundary: S = 0 (approximately)
    t_bc_lower = np.random.uniform(0.0, T, N_bc)
    S_bc_lower = np.full(N_bc, 0.01)  # Small epsilon to avoid S=0

    # Upper boundary: S = S_max
    t_bc_upper = np.random.uniform(0.0, T, N_bc)
    S_bc_upper = np.full(N_bc, S_max)

    # Boundary values for a call option
    V_bc_lower = np.zeros(N_bc)  # V(0, t) = 0 for a call
    V_bc_upper = S_max - K * np.exp(-r * (T - t_bc_upper))  # V(S_max, t) ~ S - Ke^{-r(T-t)}

    # ----- Terminal Condition (t = T) -----
    # Denser near the strike
    S_ic_uniform = np.random.uniform(0.01, S_max, N_ic // 2)
    S_ic_strike = K + np.random.normal(0, K * 0.2, N_ic // 2)
    S_ic_strike = np.clip(S_ic_strike, 0.01, S_max)
    S_ic = np.concatenate([S_ic_uniform, S_ic_strike])
    t_ic = np.full(N_ic, T)

    # Payoff at expiration
    V_ic = np.maximum(S_ic - K, 0.0)  # Call payoff

    # ----- Validation Set -----
    S_val = np.linspace(0.01, S_max, 100)
    t_val = np.linspace(0.0, T, 50)
    S_val_grid, t_val_grid = np.meshgrid(S_val, t_val)
    S_val_flat = S_val_grid.flatten()
    t_val_flat = t_val_grid.flatten()

    # Analytical solution for validation (using time-to-maturity = T - t)
    V_val_flat = np.array([
        black_scholes_analytical(s, K, T - t_v, r, sigma, "call")
        if T - t_v > 0 else max(s - K, 0.0)
        for s, t_v in zip(S_val_flat, t_val_flat)
    ])

    # Convert to tensors
    data = {
        # PDE collocation
        "S_pde": torch.tensor(S_pde, dtype=torch.float32, device=device).unsqueeze(1),
        "t_pde": torch.tensor(t_pde, dtype=torch.float32, device=device).unsqueeze(1),
        # Boundary conditions
        "S_bc_lower": torch.tensor(S_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "t_bc_lower": torch.tensor(t_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "V_bc_lower": torch.tensor(V_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "S_bc_upper": torch.tensor(S_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        "t_bc_upper": torch.tensor(t_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        "V_bc_upper": torch.tensor(V_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        # Terminal condition
        "S_ic": torch.tensor(S_ic, dtype=torch.float32, device=device).unsqueeze(1),
        "t_ic": torch.tensor(t_ic, dtype=torch.float32, device=device).unsqueeze(1),
        "V_ic": torch.tensor(V_ic, dtype=torch.float32, device=device).unsqueeze(1),
        # Validation
        "S_val": torch.tensor(S_val_flat, dtype=torch.float32, device=device).unsqueeze(1),
        "t_val": torch.tensor(t_val_flat, dtype=torch.float32, device=device).unsqueeze(1),
        "V_val": torch.tensor(V_val_flat, dtype=torch.float32, device=device).unsqueeze(1),
        # Parameters
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "S_max": S_max,
    }

    return data


def generate_put_data(
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    S_max: float = 200.0,
    N_pde: int = 10000,
    N_bc: int = 1000,
    N_ic: int = 2000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate training data for a European put option."""
    np.random.seed(seed)

    # PDE collocation (same as call)
    S_pde_uniform = np.random.uniform(0.01, S_max, N_pde // 2)
    S_pde_strike = K + np.random.normal(0, K * 0.3, N_pde // 2)
    S_pde_strike = np.clip(S_pde_strike, 0.01, S_max)
    S_pde = np.concatenate([S_pde_uniform, S_pde_strike])
    t_pde = np.random.uniform(0.0, T - 1e-6, N_pde)

    # Boundary: S = 0 -> V(0,t) = K * exp(-r*(T-t))
    t_bc_lower = np.random.uniform(0.0, T, N_bc)
    S_bc_lower = np.full(N_bc, 0.01)
    V_bc_lower = K * np.exp(-r * (T - t_bc_lower))

    # Boundary: S = S_max -> V(S_max, t) ~ 0
    t_bc_upper = np.random.uniform(0.0, T, N_bc)
    S_bc_upper = np.full(N_bc, S_max)
    V_bc_upper = np.zeros(N_bc)

    # Terminal: V(S, T) = max(K - S, 0)
    S_ic_uniform = np.random.uniform(0.01, S_max, N_ic // 2)
    S_ic_strike = K + np.random.normal(0, K * 0.2, N_ic // 2)
    S_ic_strike = np.clip(S_ic_strike, 0.01, S_max)
    S_ic = np.concatenate([S_ic_uniform, S_ic_strike])
    t_ic = np.full(N_ic, T)
    V_ic = np.maximum(K - S_ic, 0.0)

    data = {
        "S_pde": torch.tensor(S_pde, dtype=torch.float32, device=device).unsqueeze(1),
        "t_pde": torch.tensor(t_pde, dtype=torch.float32, device=device).unsqueeze(1),
        "S_bc_lower": torch.tensor(S_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "t_bc_lower": torch.tensor(t_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "V_bc_lower": torch.tensor(V_bc_lower, dtype=torch.float32, device=device).unsqueeze(1),
        "S_bc_upper": torch.tensor(S_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        "t_bc_upper": torch.tensor(t_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        "V_bc_upper": torch.tensor(V_bc_upper, dtype=torch.float32, device=device).unsqueeze(1),
        "S_ic": torch.tensor(S_ic, dtype=torch.float32, device=device).unsqueeze(1),
        "t_ic": torch.tensor(t_ic, dtype=torch.float32, device=device).unsqueeze(1),
        "V_ic": torch.tensor(V_ic, dtype=torch.float32, device=device).unsqueeze(1),
        "K": K, "r": r, "sigma": sigma, "T": T, "S_max": S_max,
    }
    return data


# =============================================================================
# Bybit Crypto Options Data
# =============================================================================

def fetch_bybit_options(
    base_coin: str = "BTC",
    limit: int = 100,
) -> List[Dict]:
    """
    Fetch options ticker data from Bybit API v5.

    Args:
        base_coin: Base coin (BTC, ETH, SOL)
        limit: Max number of instruments to return

    Returns:
        List of option ticker dictionaries
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "option",
        "baseCoin": base_coin,
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            print(f"Bybit API error: {data['retMsg']}")
            return []

        tickers = data["result"]["list"]
        return tickers

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bybit data: {e}")
        return []


def parse_bybit_option_symbol(symbol: str) -> Dict:
    """
    Parse a Bybit option symbol like 'BTC-28JUN24-70000-C'.

    Returns:
        Dictionary with base_coin, expiry, strike, option_type
    """
    parts = symbol.split("-")
    if len(parts) != 4:
        return {}

    return {
        "base_coin": parts[0],
        "expiry": parts[1],
        "strike": float(parts[2]),
        "option_type": "call" if parts[3] == "C" else "put",
    }


def fetch_bybit_spot_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Fetch current spot price from Bybit.

    Args:
        symbol: Trading pair symbol

    Returns:
        Current price or None
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "spot",
        "symbol": symbol,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] == 0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["lastPrice"])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching spot price: {e}")

    return None


def prepare_bybit_training_data(
    base_coin: str = "BTC",
    r: float = 0.05,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Fetch Bybit options data and prepare it for PINN training / validation.

    This creates market-data-informed collocation points and validation targets.

    Returns:
        Dictionary with training data derived from market prices, or None if fetch fails
    """
    # Fetch current spot
    spot_symbol = f"{base_coin}USDT"
    spot_price = fetch_bybit_spot_price(spot_symbol)
    if spot_price is None:
        print("Could not fetch spot price. Using synthetic data.")
        return None

    print(f"Current {base_coin} spot price: ${spot_price:,.2f}")

    # Fetch options tickers
    tickers = fetch_bybit_options(base_coin)
    if not tickers:
        print("Could not fetch options data. Using synthetic data.")
        return None

    print(f"Fetched {len(tickers)} option tickers from Bybit")

    # Parse and collect market data
    market_data = []
    for ticker in tickers:
        parsed = parse_bybit_option_symbol(ticker["symbol"])
        if not parsed:
            continue

        try:
            mark_price = float(ticker.get("markPrice", 0))
            mark_iv = float(ticker.get("markIv", 0))
            bid = float(ticker.get("bid1Price", 0))
            ask = float(ticker.get("ask1Price", 0))

            if mark_price > 0 and mark_iv > 0:
                market_data.append({
                    "symbol": ticker["symbol"],
                    "strike": parsed["strike"],
                    "option_type": parsed["option_type"],
                    "expiry": parsed["expiry"],
                    "mark_price": mark_price,
                    "implied_vol": mark_iv,
                    "bid": bid,
                    "ask": ask,
                    "spot": spot_price,
                })
        except (ValueError, KeyError):
            continue

    if not market_data:
        print("No valid market data found.")
        return None

    print(f"Parsed {len(market_data)} valid options")

    # Use market data to inform training parameters
    implied_vols = [d["implied_vol"] for d in market_data]
    avg_iv = np.mean(implied_vols)
    strikes = [d["strike"] for d in market_data]

    result = {
        "spot_price": spot_price,
        "avg_implied_vol": avg_iv,
        "market_data": market_data,
        "base_coin": base_coin,
        "r": r,
        "strikes": strikes,
    }

    return result


# =============================================================================
# Stock Options Data (Simulated)
# =============================================================================

def generate_stock_options_data(
    ticker: str = "AAPL",
    spot: float = 175.0,
    strikes: Optional[List[float]] = None,
    maturities: Optional[List[float]] = None,
    sigma: float = 0.25,
    r: float = 0.05,
) -> List[Dict]:
    """
    Generate simulated stock option data for training / backtesting.

    In production, you would replace this with real market data from
    an options data provider (e.g., CBOE, Interactive Brokers API).

    Args:
        ticker: Stock ticker
        spot: Current stock price
        strikes: List of strike prices
        maturities: List of maturities in years
        sigma: Implied volatility
        r: Risk-free rate

    Returns:
        List of option data dictionaries
    """
    if strikes is None:
        strikes = np.linspace(spot * 0.8, spot * 1.2, 9)
    if maturities is None:
        maturities = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0]

    options = []
    for K in strikes:
        for T in maturities:
            for opt_type in ["call", "put"]:
                price = black_scholes_analytical(spot, K, T, r, sigma, opt_type)
                greeks = black_scholes_greeks(spot, K, T, r, sigma, opt_type)

                # Add some noise to simulate bid-ask spread
                spread = price * 0.02  # 2% spread
                bid = max(price - spread / 2, 0.01)
                ask = price + spread / 2

                options.append({
                    "ticker": ticker,
                    "spot": spot,
                    "strike": K,
                    "maturity": T,
                    "option_type": opt_type,
                    "mid_price": float(price),
                    "bid": float(bid),
                    "ask": float(ask),
                    "implied_vol": sigma,
                    "delta": float(greeks["delta"]),
                    "gamma": float(greeks["gamma"]),
                    "theta": float(greeks["theta"]),
                    "vega": float(greeks["vega"]),
                })

    return options


if __name__ == "__main__":
    print("=" * 60)
    print("Black-Scholes PINN Data Loader")
    print("=" * 60)

    # Test analytical solution
    print("\n--- Analytical Black-Scholes ---")
    S_test = np.array([80, 90, 100, 110, 120], dtype=float)
    K, T, r, sigma = 100.0, 1.0, 0.05, 0.2

    call_prices = black_scholes_analytical(S_test, K, T, r, sigma, "call")
    put_prices = black_scholes_analytical(S_test, K, T, r, sigma, "put")

    print(f"K={K}, T={T}, r={r}, sigma={sigma}")
    for i, s in enumerate(S_test):
        print(f"  S={s:6.1f}  Call={call_prices[i]:7.3f}  Put={put_prices[i]:7.3f}")

    # Test synthetic data generation
    print("\n--- Synthetic Data Generation ---")
    data = generate_synthetic_data(K=100.0, r=0.05, sigma=0.2, T=1.0)
    print(f"PDE points:      {data['S_pde'].shape}")
    print(f"BC lower points: {data['S_bc_lower'].shape}")
    print(f"BC upper points: {data['S_bc_upper'].shape}")
    print(f"IC points:       {data['S_ic'].shape}")
    print(f"Validation:      {data['S_val'].shape}")

    # Test Bybit data fetching
    print("\n--- Bybit Crypto Options ---")
    bybit_data = prepare_bybit_training_data("BTC")
    if bybit_data:
        print(f"Spot: ${bybit_data['spot_price']:,.2f}")
        print(f"Avg IV: {bybit_data['avg_implied_vol']:.2%}")
        print(f"Options: {len(bybit_data['market_data'])}")

    # Test stock data
    print("\n--- Stock Options (Simulated) ---")
    stock_opts = generate_stock_options_data("AAPL", spot=175.0)
    print(f"Generated {len(stock_opts)} option contracts")
    for opt in stock_opts[:3]:
        print(f"  {opt['ticker']} {opt['option_type'].upper()} K={opt['strike']:.0f} "
              f"T={opt['maturity']:.2f} Price={opt['mid_price']:.2f} "
              f"Delta={opt['delta']:.4f}")

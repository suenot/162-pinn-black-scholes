//! Price options using the PINN model and compare with analytical Black-Scholes.
//!
//! Usage:
//!   cargo run --bin price_options
//!   cargo run --bin price_options -- --spot 50000 --strike 50000 --maturity 0.25 --sigma 0.8

use clap::Parser;
use colored::Colorize;
use pinn_black_scholes::{
    black_scholes_greeks, black_scholes_price, BSParams, OptionType, PINN,
};

#[derive(Parser, Debug)]
#[command(name = "price_options", about = "Price options using PINN and compare with analytical BS")]
struct Args {
    /// Spot price
    #[arg(long, default_value_t = 100.0)]
    spot: f64,

    /// Strike price
    #[arg(long, default_value_t = 100.0)]
    strike: f64,

    /// Risk-free rate
    #[arg(long, default_value_t = 0.05)]
    rate: f64,

    /// Volatility
    #[arg(long, default_value_t = 0.2)]
    sigma: f64,

    /// Time to maturity (years)
    #[arg(long, default_value_t = 1.0)]
    maturity: f64,

    /// Quick training epochs for PINN
    #[arg(long, default_value_t = 2000)]
    train_epochs: usize,

    /// Show crypto mode with high volatility
    #[arg(long, default_value = "false")]
    crypto: bool,
}

fn main() {
    let args = Args::parse();

    let (spot, strike, sigma, label) = if args.crypto {
        // Crypto mode: typical BTC options parameters
        let spot = if args.spot == 100.0 { 50000.0 } else { args.spot };
        let strike = if args.strike == 100.0 { 50000.0 } else { args.strike };
        let sigma = if (args.sigma - 0.2).abs() < 0.001 { 0.8 } else { args.sigma };
        (spot, strike, sigma, "Crypto (BTC)")
    } else {
        (args.spot, args.strike, args.sigma, "Stock")
    };

    let params = BSParams {
        strike,
        risk_free_rate: args.rate,
        volatility: sigma,
        maturity: args.maturity,
    };

    println!("{}", "=".repeat(70).blue());
    println!("{}", format!(" Option Pricing: {} ", label).bold().blue());
    println!("{}", "=".repeat(70).blue());
    println!("  Spot:       {:.2}", spot);
    println!("  Strike:     {:.2}", strike);
    println!("  Rate:       {:.2}%", args.rate * 100.0);
    println!("  Volatility: {:.2}%", sigma * 100.0);
    println!("  Maturity:   {:.2} years", args.maturity);

    // Analytical prices
    println!("\n{}", "--- Analytical Black-Scholes ---".yellow());

    let call_price = black_scholes_price(spot, &params, OptionType::Call);
    let put_price = black_scholes_price(spot, &params, OptionType::Put);

    println!("  Call Price: {:.4}", call_price);
    println!("  Put Price:  {:.4}", put_price);

    // Put-call parity check
    let parity = call_price - put_price;
    let expected_parity = spot - strike * (-args.rate * args.maturity).exp();
    println!("  Put-Call Parity: C-P={:.4}, S-Ke^(-rT)={:.4} (diff={:.6})",
             parity, expected_parity, (parity - expected_parity).abs());

    // Greeks
    println!("\n{}", "--- Greeks ---".yellow());
    let call_greeks = black_scholes_greeks(spot, &params, OptionType::Call);
    let put_greeks = black_scholes_greeks(spot, &params, OptionType::Put);

    println!(
        "  {:>8} {:>12} {:>12}",
        "Greek", "Call", "Put"
    );
    println!("  {}", "-".repeat(36));
    println!("  {:>8} {:12.4} {:12.4}", "Delta", call_greeks.delta, put_greeks.delta);
    println!("  {:>8} {:12.6} {:12.6}", "Gamma", call_greeks.gamma, put_greeks.gamma);
    println!("  {:>8} {:12.4} {:12.4}", "Theta", call_greeks.theta, put_greeks.theta);
    println!("  {:>8} {:12.4} {:12.4}", "Vega", call_greeks.vega, put_greeks.vega);
    println!("  {:>8} {:12.4} {:12.4}", "Rho", call_greeks.rho, put_greeks.rho);

    // Train a quick PINN and compare
    println!("\n{}", "--- PINN Pricing ---".green());
    println!("Training PINN with {} epochs...", args.train_epochs);

    // For crypto with large S values, we normalize
    let s_max = spot * 2.0;
    let mut pinn = PINN::new(&[64, 64, 64], s_max, args.maturity, 0.001);

    pinn.train(&params, args.train_epochs, 100, 30, 60, args.train_epochs / 4);

    let pinn_call = pinn.forward(spot, 0.0);
    let pinn_greeks = pinn.compute_greeks(spot, 0.0);

    println!("\n  PINN Call Price: {:.4} (BS: {:.4}, Error: {:.4})",
             pinn_call, call_price, (pinn_call - call_price).abs());
    println!("  PINN Delta:      {:.4} (BS: {:.4}, Error: {:.6})",
             pinn_greeks.delta, call_greeks.delta,
             (pinn_greeks.delta - call_greeks.delta).abs());

    // Price at multiple spots
    println!("\n{}", "--- Price Curve ---".cyan());
    let spots: Vec<f64> = (0..11)
        .map(|i| spot * (0.7 + 0.06 * i as f64))
        .collect();

    println!(
        "  {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Spot", "BS Call", "PINN Call", "Error", "Rel %"
    );
    println!("  {}", "-".repeat(60));

    for &s in &spots {
        let bs = black_scholes_price(s, &params, OptionType::Call);
        let pinn_v = pinn.forward(s, 0.0);
        let error = (pinn_v - bs).abs();
        let rel = if bs > 0.01 { error / bs * 100.0 } else { 0.0 };

        println!(
            "  {:12.2} {:12.4} {:12.4} {:12.4} {:9.2}%",
            s, bs, pinn_v, error, rel
        );
    }

    println!("\n{}", "=".repeat(70).green());
}

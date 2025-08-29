//! Training binary for the Black-Scholes PINN.
//!
//! Usage:
//!   cargo run --bin train -- --epochs 5000
//!   cargo run --bin train -- --epochs 10000 --hidden 128 128 128 --lr 0.001

use clap::Parser;
use colored::Colorize;
use pinn_black_scholes::{
    black_scholes_price, BSParams, OptionType, PINN,
};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train a PINN for Black-Scholes option pricing")]
struct Args {
    /// Number of training epochs
    #[arg(long, default_value_t = 5000)]
    epochs: usize,

    /// Hidden layer sizes
    #[arg(long, num_args = 1.., default_values_t = vec![64, 64, 64])]
    hidden: Vec<usize>,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Strike price
    #[arg(long, default_value_t = 100.0)]
    strike: f64,

    /// Risk-free rate
    #[arg(long, default_value_t = 0.05)]
    rate: f64,

    /// Volatility
    #[arg(long, default_value_t = 0.2)]
    sigma: f64,

    /// Maturity (years)
    #[arg(long, default_value_t = 1.0)]
    maturity: f64,

    /// Max spot price (for domain)
    #[arg(long, default_value_t = 200.0)]
    s_max: f64,

    /// Number of PDE collocation points
    #[arg(long, default_value_t = 200)]
    n_pde: usize,

    /// Number of boundary points
    #[arg(long, default_value_t = 50)]
    n_bc: usize,

    /// Number of terminal condition points
    #[arg(long, default_value_t = 100)]
    n_ic: usize,

    /// Print frequency
    #[arg(long, default_value_t = 500)]
    print_every: usize,
}

fn main() {
    let args = Args::parse();

    println!("{}", "=".repeat(70).blue());
    println!("{}", " PINN Black-Scholes Training ".bold().blue());
    println!("{}", "=".repeat(70).blue());

    let params = BSParams {
        strike: args.strike,
        risk_free_rate: args.rate,
        volatility: args.sigma,
        maturity: args.maturity,
    };

    println!("\nModel Configuration:");
    println!("  Hidden layers: {:?}", args.hidden);
    println!("  Learning rate: {}", args.lr);
    println!("  S_max: {}", args.s_max);
    println!("\nBlack-Scholes Parameters:");
    println!("  K={}, r={}, sigma={}, T={}", params.strike, params.risk_free_rate,
             params.volatility, params.maturity);

    // Create PINN
    let mut pinn = PINN::new(&args.hidden, args.s_max, args.maturity, args.lr);
    println!("\nNetwork Parameters: {}", pinn.num_parameters());

    // Pre-training validation
    println!("\n{}", "Pre-Training Validation:".yellow());
    pinn.print_validation_table(&params, OptionType::Call);

    // Train
    let start = Instant::now();
    let loss_history = pinn.train(
        &params,
        args.epochs,
        args.n_pde,
        args.n_bc,
        args.n_ic,
        args.print_every,
    );
    let elapsed = start.elapsed();

    println!("\n{}", "=".repeat(70).green());
    println!("{}", " Training Complete ".bold().green());
    println!("{}", "=".repeat(70).green());
    println!("  Duration: {:.2}s", elapsed.as_secs_f64());
    println!("  Final Loss: {:.6}", loss_history.last().unwrap_or(&0.0));

    // Post-training validation
    println!("\n{}", "Post-Training Validation:".green());
    pinn.print_validation_table(&params, OptionType::Call);

    // Greeks at ATM
    println!("\n{}", "Greeks at S=100 (ATM):".cyan());
    let pinn_greeks = pinn.compute_greeks(100.0, 0.0);
    let bs_greeks = pinn_black_scholes::black_scholes_greeks(100.0, &params, OptionType::Call);

    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "Greek", "Analytical", "PINN", "Error"
    );
    println!("  {}", "-".repeat(50));
    println!(
        "  {:>8} {:12.4} {:12.4} {:12.6}",
        "Delta", bs_greeks.delta, pinn_greeks.delta,
        (pinn_greeks.delta - bs_greeks.delta).abs()
    );
    println!(
        "  {:>8} {:12.6} {:12.6} {:12.6}",
        "Gamma", bs_greeks.gamma, pinn_greeks.gamma,
        (pinn_greeks.gamma - bs_greeks.gamma).abs()
    );
    println!(
        "  {:>8} {:12.4} {:12.4} {:12.6}",
        "Theta", bs_greeks.theta, pinn_greeks.theta,
        (pinn_greeks.theta - bs_greeks.theta).abs()
    );

    // Loss summary
    if loss_history.len() > 10 {
        let first_10 = loss_history[..10].iter().sum::<f64>() / 10.0;
        let last_10 = loss_history[loss_history.len() - 10..].iter().sum::<f64>() / 10.0;
        println!("\nLoss Reduction:");
        println!("  First 10 avg: {:.6}", first_10);
        println!("  Last 10 avg:  {:.6}", last_10);
        println!("  Reduction:    {:.1}x", first_10 / last_10.max(1e-10));
    }
}

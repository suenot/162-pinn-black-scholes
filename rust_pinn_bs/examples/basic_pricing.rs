//! Basic example: Price a European call option using both
//! analytical Black-Scholes and the PINN approach.
//!
//! Run with:
//!   cargo run --example basic_pricing

use pinn_black_scholes::{
    black_scholes_greeks, black_scholes_price, BSParams, OptionType, PINN,
};

fn main() {
    println!("=== PINN Black-Scholes Basic Pricing Example ===\n");

    // Define option parameters
    let params = BSParams {
        strike: 100.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
        maturity: 1.0,
    };

    let spot = 100.0;

    // 1. Analytical Black-Scholes
    println!("--- Analytical Black-Scholes ---");
    let call_price = black_scholes_price(spot, &params, OptionType::Call);
    let put_price = black_scholes_price(spot, &params, OptionType::Put);
    println!("  Call: {:.4}", call_price);
    println!("  Put:  {:.4}", put_price);

    // Verify put-call parity
    let parity_lhs = call_price - put_price;
    let parity_rhs = spot - params.strike * (-params.risk_free_rate * params.maturity).exp();
    println!(
        "  Put-Call Parity: {:.4} vs {:.4} (OK: {})",
        parity_lhs,
        parity_rhs,
        (parity_lhs - parity_rhs).abs() < 0.001
    );

    // 2. Greeks
    println!("\n--- Greeks ---");
    let greeks = black_scholes_greeks(spot, &params, OptionType::Call);
    println!("  Delta: {:.4}", greeks.delta);
    println!("  Gamma: {:.6}", greeks.gamma);
    println!("  Theta: {:.4}", greeks.theta);
    println!("  Vega:  {:.4}", greeks.vega);
    println!("  Rho:   {:.4}", greeks.rho);

    // 3. Price curve across spot prices
    println!("\n--- Price Curve ---");
    println!("  {:>8} {:>10} {:>10}", "Spot", "Call", "Put");
    println!("  {}", "-".repeat(30));

    for s in (60..=140).step_by(10) {
        let s = s as f64;
        let c = black_scholes_price(s, &params, OptionType::Call);
        let p = black_scholes_price(s, &params, OptionType::Put);
        println!("  {:8.1} {:10.4} {:10.4}", s, c, p);
    }

    // 4. Quick PINN demo
    println!("\n--- Quick PINN Training ---");
    let mut pinn = PINN::new(&[64, 64, 64], 200.0, 1.0, 0.001);
    println!("  Network parameters: {}", pinn.num_parameters());
    println!("  Training 2000 epochs...");

    let _loss = pinn.train(&params, 2000, 100, 30, 60, 2000);

    println!("\n--- PINN vs Analytical ---");
    pinn.print_validation_table(&params, OptionType::Call);

    // 5. PINN Greeks
    println!("\n--- PINN Greeks at S=100 ---");
    let pinn_greeks = pinn.compute_greeks(100.0, 0.0);
    println!("  Delta: {:.4} (analytical: {:.4})", pinn_greeks.delta, greeks.delta);
    println!("  Gamma: {:.6} (analytical: {:.6})", pinn_greeks.gamma, greeks.gamma);
    println!("  Theta: {:.4} (analytical: {:.4})", pinn_greeks.theta, greeks.theta);

    println!("\n=== Example Complete ===");
}

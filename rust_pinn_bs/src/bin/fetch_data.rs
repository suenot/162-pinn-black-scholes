//! Fetch crypto options data from Bybit for PINN training and validation.
//!
//! Usage:
//!   cargo run --bin fetch_data
//!   cargo run --bin fetch_data -- --coin ETH

use clap::Parser;
use colored::Colorize;
use pinn_black_scholes::{
    black_scholes_price, fetch_bybit_options, fetch_bybit_spot,
    parse_option_symbol, BSParams, OptionType,
};

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch crypto options data from Bybit")]
struct Args {
    /// Base coin (BTC, ETH, SOL)
    #[arg(long, default_value = "BTC")]
    coin: String,

    /// Risk-free rate assumption
    #[arg(long, default_value_t = 0.05)]
    rate: f64,

    /// Show detailed output
    #[arg(long, default_value = "false")]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    println!("{}", "=".repeat(70).blue());
    println!("{}", format!(" Bybit {} Options Data ", args.coin).bold().blue());
    println!("{}", "=".repeat(70).blue());

    // Fetch spot price
    let spot_symbol = format!("{}USDT", args.coin);
    print!("\nFetching {} spot price... ", spot_symbol);

    match fetch_bybit_spot(&spot_symbol).await {
        Ok(spot) => {
            println!("{}", format!("${:.2}", spot).green());

            // Fetch options
            print!("Fetching {} options... ", args.coin);
            match fetch_bybit_options(&args.coin).await {
                Ok(tickers) => {
                    println!("{}", format!("{} tickers found", tickers.len()).green());

                    // Parse and display
                    println!("\n{}", "--- Options Overview ---".yellow());
                    println!(
                        "  {:30} {:>8} {:>6} {:>12} {:>10} {:>10} {:>10}",
                        "Symbol", "Strike", "Type", "Mark Price", "Mark IV", "Bid", "Ask"
                    );
                    println!("  {}", "-".repeat(90));

                    let mut calls = 0;
                    let mut puts = 0;
                    let mut total_iv = 0.0;
                    let mut iv_count = 0;

                    for ticker in &tickers {
                        let parsed = match parse_option_symbol(&ticker.symbol) {
                            Some(p) => p,
                            None => continue,
                        };

                        let mark_price = ticker
                            .mark_price
                            .as_ref()
                            .and_then(|p| p.parse::<f64>().ok())
                            .unwrap_or(0.0);

                        let mark_iv = ticker
                            .mark_iv
                            .as_ref()
                            .and_then(|p| p.parse::<f64>().ok())
                            .unwrap_or(0.0);

                        let bid = ticker
                            .bid1_price
                            .as_ref()
                            .and_then(|p| p.parse::<f64>().ok())
                            .unwrap_or(0.0);

                        let ask = ticker
                            .ask1_price
                            .as_ref()
                            .and_then(|p| p.parse::<f64>().ok())
                            .unwrap_or(0.0);

                        if mark_price <= 0.0 {
                            continue;
                        }

                        match parsed.option_type {
                            OptionType::Call => calls += 1,
                            OptionType::Put => puts += 1,
                        }

                        if mark_iv > 0.0 {
                            total_iv += mark_iv;
                            iv_count += 1;
                        }

                        let type_str = match parsed.option_type {
                            OptionType::Call => "C",
                            OptionType::Put => "P",
                        };

                        if args.verbose || (mark_price > 0.001 && mark_iv > 0.0) {
                            println!(
                                "  {:30} {:>8.0} {:>6} {:>12.4} {:>9.1}% {:>10.4} {:>10.4}",
                                ticker.symbol,
                                parsed.strike,
                                type_str,
                                mark_price,
                                mark_iv * 100.0,
                                bid,
                                ask
                            );
                        }
                    }

                    // Summary
                    let avg_iv = if iv_count > 0 {
                        total_iv / iv_count as f64
                    } else {
                        0.0
                    };

                    println!("\n{}", "--- Summary ---".cyan());
                    println!("  Spot Price:     ${:.2}", spot);
                    println!("  Total Options:  {}", tickers.len());
                    println!("  Calls:          {}", calls);
                    println!("  Puts:           {}", puts);
                    println!("  Average IV:     {:.1}%", avg_iv * 100.0);

                    // Compare with BS model
                    println!("\n{}", "--- BS Model Comparison ---".cyan());
                    println!("  Using avg IV = {:.1}% for BS pricing", avg_iv * 100.0);

                    let params = BSParams {
                        strike: spot, // ATM
                        risk_free_rate: args.rate,
                        volatility: avg_iv.max(0.1),
                        maturity: 30.0 / 365.0, // ~1 month
                    };

                    let bs_call = black_scholes_price(spot, &params, OptionType::Call);
                    let bs_put = black_scholes_price(spot, &params, OptionType::Put);

                    println!("  ATM 30-day Call (BS): ${:.2}", bs_call);
                    println!("  ATM 30-day Put (BS):  ${:.2}", bs_put);

                    // Strikes distribution
                    let mut strikes: Vec<f64> = tickers
                        .iter()
                        .filter_map(|t| parse_option_symbol(&t.symbol))
                        .map(|p| p.strike)
                        .collect();
                    strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    strikes.dedup();

                    if !strikes.is_empty() {
                        println!("\n  Strike Range: ${:.0} - ${:.0} ({} unique strikes)",
                                 strikes.first().unwrap(),
                                 strikes.last().unwrap(),
                                 strikes.len());
                    }
                }
                Err(e) => {
                    println!("{}", format!("Error: {}", e).red());
                }
            }
        }
        Err(e) => {
            println!("{}", format!("Error: {}", e).red());
            println!("Check your network connection or try again later.");
        }
    }

    println!("\n{}", "=".repeat(70).green());
    Ok(())
}

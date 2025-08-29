//! # Physics-Informed Neural Network for Black-Scholes
//!
//! This crate implements a PINN-based solver for the Black-Scholes PDE
//! for European option pricing.
//!
//! The Black-Scholes PDE:
//! ```text
//! dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0
//! ```
//!
//! The PINN learns V(S, t) by minimizing a loss function that encodes:
//! 1. PDE residual (interior domain)
//! 2. Boundary conditions (S=0 and S=S_max)
//! 3. Terminal condition (payoff at t=T)

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};
use std::f64::consts::PI;

// =============================================================================
// Analytical Black-Scholes (for validation)
// =============================================================================

/// Parameters for the Black-Scholes model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSParams {
    pub strike: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub maturity: f64,
}

impl Default for BSParams {
    fn default() -> Self {
        Self {
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            maturity: 1.0,
        }
    }
}

/// Option type: Call or Put.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

/// Analytical Black-Scholes price for a European option.
pub fn black_scholes_price(spot: f64, params: &BSParams, option_type: OptionType) -> f64 {
    let BSParams {
        strike,
        risk_free_rate: r,
        volatility: sigma,
        maturity: tau,
    } = params;

    if *tau <= 0.0 {
        return match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        };
    }

    let d1 = ((spot / strike).ln() + (r + 0.5 * sigma * sigma) * tau)
        / (sigma * tau.sqrt());
    let d2 = d1 - sigma * tau.sqrt();

    let normal = StatNormal::new(0.0, 1.0).unwrap();

    match option_type {
        OptionType::Call => {
            spot * normal.cdf(d1) - strike * (-r * tau).exp() * normal.cdf(d2)
        }
        OptionType::Put => {
            strike * (-r * tau).exp() * normal.cdf(-d2) - spot * normal.cdf(-d1)
        }
    }
}

/// Analytical Black-Scholes Greeks.
#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Compute analytical Greeks for a European option.
pub fn black_scholes_greeks(spot: f64, params: &BSParams, option_type: OptionType) -> Greeks {
    let BSParams {
        strike,
        risk_free_rate: r,
        volatility: sigma,
        maturity: tau,
    } = params;

    if *tau <= 0.0 {
        let delta = match option_type {
            OptionType::Call => if spot > *strike { 1.0 } else { 0.0 },
            OptionType::Put => if spot < *strike { -1.0 } else { 0.0 },
        };
        return Greeks {
            delta,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        };
    }

    let d1 = ((spot / strike).ln() + (r + 0.5 * sigma * sigma) * tau)
        / (sigma * tau.sqrt());
    let d2 = d1 - sigma * tau.sqrt();

    let normal = StatNormal::new(0.0, 1.0).unwrap();
    let n_d1 = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt(); // PDF

    let gamma = n_d1 / (spot * sigma * tau.sqrt());
    let vega = spot * n_d1 * tau.sqrt();

    match option_type {
        OptionType::Call => {
            let delta = normal.cdf(d1);
            let theta = -spot * n_d1 * sigma / (2.0 * tau.sqrt())
                - r * strike * (-r * tau).exp() * normal.cdf(d2);
            let rho = strike * tau * (-r * tau).exp() * normal.cdf(d2);
            Greeks { delta, gamma, theta, vega, rho }
        }
        OptionType::Put => {
            let delta = normal.cdf(d1) - 1.0;
            let theta = -spot * n_d1 * sigma / (2.0 * tau.sqrt())
                + r * strike * (-r * tau).exp() * normal.cdf(-d2);
            let rho = -strike * tau * (-r * tau).exp() * normal.cdf(-d2);
            Greeks { delta, gamma, theta, vega, rho }
        }
    }
}

// =============================================================================
// Simple Neural Network (from scratch, no ML framework dependency)
// =============================================================================

/// Activation function types.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
}

impl Activation {
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => x.max(0.0),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
        }
    }

    pub fn second_derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => {
                let t = x.tanh();
                -2.0 * t * (1.0 - t * t)
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s) * (1.0 - 2.0 * s)
            }
            Activation::ReLU => 0.0,
        }
    }
}

/// A dense (fully connected) layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Option<Activation>,
    // Gradients
    pub grad_weights: Array2<f64>,
    pub grad_biases: Array1<f64>,
    // Cache for forward pass
    pub pre_activation: Array1<f64>,
    pub post_activation: Array1<f64>,
    pub input_cache: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize, activation: Option<Activation>) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let normal = Normal::new(0.0, scale).unwrap();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.sample(normal)
        });
        let biases = Array1::zeros(output_dim);

        Self {
            grad_weights: Array2::zeros((input_dim, output_dim)),
            grad_biases: Array1::zeros(output_dim),
            pre_activation: Array1::zeros(output_dim),
            post_activation: Array1::zeros(output_dim),
            input_cache: Array1::zeros(input_dim),
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input_cache = input.clone();

        // z = W^T * x + b
        let z = input.dot(&self.weights) + &self.biases;
        self.pre_activation = z.clone();

        // a = activation(z)
        let output = match self.activation {
            Some(act) => z.mapv(|v| act.forward(v)),
            None => z,
        };
        self.post_activation = output.clone();

        output
    }
}

/// Physics-Informed Neural Network for Black-Scholes.
#[derive(Debug, Clone)]
pub struct PINN {
    pub layers: Vec<DenseLayer>,
    pub s_min: f64,
    pub s_max: f64,
    pub t_max: f64,
    pub learning_rate: f64,
}

impl PINN {
    /// Create a new PINN with the given architecture.
    ///
    /// # Arguments
    /// * `hidden_sizes` - Sizes of hidden layers (e.g., [64, 64, 64])
    /// * `s_max` - Maximum spot price for normalization
    /// * `t_max` - Maximum time (maturity)
    /// * `learning_rate` - Learning rate for gradient descent
    pub fn new(
        hidden_sizes: &[usize],
        s_max: f64,
        t_max: f64,
        learning_rate: f64,
    ) -> Self {
        let mut layers = Vec::new();

        // Input layer: 2 inputs (S_norm, t_norm)
        let mut prev_size = 2;

        for &hidden_size in hidden_sizes {
            layers.push(DenseLayer::new(prev_size, hidden_size, Some(Activation::Tanh)));
            prev_size = hidden_size;
        }

        // Output layer: 1 output (V), no activation
        layers.push(DenseLayer::new(prev_size, 1, None));

        Self {
            layers,
            s_min: 0.0,
            s_max,
            t_max,
            learning_rate,
        }
    }

    /// Normalize inputs to [0, 1].
    fn normalize(&self, s: f64, t: f64) -> (f64, f64) {
        let s_norm = (s - self.s_min) / (self.s_max - self.s_min);
        let t_norm = t / self.t_max;
        (s_norm, t_norm)
    }

    /// Forward pass: compute V(S, t).
    pub fn forward(&mut self, s: f64, t: f64) -> f64 {
        let (s_norm, t_norm) = self.normalize(s, t);
        let mut x = Array1::from_vec(vec![s_norm, t_norm]);

        for layer in &mut self.layers {
            x = layer.forward(&x);
        }

        // Scale output
        x[0] * self.s_max
    }

    /// Compute V and its derivatives dV/dS, d^2V/dS^2, dV/dt using finite differences.
    ///
    /// For a production PINN, you would use automatic differentiation.
    /// Here we use numerical differentiation for the Rust implementation.
    pub fn forward_with_derivatives(&mut self, s: f64, t: f64) -> (f64, f64, f64, f64) {
        let eps_s = s.max(1.0) * 1e-4;
        let eps_t = self.t_max * 1e-4;

        let v = self.forward(s, t);
        let v_sp = self.forward(s + eps_s, t);
        let v_sm = self.forward(s - eps_s, t);
        let v_tp = self.forward(s, t + eps_t);

        // dV/dS (central difference)
        let dv_ds = (v_sp - v_sm) / (2.0 * eps_s);

        // d^2V/dS^2 (central difference)
        let d2v_ds2 = (v_sp - 2.0 * v + v_sm) / (eps_s * eps_s);

        // dV/dt (forward difference)
        let dv_dt = (v_tp - v) / eps_t;

        (v, dv_ds, d2v_ds2, dv_dt)
    }

    /// Compute the Black-Scholes PDE residual at point (S, t).
    pub fn pde_residual(&mut self, s: f64, t: f64, sigma: f64, r: f64) -> f64 {
        let (v, dv_ds, d2v_ds2, dv_dt) = self.forward_with_derivatives(s, t);

        // PDE: dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0
        dv_dt + 0.5 * sigma * sigma * s * s * d2v_ds2 + r * s * dv_ds - r * v
    }

    /// Compute numerical gradient of loss with respect to all weights using finite differences.
    /// This is a simplified training approach for the Rust PINN.
    fn numerical_gradient(
        &mut self,
        s_pde: &[f64],
        t_pde: &[f64],
        s_bc: &[f64],
        t_bc: &[f64],
        v_bc: &[f64],
        s_ic: &[f64],
        v_ic: &[f64],
        sigma: f64,
        r: f64,
        t_max: f64,
        lam_pde: f64,
        lam_bc: f64,
        lam_ic: f64,
    ) -> f64 {
        let eps = 1e-5;

        // Compute current loss
        let loss = self.compute_loss(
            s_pde, t_pde, s_bc, t_bc, v_bc, s_ic, v_ic,
            sigma, r, t_max, lam_pde, lam_bc, lam_ic,
        );

        // Update weights using stochastic perturbation (SPSA-like)
        let mut rng = rand::thread_rng();

        for layer_idx in 0..self.layers.len() {
            let (rows, cols) = self.layers[layer_idx].weights.dim();

            // Weight updates (sample a subset for efficiency)
            let n_samples = (rows * cols).min(50);
            for _ in 0..n_samples {
                let i = rng.gen_range(0..rows);
                let j = rng.gen_range(0..cols);

                self.layers[layer_idx].weights[[i, j]] += eps;
                let loss_plus = self.compute_loss(
                    s_pde, t_pde, s_bc, t_bc, v_bc, s_ic, v_ic,
                    sigma, r, t_max, lam_pde, lam_bc, lam_ic,
                );
                self.layers[layer_idx].weights[[i, j]] -= eps;

                let grad = (loss_plus - loss) / eps;
                self.layers[layer_idx].weights[[i, j]] -= self.learning_rate * grad;
            }

            // Bias updates
            let bias_len = self.layers[layer_idx].biases.len();
            for j in 0..bias_len {
                self.layers[layer_idx].biases[j] += eps;
                let loss_plus = self.compute_loss(
                    s_pde, t_pde, s_bc, t_bc, v_bc, s_ic, v_ic,
                    sigma, r, t_max, lam_pde, lam_bc, lam_ic,
                );
                self.layers[layer_idx].biases[j] -= eps;

                let grad = (loss_plus - loss) / eps;
                self.layers[layer_idx].biases[j] -= self.learning_rate * grad;
            }
        }

        loss
    }

    /// Compute total loss.
    pub fn compute_loss(
        &mut self,
        s_pde: &[f64],
        t_pde: &[f64],
        s_bc: &[f64],
        t_bc: &[f64],
        v_bc: &[f64],
        s_ic: &[f64],
        v_ic: &[f64],
        sigma: f64,
        r: f64,
        t_max: f64,
        lam_pde: f64,
        lam_bc: f64,
        lam_ic: f64,
    ) -> f64 {
        // PDE loss
        let n_pde = s_pde.len() as f64;
        let mut loss_pde = 0.0;
        for i in 0..s_pde.len() {
            let res = self.pde_residual(s_pde[i], t_pde[i], sigma, r);
            loss_pde += res * res;
        }
        loss_pde /= n_pde;

        // Boundary condition loss
        let n_bc = s_bc.len() as f64;
        let mut loss_bc = 0.0;
        for i in 0..s_bc.len() {
            let v_pred = self.forward(s_bc[i], t_bc[i]);
            let diff = v_pred - v_bc[i];
            loss_bc += diff * diff;
        }
        loss_bc /= n_bc;

        // Terminal/initial condition loss
        let n_ic = s_ic.len() as f64;
        let mut loss_ic = 0.0;
        for i in 0..s_ic.len() {
            let v_pred = self.forward(s_ic[i], t_max);
            let diff = v_pred - v_ic[i];
            loss_ic += diff * diff;
        }
        loss_ic /= n_ic;

        lam_pde * loss_pde + lam_bc * loss_bc + lam_ic * loss_ic
    }

    /// Train the PINN.
    pub fn train(
        &mut self,
        params: &BSParams,
        epochs: usize,
        n_pde: usize,
        n_bc: usize,
        n_ic: usize,
        print_every: usize,
    ) -> Vec<f64> {
        let sigma = params.volatility;
        let r = params.risk_free_rate;
        let k = params.strike;
        let t_max = params.maturity;

        let mut rng = rand::thread_rng();
        let mut loss_history = Vec::new();

        println!("Training PINN for Black-Scholes");
        println!("  Strike: {}, r: {}, sigma: {}, T: {}", k, r, sigma, t_max);
        println!("  Epochs: {}, PDE points: {}, BC points: {}, IC points: {}",
                 epochs, n_pde, n_bc, n_ic);
        println!("{}", "=".repeat(70));

        for epoch in 0..epochs {
            // Sample collocation points
            let s_pde: Vec<f64> = (0..n_pde)
                .map(|_| rng.gen_range(0.01..self.s_max))
                .collect();
            let t_pde: Vec<f64> = (0..n_pde)
                .map(|_| rng.gen_range(0.0..(t_max - 0.001)))
                .collect();

            // Boundary points (S=0 and S=S_max)
            let mut s_bc = Vec::new();
            let mut t_bc_vec = Vec::new();
            let mut v_bc = Vec::new();

            for _ in 0..n_bc {
                let t_val = rng.gen_range(0.0..t_max);
                // Lower boundary: S ≈ 0, V(0,t) = 0 for call
                s_bc.push(0.01);
                t_bc_vec.push(t_val);
                v_bc.push(0.0);

                // Upper boundary: S = S_max, V(S_max, t) ≈ S_max - K*exp(-r*(T-t))
                let t_val2 = rng.gen_range(0.0..t_max);
                s_bc.push(self.s_max);
                t_bc_vec.push(t_val2);
                v_bc.push(self.s_max - k * (-r * (t_max - t_val2)).exp());
            }

            // Terminal condition: V(S, T) = max(S - K, 0)
            let s_ic: Vec<f64> = (0..n_ic)
                .map(|_| rng.gen_range(0.01..self.s_max))
                .collect();
            let v_ic: Vec<f64> = s_ic.iter()
                .map(|&s| (s - k).max(0.0))
                .collect();

            // Adaptive loss weights
            let progress = epoch as f64 / epochs as f64;
            let lam_bc = 50.0 * (1.0 - progress) + 5.0 * progress;
            let lam_ic = 50.0 * (1.0 - progress) + 5.0 * progress;

            // Training step
            let loss = self.numerical_gradient(
                &s_pde, &t_pde,
                &s_bc, &t_bc_vec, &v_bc,
                &s_ic, &v_ic,
                sigma, r, t_max,
                1.0, lam_bc, lam_ic,
            );

            loss_history.push(loss);

            if epoch % print_every == 0 || epoch == epochs - 1 {
                // Validation
                let mae = self.validate(params, OptionType::Call);
                println!(
                    "Epoch {:6}/{} | Loss: {:.6} | Val MAE: {:.4} | LR: {:.2e}",
                    epoch, epochs, loss, mae, self.learning_rate,
                );
            }
        }

        loss_history
    }

    /// Validate against analytical solution.
    pub fn validate(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        let test_spots = vec![60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0];
        let mut total_error = 0.0;
        let mut count = 0;

        for &s in &test_spots {
            let pinn_price = self.forward(s, 0.0);
            let bs_price = black_scholes_price(s, params, option_type);
            total_error += (pinn_price - bs_price).abs();
            count += 1;
        }

        total_error / count as f64
    }

    /// Print a validation table comparing PINN with analytical prices.
    pub fn print_validation_table(&mut self, params: &BSParams, option_type: OptionType) {
        let spots = vec![70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];

        println!("\n{}", "=".repeat(65));
        println!("  Validation: PINN vs Analytical Black-Scholes");
        println!("{}", "=".repeat(65));
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>10}",
            "S", "BS Price", "PINN", "Error", "Rel %"
        );
        println!("{}", "-".repeat(65));

        let mut total_error = 0.0;

        for &s in &spots {
            let bs_price = black_scholes_price(s, params, option_type);
            let pinn_price = self.forward(s, 0.0);
            let error = (pinn_price - bs_price).abs();
            let rel_error = if bs_price > 0.01 {
                error / bs_price * 100.0
            } else {
                0.0
            };

            total_error += error;

            println!(
                "{:8.1} {:10.4} {:10.4} {:10.4} {:9.2}%",
                s, bs_price, pinn_price, error, rel_error
            );
        }

        println!("{}", "-".repeat(65));
        println!(
            "Mean Absolute Error: {:.4}",
            total_error / spots.len() as f64
        );
    }

    /// Compute Greeks at a point using finite differences.
    pub fn compute_greeks(&mut self, s: f64, t: f64) -> Greeks {
        let (v, dv_ds, d2v_ds2, dv_dt) = self.forward_with_derivatives(s, t);

        Greeks {
            delta: dv_ds,
            gamma: d2v_ds2,
            theta: dv_dt,
            vega: 0.0, // Would need sigma as input for Vega
            rho: 0.0,  // Would need r as input for Rho
        }
    }

    /// Get the total number of parameters in the network.
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| {
            l.weights.len() + l.biases.len()
        }).sum()
    }
}

// =============================================================================
// Data Generation
// =============================================================================

/// Generate collocation points for PINN training.
pub fn generate_collocation_points(
    s_max: f64,
    t_max: f64,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();

    let s_points: Vec<f64> = (0..n_points)
        .map(|_| rng.gen_range(0.01..s_max))
        .collect();

    let t_points: Vec<f64> = (0..n_points)
        .map(|_| rng.gen_range(0.0..(t_max - 0.001)))
        .collect();

    (s_points, t_points)
}

// =============================================================================
// Bybit API
// =============================================================================

/// Bybit option ticker data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTicker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: Option<String>,
    #[serde(rename = "markPrice")]
    pub mark_price: Option<String>,
    #[serde(rename = "markIv")]
    pub mark_iv: Option<String>,
    #[serde(rename = "bid1Price")]
    pub bid1_price: Option<String>,
    #[serde(rename = "ask1Price")]
    pub ask1_price: Option<String>,
    #[serde(rename = "bid1Iv")]
    pub bid1_iv: Option<String>,
    #[serde(rename = "ask1Iv")]
    pub ask1_iv: Option<String>,
}

/// Bybit API response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitResult {
    pub list: Vec<BybitTicker>,
}

/// Bybit spot ticker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitSpotTicker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitSpotResult {
    pub list: Vec<BybitSpotTicker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitSpotResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitSpotResult,
}

/// Parsed option symbol info.
#[derive(Debug, Clone)]
pub struct ParsedOption {
    pub base_coin: String,
    pub expiry: String,
    pub strike: f64,
    pub option_type: OptionType,
}

/// Parse a Bybit option symbol like "BTC-28JUN24-70000-C".
pub fn parse_option_symbol(symbol: &str) -> Option<ParsedOption> {
    let parts: Vec<&str> = symbol.split('-').collect();
    if parts.len() != 4 {
        return None;
    }

    let strike = parts[2].parse::<f64>().ok()?;
    let option_type = match parts[3] {
        "C" => OptionType::Call,
        "P" => OptionType::Put,
        _ => return None,
    };

    Some(ParsedOption {
        base_coin: parts[0].to_string(),
        expiry: parts[1].to_string(),
        strike,
        option_type,
    })
}

/// Fetch options tickers from Bybit API.
pub async fn fetch_bybit_options(base_coin: &str) -> Result<Vec<BybitTicker>, anyhow::Error> {
    let client = reqwest::Client::new();
    let url = "https://api.bybit.com/v5/market/tickers";

    let resp = client
        .get(url)
        .query(&[("category", "option"), ("baseCoin", base_coin)])
        .send()
        .await?
        .json::<BybitResponse>()
        .await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    Ok(resp.result.list)
}

/// Fetch spot price from Bybit.
pub async fn fetch_bybit_spot(symbol: &str) -> Result<f64, anyhow::Error> {
    let client = reqwest::Client::new();
    let url = "https://api.bybit.com/v5/market/tickers";

    let resp = client
        .get(url)
        .query(&[("category", "spot"), ("symbol", symbol)])
        .send()
        .await?
        .json::<BybitSpotResponse>()
        .await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let price = resp
        .result
        .list
        .first()
        .and_then(|t| t.last_price.as_ref())
        .and_then(|p| p.parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("No price data"))?;

    Ok(price)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_black_scholes_call() {
        let params = BSParams {
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            maturity: 1.0,
        };

        let price = black_scholes_price(100.0, &params, OptionType::Call);
        // Known value for ATM call: approximately 10.45
        assert_abs_diff_eq!(price, 10.4506, epsilon = 0.01);
    }

    #[test]
    fn test_black_scholes_put() {
        let params = BSParams {
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            maturity: 1.0,
        };

        let call = black_scholes_price(100.0, &params, OptionType::Call);
        let put = black_scholes_price(100.0, &params, OptionType::Put);

        // Put-call parity: C - P = S - K * exp(-rT)
        let parity = call - put;
        let expected = 100.0 - 100.0 * (-0.05_f64).exp();
        assert_abs_diff_eq!(parity, expected, epsilon = 0.01);
    }

    #[test]
    fn test_greeks_delta_bounds() {
        let params = BSParams::default();
        let greeks = black_scholes_greeks(100.0, &params, OptionType::Call);

        // Call delta should be between 0 and 1
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);

        // ATM call delta should be around 0.5-0.65
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7);
    }

    #[test]
    fn test_greeks_gamma_positive() {
        let params = BSParams::default();
        let greeks = black_scholes_greeks(100.0, &params, OptionType::Call);
        assert!(greeks.gamma > 0.0);
    }

    #[test]
    fn test_pinn_creation() {
        let pinn = PINN::new(&[64, 64, 64], 200.0, 1.0, 0.001);
        assert_eq!(pinn.layers.len(), 4); // 3 hidden + 1 output
        assert!(pinn.num_parameters() > 0);
    }

    #[test]
    fn test_pinn_forward() {
        let mut pinn = PINN::new(&[32, 32], 200.0, 1.0, 0.001);
        let v = pinn.forward(100.0, 0.0);
        // Untrained, just check it returns a finite number
        assert!(v.is_finite());
    }

    #[test]
    fn test_parse_option_symbol() {
        let parsed = parse_option_symbol("BTC-28JUN24-70000-C").unwrap();
        assert_eq!(parsed.base_coin, "BTC");
        assert_eq!(parsed.strike, 70000.0);
        assert_eq!(parsed.option_type, OptionType::Call);
    }
}

# Chapter 141: PINN Black-Scholes Explained Simply

## Imagine You're Buying Concert Tickets...

Let's understand Physics-Informed Neural Networks for option pricing through a simple story.

---

## The Concert Ticket Problem

### What are options?

Imagine your favorite band announces a concert in 6 months. Tickets will cost $100 each. But you're not sure if you want to go yet.

A friend offers you a deal:

```
"Pay me $10 now, and I'll guarantee you the RIGHT to buy
a ticket at $100 anytime in the next 6 months."
```

That's an **option**! You pay a small amount now for the *right* (but not the obligation) to buy something later at a fixed price.

```
If ticket prices go UP to $150:
  - You use your option: buy at $100, save $50
  - Your profit: $50 - $10 (option cost) = $40

If ticket prices go DOWN to $60:
  - You DON'T use your option (why pay $100 when it costs $60?)
  - You lose: $10 (what you paid for the option)

If prices stay at $100:
  - You DON'T use your option (no benefit)
  - You lose: $10
```

---

## The Big Question: What's the Fair Price?

Here's the puzzle: **How much should that option cost?**

If the option is too cheap ($1), your friend loses money. If it's too expensive ($50), nobody would buy it.

Two famous mathematicians, **Fischer Black** and **Myron Scholes**, figured out a formula for this in 1973. They won a Nobel Prize for it!

### The Black-Scholes Formula (simplified)

Their insight was that the option price depends on:

```
1. Current ticket price:     $100
2. Guaranteed buy price:     $100 (the "strike")
3. Time until concert:       6 months
4. How wild prices swing:    The "volatility" (e.g., 20%)
5. Interest rate:            If you put money in the bank instead
```

They created a math equation (a PDE -- Partial Differential Equation) that says:

```
"The option price must change over time in a way that
perfectly balances risk -- no free lunch!"
```

---

## Enter: The Neural Network That Knows Physics

### What's a PINN?

A **Physics-Informed Neural Network** is like a student who has to pass TWO tests:

```
Test 1 (Data Test):
  "Does the answer match known facts?"
  Example: At the concert date, a $100 option on a $150 ticket
  is worth exactly $50.

Test 2 (Physics Test):
  "Does the answer follow the rules of physics/math?"
  Example: The option price must satisfy Black-Scholes equation
  at EVERY point in time and price.
```

### Normal Neural Networks vs PINNs

```
NORMAL Neural Network (learns from data only):
+----------------------------------------------------+
| Give me thousands of examples of correct prices,    |
| and I'll learn the pattern.                         |
|                                                      |
| Problem: Need LOTS of data. May learn wrong          |
| patterns. Can be wildly wrong where there's          |
| no training data.                                    |
+----------------------------------------------------+

PINN (learns from data AND rules):
+----------------------------------------------------+
| Give me the math rules (Black-Scholes equation),    |
| a few boundary facts, and I'll figure out the       |
| correct price EVERYWHERE.                            |
|                                                      |
| Advantage: Needs very little data. Always obeys     |
| the rules. Smart extrapolation.                     |
+----------------------------------------------------+
```

---

## The Cooking Analogy

### How does a PINN learn?

Imagine you're learning to bake a cake:

**Step 1: Know the Recipe (the PDE)**
```
Recipe says: "Every cake must rise evenly"
Translated: dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0

(Don't worry about the math -- it's just "the recipe")
```

**Step 2: Know the Edges (Boundary Conditions)**
```
"If you have no ingredients (S=0), you get no cake (V=0)"
"If you have infinite ingredients, the cake is obvious"
```

**Step 3: Know the Finish (Terminal Condition)**
```
"At the end (concert day), the cake is done:
 it's worth max(ticket price - $100, $0)"
```

**Now the PINN trains like this:**

```
Round 1: Make a cake (guess a price)
  Chef checks: "Does it follow the recipe?" -> No! Penalty: 50
  Chef checks: "Are the edges right?"       -> No! Penalty: 30
  Chef checks: "Is the finish right?"       -> No! Penalty: 40
  Total score: 120 (bad!)

Round 1000: Make another cake
  Chef checks: "Does it follow the recipe?" -> Almost! Penalty: 2
  Chef checks: "Are the edges right?"       -> Yes! Penalty: 0.1
  Chef checks: "Is the finish right?"       -> Yes! Penalty: 0.5
  Total score: 2.6 (great!)

Round 5000: Nearly perfect cake!
  Total score: 0.01 (the neural network has learned!)
```

---

## What About Crypto?

### Bitcoin Options on Bybit

The same idea works for Bitcoin! Bybit (a crypto exchange) offers options on BTC, ETH, and SOL.

The difference?

```
Stock options:
  - Price swings: Maybe 20% per year
  - Trading hours: 9:30 AM - 4:00 PM, weekdays
  - Been around since 1973

Crypto options:
  - Price swings: Maybe 80% per year (WILD!)
  - Trading hours: 24/7, never stops
  - Relatively new
```

The PINN works the same way for both -- you just tell it different numbers for volatility!

---

## The Greeks: Option Superpowers

When you train a PINN, you get bonus information for free! These are called "the Greeks":

```
Delta: "If the stock goes up $1, how much does the option change?"
  Like: "If it rains 1 inch more, how much fuller is the bucket?"

Gamma: "How fast does Delta change?"
  Like: "Is the bucket filling faster and faster?"

Theta: "How much value does the option lose each day?"
  Like: "The ice cream is melting... how fast?"

Vega: "If prices become more wild, what happens to the option?"
  Like: "If the weather becomes more unpredictable, should
         I bring a bigger umbrella?"
```

The PINN computes these automatically because of how neural networks work (something called "automatic differentiation" -- the same math trick that lets neural networks learn).

---

## Why Is This Cool?

```
Traditional Method (Finite Differences):
  Like drawing on graph paper -- works but slow
  and limited to that specific grid

Monte Carlo:
  Like flipping a million coins -- works but noisy

Black-Scholes Formula:
  Like a magic formula -- works but only for simple cases

PINN:
  Like a smart student who knows the rules
  - Works for ANY case (even exotic options)
  - Gives you Greeks for free
  - Fast after training
  - Can handle multiple stocks at once
```

---

## Try It Yourself!

### The Simplest Example

```python
# Create the "student" (neural network)
brain = NeuralNetwork(inputs=2, hidden=[128,128,128,128], output=1)

# The "rules" it must follow:
# Rule 1: Black-Scholes equation (the recipe)
# Rule 2: At expiry, call = max(S - K, 0) (the finish)
# Rule 3: At S=0, call = 0 (the edge)

# Train for 10,000 rounds
for round in range(10000):
    # Check: does it follow the recipe?
    recipe_error = check_black_scholes(brain)

    # Check: are the edges right?
    edge_error = check_boundaries(brain)

    # Check: is the finish right?
    finish_error = check_payoff(brain)

    # Total score (lower is better)
    total_error = recipe_error + edge_error + finish_error

    # Learn from mistakes
    brain.improve(total_error)

# Now the brain knows option prices!
price = brain.predict(stock_price=100, time=0)
# -> approximately $10.45 (matches the formula!)
```

---

## Summary

```
What we learned:
1. Options = the right to buy/sell at a fixed price
2. Black-Scholes = the math rules for fair option prices
3. PINN = a neural network that learns by following the rules
4. Greeks = bonus info about how option prices change
5. Works for stocks AND crypto!

The key insight:
  Instead of solving a hard math equation directly,
  we train a neural network that MUST obey the equation.
  It's like teaching a student the laws of physics
  instead of giving them an answer sheet.
```

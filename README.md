# 🃏 Easy21: Reinforcement Learning Assignment

This repository contains a comprehensive implementation of the **Easy21** card game assignment from **David Silver's Reinforcement Learning Course (UCL)**. The project demonstrates various reinforcement learning algorithms applied to a simplified Blackjack variant.

---

## 📌 Project Overview

**Easy21** is a simplified version of Blackjack designed for RL experimentation with the following rules:

### Game Rules
- **Infinite Deck:** Cards are sampled with replacement
- **Card Values:** 1 to 10 (uniformly distributed)
- **Card Colors:**
  - ⬛ **Black (2/3 probability):** Adds to the sum
  - 🟥 **Red (1/3 probability):** Subtracts from the sum
- **Initial Deal:** Both player and dealer start with one black card
- **Actions:** Hit (draw a card) or Stick (end turn)
- **Bust Condition:** Sum < 1 or sum > 21
- **Dealer Policy:** Hits until sum ≥ 17
- **Goal:** Achieve a sum closer to 21 than the dealer without going bust

### State Space
- **Dealer's showing card:** 1-10
- **Player's sum:** 1-21
- **Total states:** 10 × 21 = 210

### Reward Structure
- **+1:** Player wins (dealer busts or player sum > dealer sum)
- **-1:** Player loses (player busts or dealer sum > player sum)
- **0:** Draw or intermediate steps

---

## 🧠 Implemented Algorithms

### 1. **Monte Carlo Control** (Part 1-2)
- **On-Policy MC:** Epsilon-greedy policy with decaying exploration
- **Off-Policy MC:** Weighted importance sampling
- **Purpose:** Generate ground truth Q* for benchmarking
- **Episodes:** 1,000,000 for convergence

### 2. **Sarsa(λ)** (Part 3)
- **On-Policy TD Learning** with eligibility traces
- **Lambda range:** 0.0 to 1.0 (11 values)
- **Features:**
  - Epsilon-greedy exploration (N₀ = 100)
  - Accumulating eligibility traces
  - MSE tracking against Q*

### 3. **Q-Learning (Watkins' Q(λ))** (Part 3)
- **Off-Policy TD Learning** with trace cutoff
- **Comparison:** Sarsa vs Q-Learning performance
- **Trace handling:** Cuts traces on non-greedy actions

### 4. **Linear Function Approximation** (Part 4)
- **Feature Representation:** Coarse coding with overlapping cuboids
  - Dealer ranges: [1-4], [4-7], [7-10]
  - Player ranges: [1-6], [4-9], [7-12], [10-15], [13-18], [16-21]
  - Total features: 3 × 6 × 2 = 36
- **Learning:** Sarsa(λ) with linear weights
- **Hyperparameters:** α = 0.01, ε = 0.05

---

## 📊 Experiments & Visualizations

The implementation runs comprehensive experiments and generates the following plots:

### Generated Plots
1. **`ground_truth_Q_star.png`** - Optimal value function V*(s) from MC
2. **`mse_vs_lambda.png`** - Performance comparison across λ values
3. **`mc_vs_td_comparison.png`** - Learning curves: MC vs Sarsa(0) vs Sarsa(1)
4. **`td_comparison.png`** - Sarsa vs Q-Learning convergence
5. **`linear_vs_tabular.png`** - Function approximation vs tabular methods
6. **`linear_value_function.png`** - Learned value function with LFA

### Experiment Pipeline
1. **Ground Truth Generation:** MC control (1M episodes) → Q*
2. **Lambda Sweep:** Sarsa(λ) for λ ∈ [0, 1] with MSE tracking
3. **MC vs TD:** Compare Sarsa(0), Sarsa(1), and MC on-policy
4. **On/Off-Policy:** Sarsa vs Q-Learning with/without traces
5. **Function Approximation:** Linear Sarsa vs tabular methods

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Libraries:** 
  - `numpy` - Numerical computations
  - `matplotlib` - Visualization
  - `tqdm` - Progress tracking

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib tqdm
```

### Run All Experiments
```bash
python easy21.py
```

### Expected Runtime
- **MC Training (1M episodes):** ~2-5 minutes
- **Sarsa Experiments (11 λ values × 10K episodes):** ~3-5 minutes
- **Total:** ~10-15 minutes (depending on hardware)

### Output
- **Console:** Progress bars and experiment status
- **Files:** 6 PNG plots in the current directory

---

## 📈 Key Results

### Optimal Lambda
The experiments typically show that **λ ≈ 0.3-0.5** provides the best balance between:
- **Bias** (low λ → high bias, fast learning)
- **Variance** (high λ → high variance, slow learning)

### MC vs TD
- **Sarsa(0):** Faster initial learning, lower variance
- **Sarsa(1):** Equivalent to MC, higher variance
- **MC On-Policy:** Similar to Sarsa(1) but episodic updates

### Linear Approximation
- **Pros:** Generalization, compact representation
- **Cons:** Higher asymptotic error due to function approximation bias

---

## 📚 Assignment Reference

Based on **David Silver's RL Course** (UCL):
- [Course Website](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Assignment focuses on implementing and comparing model-free RL methods

---

## 📂 Project Structure

```
RL_Easy21/
├── easy21.py          # Complete implementation
├── README.md          # This file
└── *.png              # Generated plots (after running)
```

---

## 🎯 Learning Objectives

This assignment demonstrates:
1. ✅ Environment design (MDP formulation)
2. ✅ Monte Carlo methods (on/off-policy)
3. ✅ Temporal Difference learning (Sarsa, Q-Learning)
4. ✅ Eligibility traces (forward/backward view)
5. ✅ Function approximation (linear features)
6. ✅ Hyperparameter tuning (λ, α, ε, N₀)
7. ✅ Performance evaluation (MSE, learning curves)

# 🃏 Easy21: Reinforcement Learning Assignment

This repository contains the implementation of the **Easy21** card game assignment from **David Silver's Reinforcement Learning Course (UCL)**. The goal is to apply various reinforcement learning algorithms to solve a modified version of Blackjack.

## 📌 Project Overview

**Easy21** is a simplified version of Blackjack with the following rules:

- **Infinite Deck:** Cards are sampled with replacement.
- **Card Values:** 1 to 10 (uniformly distributed).
- **Card Colors:**
  - ⬛ **Black (2/3 prob):** Adds to the sum.
  - 🟥 **Red (1/3 prob):** Subtracts from the sum.
- **Goal:** Achieve a sum closer to 21 than the dealer without going bust (range 1-21).

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Libraries:** `numpy`, `matplotlib`, `tqdm`
- **Environment:** Custom implementation (Gym-like `step` and `reset` interface)

## 🚀 Installation & Usage

1. **Clone the repository**
2.


---
title: "Deep Q-Learning Notes"
author: "Nicole Ku, Elliot Kim"
---

# Deep Q-Learning
To learn about Deep Q-Learning, we must review the foundational concepts behind Q-Learning. 
## Agent and Environment

An **agent** interacts with an **environment** with the goal of maximizing cumulative reward.

- **Agent**: perceives the state, selects actions
- **Environment**: provides observations and rewards based on the agent’s actions

## Markov Decision Process (MDP)

A standard framework for modeling the environment in reinforcement learning.

- **States** \( s\)
- **Actions** \( a \)
- **Rewards** \( r \)
- **Transition probabilities** \(P(s'|s, a) \)

## Optimal Values

- **State Value Function**:

$$
V^*(s) = \max_\pi \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, \pi \right]
$$

- **Action-Value Function (Q-function)**:

$$
Q^*(s, a) = \max_\pi \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a, \pi \right]
$$

- **Optimal Policy**:

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

## Q-Learning

An off-policy TD control algorithm to learn the optimal Q-function:

### Update Rule

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
$$

### Algorithm

1. Initialize \( Q(s, a) \) arbitrarily
2. At each step, select \( a \) via ε-greedy policy
3. Observe reward \( r \), next state \( s' \)
4. Update \( Q(s, a) \) using the update rule

### Limitations

While Q-Learning works well for small state-action spaces, it struggles with scalability in high-dimensional environments. Let's consider the case of Atari Breakout.

A very simple version of the game might look like this:  
![simple_breakout](./simple_breakout.png)

The game is played on a small **12×12 grid**, with **yellow bricks** arranged in the top rows, a **white ball** that moves in all directions, and a **paddle** at the bottom that can move **left**, **right**, or **stay in place**. The goal is to break all the yellow blocks using the paddle in a **single attempt** (i.e., one life).

We can represent each state as a combination of:
- Ball position: 12 × 12 = **144**
- Ball direction: 8 directions (up, down, left, right, and 4 diagonals)
- Paddle position: 11 (on a 12-column grid, assuming paddle length 2)

**Total states** = 12 × 12 × 8 × 11 = **12,672**

The agent (paddle) can take **3 actions**: `move left`, `move right`, or `stay`.

**Total state-action pairs** = 12,672 × 3 = **38,016**

This is small enough for **Q-Learning to be feasible** using a tabular approach. However, this is a very **simplified** version of Atari Breakout. 
![real_breakout](./real_breakout.png)
The **standard version** uses a much larger **84×84 grid**, which increases:

- Ball positions: from `144` to `7,056`
- Paddle positions: from `11` to `83` (assuming paddle length 2)

**Total states (standard)** = 7,056 × 8 × 83 = **4,682,112**  
**Total state-action pairs (standard)** = 4,682,112 × 3 ≈ **14 million**

This makes it **infeasible to build a Q-table** due to memory and exploration constraints.

Further complexity arises from:
- Tracking the status of each brick (on/off)
- Score multipliers or bonuses
- Ball speed or power-ups

These factors further **inflate the state space**, making the use of lookup tables in high dimensional environments infeasible.

## Deep Q-Learning

When the state-action space is too large for a table, use a neural network \( Q_\theta(s, a) \).

### State Representation

Use features instead of raw states:

```text
f(s, a) = [position of Pacman, direction, ghost locations, etc.]

---
title: "Deep Q-Learning Notes"
author: "Nicole Ku, Elliot Kim"
---

# Deep Q-Learning

## Agent and Environment

An **agent** interacts with an **environment** with the goal of maximizing cumulative reward.

- **Agent**: perceives the state, selects actions
- **Environment**: provides observations and rewards based on the agent’s actions

## Markov Decision Process (MDP)

A standard framework for modeling the environment in reinforcement learning.

- **States** \( s \)
- **Actions** \( a \)
- **Rewards** \( r \)
- **Transition probabilities** \( P(s' \mid s, a) \)

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

## Deep Q-Learning

When the state-action space is too large for a table, use a neural network \( Q_\theta(s, a) \).

### State Representation

Use features instead of raw states:

```text
f(s, a) = [position of Pacman, direction, ghost locations, etc.]

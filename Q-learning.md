---
title: "Deep Q-Learning Notes"
author: "Nicole Ku, Elliot Kim"
---

# Introduction
To learn about Deep Q-Learning, we must review the foundational concepts behind Q-Learning. Deep Q-Learning is an extension of reinforcement learning where it combines Q-learning with deep neural networks to handle environments with large or continuous state spaces, where traditional tabular Q-learning fails.


## Intuition: From Q-Tables to Deep Q-Learning

Let’s assume we are training a Pacman agent to find the optimal path to win the game. Reinforcement learning (RL) is a suitable framework for this task because it allows the agent to learn from rewards associated with its actions.

//pacman diagram

In simpler environments with a limited number of discrete states, the standard Q-learning algorithm is effective. The process typically follows these steps as we learned in the previous note:

#### Step 1: Initialize the Q-table
#### Step 2: Update the Q-values
#### Step 3: Extract the Optimal Policy
These steps ensure to select the optimal action in each state by choosing the action with the highest Q-value in the Q-table. 

Got it! Now we want to use these steps to other games that are complicated 

### A scenario where Q-learning doesnt work:

Let's consider the case of Atari Breakout.

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



### What Exactly Differentiates Deep Q-Learning from Traditional Q-Learning?

- 

---

## Scaling to Large State Spaces

As you saw it in the Atari game, when the **state space becomes exponentially large**, maintaining a Q-table becomes infeasible due to the curse of dimensionality and computationali efficiency.

---

## Generalizing with Deep Q-Learning

To address this, we replace the Q-table with a **function approximator**—typically a **deep neural network**—that learns general patterns in the data.

### 1. State Representation
Represent each state as a vector of features (e.g., Pacman’s position, ghost locations, remaining pellets) instead of a table index.

### 2. Q-Network
Train a neural network $$Q(s, a; \theta)$$ that takes a state $$s$$ and action $$a$$, and predicts the expected reward. The parameters $\theta$ are learned during training.

### 3. Experience Replay
Store past transitions $$(s, a, r, s')$$ in a replay buffer and sample batches to train the network. This helps break correlations between consecutive updates.

### 4. Updating the Network
Instead of updating a Q-table entry, we minimize the loss between the predicted Q-value and the target:




### State Representation

Use features instead of raw states:


#### passes in Ql vs DQL

#### Representing Finding the Optimal Path to Convolutional Neural Networks

#### Training DQL

Code Setup: 



---
title: "Deep Q-Learning Notes"
author: "Nicole Ku, Elliot Kim"
---

# Introduction
To understand Deep Q-Learning, it's helpful to first grasp the basics of Q-Learning.

Imagine you're teaching a robot to play a game like Pacman. In traditional Q-Learning, the robot keeps a big table (called a Q-table) that tells it how good each move is from every situation it might see. It learns this table over time by playing the game, getting rewards, and updating the table based on what worked.

But this only works if the number of possible situations (states) is small and manageable. What if Pacman plays in a huge maze, or in a 3D world with endless possibilities? The table would be way too big — you'd never finish filling it in.

That’s where Deep Q-Learning comes in.

Instead of using a table, we use a deep neural network to approximate the table. The network takes the current game screen (state) as input and predicts the value of each possible move. It learns to do this by playing the game and adjusting its weights, much like how the Q-table gets updated.


## Intuition: From Q-Tables to Deep Q-Learning

Let’s assume we are training a Pacman agent to find the optimal path to win the game. Reinforcement learning (RL) is a suitable framework for this task because it allows the agent to learn from rewards associated with its actions.

![Pacman Game](./pacman.jpg)

### Q-Learning in the Context of Pacman

| **Q-learning Concept**        | **Pacman Analogy**                                                                 |
|------------------------------|------------------------------------------------------------------------------------|
| **State \( s \)**             | Pacman's current position and nearby ghosts/pellets on the maze                    |
| **Action \( a \)**            | Move up, down, left, or right                                                      |
| **Reward \( r \)**            | +10 for eating a pellet, +50 for eating a ghost, -500 for getting caught by ghost |
| **Q-table \( Q(s, a) \)**     | A big table where Pacman stores how good each move is from each situation          |
| **Learning update equation** | \( Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \) |
| **Goal**                      | Learn which moves (actions) lead to the most points (long-term rewards)           |
| **Exploration**               | Sometimes Pacman tries random moves (ϵ-greedy) to discover better paths            |
| **Convergence**               | After many games, Pacman learns to play optimally — avoiding ghosts and eating more pellets |

![Q-learning](./Qtable.png)

These steps ensure to select the optimal action in each state by choosing the action with the highest Q-value in the Q-table. 

Now we want to check if Q-learning works for problems with more complicated state space. 

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

---

### What Exactly Differentiates Deep Q-Learning from Traditional Q-Learning?

## Deep Q-Learning = Q-learning + Key Enhancements

1. Q function Approximation via Neural Network
In **traditional Q-learning**:

> `Q(s, a)` is stored in a table.

In **Deep Q-Learning**:

> `Q(s, a) ≈ Q_ϕ(s, a)`  
> where `Q_ϕ(s, a)` is predicted by a **neural network** with parameters `ϕ`.

This introduces the need for **gradient descent** to update the network parameters

2. Gradient Descent (loss based learning)
> Since you're predicting Q-values with a network, you define a **loss function**:
> You update the network weights using **gradient descent**:

3. Replay Buffer
> Unlike tabular Q-learning (which uses each experience only once), Deep Q-Learning uses a **replay > buffer** to store past transitions.

---

## Generalizing with Deep Q-Learning

As you saw it in the Atari game, when the **state space becomes exponentially large**, maintaining a Q-table becomes infeasible due to the curse of dimensionality and computationali efficiency. To address this, we replace the Q-table with a **function approximator**—typically a **deep neural network**—that learns general patterns in the data.

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



---
title: "Deep Q-Learning Notes"
author: "Nicole Ku, Elliot Kim"
---

# Introduction
To learn about Deep Q-Learning, we must review the foundational concepts behind Q-Learning. Deep Q-Learning is an extension of reinforcement learning where it combines Q-learning with deep neural networks to handle environments with large or continuous state spaces, where traditional tabular Q-learning fails.


## Intuition: From Q-Tables to Deep Q-Learning

Let’s assume we are training a Pacman agent to find the optimal path to win the game. Reinforcement learning (RL) is a suitable framework for this task because it allows the agent to learn from rewards associated with its actions.

In simpler environments with a limited number of discrete states, the standard Q-learning algorithm is effective. The process typically follows these steps:

#### Step 1: Initialize the Q-table
Define a table $$Q(s, a)$$ to store the estimated future rewards for each state-action pair.

#### Step 2: Update the Q-values
After each action, update the Q-table using the Bellman equation:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

#### Step 3: Extract the Optimal Policy
Once trained, select the optimal action in each state by choosing the action with the highest Q-value:

$$
\pi^*(s) = \arg\max_a Q(s, a)
$$



### A scenario where Q-learning doesnt work:

Imagine playing a video game:
1. You see pixels (high-dimensional state)
2. You need to decide what action to take to win (maximize reward)
3. You can’t store every possible screen (state) in a table

So instead, you use a neural network to learn general patterns from similar states
### What will be different for Deep Q-Learning?

- 

---

## Scaling to Large State Spaces

However, when the **state space becomes exponentially large** (e.g., due to increased grid size, ghost positions, or power-ups), maintaining a Q-table becomes infeasible due to the curse of dimensionality and computationali efficiency.

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





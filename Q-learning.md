---
title: "Deep Q-Learning Notes"
author: "Nicole Ku (nnk26), Elliot Kim (emk255)"
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

<img src="./Qtable.png" alt="Breakout" width="500" height="300"/>

  Q-learning teaches Pacman how to play optimally by letting him learn from experience. At the start, Pacman initializes a Q-table that keeps track of how good each move is in every situation. At each step, he chooses an action (like moving up, down, left, or right), performs it, and receives a reward such as +10 for eating a pellet or -500 for getting caught. Then, he updates the Q-table using the learning rule, which adjusts the value based on future rewards. Over many episodes, this loop of choosing actions, measuring rewards, and updating the Q-table allows Pacman to converge on the best strategy—maximizing long-term points by avoiding ghosts and targeting pellets efficiently.


Now we want to check if Q-learning works for problems with more complicated state space. 

## A scenario where Q-learning doesn't work:

Let's consider the case of Atari Breakout.

A very simple version of the game might look like this:  
<img src="./simple_breakout.png" alt="Breakout" width="500" height="300"/>

The game is played on a small **12×12 grid**, with **yellow bricks** arranged in the top rows, a **white ball** that moves in all directions, and a **paddle** at the bottom that can move **left**, **right**, or **stay in place**. The goal is to break all the yellow blocks using the paddle in a **single attempt** (i.e., one life).

We can represent each state as a combination of:
- Ball position: 12 × 12 = **144**
- Ball direction: 8 directions (up, down, left, right, and 4 diagonals)
- Paddle position: 11 (on a 12-column grid, assuming paddle length 2)

**Total states** = 12 × 12 × 8 × 11 = **12,672**

The agent (paddle) can take **3 actions**: `move left`, `move right`, or `stay`.

**Total state-action pairs** = 12,672 × 3 = **38,016**

This is small enough for **Q-Learning to be feasible** using a tabular approach. However, this is a very **simplified** version of Atari Breakout. 
<img src="./real_breakout.png" alt="Breakout" width="500" height="300"/>

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

### From Q-learning to Deep Q-learning

# Deep Q-Learning = Q-learning + Key Enhancements

1. Q function Approximation via Neural Network

- In **traditional Q-learning**: Q value is stored in a table.

- In **Deep Q-Learning**:Q value is predicted by a **neural network** with parameters `ϕ`.

This introduces the need for **gradient descent** to update the network parameters

2. Gradient Descent (loss based learning)

- Since you're predicting Q-values with a network, you define a **loss function**.
- You update the network weights using **gradient descent**.

3. Replay Buffer

- Unlike tabular Q-learning (which uses each experience only once), Deep Q-Learning uses a **replay buffer** to store past transitions.

> So far, implementing neural network and applying gradient descent with its loss function will sound familiar since they were introduced > > multiple times in deep learning. However, what exactly is replay buffer?
> In **tabular Q-learning**:
>  - The agent observes a transition \( (s, a, r, s') \) and immediately updates the Q-table.
>  - The experience is discarded after one use.
>
>  In **Deep Q-learning**:
>  - Each transition \( (s, a, r, s') \) is stored in a **replay buffer**.
>  - During training, the algorithm randomly samples mini-batches from this buffer to update the neural network.
>
>      With a replay buffer, Deep Q-Learning stores past experiences and randomly picks from them to train. This breaks the pattern of learning from one moment to the next, so the Q-values don't depend on each other too much. It also lets the model use the same data more than once and helps it learn more steadily.


---

Let’s now take a closer look at how Deep Q-Learning works in detail.

---

## Generalizing with Deep Q-Learning

As you saw it in the Atari Breakout example, when the **state space becomes exponentially large**, maintaining a Q-table becomes infeasible due to the curse of dimensionality and computationali efficiency. To address this, we replace the Q-table with a **function approximator**—typically a **deep neural network**—that learns general patterns in the data.

The model architecture 

### 1. State Representation
Represent each state as a vector of features instead of a table index. In Atari Breakout, this corresponds to a visual frame of the game. To convert this into a suitable input for a neural network:
- We take the **last 4 grayscale frames** of the game screen.
- Each frame is **84×84 pixels**, and stacking 4 of them forms a tensor of shape **4 × 84 × 84**.
- This allows the model to infer **motion and direction** (e.g., of the ball and paddle) and goes by the temporal limitations presented by inputting just one frame.
- A **Convolutional Neural Netowrk (CNN)** is used to apply filters to detect edges, shapes, and motion and reduce spatial dimensions while increasing semantic abstraction and output a flattened state representation vector.

The flattened feature vector is then passed to the Deep Q-Network to estiamte Q-values for each action.

### 2. Deep Q-Network
Deep Q-Learning utilizes two different Deep Q-Networks. 
#### 1. Main/Current/Policy Q-Network
Approximates $Q(s, a; \theta)$ that takes the flattened state vector$ and action $a$, and predicts the expected reward. Actions are determined through the epsilon-greedy policy, similar to that of traditional Q-Learning.

The parameters $\theta$ are learned during training using gradient descent. The network outputs Q-values for **all possible actions** in the action-space. In Atari Breakout, this corresponds to the paddle moving left, right, or staying.
#### 2. Target Q-Network
This network has the same architecture as the main Q-network, but with a separate set of weights denoted $\theta^{-}$. It is used to compute the **target Q-value** during training:

$y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$

- Unlike the main network, **the target network is not updated every step**.
- Instead, its parameters $\theta^{-}$ are **periodically copied** from the main network (e.g., every 10,000 steps).
- This helps prevent instability caused by having both predicted and target values depend on rapidly changing parameters.


### 3. Experience Replay
Stores past transitions $(s, a, r, s')$ in a replay buffer $D$ and sample random mini-batch of transitions from $D$. This helps break correlations between consecutive updates.

---
Since we learned each component of Deep Q-Learning model architecture and how each components function, let's apply it in Atari Breakout game. 

## Training DQL in Atari Breakout

#### 1. **Initialize components**
- A **Replay Buffer** `D` to store past transitions: `(s, a, r, s')`
- A **Main Q-Network** with parameters `θ` (randomly initialized)
- A **Target Q-Network** with parameters `θ⁻ = θ` (initially copied from main)

---

#### 2. **Preprocess game input**
- Capture **grayscale game frames** (each of size `84 × 84` pixels)
- **Stack the last 4 frames** to encode motion → input tensor: `4 × 84 × 84`
- Feed this to the **CNN-based Q-network** to get Q-values for actions:
  - Move Left
  - Move Right
  - Do Nothing

---

#### 3. **Action Selection (Exploration vs Exploitation)**
Use **ϵ-greedy policy**:
- With probability **ϵ**, select a **random** action (explore)
- Otherwise, select action with **highest predicted Q-value** (exploit)

Initially, ϵ is high (e.g., 1.0) and **decays over time** (e.g., to 0.1) to reduce exploration

---

#### 4. **Play and Store Experience**
- Execute chosen action `aₜ`, observe reward `rₜ` and next state `sₜ₊₁`
- Store transition `(sₜ, aₜ, rₜ, sₜ₊₁)` in replay buffer `D`

---

#### 5. **Sample Mini-Batch & Compute Targets**
- Randomly sample a batch of transitions from `D`
- For each:
  ```math
  yᵢ = 
  \begin{cases}
  rᵢ & \text{if } s'_i \text{ is terminal} \\
  rᵢ + γ \max_{a'} Q(s'_i, a'; θ⁻) & \text{otherwise}
  \end{cases}

#### 6. **Update Main Q-Network**

- Use gradient descent to **minimize squared error** between predicted and target Q-values:

  $$
  L(\theta) = \frac{1}{N} \sum_i \left( y_i - Q(s_i, a_i; \theta) \right)^2
  $$

- Backpropagate to update $\theta$

---

#### 7. **Periodically update Target Network**

- Every $C$ steps, copy weights from main to target network: $\theta^{-} \leftarrow \theta$

## Overestimation in DQN
Deep Q-Networks are not always perfect. They sometimes suffer from **overestimation bias**.
This happens because the target Q-value is computed using: $y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$. This indicates that both the **action selection** (`\max`) and **Q-value estimation** are done using the **same target network** $\theta^{-}$. If Q-values are noisy or imprecise, the max operation tends to selection action with overestimated values, leading to overly optimistic Q-value updates over time. The accumulation of such biases may destabilize learning over time.

### Double DQN
![Double DQN](./double_dqn.png)
To go about the issue of the overestimation bias, we can use a Double DQN, which decouples action selection and evaluation through two networks as shown in the figure above. Research shows that by doing so, it reduces over-estimation and stabilizes overall training.

#### Analogy: Archery Contest
Assume there are 100 archers who have the same skill level and are assigned to shoot at a target. However, there is always a gust of wind blowing 3mph. 

Method 1 (DQN-style) lets each archer fire one arrow, and chooses the archer whose arrow landed closest to the center of the target. Method 2 (Double DQN) selects the same archer that landed the closest, but makes the archer fire again to evaluate their accuracy. The problem with method 1 is that the archer might just have gotten lucky due to the uncontrollable wind, leading to the method potentially overestimating the archer's true skill. The second method goes about this problem by selecting the same archer, but makes sure to not "double-count" the wind noise in the second round. 


#### Final Overview

At the end, you might have felt overwhelmed by the amount of new information in this notes. We will make it clean for you on what are the key points.

Q learning --> Deep Q-learning --> Double Deep Q-learning
1. Higher overview

2. Algorithm Comparison


Major technicals that make Deep Q-learning better than Q-learning

replay buffer



#### Representing Finding the Optimal Path to Convolutional Neural Networks





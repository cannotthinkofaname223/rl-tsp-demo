# Reinforcement Learning for Traveling Salesman Problem (TSP)

This is a small experimental project that explores how reinforcement learning can be applied to solve a simple Traveling Salesman Problem (TSP).

The goal of this project is not to build a state-of-the-art solver, but to understand the basic workflow of reinforcement learning experiments.

---

## Problem

The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where a salesman must visit a set of cities exactly once and return to the starting point while minimizing the total travel distance.

In this project, cities are randomly generated in a 2D plane.

---

## Method

A simple policy network is trained using policy gradient reinforcement learning.

The model learns to generate a sequence of cities (a route) step by step. At each step, the policy outputs a probability distribution over the remaining cities.

The total route distance is used to compute the reward signal.

Training process:

1. Sample a route using the policy network
2. Compute the total route distance
3. Use the negative distance as reward
4. Update the policy using policy gradient

---

## Baseline

To evaluate the model, the learned policy is compared with a random route baseline.

Example comparison:

Random route distance: ~5.3  
Best RL route distance: ~3.4

This shows that the RL policy is able to learn shorter routes than random sampling.

---

## Visualization

The project includes several visualizations:

- Random route visualization
- Learned route visualization
- Training curve of route length
- Smoothed training curve using moving average

These visualizations help observe how the policy improves during training.

---

## Example Training Curve

The following figure shows the smoothed training curve of route distance during RL training.

(Insert training curve image here)

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## Purpose of the Project

This project was created as a learning exercise to better understand:

- Reinforcement learning training loops
- Policy gradient methods
- Experimental evaluation with baselines
- Visualization of training dynamics

Future improvements could include:

- Greedy baseline comparison
- Route evolution animation
- More advanced RL architectures# rl-tsp-demo

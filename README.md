# RL CartPole Lab

A compact experimental repository for studying continuous-control CartPole with Proximal Policy Optimization (PPO), built on top of [safe-control-gym](https://github.com/learnsyslab/safe-control-gym).

## Overview

This project uses the existing CartPole environment and PPO implementation from `safe-control-gym` as the learning backend. The goal is not to reimplement PPO from scratch, but to build an experimental RL laboratory around it and study how important Reinforcement Learning design choices affect the learned controller.

The stabilization target is

\[
s_{\mathrm{ref}} = [0,\;0,\;0,\;0]^T
\]

with state

\[
s = [x,\;\dot{x},\;\theta,\;\dot{\theta}]^T.
\]

The actor outputs a continuous action which is mapped to the physical cart force.

## Main additions in this repository

Compared with the basic CartPole RL example, this project adds an experiment-oriented interface and analysis workflow:

- CartPole stabilization around the zero target
- PPO retraining for different experiment configurations
- Custom RL CartPole Lab GUI
- Real-time 2D CartPole animation
- User-defined initial conditions
- Live plots of `x(t)`, `theta(t)`, applied force `F(t)`, and reward `r(t)`
- PPO raw action and physical force monitoring
- Reward-weight tuning
- Discount factor, GAE and PPO clipping controls
- Observation-noise tests
- External-disturbance tests
- Action-delay tests
- Automatic performance metrics
- Policy naming and model selection
- A/B experiment comparison
- Theory panel for selected RL equations

## Experiments

The report focuses on five experiments:

1. **Initial-State Generalization** — evaluate the same frozen policy from different initial states.
2. **Position Reward Weight** — study the effect of changing `q_x`.
3. **Control-Effort Penalty** — study the trade-off introduced by changing `R_u`.
4. **Discount Factor** — compare policies trained with different `gamma` values.
5. **Observation-Noise Robustness** — evaluate the same frozen policy with nominal and noisy observations.

## Project structure

```text
rl-cartpole-lab/
├── README.md
├── environment.yml
├── .gitignore
├── src/                  # Custom GUI / experiment code
├── configs/              # CartPole and PPO configuration files
├── scripts/              # Convenience launch scripts
├── models/               # Selected trained policies
├── results/              # Experiment tables / exported metrics
├── figures/              # Report and README figures
└── docs/                 # Project report
```

## Installation

A Conda environment is recommended.

```bash
conda env create -f environment.yml
conda activate rl-cartpole-lab
```

The project depends on `safe-control-gym`. If you prefer an editable local installation:

```bash
git clone https://github.com/learnsyslab/safe-control-gym.git
cd safe-control-gym
pip install -e .
```

Then clone this repository separately:

```bash
git clone https://github.com/omm87/rl-cartpole-lab.git
cd rl-cartpole-lab
```

## Configuration

`configs/cartpole/cartpole_stab_zero.yaml` contains the zero-reference CartPole task configuration used as the baseline task setup.

`configs/cartpole/ppo_cartpole.yaml` contains the PPO configuration based on the upstream `safe-control-gym` CartPole example.

Experiment-specific variants can be stored under `configs/experiments/` so that each result can be reproduced without changing the baseline files.

## Running

After the custom Python files are placed under `src/`, the intended workflow is:

```bash
bash scripts/run_gui.sh
```

and, for direct training:

```bash
bash scripts/train_policy.sh
```

The scripts are intentionally small wrappers so the actual configuration passed to `safe-control-gym` remains visible.

## RL / control interpretation

The project also connects RL design choices with familiar optimal-control concepts. The stage-cost structure

\[
J_t = q_x e_x^2 + q_{\dot{x}}\dot{x}^2 + q_\theta e_\theta^2 + q_{\dot{\theta}}\dot{\theta}^2 + R_u u^2
\]

is conceptually similar to the LQR objective

\[
J_{\mathrm{LQR}} = \sum_k (x_k^T Q x_k + u_k^T R u_k).
\]

The analogy is useful for interpreting reward-weight experiments, while the main distinction remains that PPO learns a nonlinear policy from sampled interaction rather than computing an analytic linear feedback gain from a model.

## Attribution

This repository uses the CartPole environment and PPO implementation provided by [safe-control-gym](https://github.com/learnsyslab/safe-control-gym).

The PPO algorithm itself was **not implemented from scratch** in this project. The work here focuses on configuration, retraining, visualization, experimental design, robustness tests, performance analysis, and interpretation.

Relevant upstream references:

- Z. Yuan et al., *Safe-Control-Gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics*, IEEE RA-L, 2022.
- J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.

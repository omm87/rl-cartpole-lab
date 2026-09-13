# RL CartPole Lab

A compact experimental laboratory for studying continuous-control CartPole with Proximal Policy Optimization (PPO), built on top of [safe-control-gym](https://github.com/learnsyslab/safe-control-gym).

## Overview

This project uses the existing CartPole environment and PPO implementation from `safe-control-gym` as the learning backend. PPO itself is **not reimplemented from scratch**. The purpose of this repository is to provide an experiment-oriented layer for training, evaluation, visualization, robustness testing, and control-oriented interpretation.

The stabilization target is

$$
\mathbf{s}_{\mathrm{ref}} = [0,\ 0,\ 0,\ 0]^T
$$

with state

$$
\mathbf{s} = [x,\ \dot{x},\ \theta,\ \dot{\theta}]^T.
$$

The PPO actor produces a continuous normalized action which is mapped to the physical cart force.

## Main additions

The custom RL lab adds:

- zero-reference CartPole stabilization
- PPO retraining from the GUI
- real-time 2D CartPole animation
- user-defined evaluation initial conditions
- live plots of `x(t)`, `theta(t)`, applied force `F(t)`, and reward `r(t)`
- PPO raw-action and physical-force monitoring
- reward-weight tuning
- Discount Factor, GAE, and PPO clipping controls
- Observation Noise tests
- External Disturbance tests
- Action Delay tests
- automatic performance metrics
- policy naming and model selection
- A/B experiment comparison
- a Theory panel for selected RL equations

## Experiments

The current report focuses on five experiments:

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
├── configs/
│   └── cartpole/
│       ├── cartpole_stab_zero.yaml
│       └── ppo_cartpole.yaml
├── scripts/
│   ├── install_miniforge.sh
│   ├── bootstrap.sh
│   ├── prepare_runtime_layout.sh
│   ├── check_install.sh
│   ├── run_gui.sh
│   └── train_baseline.sh
├── src/
│   ├── cartpole_rl_lab.py
│   ├── run_lab.py
│   └── rl_experiment.py
├── results/
├── models/                  # created locally when models are saved/trained
└── safe_control_gym/        # created locally by bootstrap.sh; ignored by Git
```

`src/cartpole_rl_lab.py` contains the main custom GUI implementation. `src/run_lab.py` is the standalone entry point that adapts the original path assumptions to this independent repository layout.

---

# Installation from a clean machine

The intended workflow is:

```text
Git + Miniforge
      ↓
clone rl-cartpole-lab
      ↓
bootstrap.sh
      ↓
Conda environment + safe-control-gym
      ↓
check_install.sh
      ↓
run_gui.sh
```

The instructions below assume a machine with no Python/Conda project environment already prepared.

## macOS

Supported targets are Apple Silicon (`arm64`) and Intel (`x86_64`) Macs.

### 1. Install Git command-line tools

Open Terminal and run:

```bash
xcode-select --install
```

If the command-line tools are already installed, macOS will tell you.

Check Git:

```bash
git --version
```

### 2. Clone this repository

```bash
git clone https://github.com/omm87/rl-cartpole-lab.git
cd rl-cartpole-lab
```

### 3. Install Miniforge

If `conda --version` already works, skip this step.

Otherwise run:

```bash
bash scripts/install_miniforge.sh
```

Then either reopen Terminal or activate Miniforge in the current shell:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

### 4. Bootstrap the project

```bash
bash scripts/bootstrap.sh
```

This command:

- creates/updates the `rl-cartpole-lab` Conda environment,
- installs the Python dependencies from `environment.yml`,
- clones `safe-control-gym` into `./safe_control_gym`,
- installs `safe-control-gym` in editable mode,
- prepares the runtime config layout expected by the GUI,
- runs an import and syntax smoke test.

### 5. Start the GUI

```bash
bash scripts/run_gui.sh
```

You do **not** need to activate the Conda environment manually when using this launcher; it runs the application through the correct environment.

---

## Linux

The GUI requires a desktop/display environment because it uses Tkinter and Matplotlib's `TkAgg` backend. A headless SSH-only machine will need X forwarding or another display solution.

The commands below use Ubuntu/Debian package names. On Fedora/Arch/etc., install the equivalent `git` and `curl` packages with the distribution package manager.

### 1. Install system prerequisites

```bash
sudo apt update
sudo apt install -y git curl
```

Check Git:

```bash
git --version
```

### 2. Clone this repository

```bash
git clone https://github.com/omm87/rl-cartpole-lab.git
cd rl-cartpole-lab
```

### 3. Install Miniforge

If `conda --version` already works, skip this step.

Otherwise:

```bash
bash scripts/install_miniforge.sh
```

Then either reopen the terminal or run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

### 4. Bootstrap the project

```bash
bash scripts/bootstrap.sh
```

The bootstrap script automatically detects the repository path and prepares the same project layout used on macOS.

### 5. Start the GUI

```bash
bash scripts/run_gui.sh
```

---

# Verify the installation

At any time, run:

```bash
bash scripts/check_install.sh
```

A correct installation ends with:

```text
Python imports: OK
Installation check: PASS
```

The check verifies the Conda environment, required files, the local `safe-control-gym` checkout, Python imports, generated runtime configs, and Python syntax.

## Manual environment activation

The launch scripts do not require manual activation, but if you want an interactive project shell:

```bash
conda activate rl-cartpole-lab
```

Then check:

```bash
python -c "import torch, pybullet, gymnasium, casadi, safe_control_gym; print('imports OK')"
```

---

# Running and training

## Start the RL Lab GUI

```bash
bash scripts/run_gui.sh
```

The launcher passes the baseline CartPole and PPO YAML files to the GUI automatically.

Training parameters such as Reward weights, Discount Factor `gamma`, GAE parameter `lambda`, and PPO clipping parameter `epsilon` can then be modified directly from the GUI before pressing **TRAIN NEW POLICY**.

Observation Noise, External Disturbance, and Action Delay are evaluation-side robustness settings in the current workflow.

## Train the baseline policy directly

The GUI is the main workflow, but a baseline training run can also be started without the GUI:

```bash
bash scripts/train_baseline.sh
```

The default output directory is:

```text
models/baseline_training/
```

A different output directory may be supplied as the first argument:

```bash
bash scripts/train_baseline.sh models/my_baseline
```

## Runtime path compatibility

The GUI was originally developed inside the upstream `safe-control-gym` example tree. In this repository that dependency is kept local and reproducible:

```text
rl-cartpole-lab/
├── src/
└── safe_control_gym/
    └── safe_control_gym/
        └── experiments/
            └── train_rl_controller.py
```

`bootstrap.sh` creates this local checkout automatically, while `run_lab.py` resolves the project root for the training code. `prepare_runtime_layout.sh` generates the small compatibility config directory used by the original GUI implementation. These generated files are ignored by Git.

---

# Configuration

`configs/cartpole/cartpole_stab_zero.yaml` contains the baseline CartPole task configuration with zero stabilization target.

`configs/cartpole/ppo_cartpole.yaml` contains the baseline PPO configuration derived from the upstream CartPole PPO example.

Experiment-specific training parameters are selected from the GUI rather than maintained as many duplicate YAML files.

## Baseline environment

The Conda environment is defined in `environment.yml` and uses Python 3.10. It includes the direct dependencies required by the custom code, including PyTorch, PyBullet, NumPy, SciPy, Matplotlib, PyYAML, Tk, Gymnasium, and CasADi. `safe-control-gym` is installed separately in editable mode by the bootstrap script because the GUI also invokes its training script directly.

---

# RL / control interpretation

The stage-cost structure used in the experiments is

$$
J_t = q_x e_x^2 + q_{\dot{x}}\dot{x}^2 + q_\theta e_\theta^2 + q_{\dot{\theta}}\dot{\theta}^2 + R_u u^2.
$$

This is conceptually similar to the LQR objective

$$
J_{\mathrm{LQR}} = \sum_k \left(x_k^T Q x_k + u_k^T R u_k\right).
$$

The analogy is useful for interpreting Reward-weight experiments. The important distinction is that PPO learns a nonlinear Policy from sampled interaction, whereas LQR computes a linear feedback gain from a model under linear-system and quadratic-cost assumptions.

---

# Troubleshooting

If the GUI reports that it cannot find `train_rl_controller.py`, run:

```bash
bash scripts/bootstrap.sh
```

If Python imports fail, run:

```bash
bash scripts/check_install.sh
```

If `conda` is not found after installing Miniforge, reopen the terminal or run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

On Linux, if Tkinter cannot open a window, make sure the machine has a graphical desktop/display available.

---

# Reproducibility note

The repository contains the custom code, baseline configuration, installation scripts, and experiment results. Trained PPO checkpoints may be added separately under `models/` when a specific frozen Policy needs to be reproduced exactly.

For stochastic RL training, reproducing the same configuration and seed improves repeatability but does not imply bit-for-bit identical learning trajectories across different hardware/software stacks.

## Attribution

This repository uses the CartPole environment and PPO implementation provided by [safe-control-gym](https://github.com/learnsyslab/safe-control-gym).

The PPO algorithm itself was **not implemented from scratch** in this project. The custom work focuses on configuration, retraining, visualization, experimental design, robustness tests, performance analysis, and interpretation.

Relevant references:

- Z. Yuan et al., *Safe-Control-Gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics*, IEEE Robotics and Automation Letters, 2022.
- J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.

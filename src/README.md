# Source code

This directory contains the custom experimental layer built on top of `safe-control-gym`.

- `cartpole_rl_lab.py` — main GUI implementation for PPO training, evaluation, real-time CartPole animation, plots, metrics, robustness tests, and policy comparison.
- `run_lab.py` — standalone entry point used by `scripts/run_gui.sh`. It adapts the original path assumptions so the GUI can run from this independent repository.
- `rl_experiment.py` — lower-level RL experiment/evaluation helper retained from the safe-control-gym workflow.

The PPO algorithm itself remains provided by `safe-control-gym`.

Generated runtime directories such as `config_overrides/`, `lab_configs/`, `lab_runs/`, and `temp_rl_lab_eval/` are created locally by the setup/GUI workflow and are ignored by Git.

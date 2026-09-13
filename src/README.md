# Source code

Place the custom experiment code in this directory.

Planned files:

- `cartpole_rl_lab.py` — main GUI for training, evaluation, real-time animation, plots, metrics, robustness tests, and policy comparison.
- `cartpole_gui.py` — earlier/simpler evaluation GUI, if you want to keep it for reference.
- `rl_experiment.py` — direct training/evaluation experiment entry point.

The PPO implementation itself remains provided by `safe-control-gym`; these files are the custom experimental layer built around it.

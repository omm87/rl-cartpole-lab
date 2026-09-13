# Trained policies

This directory is intended for selected PPO checkpoints that should be kept with the project.

The GUI normally writes new training runs under `src/lab_runs/`. The standalone baseline training helper writes to `models/baseline_training/` by default.

Large or temporary checkpoints do not need to be committed. Add only the frozen policies required to reproduce a specific reported result.

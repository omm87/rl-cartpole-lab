#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rl-cartpole-lab"
SAFE_CONTROL_GYM_DIR="$ROOT_DIR/safe_control_gym"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

fail() {
  echo "ERROR: $1"
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is not available"
command -v conda >/dev/null 2>&1 || fail "conda is not available"

conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" \
  || fail "Conda environment '$ENV_NAME' does not exist"

[ -f "$ROOT_DIR/configs/cartpole/cartpole_stab_zero.yaml" ] \
  || fail "baseline task config is missing"
[ -f "$ROOT_DIR/configs/cartpole/ppo_cartpole.yaml" ] \
  || fail "baseline PPO config is missing"
[ -f "$ROOT_DIR/src/cartpole_rl_lab.py" ] \
  || fail "main GUI implementation is missing"
[ -f "$ROOT_DIR/src/run_lab.py" ] \
  || fail "standalone GUI entry point is missing"
[ -f "$SAFE_CONTROL_GYM_DIR/safe_control_gym/experiments/train_rl_controller.py" ] \
  || fail "safe-control-gym training script is missing; run bootstrap.sh"

bash "$ROOT_DIR/scripts/prepare_runtime_layout.sh" >/dev/null

[ -f "$ROOT_DIR/src/config_overrides/cartpole/cartpole_stab.yaml" ] \
  || fail "runtime CartPole config was not generated"
[ -f "$ROOT_DIR/src/config_overrides/cartpole/ppo_cartpole.yaml" ] \
  || fail "runtime PPO config was not generated"

echo "Checking Python imports..."
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import casadi
import gymnasium
import matplotlib
import numpy
import pybullet
import torch
import tkinter
import yaml
import safe_control_gym
print("Python imports: OK")
print("PyTorch:", torch.__version__)
PY

echo "Checking Python syntax..."
conda run --no-capture-output -n "$ENV_NAME" \
  python -m py_compile \
  "$ROOT_DIR/src/cartpole_rl_lab.py" \
  "$ROOT_DIR/src/run_lab.py" \
  "$ROOT_DIR/src/rl_experiment.py"

echo "Installation check: PASS"

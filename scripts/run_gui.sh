#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rl-cartpole-lab"

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

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Run: bash scripts/install_miniforge.sh"
  exit 1
fi

if [ ! -f "$ROOT_DIR/safe_control_gym/safe_control_gym/experiments/train_rl_controller.py" ]; then
  echo "safe-control-gym dependency is not prepared."
  echo "Run: bash scripts/bootstrap.sh"
  exit 1
fi

bash "$ROOT_DIR/scripts/prepare_runtime_layout.sh" >/dev/null

exec conda run --no-capture-output -n "$ENV_NAME" \
  python "$ROOT_DIR/src/run_lab.py" \
  --algo ppo \
  --task cartpole \
  --overrides \
    "$ROOT_DIR/configs/cartpole/cartpole_stab_zero.yaml" \
    "$ROOT_DIR/configs/cartpole/ppo_cartpole.yaml"

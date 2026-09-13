#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rl-cartpole-lab"
SAFE_CONTROL_GYM_DIR="$ROOT_DIR/safe_control_gym"
OUTPUT_DIR="${1:-$ROOT_DIR/models/baseline_training}"

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
  echo "Conda was not found. Run: bash scripts/bootstrap.sh"
  exit 1
fi

TRAIN_SCRIPT="$SAFE_CONTROL_GYM_DIR/safe_control_gym/experiments/train_rl_controller.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "safe-control-gym dependency is not prepared. Run: bash scripts/bootstrap.sh"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "Training baseline PPO policy"
echo "Output directory: $OUTPUT_DIR"

exec conda run --no-capture-output -n "$ENV_NAME" \
  python -u "$TRAIN_SCRIPT" \
  --algo ppo \
  --task cartpole \
  --output_dir "$OUTPUT_DIR" \
  --overrides \
    "$ROOT_DIR/configs/cartpole/cartpole_stab_zero.yaml" \
    "$ROOT_DIR/configs/cartpole/ppo_cartpole.yaml"

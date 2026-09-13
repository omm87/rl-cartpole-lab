#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE_CONTROL_GYM_DIR="$ROOT_DIR/safe_control_gym"
ENV_NAME="rl-cartpole-lab"

# Make conda available in non-interactive shells when installed by Miniforge.
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
  echo "Conda/Miniforge was not found."
  echo "Run: bash scripts/install_miniforge.sh"
  echo "Then reopen the terminal or source Miniforge and run this script again."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Git was not found. Install Git first and rerun bootstrap.sh."
  exit 1
fi

echo "==> Creating/updating Conda environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ROOT_DIR/environment.yml" --prune
else
  conda env create -f "$ROOT_DIR/environment.yml"
fi

echo "==> Preparing repository-local safe-control-gym dependency"
if [ -d "$SAFE_CONTROL_GYM_DIR/.git" ]; then
  echo "safe-control-gym already cloned at $SAFE_CONTROL_GYM_DIR"
else
  if [ -e "$SAFE_CONTROL_GYM_DIR" ]; then
    echo "Path exists but is not a Git checkout: $SAFE_CONTROL_GYM_DIR"
    echo "Remove or rename it, then rerun bootstrap.sh."
    exit 1
  fi
  git clone https://github.com/learnsyslab/safe-control-gym.git "$SAFE_CONTROL_GYM_DIR"
fi

echo "==> Installing safe-control-gym in editable mode"
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install -e "$SAFE_CONTROL_GYM_DIR"

echo "==> Preparing runtime config layout"
bash "$ROOT_DIR/scripts/prepare_runtime_layout.sh"

echo "==> Running installation checks"
bash "$ROOT_DIR/scripts/check_install.sh"

echo
echo "Bootstrap completed successfully."
echo "Start the GUI with:"
echo "  bash scripts/run_gui.sh"

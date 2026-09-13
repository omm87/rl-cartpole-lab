#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
RUNTIME_CONFIG_DIR="$SRC_DIR/config_overrides/cartpole"

mkdir -p "$RUNTIME_CONFIG_DIR"
mkdir -p "$SRC_DIR/lab_runs" "$SRC_DIR/lab_configs" "$SRC_DIR/temp_rl_lab_eval"

cp "$ROOT_DIR/configs/cartpole/cartpole_stab_zero.yaml" \
   "$RUNTIME_CONFIG_DIR/cartpole_stab.yaml"

cp "$ROOT_DIR/configs/cartpole/ppo_cartpole.yaml" \
   "$RUNTIME_CONFIG_DIR/ppo_cartpole.yaml"

printf 'Runtime layout prepared.\n'
printf '  Task config: %s\n' "$RUNTIME_CONFIG_DIR/cartpole_stab.yaml"
printf '  PPO config:  %s\n' "$RUNTIME_CONFIG_DIR/ppo_cartpole.yaml"

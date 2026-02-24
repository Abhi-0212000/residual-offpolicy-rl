#!/usr/bin/env bash
# ============================================================================
# docker-run.sh – Quick start script for running on a remote GPU server
# ============================================================================
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit installed
#   - Your .env file with WANDB_API_KEY and HF_TOKEN
#
# Usage:
#   ./docker-run.sh                          # interactive bash
#   ./docker-run.sh train_bc                 # run BC training
#   ./docker-run.sh train_rl                 # run residual RL training
#   ./docker-run.sh custom "python ..."      # run any command
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check .env exists
if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found!"
    echo "  cp .env.example .env"
    echo "  # Then fill in WANDB_API_KEY and HF_TOKEN"
    exit 1
fi

# Ensure output directories exist on host
mkdir -p docker-volumes/outputs docker-volumes/runs docker-volumes/wandb

MODE="${1:-bash}"

case "$MODE" in
    bash)
        echo "Starting interactive shell..."
        docker compose run --rm train /bin/bash
        ;;
    train_bc)
        echo "Starting BC training..."
        docker compose run --rm train python resfit/lerobot/scripts/train_bc_dexmg.py \
            --dataset ankile/dexmg-two-arm-coffee \
            --policy act \
            --steps 200000 \
            --batch_size 128 \
            --wandb_project dexmg-bc \
            --eval_env TwoArmCoffee \
            --rollout_freq 5000 \
            --eval_video_key observation.images.frontview \
            --eval_render_size 224 \
            --eval_num_envs 4 \
            --eval_num_episodes 20 \
            --wandb_enable
        ;;
    train_rl)
        echo "Starting Residual RL training..."
        docker compose run --rm train python resfit/rl_finetuning/scripts/train_residual_td3.py \
            --env TwoArmCoffee \
            --wandb_enable
        ;;
    custom)
        shift
        echo "Running: $*"
        docker compose run --rm train "$@"
        ;;
    *)
        echo "Usage: $0 {bash|train_bc|train_rl|custom \"command\"}"
        exit 1
        ;;
esac

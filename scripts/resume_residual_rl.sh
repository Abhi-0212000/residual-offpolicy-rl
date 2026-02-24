#!/usr/bin/env bash
# ============================================================================
# resume_residual_rl.sh — Resume a crashed/stopped Residual RL run
# ============================================================================
# Resumes training from a saved checkpoint. Useful when:
#   - Training was interrupted (OOM, server restart, timeout)
#   - You want to extend a run beyond its original total_timesteps
#   - You want to continue with different hyperparameters (careful!)
#
# Prerequisites:
#   - A previous run directory with models/latest/ or models/policy_step_N/
#   - The WandB run ID (for continuing the same WandB run)
#
# Usage:
#   # Edit RESUME_CKPT and WANDB_CONTINUE_ID below, then:
#   bash scripts/resume_residual_rl.sh
#
#   # From host:
#   docker compose run --rm --name qte9489-resfit-train train bash scripts/resume_residual_rl.sh
#
# Finding the values:
#   RESUME_CKPT: ls $CACHE_DIR/run_*/models/  → pick "latest" or "policy_step_N"
#   WANDB_CONTINUE_ID: WandB dashboard → run page → URL has the 8-char ID
# ============================================================================
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# REQUIRED — You MUST set these for each resume
# ══════════════════════════════════════════════════════════════════════════════

# ── Checkpoint to resume from ────────────────────────────────────────────────
# Path to the checkpoint directory (relative to CACHE_DIR or absolute)
# Examples:
#   "run_2026-02-22_.../models/latest"       ← most recent auto-save
#   "run_2026-02-22_.../models/best"          ← best success rate so far
#   "run_2026-02-22_.../models/policy_step_50000"  ← specific step
RESUME_CKPT=""   # ← FILL THIS IN

# ── WandB run ID to continue ────────────────────────────────────────────────
# 8-character ID from WandB (e.g., "1wrldnus")
# Find it: WandB dashboard → your run → URL: wandb.ai/.../runs/<THIS_ID>
# Leave empty to start a NEW WandB run (old one will be marked as crashed)
WANDB_CONTINUE_ID=""   # ← FILL THIS IN

# ── Seed ─────────────────────────────────────────────────────────────────────
# MUST match the original run's seed for reproducibility
# Find it: WandB config tab → "seed", or in the run directory name
SEED=""   # ← FILL THIS IN (e.g., 1987747100)

# ══════════════════════════════════════════════════════════════════════════════
# Same hyperparameters as original run (should match for consistency)
# ══════════════════════════════════════════════════════════════════════════════

# ── Base BC Policy (must match original run) ─────────────────────────────────
BASE_WANDB_ID="dexmg-bc/zp7niccu"
BASE_WT_TYPE="best"

# ── Hydra Config ─────────────────────────────────────────────────────────────
CONFIG_NAME="residual_td3_coffee_config"

# ── Hyperparameters (keep same as original unless intentionally changing) ────
TOTAL_TIMESTEPS=500000                     # Can INCREASE to extend training
OFFLINE_EPISODES=100

# ── Evaluation & Checkpointing ───────────────────────────────────────────────
EVAL_NUM_ENVS=4
HEADLESS="true"
SAVE_FREQ=10000
NO_CLEANUP="true"
DEBUG="false"

# ── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT="dexmg-coffee"

# ============================================================================
# Validation
# ============================================================================
if [[ -z "${RESUME_CKPT}" ]]; then
    echo "ERROR: RESUME_CKPT is not set!"
    echo "  Set it to the checkpoint path, e.g.:"
    echo "  RESUME_CKPT=\"run_2026-02-22_.../models/latest\""
    echo ""
    echo "Available runs:"
    ls -d "${CACHE_DIR:-.}"/run_* 2>/dev/null || echo "  (none found in CACHE_DIR=${CACHE_DIR:-.})"
    exit 1
fi

if [[ -z "${SEED}" ]]; then
    echo "ERROR: SEED is not set!"
    echo "  Must match the original run's seed."
    echo "  Check the run directory name or WandB config."
    exit 1
fi

if [[ -z "${WANDB_CONTINUE_ID}" ]]; then
    echo "WARNING: WANDB_CONTINUE_ID is not set."
    echo "  Training will start a NEW WandB run instead of continuing the old one."
    echo "  Press Ctrl+C in 5s to abort, or wait to continue..."
    sleep 5
fi

# ============================================================================
# Run training (resume)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESUMING Residual RL Training                              "
echo "║  Checkpoint: ${RESUME_CKPT}                                 "
echo "║  WandB continue: ${WANDB_CONTINUE_ID:-NEW RUN}             "
echo "║  Seed: ${SEED}                                              "
echo "╚══════════════════════════════════════════════════════════════╝"

# Build the command
CMD=(
    python resfit/rl_finetuning/scripts/train_residual_td3.py
    --config-name="${CONFIG_NAME}"
    base_policy.wandb_id="${BASE_WANDB_ID}"
    base_policy.wt_type="${BASE_WT_TYPE}"
    algo.total_timesteps="${TOTAL_TIMESTEPS}"
    offline_data.num_episodes="${OFFLINE_EPISODES}"
    seed="${SEED}"
    resume_ckpt="${RESUME_CKPT}"
    headless="${HEADLESS}"
    eval_num_envs="${EVAL_NUM_ENVS}"
    no_cleanup="${NO_CLEANUP}"
    save_freq="${SAVE_FREQ}"
    debug="${DEBUG}"
    wandb.project="${WANDB_PROJECT}"
)

# Add WandB continue ID if set
if [[ -n "${WANDB_CONTINUE_ID}" ]]; then
    CMD+=(wandb.continue_run_id="${WANDB_CONTINUE_ID}")
fi

# Execute
"${CMD[@]}"

echo ""
echo "✓ Resumed training complete."

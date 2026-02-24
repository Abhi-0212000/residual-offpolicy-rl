#!/usr/bin/env bash
# ============================================================================
# train_bc_act.sh — Train a BC (Behavior Cloning) policy using ACT
# ============================================================================
# Trains a base policy from offline demonstrations. This must be done BEFORE
# residual RL training, as the RL policy uses this as its frozen base.
#
# Usage:
#   # Inside container:
#   bash scripts/train_bc_act.sh
#
#   # From host (with docker compose):
#   docker compose run --rm --name qte9489-resfit-train train bash scripts/train_bc_act.sh
#
# After training:
#   - Best checkpoint: $CACHE_DIR/bc_run_<timestamp>/best/
#   - WandB run ID: check WandB dashboard → use as base_policy.wandb_id in RL
#   - Put "wandb_project/run_id" into residual_td3.py config
# ============================================================================
set -euo pipefail

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET="ankile/dexmg-two-arm-coffee"    # HuggingFace dataset ID
                                          # Other options: ankile/dexmg-two-arm-transport, etc.

# ── Policy ───────────────────────────────────────────────────────────────────
POLICY="act"                              # Policy architecture: "act" (Action Chunking Transformer)
                                          # This is the only one tested for DexMG tasks

# ── Training ─────────────────────────────────────────────────────────────────
STEPS=200000                              # Total training steps (200K is standard for coffee)
BATCH_SIZE=128                            # Batch size. 128 fits 16GB GPU. Try 256 on 48GB
                                          # Larger = faster convergence but more VRAM

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_ENV="TwoArmCoffee"                  # Robosuite environment for rollout evaluation
                                          # Must match the dataset task
ROLLOUT_FREQ=5000                         # Evaluate every N steps (5K = good balance)
                                          # Lower = more frequent eval but slower training
EVAL_NUM_ENVS=4                           # Parallel eval environments. More = faster eval
                                          # 4 is safe on 16GB, try 8-16 on 48GB
EVAL_NUM_EPISODES=100                     # Episodes per evaluation. 100 = statistically robust
                                          # 20 for quick sanity checks, 100 for real runs
EVAL_VIDEO_KEY="observation.images.frontview"   # Camera view for recorded eval videos
EVAL_RENDER_SIZE=224                      # Resolution of eval video frames (pixels)

# ── WandB Logging ────────────────────────────────────────────────────────────
WANDB_PROJECT="dexmg-bc"                 # WandB project name (group BC runs together)
WANDB_ENABLE="--wandb_enable"            # Set to "" to disable WandB logging

# ── Cleanup ──────────────────────────────────────────────────────────────────
NO_CLEANUP="--no_cleanup"                 # Keep all checkpoint files after training
                                          # Remove this flag to auto-delete intermediate checkpoints

# ============================================================================
# Run training
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BC Policy Training — ACT on ${EVAL_ENV}                    "
echo "║  Steps: ${STEPS}  |  Batch: ${BATCH_SIZE}  |  Eval envs: ${EVAL_NUM_ENVS}"
echo "║  WandB: ${WANDB_PROJECT}                                    "
echo "╚══════════════════════════════════════════════════════════════╝"

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset "${DATASET}" \
    --policy "${POLICY}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --eval_env "${EVAL_ENV}" \
    --rollout_freq "${ROLLOUT_FREQ}" \
    --eval_video_key "${EVAL_VIDEO_KEY}" \
    --eval_render_size "${EVAL_RENDER_SIZE}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_num_episodes "${EVAL_NUM_EPISODES}" \
    ${WANDB_ENABLE} \
    ${NO_CLEANUP}

echo ""
echo "✓ BC training complete. Check WandB for the run ID."
echo "  Use it as: base_policy.wandb_id=${WANDB_PROJECT}/<run_id>"

#!/usr/bin/env bash
# ============================================================================
# train_residual_rl.sh — Residual RL fine-tuning with TD3
# ============================================================================
# Fine-tunes a frozen BC base policy with a small residual action correction
# using TD3 (Twin Delayed DDPG) off-policy RL.
#
# Prerequisites:
#   - A trained BC policy (from train_bc_act.sh) uploaded to WandB
#   - Set BASE_WANDB_ID below to "project/run_id" from your BC run
#
# Usage:
#   # Inside container:
#   bash scripts/train_residual_rl.sh
#
#   # From host:
#   docker compose run --rm --name qte9489-resfit-train train bash scripts/train_residual_rl.sh
#
# Output:
#   - Run directory: $CACHE_DIR/run_<timestamp>_resfit__<params>/
#     ├── models/         (checkpoints: best/, latest/, policy_step_N/, final/)
#     └── outputs/        (eval videos, Q-value plots)
# ============================================================================
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE THESE — most important parameters
# ══════════════════════════════════════════════════════════════════════════════

# ── Base BC Policy ───────────────────────────────────────────────────────────
BASE_WANDB_ID="dexmg-bc/zp7niccu"        # WandB "project/run_id" of your trained BC policy
                                           # Find this in WandB dashboard → BC training run
BASE_WT_TYPE="best"                        # Which checkpoint: "best" (highest success rate)
                                           #                   "latest" (last saved)
                                           #                   "final" (end of training)

# ── Hydra Config ─────────────────────────────────────────────────────────────
CONFIG_NAME="residual_td3_coffee_config"   # Hydra config name from residual_td3.py
                                           # Options: residual_td3_coffee_config (TwoArmCoffee)

# ── Core RL Hyperparameters ──────────────────────────────────────────────────
TOTAL_TIMESTEPS=500000                     # Total env steps. 500K is standard for coffee
                                           # 300K for quick experiments, 1M for thorough
N_STEP=5                                   # N-step returns. 5 = good balance of bias/variance
                                           # 1 = standard TD, 3-10 = typical range
GAMMA=0.995                                # Discount factor. 0.995 = long horizon (~200 effective steps)
                                           # 0.99 = shorter horizon, 0.999 = very long
UTD=4                                      # Updates-to-data ratio (critic updates per env step)
                                           # 4 = standard. Higher = more sample efficient but slower
                                           # 1 = on-policy ratio, 8-16 for very sample-efficient

# ── Exploration Noise ────────────────────────────────────────────────────────
STDDEV_MAX=0.025                           # Max exploration noise (std of Gaussian added to actions)
STDDEV_MIN=0.025                           # Min exploration noise (same = constant noise)
                                           # 0.025 = small/careful. 0.1 = more exploration
                                           # Set max > min for noise annealing schedule

# ── Residual Action Scale ───────────────────────────────────────────────────
ACTION_SCALE=0.2                           # Max magnitude of the residual correction
                                           # 0.2 = residual can adjust up to ±20% of action range
                                           # Smaller = safer/conservative, larger = more freedom
                                           # 0.1 for stable tasks, 0.3 for harder tasks

# ── Replay Buffer ────────────────────────────────────────────────────────────
BUFFER_SIZE=80000                          # Total replay buffer capacity (transitions)
                                           # 80K fits ~16GB RAM. 200K for 48GB servers
BATCH_SIZE=256                             # Batch size for critic/actor updates
                                           # 128 = standard. 256 if you have VRAM headroom
SAMPLING="uniform"                         # Sampling strategy: "uniform" or "prioritized"
                                           # uniform = simpler, prioritized = focus on high-TD-error

# ── Offline Data (demonstrations mixed into buffer) ──────────────────────────
OFFLINE_EPISODES=1000                       # How many demo episodes to load into the buffer
                                           # 100 = standard. 250 = more demos (more conservative)
                                           # More demos = stabler but slower to improve

# ── Learning Rates ───────────────────────────────────────────────────────────
ACTOR_LR=1e-6                              # Actor learning rate. 1e-6 = very conservative (recommended)
                                           # Too high → base policy degrades. 1e-5 max.
PREFETCH=4                                 # Number of batches to prefetch for GPU pipeline
                                           # 4 = good default. Increase if GPU is idle

# ── Warmup ───────────────────────────────────────────────────────────────────
LEARNING_STARTS=10000                      # Random actions before learning starts (exploration)
                                           # 10K = standard. 5K for quick tests
CRITIC_WARMUP=10000                        # Train critic only (no actor) for this many steps
                                           # 10K = lets critic stabilize before actor uses it

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_NUM_ENVS=10                            # Parallel eval envs. 4 = safe, 8-16 on 48GB GPU
HEADLESS="true"                            # true = no display (server). false = show MuJoCo viewer

# ── Checkpointing ────────────────────────────────────────────────────────────
SAVE_FREQ=10000                            # Save checkpoint every N env steps
                                           # 10K = good balance. 5K for more granular recovery
NO_CLEANUP="true"                          # Keep all checkpoints (true) or auto-delete old ones (false)

# ── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT="dexmg-coffee"               # WandB project for RL runs
WANDB_NAME="resfit"                        # Run name prefix in WandB
WANDB_GROUP="resfit"                       # Group name for organizing related runs

# ── Debug ────────────────────────────────────────────────────────────────────
DEBUG="false"                              # true = verbose logging, shorter runs for testing

# ============================================================================
# Run training
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Residual RL Training — TD3                                 "
echo "║  Base policy: ${BASE_WANDB_ID} (${BASE_WT_TYPE})            "
echo "║  Steps: ${TOTAL_TIMESTEPS}  |  UTD: ${UTD}  |  Buffer: ${BUFFER_SIZE}"
echo "║  Action scale: ${ACTION_SCALE}  |  Actor LR: ${ACTOR_LR}   "
echo "║  Offline episodes: ${OFFLINE_EPISODES}                      "
echo "║  WandB: ${WANDB_PROJECT}/${WANDB_NAME}                      "
echo "╚══════════════════════════════════════════════════════════════╝"

python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name="${CONFIG_NAME}" \
    base_policy.wandb_id="${BASE_WANDB_ID}" \
    base_policy.wt_type="${BASE_WT_TYPE}" \
    algo.total_timesteps="${TOTAL_TIMESTEPS}" \
    algo.prefetch_batches="${PREFETCH}" \
    algo.n_step="${N_STEP}" \
    algo.gamma="${GAMMA}" \
    algo.learning_starts="${LEARNING_STARTS}" \
    algo.critic_warmup_steps="${CRITIC_WARMUP}" \
    algo.num_updates_per_iteration="${UTD}" \
    algo.stddev_max="${STDDEV_MAX}" \
    algo.stddev_min="${STDDEV_MIN}" \
    algo.buffer_size="${BUFFER_SIZE}" \
    algo.batch_size="${BATCH_SIZE}" \
    algo.sampling_strategy="${SAMPLING}" \
    agent.actor.action_scale="${ACTION_SCALE}" \
    agent.actor_lr="${ACTOR_LR}" \
    offline_data.num_episodes="${OFFLINE_EPISODES}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.name="${WANDB_NAME}" \
    wandb.group="${WANDB_GROUP}" \
    headless="${HEADLESS}" \
    eval_num_envs="${EVAL_NUM_ENVS}" \
    no_cleanup="${NO_CLEANUP}" \
    save_freq="${SAVE_FREQ}" \
    debug="${DEBUG}"

echo ""
echo "✓ Residual RL training complete."
echo "  Check WandB: ${WANDB_PROJECT}/${WANDB_NAME}"

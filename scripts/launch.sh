#!/usr/bin/env bash
# ============================================================================
# launch.sh — Start training in a tmux session (survives SSH disconnect)
# ============================================================================
# Usage (run on server host, NOT inside the container):
#
#   ./scripts/launch.sh train         # Residual RL training
#   ./scripts/launch.sh bc            # BC policy training
#   ./scripts/launch.sh resume        # Resume interrupted RL run
#
#   # Detach from tmux:       Ctrl+B, then D
#   # Reattach after SSH:     tmux attach -t resfit
#   # Kill training:          tmux kill-session -t resfit
#   # List tmux sessions:     tmux ls
# ============================================================================
set -euo pipefail

SESSION="resfit"

usage() {
    echo "Usage: $0 {train|bc|resume}"
    echo ""
    echo "  train   — Residual RL fine-tuning (scripts/train_residual_rl.sh)"
    echo "  train_res_rl_lift — Residual RL fine-tuning for robosuite lift task (scripts/train_residual_rl_lift.sh)"
    echo "  train_res_rl_lift_sparse — Residual RL fine-tuning for robosuite lift task with sparse rewards (scripts/train_residual_rl_lift_sparse.sh)"
    echo "  train_res_rl_can  — Residual RL fine-tuning for robosuite can task (scripts/train_residual_rl_can.sh)"
    echo "  train_res_rl_can_sparse  — Residual RL fine-tuning for robosuite can task with sparse rewards (scripts/train_residual_rl_can_sparse.sh)"
    echo "  train_res_rl_can_sparse_paper_run  — Paper run: sparse Can residual RL with hardcoded params (scripts/train_residual_rl_can_sparse_paper_run.sh)"
    echo "  bc      — BC policy training (scripts/train_bc_act.sh)"
    echo "  bc_lift — BC policy (ACT) training for robosuite lift task (scripts/train_bc_lift.sh)"
    echo "  bc_can  — BC policy (ACT) training for robosuite can task (scripts/train_bc_can.sh)"
    echo '  bc_transport  — BC policy (ACT) training for robosuite transport task (scripts/train_bc_transport.sh)'
    echo "  resume  — Resume RL from checkpoint (scripts/resume_residual_rl.sh)"
    echo ""
    echo "Training runs inside a tmux session named '$SESSION'."
    echo "Safe to close SSH — reattach with: tmux attach -t $SESSION"
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    train)  SCRIPT="scripts/train_residual_rl.sh" ;;
    train_res_rl_lift) SCRIPT="scripts/train_residual_rl_lift.sh" ;;
    train_res_rl_lift_sparse) SCRIPT="scripts/train_residual_rl_lift_sparse.sh" ;;
    train_res_rl_can) SCRIPT="scripts/train_residual_rl_can.sh" ;;
    train_res_rl_can_sparse) SCRIPT="scripts/train_residual_rl_can_sparse.sh" ;;
    train_res_rl_can_sparse_paper_run) SCRIPT="scripts/train_residual_rl_can_sparse_paper_run.sh" ;;
    bc)     SCRIPT="scripts/train_bc_act.sh" ;;
    bc_lift) SCRIPT="scripts/train_bc_lift.sh" ;;
    bc_can) SCRIPT="scripts/train_bc_can.sh" ;;
    bc_transport) SCRIPT="scripts/train_bc_transport.sh" ;;
    resume) SCRIPT="scripts/resume_residual_rl.sh" ;;
    *)      usage ;;
esac

# Check tmux is available
if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux not found. Install it: sudo apt install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠  tmux session '$SESSION' already exists!"
    echo ""
    echo "  Attach to it:   tmux attach -t $SESSION"
    echo "  Kill it first:  tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting training in tmux session '$SESSION'..."
echo "  Script:    $SCRIPT"
echo "  Container: qte9489-resfit-train"
echo ""

# Create tmux session and attach immediately (avoids "size missing" warning)
# The training command and monitor pane are set up after attach via send-keys
tmux new-session -d -s "$SESSION" -x "$(tput cols)" -y "$(tput lines)" \
    "docker compose run --rm --name qte9489-resfit-train train bash $SCRIPT; echo ''; echo 'Training finished. Press Enter to close.'; read"

# Split: monitoring pane on the right (30% width)
tmux split-window -h -t "$SESSION" -p 30 \
    "sleep 10 && bash scripts/monitor.sh --live 10; read"

# Select left pane (training output) as active
tmux select-pane -t "$SESSION:0.0"

echo "✓ tmux session '$SESSION' started with 2 panes:"
echo "  Left:  training output"
echo "  Right: live resource monitor"
echo ""
echo "  Attach now:      tmux attach -t $SESSION"
echo "  Detach later:    Ctrl+B, then D"
echo "  Reattach (SSH):  tmux attach -t $SESSION"
echo "  Kill training:   tmux kill-session -t $SESSION"

# Auto-attach
tmux attach -t "$SESSION"

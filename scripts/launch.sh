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
    echo "  bc      — BC policy training (scripts/train_bc_act.sh)"
    echo "  resume  — Resume RL from checkpoint (scripts/resume_residual_rl.sh)"
    echo ""
    echo "Training runs inside a tmux session named '$SESSION'."
    echo "Safe to close SSH — reattach with: tmux attach -t $SESSION"
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    train)  SCRIPT="scripts/train_residual_rl.sh" ;;
    bc)     SCRIPT="scripts/train_bc_act.sh" ;;
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

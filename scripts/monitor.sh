#!/usr/bin/env bash
# ============================================================================
# monitor.sh — Container-scoped resource monitor for qte9489-resfit-train
# ============================================================================
# Usage:
#   ./scripts/monitor.sh              # one-shot snapshot
#   ./scripts/monitor.sh --live       # auto-refresh every 5s
#   ./scripts/monitor.sh --live 2     # auto-refresh every 2s
# ============================================================================
set -euo pipefail

IMAGE="qte9489-resfit:latest"
LIVE=false
INTERVAL=5

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live|-l) LIVE=true; shift
            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                INTERVAL=$1; shift
            fi ;;
        *) echo "Usage: $0 [--live [interval_secs]]"; exit 1 ;;
    esac
done

# Colors
BOLD='\033[1m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
DIM='\033[2m'
RESET='\033[0m'

print_snapshot() {
    # Auto-detect container by image name (works regardless of random suffix)
    CONTAINER=$(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' | head -1)
    if [[ -z "$CONTAINER" ]]; then
        echo -e "${RED}No running container found for image '$IMAGE'.${RESET}"
        return 1
    fi

    local WIDTH=78
    local SEP=$(printf '═%.0s' $(seq 1 $WIDTH))
    local THIN=$(printf '─%.0s' $(seq 1 $WIDTH))

    echo -e "${CYAN}${SEP}${RESET}"
    echo -e "${BOLD}  Container Monitor: ${GREEN}${CONTAINER}${RESET}"
    echo -e "${DIM}  $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
    echo -e "${CYAN}${SEP}${RESET}"

    # ── 1. Container Stats (CPU / RAM / Net / Disk) ─────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Container Resources${RESET}"
    echo -e "${THIN}"
    docker stats --no-stream --format \
        "  CPU:    {{.CPUPerc}}
  RAM:    {{.MemUsage}}  ({{.MemPerc}})
  Net IO: {{.NetIO}}
  Blk IO: {{.BlockIO}}
  PIDs:   {{.PIDs}}" "$CONTAINER" 2>/dev/null || echo "  (stats unavailable)"

    # ── 2. GPU Usage ────────────────────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ GPU Usage (container processes only)${RESET}"
    echo -e "${THIN}"

    # Get PIDs inside the container
    local CONTAINER_PID
    CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null)

    # Get nvidia-smi output from inside the container
    docker exec "$CONTAINER" nvidia-smi \
        --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name gpu_util mem_util mem_used mem_total temp power; do
        # Trim whitespace
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        temp=$(echo "$temp" | xargs)
        power=$(echo "$power" | xargs)

        echo -e "  GPU $idx: $name"
        echo -e "  ├─ Utilization:  ${gpu_util}% GPU  |  ${mem_util}% MEM"
        echo -e "  ├─ VRAM:         ${mem_used} / ${mem_total} MiB"
        echo -e "  ├─ Temperature:  ${temp}°C"
        echo -e "  └─ Power:        ${power} W"
    done

    # Show per-process GPU memory from inside container
    echo ""
    local GPU_PROCS
    GPU_PROCS=$(docker exec "$CONTAINER" nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_PROCS" ]]; then
        echo -e "  ${DIM}Per-process GPU memory:${RESET}"
        echo "$GPU_PROCS" | while IFS=',' read -r pid mem proc; do
            pid=$(echo "$pid" | xargs)
            mem=$(echo "$mem" | xargs)
            proc=$(echo "$proc" | xargs)
            proc_short=$(basename "$proc")
            echo -e "    PID $pid  ${mem}  $proc_short"
        done
    else
        echo -e "  ${DIM}No GPU processes running in container${RESET}"
    fi

    # ── 3. CPU & Memory per process ────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Top Processes (by CPU)${RESET}"
    echo -e "${THIN}"
    printf "  ${DIM}%-8s %-6s %-6s %-10s %-8s %s${RESET}\n" "PID" "%CPU" "%MEM" "RSS(MB)" "THREADS" "COMMAND"
    docker exec "$CONTAINER" ps -eo pid,pcpu,pmem,rss,nlwp,comm --sort=-pcpu --no-headers 2>/dev/null \
        | head -15 \
        | while read -r pid cpu mem rss threads cmd; do
            rss_mb=$(awk "BEGIN {printf \"%.1f\", $rss/1024}")
            printf "  %-8s %-6s %-6s %-10s %-8s %s\n" "$pid" "${cpu}%" "${mem}%" "$rss_mb" "$threads" "$cmd"
        done

    # ── 4. Thread summary ──────────────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Thread Summary${RESET}"
    echo -e "${THIN}"
    local TOTAL_THREADS
    TOTAL_THREADS=$(docker exec "$CONTAINER" bash -c 'ls /proc/[0-9]*/task 2>/dev/null | wc -l' 2>/dev/null || echo "?")
    local TOTAL_PROCS
    TOTAL_PROCS=$(docker exec "$CONTAINER" bash -c 'ls -d /proc/[0-9]* 2>/dev/null | wc -l' 2>/dev/null || echo "?")
    echo -e "  Total processes: $TOTAL_PROCS"
    echo -e "  Total threads:   $TOTAL_THREADS"

    # Top thread-spawning processes
    echo -e "  ${DIM}Heaviest thread spawners:${RESET}"
    docker exec "$CONTAINER" ps -eo nlwp,comm --sort=-nlwp --no-headers 2>/dev/null \
        | head -5 \
        | while read -r threads cmd; do
            printf "    %-6s threads → %s\n" "$threads" "$cmd"
        done

    # ── 5. Memory breakdown ────────────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Container Memory${RESET}"
    echo -e "${THIN}"
    docker exec "$CONTAINER" bash -c '
        awk "/MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree/" /proc/meminfo \
            | while read -r label value unit; do
                printf "  %-18s %10s %s\n" "$label" "$value" "$unit"
            done
    ' 2>/dev/null

    # ── 6. Disk usage for key dirs ─────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Disk Usage (key dirs)${RESET}"
    echo -e "${THIN}"
    docker exec "$CONTAINER" bash -c '
        for d in /app/data /app/artifacts /app/wandb /app/outputs; do
            if [ -d "$d" ]; then
                size=$(du -sh "$d" 2>/dev/null | cut -f1)
                printf "  %-25s %s\n" "$d" "$size"
            fi
        done
        # Also show overall /app/data breakdown
        if [ -d /app/data ]; then
            echo ""
            echo "  /app/data breakdown:"
            du -sh /app/data/*/ 2>/dev/null | while read -r size dir; do
                printf "    %-40s %s\n" "$dir" "$size"
            done
        fi
    ' 2>/dev/null

    echo -e "\n${CYAN}${SEP}${RESET}"
}

# Main loop
if $LIVE; then
    while true; do
        clear
        print_snapshot
        echo -e "\n${DIM}  Refreshing every ${INTERVAL}s — Ctrl+C to stop${RESET}"
        sleep "$INTERVAL"
    done
else
    print_snapshot
fi

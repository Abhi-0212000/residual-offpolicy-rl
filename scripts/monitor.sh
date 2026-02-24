#!/usr/bin/env bash
# ============================================================================
# monitor.sh — Container-scoped resource monitor for qte9489-resfit-train
# ============================================================================
# Usage:
#   ./scripts/monitor.sh              # one-shot snapshot
#   ./scripts/monitor.sh --live       # auto-refresh every 5s
#   ./scripts/monitor.sh --live 2     # auto-refresh every 2s
#   ./scripts/monitor.sh --profile    # snapshot + full Python stack traces per thread
#   ./scripts/monitor.sh --top        # py-spy top (live htop-like view of Python funcs)
# ============================================================================
set -euo pipefail

IMAGE="qte9489-resfit:latest"
LIVE=false
PROFILE=false
TOP_MODE=false
INTERVAL=5

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live|-l) LIVE=true; shift
            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                INTERVAL=$1; shift
            fi ;;
        --profile|-p) PROFILE=true; shift ;;
        --top|-t) TOP_MODE=true; shift ;;
        *) echo "Usage: $0 [--live [interval_secs]] [--profile] [--top]"; exit 1 ;;
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
            # Try to get full cmdline for this PID
            full_cmd=$(docker exec "$CONTAINER" cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | sed 's|/usr/bin/||g; s|/app/||g' | cut -c1-60 || echo "$proc")
            echo -e "    PID $pid  ${mem}  $full_cmd"
        done
    else
        echo -e "  ${DIM}No GPU processes running in container${RESET}"
    fi

    # ── 3. CPU & Memory per process ────────────────────────────────────────
    echo -e "\n${BOLD}${YELLOW}▶ Top Processes (by CPU)${RESET}"
    echo -e "${THIN}"
    printf "  ${DIM}%-7s %-6s %-6s %-9s %-5s %s${RESET}\n" "PID" "%CPU" "%MEM" "RSS(MB)" "THR" "COMMAND"
    docker exec "$CONTAINER" ps -eo pid,pcpu,pmem,rss,nlwp,args --sort=-pcpu --no-headers 2>/dev/null \
        | head -15 \
        | while read -r pid cpu mem rss threads cmd; do
            rss_mb=$(awk "BEGIN {printf \"%.1f\", $rss/1024}")
            # Shorten: strip /usr/bin/ prefixes, truncate long paths to script basename
            short_cmd=$(echo "$cmd" | sed 's|/usr/bin/||g; s|/app/||g; s|/opt/[^ ]*/||g' | cut -c1-70)
            printf "  %-7s %-6s %-6s %-9s %-5s %s\n" "$pid" "${cpu}%" "${mem}%" "$rss_mb" "$threads" "$short_cmd"
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
    docker exec "$CONTAINER" ps -eo nlwp,args --sort=-nlwp --no-headers 2>/dev/null \
        | head -5 \
        | while read -r threads cmd; do
            short_cmd=$(echo "$cmd" | sed 's|/usr/bin/||g; s|/app/||g; s|/opt/[^ ]*/||g' | cut -c1-60)
            printf "    %-6s threads → %s\n" "$threads" "$short_cmd"
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

    # ── 7. Python stack traces (only with --profile) ───────────────────────
    if $PROFILE; then
        echo -e "\n${BOLD}${YELLOW}▶ Python Stack Traces (py-spy dump)${RESET}"
        echo -e "${THIN}"
        echo -e "${DIM}  Shows every Python thread, what function it's in, and the full call stack${RESET}\n"

        # Find all Python PIDs inside the container
        local PY_PIDS
        PY_PIDS=$(docker exec "$CONTAINER" bash -c 'pgrep -x python || pgrep -f "python3"' 2>/dev/null | sort -u || true)

        if [[ -z "$PY_PIDS" ]]; then
            echo -e "  ${DIM}No Python processes found${RESET}"
        else
            for ppid in $PY_PIDS; do
                # Get the script name for this PID
                local script_name
                script_name=$(docker exec "$CONTAINER" cat /proc/$ppid/cmdline 2>/dev/null | tr '\0' ' ' | sed 's|/app/||g' | cut -c1-80 || echo "PID $ppid")
                echo -e "  ${GREEN}━━━ PID $ppid: $script_name${RESET}"

                # py-spy dump shows all threads with their Python stack traces
                docker exec "$CONTAINER" py-spy dump --pid "$ppid" 2>/dev/null \
                    | sed 's/^/    /' \
                    || echo -e "    ${DIM}Could not attach to PID $ppid (may have exited)${RESET}"
                echo ""
            done
        fi
        echo -e "${CYAN}${SEP}${RESET}"
    fi
}

# Main

# --top mode: launch py-spy top directly (interactive, replaces this script)
if $TOP_MODE; then
    CONTAINER=$(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' | head -1)
    if [[ -z "$CONTAINER" ]]; then
        echo "No running container found for image '$IMAGE'."
        exit 1
    fi
    # Find main training PID (highest CPU python process)
    MAIN_PID=$(docker exec "$CONTAINER" bash -c 'ps -eo pid,pcpu,comm --sort=-pcpu --no-headers | grep python | head -1 | awk "{print \$1}"' 2>/dev/null)
    if [[ -z "$MAIN_PID" ]]; then
        echo "No Python process found in container."
        exit 1
    fi
    echo "Launching py-spy top for PID $MAIN_PID in $CONTAINER..."
    echo "(like htop but for Python functions — press Ctrl+C to exit)"
    docker exec -it "$CONTAINER" py-spy top --pid "$MAIN_PID"
    exit 0
fi

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

#!/usr/bin/env python
"""
Inspect BC training runs — local checkpoints and/or WandB runs.

Usage examples:

  # Inspect a local run directory
  python inspect_run.py --local bc_run_2026-02-21_08-20-22_dexmg-two-arm-coffee_act

  # Inspect a WandB run by project/run_id
  python inspect_run.py --wandb dexmg-bc/zp7niccu

  # Both at once
  python inspect_run.py \
      --local bc_run_2026-02-21_08-20-22_dexmg-two-arm-coffee_act \
      --wandb dexmg-bc/zp7niccu

  # List all runs in a WandB project
  python inspect_run.py --wandb-project dexmg-bc

  # Sync / re-upload local checkpoints to WandB (safety net)
  python inspect_run.py \
      --local bc_run_2026-02-21_08-20-22_dexmg-two-arm-coffee_act \
      --wandb dexmg-bc/zp7niccu \
      --sync

  # Sync a local wandb offline run
  python inspect_run.py --sync-offline wandb/run-20260221_082026-zp7niccu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Terminal colors
# ─────────────────────────────────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def _hr(char="─", width=80):
    print(C.DIM + char * width + C.END)


def _title(text: str):
    _hr("═")
    print(f"{C.BOLD}{C.HEADER} {text}{C.END}")
    _hr("═")


def _subtitle(text: str):
    print(f"\n{C.BOLD}{C.BLUE}▸ {text}{C.END}")
    _hr()


# ─────────────────────────────────────────────────────────────────────────────
# Local inspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_local(run_dir: Path):
    """Inspect a local BC training run directory."""
    _title(f"LOCAL RUN: {run_dir.name}")

    if not run_dir.exists():
        print(f"{C.RED}  ERROR: Directory does not exist: {run_dir}{C.END}")
        return

    # ── Discover all checkpoints ──────────────────────────────────────────
    _subtitle("Checkpoints")
    checkpoints = []

    # best/ and latest/
    for special in ["best", "latest"]:
        d = run_dir / special
        if d.exists():
            state_file = d / "trainer_state.pt"
            step = None
            if state_file.exists():
                state = torch.load(state_file, map_location="cpu", weights_only=True)
                step = state.get("step")
            has_policy = (d / "policy" / "model.safetensors").exists() or (d / "policy" / "config.json").exists()
            checkpoints.append({
                "name": special,
                "path": str(d),
                "step": step,
                "has_policy": has_policy,
                "has_trainer_state": state_file.exists(),
            })

    # best_step_*
    for d in sorted(run_dir.glob("best_step_*"), key=lambda p: _extract_step(p.name)):
        step = _extract_step(d.name)
        has_policy = (d / "policy" / "model.safetensors").exists() or (d / "policy" / "config.json").exists()
        checkpoints.append({
            "name": d.name,
            "path": str(d),
            "step": step,
            "has_policy": has_policy,
            "has_trainer_state": (d / "trainer_state.pt").exists(),
        })

    # policy_step_*
    for d in sorted(run_dir.glob("policy_step_*"), key=lambda p: _extract_step(p.name)):
        step = _extract_step(d.name)
        has_policy = (d / "policy" / "model.safetensors").exists() or (d / "policy" / "config.json").exists()
        checkpoints.append({
            "name": d.name,
            "path": str(d),
            "step": step,
            "has_policy": has_policy,
            "has_trainer_state": (d / "trainer_state.pt").exists(),
        })

    if not checkpoints:
        print(f"  {C.YELLOW}No checkpoints found{C.END}")
    else:
        # Table header
        print(f"  {'Name':<25} {'Step':>10} {'Policy':>8} {'State':>8}")
        print(f"  {'─' * 25} {'─' * 10} {'─' * 8} {'─' * 8}")
        for ckpt in checkpoints:
            name = ckpt["name"]
            step_str = str(ckpt["step"]) if ckpt["step"] is not None else "?"
            pol = f"{C.GREEN}✓{C.END}" if ckpt["has_policy"] else f"{C.RED}✗{C.END}"
            state = f"{C.GREEN}✓{C.END}" if ckpt["has_trainer_state"] else f"{C.DIM}–{C.END}"
            # Highlight best
            if name == "best":
                name = f"{C.GREEN}{C.BOLD}{name}{C.END}"
                step_str = f"{C.GREEN}{C.BOLD}{step_str}{C.END}"
            elif name == "latest":
                name = f"{C.CYAN}{name}{C.END}"
                step_str = f"{C.CYAN}{step_str}{C.END}"
            elif name.startswith("best_step"):
                name = f"{C.GREEN}{name}{C.END}"
            print(f"  {name:<40} {step_str:>20} {pol:>16} {state:>16}")

    # ── Best checkpoint info ──────────────────────────────────────────────
    best_dir = run_dir / "best"
    if best_dir.exists():
        state = torch.load(best_dir / "trainer_state.pt", map_location="cpu", weights_only=True)
        best_step = state.get("step", "?")
        print(f"\n  {C.GREEN}{C.BOLD}★ Best checkpoint saved at step: {best_step}{C.END}")

    # ── Policy config ─────────────────────────────────────────────────────
    _subtitle("Policy Config (from best or latest)")
    config_path = None
    for candidate in [run_dir / "best" / "policy" / "config.json",
                      run_dir / "latest" / "policy" / "config.json"]:
        if candidate.exists():
            config_path = candidate
            break
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        # Print key architecture params
        keys_of_interest = [
            "type", "vision_backbone", "dim_model", "n_heads", "dim_feedforward",
            "n_encoder_layers", "n_decoder_layers", "chunk_size", "n_action_steps",
            "use_vae", "latent_dim", "dropout", "kl_weight",
            "optimizer_lr", "optimizer_weight_decay",
        ]
        for k in keys_of_interest:
            if k in cfg:
                print(f"  {k:<30} = {cfg[k]}")
        # Print input/output shapes
        if "input_features" in cfg:
            print(f"\n  Input features:")
            for feat_name, feat_info in cfg["input_features"].items():
                print(f"    {feat_name}: {feat_info}")
        if "output_features" in cfg:
            print(f"  Output features:")
            for feat_name, feat_info in cfg["output_features"].items():
                print(f"    {feat_name}: {feat_info}")
    else:
        print(f"  {C.YELLOW}No config.json found{C.END}")

    # ── Eval videos ───────────────────────────────────────────────────────
    _subtitle("Evaluation Videos")
    eval_dirs = list(run_dir.glob("eval_*"))
    if not eval_dirs:
        print(f"  {C.YELLOW}No eval videos found{C.END}")
    else:
        for eval_dir in eval_dirs:
            print(f"  {C.CYAN}{eval_dir.name}/{C.END}")
            videos = sorted(eval_dir.rglob("*.mp4"))
            if not videos:
                print(f"    No .mp4 files")
            else:
                print(f"    {len(videos)} video(s)")
                for v in videos[:5]:
                    size_mb = v.stat().st_size / (1024 * 1024)
                    print(f"    {v.name:<50} {size_mb:>8.1f} MB")
                if len(videos) > 5:
                    print(f"    ... and {len(videos) - 5} more")

    # ── Parse eval success rates from wandb output.log if available ───────
    _subtitle("Eval Success Rate History (from local wandb logs)")
    success_history = _parse_local_eval_history(run_dir)
    if success_history:
        _print_eval_table(success_history)
    else:
        # Try finding wandb log from the wandb/ directory
        wandb_dir = run_dir.parent / "wandb"
        if wandb_dir.exists():
            found = False
            for log_dir in sorted(wandb_dir.glob("run-*")):
                output_log = log_dir / "files" / "output.log"
                if output_log.exists():
                    history = _parse_eval_log(output_log)
                    if history:
                        # Match by checking if the timestamps roughly correspond
                        print(f"  {C.DIM}(parsed from {output_log}){C.END}")
                        found = True
                        _print_eval_table(history)
                        break
            if not found:
                print(f"  {C.YELLOW}No eval history found in local logs{C.END}")
        else:
            print(f"  {C.YELLOW}No eval history found{C.END}")

    # ── Disk usage ────────────────────────────────────────────────────────
    _subtitle("Disk Usage")
    total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    print(f"  Total: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.0f} MB)")


def _extract_step(name: str) -> int:
    """Extract step number from a directory name like policy_step_10000."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _parse_local_eval_history(run_dir: Path) -> list[dict] | None:
    """Try to find and parse eval success rates from local wandb output logs matching this run."""
    # The run dir name has a timestamp like bc_run_2026-02-21_08-20-22_...
    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir.name)
    if not m:
        return None

    run_timestamp = m.group(1)
    # Convert to datetime for matching
    run_dt = datetime.strptime(run_timestamp, "%Y-%m-%d_%H-%M-%S")

    wandb_dir = run_dir.parent / "wandb"
    if not wandb_dir.exists():
        return None

    # Find the wandb run that started closest to our training run
    best_match = None
    best_diff = float("inf")
    for log_dir in wandb_dir.glob("run-*"):
        # Parse wandb run timestamp from dir name: run-20260221_082026-zp7niccu
        m2 = re.search(r"run-(\d{8}_\d{6})", log_dir.name)
        if m2:
            wandb_dt = datetime.strptime(m2.group(1), "%Y%m%d_%H%M%S")
            diff = abs((wandb_dt - run_dt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_match = log_dir

    if best_match is None or best_diff > 600:  # within 10 minutes
        return None

    output_log = best_match / "files" / "output.log"
    if not output_log.exists():
        return None

    print(f"  {C.DIM}(matched wandb run: {best_match.name}, Δt={best_diff:.0f}s){C.END}")
    return _parse_eval_log(output_log)


def _parse_eval_log(log_path: Path) -> list[dict]:
    """Parse eval success rates from a wandb output.log file."""
    # Pattern: [step  15000] eval success-rate: 66.0% | rollout: 304.96 s | 112.2 fps
    pattern = re.compile(
        r"\[step\s+(\d+)\]\s+eval success-rate:\s+([\d.]+)%\s*\|.*?rollout:\s+([\d.]+)\s*s\s*\|\s*([\d.]+)\s*fps"
    )
    results = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                results.append({
                    "step": int(m.group(1)),
                    "success_rate": float(m.group(2)) / 100.0,
                    "rollout_time_s": float(m.group(3)),
                    "fps": float(m.group(4)),
                })

    # Also parse loss from log
    loss_pattern = re.compile(r"\[step\s+(\d+)/\d+\]\s+loss:\s+([\d.]+)")
    loss_at_step = {}
    with open(log_path) as f:
        for line in f:
            m = loss_pattern.search(line)
            if m:
                loss_at_step[int(m.group(1))] = float(m.group(2))

    for r in results:
        # Find closest loss step
        closest_step = min(loss_at_step.keys(), key=lambda s: abs(s - r["step"]), default=None)
        if closest_step is not None and abs(closest_step - r["step"]) <= 100:
            r["loss"] = loss_at_step[closest_step]

    return results


def _print_eval_table(history: list[dict]):
    """Pretty-print evaluation history as a table with a bar chart."""
    if not history:
        print(f"  {C.YELLOW}No eval data{C.END}")
        return

    # Find best
    best = max(history, key=lambda r: r["success_rate"])
    best_step = best["step"]
    best_sr = best["success_rate"]

    print(f"  {'Step':>8}  {'Success%':>9}  {'Loss':>10}  {'Rollout(s)':>11}  {'FPS':>6}  {'Bar'}")
    print(f"  {'─' * 8}  {'─' * 9}  {'─' * 10}  {'─' * 11}  {'─' * 6}  {'─' * 30}")

    for r in history:
        step = r["step"]
        sr = r["success_rate"]
        loss_str = f"{r['loss']:.5f}" if "loss" in r else "—"
        rt_str = f"{r['rollout_time_s']:.1f}" if "rollout_time_s" in r else "—"
        fps_str = f"{r['fps']:.0f}" if "fps" in r else "—"

        bar_len = int(sr * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        # Color code
        if sr >= 0.50:
            color = C.GREEN
        elif sr >= 0.25:
            color = C.YELLOW
        else:
            color = C.RED

        marker = f" {C.GREEN}{C.BOLD}★ BEST{C.END}" if step == best_step else ""
        print(f"  {step:>8}  {color}{sr * 100:>8.1f}%{C.END}  {loss_str:>10}  {rt_str:>11}  {fps_str:>6}  {color}{bar}{C.END}{marker}")

    print(f"\n  {C.GREEN}{C.BOLD}★ Best: step {best_step}, success_rate = {best_sr * 100:.1f}%{C.END}")

    # Basic stats
    srs = [r["success_rate"] for r in history]
    avg = sum(srs) / len(srs)
    last_5_avg = sum(srs[-5:]) / min(5, len(srs))
    print(f"  {C.DIM}  Mean: {avg * 100:.1f}%  |  Last 5 avg: {last_5_avg * 100:.1f}%  |  Total evals: {len(history)}{C.END}")


# ─────────────────────────────────────────────────────────────────────────────
# WandB inspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_wandb_run(run_path: str):
    """Inspect a WandB run by project/run_id."""
    import wandb

    _title(f"WANDB RUN: {run_path}")

    api = wandb.Api()

    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"{C.RED}  ERROR: Could not fetch run: {e}{C.END}")
        return

    # ── Run info ──────────────────────────────────────────────────────────
    _subtitle("Run Info")
    print(f"  {'ID':<20} {run.id}")
    print(f"  {'Name':<20} {run.name}")
    print(f"  {'State':<20} {_colorize_state(run.state)}")
    print(f"  {'Created':<20} {run.created_at}")
    print(f"  {'URL':<20} {run.url}")

    # Run duration
    summary = dict(run.summary)
    runtime = summary.get("_wandb", {}).get("runtime")
    if runtime:
        hours = runtime // 3600
        mins = (runtime % 3600) // 60
        print(f"  {'Duration':<20} {hours}h {mins}m ({runtime}s)")

    # ── Config ────────────────────────────────────────────────────────────
    _subtitle("Training Config")
    cfg = run.config
    config_keys = [
        "dataset", "policy", "steps", "batch_size", "grad_clip_norm",
        "num_workers", "seed", "device", "rollout_freq", "eval_env",
        "eval_num_envs", "eval_num_episodes", "eval_camera_size",
        "eval_render_size", "eval_video_key",
    ]
    for k in config_keys:
        if k in cfg:
            print(f"  {k:<25} = {cfg[k]}")

    # Policy config
    policy_config = cfg.get("policy_config", {})
    if policy_config:
        print(f"\n  {C.CYAN}Policy Architecture:{C.END}")
        arch_keys = [
            "vision_backbone", "dim_model", "n_heads", "dim_feedforward",
            "n_encoder_layers", "n_decoder_layers", "chunk_size", "n_action_steps",
            "use_vae", "latent_dim", "optimizer_lr",
        ]
        for k in arch_keys:
            if k in policy_config:
                print(f"    {k:<25} = {policy_config[k]}")

    # ── Final Summary Metrics ─────────────────────────────────────────────
    _subtitle("Final Summary Metrics")
    for k in sorted(summary.keys()):
        if k.startswith("_"):
            continue
        v = summary[k]
        if isinstance(v, (int, float)):
            if "rate" in k:
                print(f"  {k:<35} = {C.GREEN}{v * 100:.1f}%{C.END}")
            elif "loss" in k:
                print(f"  {k:<35} = {v:.6f}")
            elif "ms" in k:
                print(f"  {k:<35} = {v:.1f} ms")
            else:
                print(f"  {k:<35} = {v}")

    # ── Eval History ──────────────────────────────────────────────────────
    _subtitle("Eval Success Rate History (from WandB)")
    try:
        history_rows = list(run.scan_history(
            keys=["eval/success_rate", "train/loss"],
            page_size=1000,
        ))
        eval_history = []
        for row in history_rows:
            sr = row.get("eval/success_rate")
            if sr is not None:
                eval_history.append({
                    "step": row.get("_step", 0),
                    "success_rate": sr,
                    "loss": row.get("train/loss"),
                })
        if eval_history:
            # Adapt to _print_eval_table format
            for r in eval_history:
                if r["loss"] is None:
                    r.pop("loss", None)
            _print_eval_table(eval_history)
        else:
            print(f"  {C.YELLOW}No eval data logged{C.END}")
    except Exception as e:
        print(f"  {C.YELLOW}Could not fetch history: {e}{C.END}")

    # ── Artifacts ─────────────────────────────────────────────────────────
    _subtitle("Artifacts")
    try:
        artifacts = list(run.logged_artifacts())
        if not artifacts:
            print(f"  {C.YELLOW}No artifacts{C.END}")
        else:
            print(f"  {'Name':<50} {'Type':<10} {'Size':>10}")
            print(f"  {'─' * 50} {'─' * 10} {'─' * 10}")
            for art in artifacts:
                size_str = f"{art.size / (1024**2):.1f} MB" if art.size else "?"
                print(f"  {art.name:<50} {art.type:<10} {size_str:>10}")
    except Exception as e:
        print(f"  {C.YELLOW}Could not list artifacts: {e}{C.END}")


def list_wandb_project(project: str):
    """List all runs in a WandB project."""
    import wandb

    _title(f"WANDB PROJECT: {project}")

    api = wandb.Api()
    try:
        runs = api.runs(project)
    except Exception as e:
        print(f"{C.RED}  ERROR: {e}{C.END}")
        return

    print(f"  {'Run ID':<12} {'Name':<35} {'State':<12} {'Success%':>9} {'Loss':>10} {'Steps':>8} {'Created'}")
    print(f"  {'─' * 12} {'─' * 35} {'─' * 12} {'─' * 9} {'─' * 10} {'─' * 8} {'─' * 20}")

    for r in runs:
        s = dict(r.summary)
        sr = s.get("eval/success_rate")
        sr_str = f"{sr * 100:.1f}%" if sr is not None else "—"
        loss = s.get("train/loss")
        loss_str = f"{loss:.5f}" if loss is not None else "—"
        steps = s.get("_step", "—")
        state = _colorize_state(r.state)
        print(f"  {r.id:<12} {r.name:<35} {state:<22} {sr_str:>9} {loss_str:>10} {steps:>8} {r.created_at}")

    print(f"\n  Total runs: {len(runs)}")


def _colorize_state(state: str) -> str:
    if state == "finished":
        return f"{C.GREEN}{state}{C.END}"
    elif state == "running":
        return f"{C.CYAN}{state}{C.END}"
    elif state == "failed":
        return f"{C.RED}{state}{C.END}"
    elif state == "crashed":
        return f"{C.RED}{state}{C.END}"
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Sync local checkpoints to WandB
# ─────────────────────────────────────────────────────────────────────────────
def sync_to_wandb(local_dir: Path, wandb_run_path: str):
    """Upload local checkpoints as artifacts to an existing WandB run."""
    import wandb

    _title(f"SYNC: {local_dir.name} → {wandb_run_path}")

    if not local_dir.exists():
        print(f"{C.RED}  ERROR: Local directory not found: {local_dir}{C.END}")
        return

    api = wandb.Api()
    try:
        run = api.run(wandb_run_path)
    except Exception as e:
        print(f"{C.RED}  ERROR: Could not find WandB run: {e}{C.END}")
        return

    # Check what artifacts already exist
    existing_artifacts = {art.name for art in run.logged_artifacts()}
    print(f"  Existing artifacts: {len(existing_artifacts)}")

    # Resume the run to log artifacts
    project, run_id = wandb_run_path.split("/")
    wandb.init(project=project, id=run_id, resume="must")

    uploaded = 0
    skipped = 0

    # Upload best/ checkpoint
    best_dir = local_dir / "best"
    if best_dir.exists():
        art_name = f"run_{run_id}_best"
        print(f"  Uploading {C.GREEN}best{C.END} checkpoint as {art_name}...")
        art = wandb.Artifact(name=art_name, type="model")
        art.add_dir(str(best_dir))
        wandb.log_artifact(art, aliases=["best", "latest"])
        uploaded += 1

    # Upload latest/ checkpoint
    latest_dir = local_dir / "latest"
    if latest_dir.exists():
        art_name = f"run_{run_id}_latest"
        print(f"  Uploading {C.CYAN}latest{C.END} checkpoint as {art_name}...")
        art = wandb.Artifact(name=art_name, type="model")
        art.add_dir(str(latest_dir))
        wandb.log_artifact(art, aliases=["latest"])
        uploaded += 1

    # Upload policy_step_* checkpoints
    for ckpt_dir in sorted(local_dir.glob("policy_step_*"), key=lambda p: _extract_step(p.name)):
        step = _extract_step(ckpt_dir.name)
        art_name = f"run_{run_id}_model_step_{step}"
        if art_name in existing_artifacts:
            print(f"  {C.DIM}Skipping {ckpt_dir.name} (already exists on WandB){C.END}")
            skipped += 1
            continue
        print(f"  Uploading {ckpt_dir.name} as {art_name}...")
        art = wandb.Artifact(name=art_name, type="model")
        art.add_dir(str(ckpt_dir))
        wandb.log_artifact(art)
        uploaded += 1

    # Upload best_step_* checkpoints
    for ckpt_dir in sorted(local_dir.glob("best_step_*"), key=lambda p: _extract_step(p.name)):
        step = _extract_step(ckpt_dir.name)
        art_name = f"run_{run_id}_best_step_{step}"
        if art_name in existing_artifacts:
            print(f"  {C.DIM}Skipping {ckpt_dir.name} (already exists){C.END}")
            skipped += 1
            continue
        print(f"  Uploading {ckpt_dir.name}...")
        art = wandb.Artifact(name=art_name, type="model")
        art.add_dir(str(ckpt_dir))
        wandb.log_artifact(art)
        uploaded += 1

    wandb.finish()
    print(f"\n  {C.GREEN}Done! Uploaded: {uploaded}, Skipped: {skipped}{C.END}")


def sync_offline_wandb(wandb_run_dir: Path):
    """Sync an offline/interrupted wandb run directory."""
    _title(f"SYNC OFFLINE: {wandb_run_dir.name}")

    if not wandb_run_dir.exists():
        print(f"{C.RED}  ERROR: Directory not found: {wandb_run_dir}{C.END}")
        return

    print(f"  Running: wandb sync {wandb_run_dir}")
    os.system(f"wandb sync {wandb_run_dir}")
    print(f"  {C.GREEN}Sync complete{C.END}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Inspect BC training runs (local checkpoints and/or WandB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect local run
  python inspect_run.py --local bc_run_2026-02-21_08-20-22_dexmg-two-arm-coffee_act

  # Inspect WandB run
  python inspect_run.py --wandb dexmg-bc/zp7niccu

  # List all runs in project
  python inspect_run.py --wandb-project dexmg-bc

  # Sync local checkpoints to WandB
  python inspect_run.py --local <dir> --wandb <project/run_id> --sync

  # Sync offline wandb run
  python inspect_run.py --sync-offline wandb/run-20260221_082026-zp7niccu
        """,
    )
    parser.add_argument("--local", type=str, help="Path to local run directory")
    parser.add_argument("--wandb", type=str, help="WandB run path: project/run_id")
    parser.add_argument("--wandb-project", type=str, help="List all runs in a WandB project")
    parser.add_argument("--sync", action="store_true", help="Sync local checkpoints to WandB (requires both --local and --wandb)")
    parser.add_argument("--sync-offline", type=str, help="Sync a local wandb run directory")

    args = parser.parse_args()

    if not any([args.local, args.wandb, args.wandb_project, args.sync_offline]):
        parser.print_help()
        sys.exit(1)

    # ── List project ──────────────────────────────────────────────────────
    if args.wandb_project:
        list_wandb_project(args.wandb_project)

    # ── Inspect local ─────────────────────────────────────────────────────
    if args.local:
        local_path = Path(args.local).resolve()
        inspect_local(local_path)

    # ── Inspect WandB ─────────────────────────────────────────────────────
    if args.wandb and not args.sync:
        inspect_wandb_run(args.wandb)

    # ── Sync ──────────────────────────────────────────────────────────────
    if args.sync:
        if not args.local or not args.wandb:
            print(f"{C.RED}ERROR: --sync requires both --local and --wandb{C.END}")
            sys.exit(1)
        sync_to_wandb(Path(args.local).resolve(), args.wandb)

    # ── Sync offline ──────────────────────────────────────────────────────
    if args.sync_offline:
        sync_offline_wandb(Path(args.sync_offline).resolve())


if __name__ == "__main__":
    main()

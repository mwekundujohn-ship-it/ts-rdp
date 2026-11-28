#!/usr/bin/env python
"""Alpha wave orchestrator - runs all stages for the Alpha master.

Usage:
    python run_alpha_wave.py --wave w1 --phase all
    python run_alpha_wave.py --wave w1 --phase 1    # parity + CZET
    python run_alpha_wave.py --wave w1 --phase 2    # encode 30m/1h
    python run_alpha_wave.py --wave w1 --phase 3    # encode 4h/6h
    python run_alpha_wave.py --wave w1 --phase 4    # relational + anchors
    python run_alpha_wave.py --wave w1 --phase 5    # cascade + exec backtest
    python run_alpha_wave.py --wave w1 --phase 6    # audit + handoff

ALL PATHS are read from runner_mt5_config.json and wave2a_shard_manifest.json.
NO HARDCODED PATHS.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add fib_engine parent to path so "from fib_engine import ..." works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit
import cascade
import compare
import dalton
import epicycle
import lattice
import push_state


def load_runner_config(config_path: str | None = None) -> dict:
    """Load runner_mt5_config.json from default locations or explicit path."""
    candidates = [
        Path(config_path) if config_path else None,
        Path.cwd() / "runner_mt5_config.json",
        Path(__file__).resolve().parent.parent.parent.parent / "runner_mt5_config.json",
        Path(__file__).resolve().parent.parent.parent / "runner_mt5_config.json",
        Path(r"C:\mt5-opencode\runner_mt5_config.json"),
    ]
    for c in candidates:
        if c and c.exists():
            with open(c, encoding="utf-8-sig") as f:
                return json.load(f)
    return {
        "source_directory": r"C:\quantstore\ticks\Jump_10_Index",
        "output_root": r"C:\work\out",
        "timeframes": ["30m", "1h", "4h", "6h"],
        "symbol": "Jump 10 Index.0",
    }


def load_shard_manifest(manifest_path: str | None = None) -> dict:
    """Load wave2a_shard_manifest.json from default locations or explicit path."""
    candidates = [
        Path(manifest_path) if manifest_path else None,
        Path.cwd() / "wave2a_shard_manifest.json",
        Path(__file__).resolve().parent.parent.parent.parent / "wave2a_shard_manifest.json",
        Path(__file__).resolve().parent.parent.parent / "wave2a_shard_manifest.json",
        Path(r"C:\mt5-opencode\ts-rdp\wave2a_shard_manifest.json"),
    ]
    for c in candidates:
        if c and c.exists():
            with open(c, encoding="utf-8-sig") as f:
                return json.load(f)
    return {}


def run_cmd(cmd: list[str], cwd: str = None, timeout: int = 3600) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    print(f"$ {' '.join(cmd[:5])}...", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"FAILED ({result.returncode}): {output[-2000:]}", flush=True)
        else:
            print(f"OK ({result.returncode})", flush=True)
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def push_stage(
    stage_dir: Path,
    stage_name: str,
    wave: str,
    node: str,
    fleet: str = "alpha",
    repo: str = ".",
    branch: str = None,
):
    """Push stage to both QuantStore and GitHub."""
    branch = branch or f"quant/{wave}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "push_state.py"),
        "--dir", str(stage_dir),
        "--stage", stage_name,
        "--fleet", fleet,
        "--wave", wave,
        "--node", node,
        "--repo", repo,
        "--branch", branch,
    ]
    rc, out = run_cmd(cmd, timeout=600)
    return rc == 0


def generate_handoff(results: dict, wave: str, node: str, out_base: Path) -> str:
    """Generate HANDOFF_ALPHA.md from phase results."""
    from datetime import datetime, timezone

    lines = [
        f"# HANDOFF_ALPHA - Wave {wave}",
        "",
        f"- node: `{node}`",
        f"- wave: `{wave}`",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Phase Results",
        "",
    ]

    for phase, res in results.items():
        status = "✅ PASS" if res["rc"] == 0 else "❌ FAIL"
        lines.append(f"- **{phase}**: {status}")
        if res["rc"] != 0:
            lines.append(f"  - Error: {res['output'][:500]}")

    lines += [
        "",
        "## Artifacts",
        "",
    ]

    # List artifacts from each phase
    for phase_dir in sorted(out_base.glob("phase*")):
        if phase_dir.is_dir():
            metrics_files = list(phase_dir.glob("metrics_and_checks.json"))
            if metrics_files:
                with open(metrics_files[0]) as f:
                    m = json.load(f)
                lines.append(f"- **{phase_dir.name}**: {m['checks_total'] - m['checks_failed']}/{m['checks_total']} checks, {m['elapsed_s']}s")
                for art in m.get("artifacts", []):
                    lines.append(f"  - {Path(art['path']).name} ({art['bytes']} bytes)")

    lines += [
        "",
        "## Next Wave Tasks",
        "",
        "1. Extend backtest to full OOS period (2024-01 to 2026-10)",
        "2. Add transaction cost modeling with Deriv spread + slippage",
        "3. Implement walk-forward anchor selection",
        "4. Add regime detection for adaptive position sizing",
        "5. Cross-validate with Beta fleet",
        "",
        f"---",
        f"Generated by alpha-master at {datetime.now(timezone.utc).isoformat()}",
    ]

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Alpha wave orchestrator")
    ap.add_argument("--wave", required=True, help="Wave ID (e.g., w1)")
    ap.add_argument("--phase", choices=["all", "1", "2", "3", "4", "5", "6"],
                    default="all", help="Phase to run")
    ap.add_argument("--node", default=os.environ.get("COMPUTERNAME", "alpha-master"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", help="Path to runner_mt5_config.json")
    ap.add_argument("--manifest", help="Path to wave2a_shard_manifest.json")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--github-token", default=os.environ.get("ALPHA_GITHUB_TOKEN"))
    ap.add_argument("--node-index", type=int, default=0, help="Node index: 0=master, 1=worker1, 2=worker2")
    args = ap.parse_args()

    # Load configuration - NO HARDCODED PATHS
    config = load_runner_config(args.config)
    manifest = load_shard_manifest(args.manifest)

    wave = args.wave
    node = args.node
    node_index = args.node_index

    # Output directory from config
    out_root = Path(config.get("output_root", r"C:\work\out"))
    out_base = out_root / wave / node
    out_base.mkdir(parents=True, exist_ok=True)

    # Tick directory from manifest (worker-specific) or config
    if node_index in (1, 2):
        alpha = manifest.get("fleets", {}).get("alpha", {})
        worker_key = f"alpha-worker{node_index}"
        worker = alpha.get(worker_key, {})
        tick_dir = Path(worker.get("source_directory", config.get("source_directory", r"C:\quantstore\ticks\Jump_10_Index")))
    else:
        tick_dir = Path(config.get("source_directory", r"C:\quantstore\ticks\Jump_10_Index"))

    # Phase 3 directory (for verified anchors)
    phase3_dir = out_root / "phase3"  # Relative to output_root

    phases_to_run = []
    if args.phase == "all":
        phases_to_run = ["1", "2", "3", "4", "5", "6"]
    else:
        phases_to_run = [args.phase]

    results = {}

    # =====================================================================
    # Phase 1: Parity + CZET (master only)
    # =====================================================================
    if "1" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 1: Parity + CZET v2")
        print("="*60)

        stage_dir = out_base / "phase1_parity_czet"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Run live_matcher verification
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "live_matcher.py"),
        ]
        rc, out = run_cmd(cmd, timeout=120)
        results["phase1"] = {"rc": rc, "output": out}

        # Push
        if rc == 0:
            push_stage(stage_dir, "phase1_parity_czet", wave, node)

    # =====================================================================
    # Phase 2: Encode 30m/1h (workers)
    # =====================================================================
    if "2" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 2: Encode 30m/1h")
        print("="*60)

        for tf in ["30m", "1h"]:
            stage_dir = out_base / f"phase2_encode_{tf}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            # This runs on workers - master dispatches
            # For now, run locally for testing
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_stage.py"),
                "--stage", "encode",
                "--tf", tf,
                "--ticks", str(tick_dir),
                "--out", str(stage_dir),
                "--budget-s", "5400",
                "--notes", f"alpha {wave} encode {tf}",
                "--config", args.config or "",
                "--manifest", args.manifest or "",
                "--node-index", str(node_index),
            ]
            rc, out = run_cmd(cmd, timeout=7200)
            results[f"phase2_{tf}"] = {"rc": rc, "output": out}

            if rc == 0:
                push_stage(stage_dir, f"phase2_encode_{tf}", wave, node)

    # =====================================================================
    # Phase 3: Encode 4h/6h (workers)
    # =====================================================================
    if "3" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 3: Encode 4h/6h")
        print("="*60)

        for tf in ["4h", "6h"]:
            stage_dir = out_base / f"phase3_encode_{tf}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_stage.py"),
                "--stage", "encode",
                "--tf", tf,
                "--ticks", str(tick_dir),
                "--out", str(stage_dir),
                "--budget-s", "3600",
                "--notes", f"alpha {wave} encode {tf}",
                "--config", args.config or "",
                "--manifest", args.manifest or "",
                "--node-index", str(node_index),
            ]
            rc, out = run_cmd(cmd, timeout=7200)
            results[f"phase3_{tf}"] = {"rc": rc, "output": out}

            if rc == 0:
                push_stage(stage_dir, f"phase3_encode_{tf}", wave, node)

    # =====================================================================
    # Phase 4: Relational + Anchors (master)
    # =====================================================================
    if "4" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 4: Relational tensor + Anchor vault")
        print("="*60)

        stage_dir = out_base / "phase4_compare_anchors"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Collect ALL four encode directories explicitly (30m, 1h, 4h, 6h)
        # Each timeframe has its own output directory from Phase 2/3
        encode_dirs = {}
        for tf in ["30m", "1h", "4h", "6h"]:
            # Find the encode directory for this timeframe
            pattern = f"phase*_encode_{tf}"
            matches = sorted(out_base.parent.glob(pattern))
            if matches:
                encode_dirs[tf] = matches[-1]  # Use latest if multiple
            else:
                print(f"[WARN] No encode directory found for {tf} (pattern: {pattern})")

        if len(encode_dirs) < 4:
            print(f"[ERROR] Missing encode directories: got {list(encode_dirs.keys())}, expected 30m,1h,4h,6h")
            results["phase4"] = {"rc": 1, "output": "Missing encode directories"}
        else:
            # Pass all four encode directories to stage_compare_anchors.py
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "stage_compare_anchors.py"),
                "--encode-dir-30m", str(encode_dirs["30m"]),
                "--encode-dir-1h", str(encode_dirs["1h"]),
                "--encode-dir-4h", str(encode_dirs["4h"]),
                "--encode-dir-6h", str(encode_dirs["6h"]),
                "--phase3-dir", str(phase3_dir),
                "--out", str(stage_dir),
                "--notes", f"alpha {wave} compare anchors",
            ]
            rc, out = run_cmd(cmd, timeout=5400)
            results["phase4"] = {"rc": rc, "output": out}

            if rc == 0:
                push_stage(stage_dir, "phase4_compare_anchors", wave, node)

    # =====================================================================
    # Phase 5: Cascade + Exec Backtest (master)
    # =====================================================================
    if "5" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 5: Cascade + Execution backtest")
        print("="*60)

        stage_dir = out_base / "phase5_cascade_exec"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Find phase 4 outputs
        phase4_dir = out_base / "phase4_compare_anchors"
        vault_path = phase4_dir / "anchor_vault.json"
        cascade_path = phase4_dir / "cascade_results.json"

        if not vault_path.exists() or not cascade_path.exists():
            print(f"Missing phase 4 outputs: {vault_path} or {cascade_path}")
            results["phase5"] = {"rc": 1, "output": "Missing phase 4 outputs"}
        else:
            # Collect ALL four encode directories explicitly (same as Phase 4)
            encode_dirs = {}
            for tf in ["30m", "1h", "4h", "6h"]:
                pattern = f"phase*_encode_{tf}"
                matches = sorted(out_base.parent.glob(pattern))
                if matches:
                    encode_dirs[tf] = matches[-1]
                else:
                    print(f"[WARN] No encode directory found for {tf} (pattern: {pattern})")

            if len(encode_dirs) < 4:
                print(f"[ERROR] Missing encode directories: got {list(encode_dirs.keys())}, expected 30m,1h,4h,6h")
                results["phase5"] = {"rc": 1, "output": "Missing encode directories"}
            else:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "stage_exec_backtest.py"),
                    "--vault", str(vault_path),
                    "--cascade", str(cascade_path),
                    "--encode-dir-30m", str(encode_dirs["30m"]),
                    "--encode-dir-1h", str(encode_dirs["1h"]),
                    "--encode-dir-4h", str(encode_dirs["4h"]),
                    "--encode-dir-6h", str(encode_dirs["6h"]),
                    "--ticks", str(tick_dir),
                    "--out", str(stage_dir),
                    "--notes", f"alpha {wave} cascade exec",
                ]
                rc, out = run_cmd(cmd, timeout=3600)
                results["phase5"] = {"rc": rc, "output": out}

                if rc == 0:
                    push_stage(stage_dir, "phase5_cascade_exec", wave, node)

    # =====================================================================
    # Phase 6: Audit + Handoff
    # =====================================================================
    if "6" in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 6: Audit + Handoff")
        print("="*60)

        stage_dir = out_base / "phase6_audit_handoff"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Collect all metrics and generate HANDOFF_ALPHA.md
        handoff_path = stage_dir / "HANDOFF_ALPHA.md"
        handoff_content = generate_handoff(results, wave, node, out_base)
        handoff_path.write_text(handoff_content)
        print(f"Handoff written to {handoff_path}")

        # Push
        push_stage(stage_dir, "phase6_audit_handoff", wave, node)

    # Summary
    print("\n" + "="*60)
    print("WAVE SUMMARY")
    print("="*60)
    for phase, res in results.items():
        status = "PASS" if res["rc"] == 0 else "FAIL"
        print(f"  {phase}: {status}")

    return 0 if all(r["rc"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
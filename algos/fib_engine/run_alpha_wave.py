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
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add fib_engine to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fib_engine import push_state


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


def main():
    ap = argparse.ArgumentParser(description="Alpha wave orchestrator")
    ap.add_argument("--wave", required=True, help="Wave ID (e.g., w1)")
    ap.add_argument("--phase", choices=["all", "1", "2", "3", "4", "5", "6"],
                    default="all", help="Phase to run")
    ap.add_argument("--node", default=os.environ.get("COMPUTERNAME", "alpha-master"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ticks", default=r"C:\work\ticks\Jump_10_Index")
    ap.add_argument("--phase2", default=r"C:\work\phase2")
    ap.add_argument("--phase3", default=r"C:\work\phase3")
    ap.add_argument("--out", default=r"C:\work\out")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--github-token", default=os.environ.get("ALPHA_GITHUB_TOKEN"))
    args = ap.parse_args()

    wave = args.wave
    node = args.node
    out_base = Path(args.out) / wave / node
    out_base.mkdir(parents=True, exist_ok=True)

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
                "--ticks", args.ticks,
                "--out", str(stage_dir),
                "--budget-s", "5400",
                "--notes", f"alpha {wave} encode {tf}",
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
                "--ticks", args.ticks,
                "--out", str(stage_dir),
                "--budget-s", "3600",
                "--notes", f"alpha {wave} encode {tf}",
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

        encode_dir = out_base.parent / "phase2_encode_30m"  # parent wave dir
        # Find the latest encode outputs
        encode_dirs = sorted(out_base.parent.glob("phase*_encode_*"))
        encode_dir = encode_dirs[-1].parent if encode_dirs else out_base

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "stage_compare_anchors.py"),
            "--encode-dir", str(encode_dir),
            "--phase3-dir", args.phase3,
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
            encode_dirs = sorted(out_base.parent.glob("phase*_encode_*"))
            encode_dir = encode_dirs[-1].parent if encode_dirs else out_base

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "stage_exec_backtest.py"),
                "--vault", str(vault_path),
                "--cascade", str(cascade_path),
                "--encode-dir", str(encode_dir),
                "--ticks", args.ticks,
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


if __name__ == "__main__":
    sys.exit(main())
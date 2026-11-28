"""Audit packet writer.

Every stage writes three things next to its artifacts:

    progress_chat.md      narrative: what ran, why, what broke, what was decided
    metrics_and_checks.json  exact counts/shapes/NaN rates/timings - no bare "PASS"
    manifest.sha256       checksum of every file the stage produced

The local audit loop pulls these and re-derives the numbers, so a stage that silently
skips work cannot hide: the counts are recomputed from the artifacts, not asserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Audit:
    """Accumulates checks during a stage, then writes the packet."""

    def __init__(self, out_dir: str | os.PathLike, stage: str, node: str = "", notes: str = ""):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.node = node or platform.node()
        self.started = time.time()
        self.notes = notes
        self.checks: list[dict] = []
        self.metrics: dict = {}
        self.artifacts: list[Path] = []
        self.log: list[str] = []

    # ---------------------------------------------------------------- record
    def say(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}"
        self.log.append(line)
        print(line, flush=True)

    def check(self, name: str, ok: bool, detail: str = "", **extra) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail), **extra})
        self.say(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
        return bool(ok)

    def metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def artifact(self, path: str | os.PathLike) -> None:
        p = Path(path)
        if p.exists():
            self.artifacts.append(p)

    # ---------------------------------------------------------------- write
    def _packet(self) -> dict:
        elapsed = time.time() - self.started
        failed = [c["name"] for c in self.checks if not c["ok"]]
        
        # Provenance block (required by stage_gate.py)
        import os
        provenance = {
            "runner_name": os.environ.get("RUNNER_NAME", self.node),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "hostname": platform.node(),
            "shard_id": os.environ.get("SHARD_ID", ""),
            "pid": os.getpid(),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return {
            "stage": self.stage,
            "node": self.node,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_utc": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 3),
            "checks_total": len(self.checks),
            "checks_failed": len(failed),
            "failed": failed,
            "checks": self.checks,
            "metrics": self.metrics,
            "provenance": provenance,
            "artifacts": [
                {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in self.artifacts
            ],
        }

    def finish(self, summary: str = "") -> Path:
        pkt = self._packet()

        mpath = self.dir / "metrics_and_checks.json"
        mpath.write_text(json.dumps(pkt, indent=2), encoding="utf-8", newline="\n")

        mpath_sha = self.dir / "manifest.sha256"
        lines = [f"{a['sha256']}  {Path(a['path']).name}" for a in pkt["artifacts"]]
        lines.append(f"{sha256_file(mpath)}  {mpath.name}")
        mpath_sha.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

        chat = [
            f"# {self.stage} - progress chat",
            "",
            f"- node: `{pkt['node']}`",
            f"- started: {pkt['started_utc']}",
            f"- elapsed: {pkt['elapsed_s']}s",
            f"- checks: {pkt['checks_total'] - pkt['checks_failed']}/{pkt['checks_total']} passed",
            "",
            "## What ran",
            "",
        ]
        chat += [f"- {l}" for l in self.log] or ["- (nothing logged)"]
        chat += ["", "## Metrics", "", "```json", json.dumps(self.metrics, indent=2), "```", ""]
        if pkt["failed"]:
            chat += ["## Failures", ""] + [f"- {n}" for n in pkt["failed"]] + [""]
        if self.notes:
            chat += ["## Notes", "", self.notes, ""]
        if summary:
            chat += ["## Summary", "", summary, ""]
        (self.dir / "progress_chat.md").write_text("\n".join(chat), encoding="utf-8", newline="\n")

        self.say(f"packet written to {self.dir}")
        return self.dir

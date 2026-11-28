#!/usr/bin/env python3
"""Hard promotion gate for a research stage (Wave 2A contract).

Reads <stage-dir>/metrics_and_checks.json and refuses promotion unless every
contract check passes. On success it writes manifest.sha256 over the artifact
bytes, hashing text artifacts in LF-normalized form so the QuantStore copy and
the GitHub copy produce identical hashes.

Hardened against the four holes found in the 2026-11-28 recon:
  H1 artifact paths must resolve STRICTLY inside --stage-dir (no basename
     fallback, no absolute runner paths, no ".." escape).
  H2 placeholder provenance is rejected: github_run_id matching
     test*/dummy/local/none/todo/xxx/example/sample, and non-numeric ids when
     GITHUB_ACTIONS=true. tailscale_ip / node_index / shard_id are mandatory.
  H3 shard_id must be exactly "0/2", "1/2" or "merged" (rejects "0/3").
  H4 merged stages must carry a numeric timestamp_equality_proof with
     matched == total >= 1000, and every artifact sha256 must match the bytes
     on disk.

Exit codes: 0 = pass (manifest written), 1 = blocked (no manifest written).

Usage:
    python tools/stage_gate.py --stage-dir out/stage --expected-months 68
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROVENANCE_KEYS = (
    "runner_name",
    "github_run_id",
    "hostname",
    "tailscale_ip",
    "node_index",
    "shard_id",
    "pid",
    "utc_timestamp",
)

ALLOWED_SHARDS = {"0/2", "1/2", "merged"}
PLACEHOLDER_RUN_ID = re.compile(
    r"^(test|dummy|local|none|null|todo|tbd|xxx|example|sample|placeholder|fake|mock)",
    re.IGNORECASE,
)
NUMERIC = re.compile(r"^\d+$")
MIN_PROOF_ROWS = 1000

TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".sha256", ".py", ".txt", ".log",
    ".yaml", ".yml", ".ini", ".cfg", ".tsv",
}
BINARY_SUFFIXES = {
    ".npz", ".npy", ".gz", ".parquet", ".pt", ".pth", ".db",
    ".xlsx", ".exe", ".zip", ".png", ".jpg", ".pdf",
}


def is_text(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return False
    if suffix in TEXT_SUFFIXES:
        return True
    return True  # unknown small files are treated as text and checked for CRLF


def norm_rel(path: Path, stage_dir: Path) -> str:
    return path.relative_to(stage_dir).as_posix()


def resolve_artifact(candidate: str, stage_dir: Path, errors: list[str]) -> Path | None:
    """H1: resolve strictly inside stage_dir. No fallback, no escape, no absolute."""
    raw = candidate.strip()
    if not raw:
        errors.append("artifact entry is empty")
        return None
    p = Path(raw)
    if p.is_absolute():
        errors.append(f"artifact path is absolute (must be stage-relative): {raw}")
        return None
    if any(part == ".." for part in p.parts):
        errors.append(f"artifact path escapes stage dir with '..': {raw}")
        return None
    resolved = (stage_dir / p).resolve()
    try:
        resolved.relative_to(stage_dir)
    except ValueError:
        errors.append(f"artifact path resolves outside stage dir: {raw} -> {resolved}")
        return None
    if not resolved.exists():
        errors.append(f"artifact missing on disk: {raw}")
        return None
    if not resolved.is_file():
        errors.append(f"artifact is not a file: {raw}")
        return None
    return resolved


def iter_checks(metrics: dict):
    checks = metrics.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict):
                yield item.get("name", "<unnamed>"), item.get("ok")
            else:
                yield "<unnamed>", item
    elif isinstance(checks, dict):
        for name, value in checks.items():
            yield name, value
    else:
        yield "<checks>", None


def get_metric(metrics: dict, key: str):
    if key in metrics:
        return metrics[key]
    inner = metrics.get("metrics")
    if isinstance(inner, dict) and key in inner:
        return inner[key]
    return None


def check_provenance(prov: dict, errors: list[str]) -> None:
    """H2: reject placeholders and require the full provenance surface."""
    for key in PROVENANCE_KEYS:
        value = prov.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"provenance.{key} missing or empty")

    run_id = prov.get("github_run_id")
    if isinstance(run_id, str) and PLACEHOLDER_RUN_ID.match(run_id.strip()):
        errors.append(f"provenance.github_run_id is a placeholder: {run_id!r}")
    on_gha = os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
    if on_gha and isinstance(run_id, str) and run_id.strip() and not NUMERIC.match(run_id.strip()):
        errors.append(f"provenance.github_run_id must be numeric on GitHub Actions: {run_id!r}")

    node_index = prov.get("node_index")
    if not isinstance(node_index, int) or isinstance(node_index, bool):
        if not (isinstance(node_index, str) and node_index.strip().isdigit()):
            errors.append(f"provenance.node_index is not an integer: {node_index!r}")

    shard = prov.get("shard_id")
    if shard not in ALLOWED_SHARDS:
        errors.append(f"provenance.shard_id={shard!r} not in {sorted(ALLOWED_SHARDS)}")

    ts = prov.get("utc_timestamp")
    if isinstance(ts, str) and not re.match(r"^\d{4}-\d{2}-\d{2}T", ts.strip()):
        errors.append(f"provenance.utc_timestamp is not ISO-8601: {ts!r}")


def check_timestamp_proof(metrics: dict, shard: str, errors: list[str]) -> None:
    """H4: merged stages need a numeric 1000-row equality proof."""
    proof = metrics.get("timestamp_equality_proof")
    if proof is None:
        inner = metrics.get("metrics")
        proof = inner.get("timestamp_equality_proof") if isinstance(inner, dict) else None

    if shard != "merged":
        if proof is not None:
            if not isinstance(proof, dict):
                errors.append("timestamp_equality_proof is not an object")
            else:
                m, t = proof.get("matched"), proof.get("total")
                if not (isinstance(m, int) and isinstance(t, int) and m == t):
                    errors.append(f"timestamp_equality_proof mismatch: {m}/{t}")
        return

    if not isinstance(proof, dict):
        errors.append("merged stage requires a numeric timestamp_equality_proof object")
        return
    matched, total = proof.get("matched"), proof.get("total")
    if not isinstance(matched, int) or isinstance(matched, bool):
        errors.append(f"timestamp_equality_proof.matched is not an integer: {matched!r}")
    if not isinstance(total, int) or isinstance(total, bool):
        errors.append(f"timestamp_equality_proof.total is not an integer: {total!r}")
    if isinstance(matched, int) and isinstance(total, int):
        if matched != total:
            errors.append(f"timestamp_equality_proof {matched}/{total} (mismatch)")
        if total < MIN_PROOF_ROWS:
            errors.append(f"timestamp_equality_proof total={total} < {MIN_PROOF_ROWS}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hard stage promotion gate")
    ap.add_argument("--stage-dir", required=True, help="directory containing metrics_and_checks.json")
    ap.add_argument("--expected-months", type=int, default=None,
                    help="override expected month count; default from metrics")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    stage_dir = Path(args.stage_dir).resolve()
    metrics_path = stage_dir / "metrics_and_checks.json"
    errors: list[str] = []

    if not metrics_path.exists():
        print(f"GATE FAIL: missing {metrics_path}")
        return 1
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"GATE FAIL: cannot parse {metrics_path}: {exc}")
        return 1
    if not isinstance(metrics, dict):
        print("GATE FAIL: metrics root is not an object")
        return 1

    # (a) checks_failed and strict boolean check values
    checks_failed = metrics.get("checks_failed")
    if not isinstance(checks_failed, int) or isinstance(checks_failed, bool):
        errors.append("checks_failed missing or not an integer")
    elif checks_failed > 0:
        errors.append(f"checks_failed={checks_failed}")

    check_count = 0
    for name, value in iter_checks(metrics):
        check_count += 1
        if not isinstance(value, bool):
            errors.append(f"check {name!r} is non-boolean: {value!r}")
        elif value is not True:
            errors.append(f"check {name!r} is false")
    if check_count == 0:
        errors.append("no checks recorded")

    # (b) artifacts: strict in-dir resolution + exact sha256/bytes verification
    raw_artifacts = metrics.get("artifacts")
    artifacts: list[Path] = []
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        errors.append("artifacts list is empty")
        raw_artifacts = []
    for item in raw_artifacts:
        if isinstance(item, str):
            cand, rec_sha, rec_bytes = item, None, None
        elif isinstance(item, dict):
            cand = item.get("path") or item.get("file") or item.get("name")
            rec_sha = item.get("sha256")
            rec_bytes = item.get("bytes")
        else:
            errors.append(f"artifact entry is not a string or object: {item!r}")
            continue
        if not cand:
            errors.append("artifact entry has no path")
            continue
        p = resolve_artifact(str(cand), stage_dir, errors)
        if p is None:
            continue
        artifacts.append(p)
        if rec_bytes is not None and int(rec_bytes) != p.stat().st_size:
            errors.append(f"artifact size mismatch for {norm_rel(p, stage_dir)}: "
                          f"recorded={rec_bytes} actual={p.stat().st_size}")
        if rec_sha is None:
            errors.append(f"artifact {norm_rel(p, stage_dir)} has no recorded sha256")
        else:
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual.lower() != str(rec_sha).lower():
                errors.append(f"artifact sha256 mismatch for {norm_rel(p, stage_dir)}")

    # (c) elapsed_s
    elapsed = metrics.get("elapsed_s")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        errors.append("elapsed_s missing or non-numeric")
    elif elapsed <= 0:
        errors.append(f"elapsed_s={elapsed} (copied/cached JSON)")

    # (d) provenance (hardened)
    prov = metrics.get("provenance")
    if not isinstance(prov, dict):
        inner = metrics.get("metrics")
        prov = inner.get("provenance") if isinstance(inner, dict) else None
    if not isinstance(prov, dict):
        errors.append("provenance block missing")
        prov = {}
    else:
        check_provenance(prov, errors)

    shard = prov.get("shard_id") if isinstance(prov, dict) else None

    # (e) month coverage
    expected = args.expected_months
    if expected is None:
        expected = get_metric(metrics, "expected_months_count")
    encoded = get_metric(metrics, "encoded_months_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        errors.append("expected_months_count missing or invalid")
    if not isinstance(encoded, int) or isinstance(encoded, bool):
        errors.append("encoded_months_count missing or invalid")
    elif isinstance(expected, int) and not isinstance(expected, bool) and encoded < expected:
        errors.append(f"encoded_months_count={encoded} < expected={expected}")

    # (f) nan_rate
    nan_rate = get_metric(metrics, "nan_rate")
    if nan_rate is None:
        errors.append("nan_rate missing")
    elif not isinstance(nan_rate, (int, float)) or isinstance(nan_rate, bool):
        errors.append(f"nan_rate non-numeric: {nan_rate!r}")
    elif float(nan_rate) > 0.0:
        errors.append(f"nan_rate={nan_rate} > 0.0")

    # (g) timestamp equality proof (hardened)
    check_timestamp_proof(metrics, shard, errors)

    # (h) CRLF in text artifacts + metrics file
    for p in artifacts:
        if not is_text(p):
            continue
        try:
            data = p.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read {p}: {exc}")
            continue
        if b"\r\n" in data:
            errors.append(f"CRLF bytes in text artifact: {norm_rel(p, stage_dir)}")
    if b"\r\n" in metrics_path.read_bytes():
        errors.append("CRLF bytes in metrics_and_checks.json")

    if errors:
        print("GATE FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    # Manifest over LF-normalized artifact bytes.
    lines = []
    seen = set()
    for p in sorted(artifacts, key=lambda x: norm_rel(x, stage_dir)):
        rel = norm_rel(p, stage_dir)
        if rel == "manifest.sha256" or rel in seen:
            continue
        seen.add(rel)
        data = p.read_bytes()
        if is_text(p):
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = hashlib.sha256(data).hexdigest()
        lines.append(f"{digest}  {rel}")
    manifest = "\n".join(lines) + "\n"
    (stage_dir / "manifest.sha256").write_text(manifest, encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"GATE PASS: {stage_dir.name} ({len(lines)} artifacts, {check_count} checks, shard={shard})")
        print(f"manifest: {stage_dir / 'manifest.sha256'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

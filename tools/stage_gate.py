#!/usr/bin/env python3
"""Hard promotion gate for a research stage.

Reads <stage-dir>/metrics_and_checks.json and refuses promotion unless every
contract check passes. On success it writes manifest.sha256 over the artifact
bytes, hashing text artifacts in LF-normalized form so the QuantStore copy and
the GitHub copy produce identical hashes.

Exit codes: 0 = pass (manifest written), 1 = blocked (no manifest written).

Usage:
    python tools/stage_gate.py --stage-dir out/stage --expected-months 68
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROVENANCE_KEYS = (
    "runner_name",
    "github_run_id",
    "hostname",
    "shard_id",
    "pid",
    "utc_timestamp",
)

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
    try:
        return path.resolve().relative_to(stage_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def artifact_paths(metrics: dict, stage_dir: Path) -> list[Path]:
    raw = metrics.get("artifacts")
    if not isinstance(raw, list) or not raw:
        return []
    out: list[Path] = []
    for item in raw:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = item.get("path") or item.get("file") or item.get("name")
        else:
            candidate = None
        if not candidate:
            continue
        p = Path(candidate)
        if not p.is_absolute():
            p = stage_dir / p
        if not p.exists():
            # Runner-local absolute paths are common; fall back to basename.
            alt = stage_dir / Path(candidate).name
            if alt.exists():
                p = alt
        out.append(p)
    return out


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

    # (a) checks_failed and boolean check values
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

    # (b) artifacts
    artifacts = artifact_paths(metrics, stage_dir)
    if not artifacts:
        errors.append("artifacts list is empty")
    for p in artifacts:
        if not p.exists():
            errors.append(f"artifact missing on disk: {p}")
        elif not p.is_file():
            errors.append(f"artifact is not a file: {p}")

    # (c) elapsed_s
    elapsed = metrics.get("elapsed_s")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        errors.append("elapsed_s missing or non-numeric")
    elif elapsed <= 0:
        errors.append(f"elapsed_s={elapsed} (copied/cached JSON)")

    # (d) provenance
    prov = metrics.get("provenance")
    if not isinstance(prov, dict):
        inner = metrics.get("metrics")
        prov = inner.get("provenance") if isinstance(inner, dict) else None
    if not isinstance(prov, dict):
        errors.append("provenance block missing")
    else:
        for key in PROVENANCE_KEYS:
            value = prov.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"provenance.{key} missing or empty")

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

    # (g) CRLF in text artifacts
    for p in artifacts:
        if not p.exists() or not p.is_file():
            continue
        if not is_text(p):
            continue
        try:
            data = p.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read {p}: {exc}")
            continue
        if b"\r\n" in data:
            errors.append(f"CRLF bytes in text artifact: {norm_rel(p, stage_dir)}")

    # metrics file itself must be LF too
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
        print(f"GATE PASS: {stage_dir.name} ({len(lines)} artifacts, {check_count} checks)")
        print(f"manifest: {stage_dir / 'manifest.sha256'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

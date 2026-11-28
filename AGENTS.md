# Wave 2A Cluster Agent Rules (Windows PowerShell only)

## Shell discipline
- Windows PowerShell / pwsh only. No bash, no `/dev/null`, no `| head`, no `2>/dev/null`.
- Long jobs must be detached. Use `Start-Process` with redirected stdout/stderr, or `python -u script.py > log.txt 2>&1` inside a detached PowerShell, then poll with `Get-Content -Tail`.
- Never block the 30-second command channel on a long-running process: upload with `put`, launch with `job`, inspect with `tail`.
- All text files created by agents use LF (`newline="\n"`). Never commit `__pycache__` or `.pyc`.

## Timeframes and shards
- ONE timeframe at a time per worker. Never run two `--tf` stages concurrently on the same node.
- Master (`node_index=0`) has `pull=false`: it dispatches, gates and merges only. Workers pull their own shard.
- The manifest has TWO shards: `0/2`, `1/2`, plus `merged`. `0/3` and `1/3` are invalid and must be rejected.
- Pull contiguous months only: `--only "Jump_10_Index_ticks_202[1-3]-*.csv.gz"` and `--only "Jump_10_Index_ticks_202[4-6]-*.csv.gz"`. Never use `pull_store.py --shard` (modulo, non-contiguous).

## Storage split
- QuantStore (`\\100.125.90.54\quantstore`, SFTP `quantstore@100.125.90.54`, writable `/data`) holds heavy `.npz` and `.csv` / `.csv.gz` artifacts.
- GitHub holds code, JSON metrics, SHA-256 manifests and HANDOFF metadata only. Never commit tick data or large arrays.
- Credentials and tokens come from secrets / environment. Never hardcode or echo them. `QUANTSTORE_PW` is injected as a GitHub secret into env only.

## Worker supervision
- `C:\cluster\worker_node.py` runs under `C:\cluster\run_worker_watchdog.ps1`, logs to `C:\cluster\worker_node.log`, and restarts 3 seconds after any exit.
- Agents do not restart, stop, or edit the worker; the watchdog owns its lifecycle.

## Wave 2A contract
- Canonical encoder: `algos/fib_engine/run_stage.py`, byte-identical on every node. Contract `wave2a-canonical-1`.
- True epoch-aligned 50%-overlap windows; global key `tf|t_open_ms|t_close_ms`.
- M=160 lattice: causal `[-4,0]`, forward `[0,+4]` in causal sigma units. Payout-agnostic, zero ONETOUCH.
- `tools/stage_gate.py` must PASS before promotion. Placeholder provenance, `0/3`, non-numeric/absent proof, runner-local artifact paths and SHA-256 mismatches are hard fails.
- Keep `mega_enabled: false`. Do not touch MEGA folders, network daemons, or live MT5 / `ChartReaderEA` unless explicitly instructed.

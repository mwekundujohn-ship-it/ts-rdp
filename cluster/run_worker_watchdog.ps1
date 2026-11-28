# Supervisor for C:\cluster\worker_node.py. Restarts the worker 3s after any exit.
# All text files written by this cluster use LF line endings.
$ErrorActionPreference = 'Continue'
$clusterDir = 'C:\cluster'
$worker = Join-Path $clusterDir 'worker_node.py'
$log = Join-Path $clusterDir 'worker_node.log'
New-Item -ItemType Directory -Force -Path $clusterDir | Out-Null

function Write-WorkerLog([string]$message) {
    $line = '[{0}] {1}' -f (Get-Date -Format o), $message
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

# Only one supervisor may own the worker.
$others = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
    Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*run_worker_watchdog.ps1*' })
if ($others.Count -gt 0) {
    Write-WorkerLog "another watchdog is already running (pid=$($others[0].ProcessId)); exit"
    exit 0
}

Write-WorkerLog "watchdog start pid=$PID worker=$worker"
while ($true) {
    $code = 0
    try {
        Write-WorkerLog 'worker start'
        & python -u $worker *>> $log
        $code = $LASTEXITCODE
    }
    catch {
        Write-WorkerLog ("worker exception: " + $_.Exception.Message)
        $code = -1
    }
    Write-WorkerLog "worker exit code=$code; restart in 3s"
    Start-Sleep -Seconds 3
}

param(
  [string]$ConfigPath = (Join-Path $PSScriptRoot "config.json"),
  [switch]$Connect,
  [switch]$NoConnect,
  [switch]$Replace,
  [switch]$Cluster,
  [switch]$UseFleetToken,
  [string[]]$Roles = @("master", "worker1", "worker2"),
  [int]$PollSeconds = 10,
  [int]$ReadyTimeoutMinutes = 15
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) { throw "config not found: $ConfigPath" }
$script:cfg = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ($env:GITHUB_TOKEN) { $script:cfg.token = $env:GITHUB_TOKEN }
if ($env:TS_AUTHKEY) { $script:cfg.authkey = $env:TS_AUTHKEY }
if ($UseFleetToken -and $script:cfg.fleetToken) { $script:cfg.token = $script:cfg.fleetToken }
if ($UseFleetToken -and $script:cfg.fleetOwner) { $script:cfg.owner = $script:cfg.fleetOwner }

$script:ApiBase = "https://api.github.com/repos/$($script:cfg.owner)/$($script:cfg.repo)"

function Gh-Headers {
  @{ Authorization = "token $($script:cfg.token)"; "User-Agent" = "rdp-auto"; Accept = "application/vnd.github+json" }
}

function Get-Runs {
  $u = "$script:ApiBase/actions/workflows/$($script:cfg.workflow)/runs?per_page=20"
  (Invoke-RestMethod -Uri $u -Headers (Gh-Headers)).workflow_runs
}

function Get-Run([string]$id) {
  Invoke-RestMethod -Uri "$script:ApiBase/actions/runs/$id" -Headers (Gh-Headers)
}

function Get-ActiveRuns {
  @(Get-Runs | Where-Object { $_.status -eq 'queued' -or $_.status -eq 'in_progress' })
}

function Get-RemoteStatus([string]$file) {
  try {
    $c = Invoke-RestMethod -Uri "$script:ApiBase/contents/$file" -Headers (Gh-Headers)
    $txt = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($c.content))
    $txt | ConvertFrom-Json
  } catch { $null }
}

function Start-RdpRun([string]$nodeRole) {
  $fleet = if ($script:cfg.fleet) { [string]$script:cfg.fleet } else { "alpha" }
  $inputs = [ordered]@{
    fleet = $fleet
    node_role = [string]$nodeRole
    duration_minutes = [string]$script:cfg.durationMinutes
    rdp_password = [string]$script:cfg.password
    ts_authkey = [string]$script:cfg.authkey
    nvidia_api_key = [string]$script:cfg.nvidiaKey
  }
  if ($script:cfg.mt5Login) { $inputs.mt5_login = [string]$script:cfg.mt5Login }
  if ($script:cfg.mt5Password) { $inputs.mt5_password = [string]$script:cfg.mt5Password }
  if ($script:cfg.mt5Server) { $inputs.mt5_server = [string]$script:cfg.mt5Server }
  if ($script:cfg.megaEmail -and $script:cfg.megaEmail -notmatch '^<') { $inputs.mega_email = [string]$script:cfg.megaEmail }
  if ($script:cfg.megaPassword -and $script:cfg.megaPassword -notmatch '^<') { $inputs.mega_password = [string]$script:cfg.megaPassword }
  $body = @{ ref = $script:cfg.ref; inputs = $inputs } | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri "$script:ApiBase/actions/workflows/$($script:cfg.workflow)/dispatches" -Method Post -Headers (Gh-Headers) -Body $body -ContentType "application/json"
}

function Stop-Run([string]$id) {
  try { Invoke-RestMethod -Uri "$script:ApiBase/actions/runs/$id/cancel" -Method Post -Headers (Gh-Headers) | Out-Null } catch {}
}

function Test-Port([string]$ip, [int]$port, [int]$timeoutMs = 4000) {
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $ar = $client.BeginConnect($ip, $port, $null, $null)
    if (-not $ar.AsyncWaitHandle.WaitOne($timeoutMs)) { return $false }
    $client.EndConnect($ar)
    return $true
  } catch { return $false } finally { $client.Close() }
}

function Get-DiesAt($createdAt) {
  $created = [DateTime]::Parse($createdAt).ToUniversalTime()
  $created.AddMinutes([double]$script:cfg.durationMinutes + 2)
}

function Ensure-Tailnet {
  $exe = $script:cfg.tailscaleExe
  if (-not $exe -or -not (Test-Path -LiteralPath $exe)) { return }
  if (-not $script:cfg.tailnet) { return }
  $raw = & $exe switch --list --json 2>$null | Out-String
  if (-not $raw.Trim()) { return }
  try { $profiles = $raw | ConvertFrom-Json } catch { return }
  $active = $profiles | Where-Object { $_.selected } | Select-Object -First 1
  if ($active -and $active.tailnet -eq $script:cfg.tailnet) { return }
  $target = $profiles | Where-Object { $_.tailnet -eq $script:cfg.tailnet -or $_.nickname -eq $script:cfg.tailnet } | Select-Object -First 1
  if (-not $target) {
    Write-Host "tailnet '$($script:cfg.tailnet)' not found; active profile is '$($active.tailnet)'"
    return
  }
  Write-Host "switching tailnet -> $($target.tailnet)"
  & $exe switch $target.id | Out-Null
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 750
    try { $r = & $exe switch --list --json 2>$null | Out-String | ConvertFrom-Json } catch { continue }
    $a = $r | Where-Object { $_.selected } | Select-Object -First 1
    if ($a -and $a.tailnet -eq $script:cfg.tailnet) { return }
  }
}

function Wait-Ready([string]$runId, [int]$timeoutMin, [string]$file) {
  if (-not $file) { $file = "status-$runId.json" }
  $deadline = (Get-Date).AddMinutes($timeoutMin)
  while ((Get-Date) -lt $deadline) {
    $st = Get-RemoteStatus $file
    if ($st -and ([string]$st.run_id -eq $runId)) {
      if ($st.status -eq 'ready') { return $st }
      if ($st.status -eq 'failed' -or $st.status -eq 'wrong_tailnet') { throw "run reported '$($st.status)'" }
    }
    try {
      $r = Get-Run $runId
      if ($r.status -eq 'completed' -and $r.conclusion -ne 'success') { throw "run $runId finished '$($r.conclusion)' before ready" }
    } catch {
      if ($_.Exception.Message -match 'finished') { throw }
    }
    Start-Sleep -Seconds $PollSeconds
  }
  throw "timeout waiting for run $runId to become ready"
}

function Wait-ForRunCreated([int64]$before) {
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 3
    $r = @(Get-Runs | Where-Object { [int64]$_.id -gt $before } | Sort-Object { [int64]$_.id })
    if ($r.Count -gt 0) { return $r[-1] }
  }
  throw "dispatch did not create a run"
}

function Start-NewRdp([switch]$Replace) {
  Ensure-Tailnet
  $existing = @(Get-Runs | ForEach-Object { [int64]$_.id })
  $before = 0
  if ($existing.Count -gt 0) { $before = ($existing | Measure-Object -Maximum).Maximum }
  $old = @(Get-ActiveRuns | ForEach-Object { [string]$_.id })
  $role = if ($script:cfg.nodeRole) { [string]$script:cfg.nodeRole } else { "master" }
  Start-RdpRun $role | Out-Null
  $newRun = Wait-ForRunCreated $before
  Write-Host "run $($newRun.id) [$role] queued; waiting for tailnet..."
  $st = Wait-Ready ([string]$newRun.id) $ReadyTimeoutMinutes $null
  for ($i = 0; $i -lt 30; $i++) {
    if (Test-Port $st.ip 3389) { break }
    Start-Sleep -Seconds 3
  }
  if ($Replace) {
    foreach ($id in $old) { if ($id -ne [string]$newRun.id) { Stop-Run $id } }
  }
  [pscustomobject]@{
    RunId = [string]$newRun.id
    Role = $role
    CreatedAt = $newRun.created_at
    Status = $st
    Ready = $true
    DiesAt = (Get-DiesAt $newRun.created_at)
  }
}

function Start-NewCluster([switch]$Replace) {
  Ensure-Tailnet
  $existing = @(Get-Runs | ForEach-Object { [int64]$_.id })
  $before = 0
  if ($existing.Count -gt 0) { $before = ($existing | Measure-Object -Maximum).Maximum }
  $old = @(Get-ActiveRuns | ForEach-Object { [string]$_.id })
  $launched = @()
  foreach ($role in $Roles) {
    $existing = @(Get-Runs | ForEach-Object { [int64]$_.id })
    $before = 0
    if ($existing.Count -gt 0) { $before = ($existing | Measure-Object -Maximum).Maximum }
    Start-RdpRun $role | Out-Null
    $newRun = Wait-ForRunCreated $before
    $launched += [pscustomobject]@{ Role = [string]$role; RunId = [string]$newRun.id; CreatedAt = $newRun.created_at }
    Write-Host "run $($newRun.id) [$role] queued"
  }
  $nodes = @()
  $fleet = if ($script:cfg.fleet) { [string]$script:cfg.fleet } else { "alpha" }
  foreach ($node in $launched) {
    $st = Wait-Ready $node.RunId $ReadyTimeoutMinutes "node-$fleet-$($node.Role).json"
    for ($i = 0; $i -lt 30; $i++) {
      if ((Test-Port $st.ip 5555) -and ($node.Role -ne 'master' -or (Test-Port $st.ip 3389))) { break }
      Start-Sleep -Seconds 3
    }
    Write-Host "$($node.Role): $($st.ip) ($($st.host))"
    $nodes += [pscustomobject]@{
      Role = $node.Role
      RunId = $node.RunId
      Ip = $st.ip
      Host = $st.host
      Self = $st.self
      Ready = $true
      DiesAt = (Get-DiesAt $node.CreatedAt)
    }
  }
  if ($Replace) {
    $newIds = @($launched | ForEach-Object { $_.RunId })
    foreach ($id in $old) { if ($newIds -notcontains $id) { Stop-Run $id } }
  }
  $nodes | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot "cloud_nodes.json") -Encoding UTF8
  $masterNode = $nodes | Where-Object { $_.Role -eq 'master' } | Select-Object -First 1
  if (-not $masterNode) { throw "master role missing from cluster" }
  $masterRun = Get-Run $masterNode.RunId
  [pscustomobject]@{
    RunId = $masterNode.RunId
    Role = "master"
    CreatedAt = $masterRun.created_at
    Status = [pscustomobject]@{ ip = $masterNode.Ip; self = $masterNode.Self; host = $masterNode.Host; user = "rdp" }
    Ready = $true
    DiesAt = $masterNode.DiesAt
  }
}

function Save-RdpCreds([string]$ip) {
  $target = "TERMSRV/$ip"
  $pass = [string]$script:cfg.password
  $cmdArgs = @("/add:$target", "/user:rdp", "/pass:$pass")
  & cmdkey /delete:$target 2>$null | Out-Null
  & cmdkey @cmdArgs | Out-Null
}

function Show-Rdp($cur) {
  $ip = [string]$cur.Status.ip
  $hostName = ([string]$cur.Status.self).TrimEnd('.')
  $left = [int][Math]::Max(0, ($cur.DiesAt.ToUniversalTime() - (Get-Date).ToUniversalTime()).TotalMinutes)
  $model = if ($cur.Status.model) { [string]$cur.Status.model } else { "nvidia/nemotron-3-ultra-550b-a55b" }
  $variant = if ($cur.Status.variant) { [string]$cur.Status.variant } else { "high" }
  $lines = @(
    "role: $($cur.Role)",
    "host: $ip ($hostName)",
    "user: $($cur.Status.user)",
    "pass: $($script:cfg.password)",
    "opencode: ready ($model#$variant)",
    "connect: mstsc /v:$ip /f",
    "expires: $($cur.DiesAt.ToString('yyyy-MM-dd HH:mm'))Z (~$left min left)"
  )
  $lines | ForEach-Object { Write-Host $_ }
  $rdpFile = Join-Path $PSScriptRoot "rdp-$($cur.RunId).rdp"
  try {
    $text = $lines -join "`r`n"
    Set-Content -LiteralPath (Join-Path $PSScriptRoot "last-rdp.txt") -Value $text -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $PSScriptRoot "rdp-$($cur.RunId).txt") -Value $text -Encoding ASCII
    $rdpText = @(
      "screen mode id:i:2",
      "full address:s:$ip",
      "username:s:rdp",
      "authentication level:i:2",
      "enablecredsspsupport:i:1"
    ) -join "`r`n"
    Set-Content -LiteralPath $rdpFile -Value $rdpText -Encoding ASCII
  } catch {}
  if (-not $NoConnect) {
    Save-RdpCreds $ip
    Start-Sleep -Milliseconds 500
    Start-Process "mstsc.exe" -ArgumentList "`"$rdpFile`" /f"
  }
}

$cur = if ($Cluster) { Start-NewCluster -Replace:$Replace } else { Start-NewRdp -Replace:$Replace }
Show-Rdp $cur

[CmdletBinding()]
param([string]$Preset = 'rfc-01')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$shellExe = (Get-Process -Id $PID).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$runRoot = Join-Path $projectRoot ".artifacts\cold-start\$runId"
$backupRoot = Join-Path $projectRoot ".backup_indexes\$runId"
$backupPath = Join-Path $backupRoot 'original'
$testPath = Join-Path $backupRoot 'test-generated'
function Assert-WorkspacePath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside workspace: $resolved" }
    $ancestor = $resolved
    while ($ancestor -and $ancestor -ne $projectRoot) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point forbidden: $ancestor" }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
}
function Get-Manifest([string]$Path) {
    $records = @()
    foreach ($item in @(Get-ChildItem -LiteralPath $Path -Force -Recurse | Sort-Object FullName)) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point in index: $($item.FullName)" }
        if (-not $item.PSIsContainer) {
            $records += @{ path=$item.FullName.Substring($Path.Length); bytes=$item.Length; sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash }
        }
    }
    return ConvertTo-Json -InputObject $records -Depth 4 -Compress
}
$moved = $false
$testStarted = $false
$restored = $false
$failure = $null
Push-Location $projectRoot
try {
    $settingsJson = & $pythonExe -c "from settings import RAGSettings; import json; s=RAGSettings(); print(json.dumps({'index':str(s.qdrant_storage_path),'documents':str(s.document_root)}))"
    if ($LASTEXITCODE -ne 0) { throw 'Settings validation failed.' }
    $config = $settingsJson | ConvertFrom-Json
    $indexPath = [IO.Path]::GetFullPath($config.index)
    foreach ($path in @($indexPath,$backupRoot,$backupPath,$testPath,$runRoot)) { Assert-WorkspacePath $path }
    $docsPath = [IO.Path]::GetFullPath($config.documents)
    if ($docsPath -eq $indexPath -or $docsPath.StartsWith($indexPath + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Index path includes the document corpus.' }
    if (-not (Test-Path -LiteralPath $indexPath -PathType Container)) { throw 'This rollback test requires an existing working index.' }
    & $shellExe -NoProfile -File (Join-Path $PSScriptRoot 'stop_rag.ps1') -Force
    if ($LASTEXITCODE -ne 0) { throw 'Unable to stop the owned server.' }
    if (@(Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue).Count -gt 0) { throw 'Port 8080 still occupied; no unowned process will be terminated.' }
    if (@(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'").Count -gt 0) { throw 'Another llama-server is active; cold-start precondition failed.' }
    New-Item -ItemType Directory -Path $backupRoot,$runRoot | Out-Null
    $before = Get-Manifest $indexPath
    $before | Set-Content -LiteralPath (Join-Path $runRoot 'manifest-before.json') -Encoding UTF8
    @{ index=$indexPath; backup=$backupPath; test=$testPath } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'recovery.json') -Encoding UTF8
    # All absolute source/destination paths were checked above; no cross-shell moves.
    Move-Item -LiteralPath $indexPath -Destination $backupPath
    $moved = $true
    if (Test-Path -LiteralPath $indexPath) { throw 'Index still exists before cold start.' }
    $testStarted = $true
    $timer = [Diagnostics.Stopwatch]::StartNew()
    if ($Preset -notmatch '^[A-Za-z0-9_-]+$') { throw 'Preset must be an alphanumeric dataset ID.' }
    $launchArgs = @('-NoProfile','-File', ('"' + (Join-Path $PSScriptRoot 'start_rag.ps1') + '"'), '-Preset',$Preset,'-Export',('"' + (Join-Path $runRoot 'demo.json') + '"'),'-NoColor')
    # Wait for the launcher PID, not inherited native-output pipe handles held by the server.
    $launcher = Start-Process -FilePath $shellExe -ArgumentList $launchArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'cold-start.log') -RedirectStandardError (Join-Path $runRoot 'cold-start.stderr.log')
    if (-not $launcher.WaitForExit(1800000)) { throw 'Launcher timeout; inspect running processes before manual recovery.' }
    $startCode = $launcher.ExitCode
    $wallSeconds = $timer.Elapsed.TotalSeconds
    Copy-Item -LiteralPath (Join-Path $projectRoot '.artifacts\stage15\launch.json') -Destination (Join-Path $runRoot 'launch.json')
    Copy-Item -LiteralPath (Join-Path $projectRoot '.artifacts\stage15\llama.stderr.log') -Destination (Join-Path $runRoot 'llama.stderr.log')
    if ($startCode -ne 0) { throw "Cold start failed with exit $startCode; see $runRoot" }
    $launch = Get-Content -LiteralPath (Join-Path $runRoot 'launch.json') -Raw | ConvertFrom-Json
    if (-not $launch.server_started -or -not $launch.ready) { throw 'Cold server startup was not recorded.' }
    $response = (Get-Content -LiteralPath (Join-Path $runRoot 'demo.json') -Raw | ConvertFrom-Json).response
    if ($response.refused) { throw 'The first query was refused; inspect evidence.' }
    $indexLines = @(Get-Content -LiteralPath (Join-Path $runRoot 'cold-start.log') | Where-Object { $_.StartsWith('{"action":') })
    if ($indexLines.Count -ne 1) { throw 'Missing structured indexing statistics.' }
    $indexStats = $indexLines[0] | ConvertFrom-Json
    if ($indexStats.action -ne 'indexed' -or $indexStats.indexed_chunks -le 0) { throw 'Fresh indexing was not performed.' }
    @{ launcher_wall_seconds=$wallSeconds; first_query_ms=$response.execution_stats.total_ms; readiness_seconds=$launch.readiness_seconds; indexing_ms=$indexStats.elapsed_ms; files=$indexStats.processed_files; chunks=$indexStats.indexed_chunks; response_refused=$response.refused } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'timings.json') -Encoding UTF8
} catch { $failure = $_.Exception.Message }
finally {
    if ($testStarted) {
        & $shellExe -NoProfile -File (Join-Path $PSScriptRoot 'stop_rag.ps1') -Force
        if ($LASTEXITCODE -ne 0) { $failure = "$failure; server stop failed" }
    }
    if ($moved) {
        try {
            foreach ($path in @($indexPath,$backupPath,$testPath)) { Assert-WorkspacePath $path }
            if (Test-Path -LiteralPath $indexPath) { Move-Item -LiteralPath $indexPath -Destination $testPath }
            Move-Item -LiteralPath $backupPath -Destination $indexPath
            $after = Get-Manifest $indexPath
            $after | Set-Content -LiteralPath (Join-Path $runRoot 'manifest-after.json') -Encoding UTF8
            if ($before -cne $after) { throw 'Restored index manifest differs; test copy retained.' }
            $restored = $true
            # Only the disposable test index is removed, after byte-hash verified restoration.
            if (Test-Path -LiteralPath $testPath) {
                Assert-WorkspacePath $testPath
                $null = Get-Manifest $testPath
                Remove-Item -LiteralPath $testPath -Recurse -Force
            }
        } catch { $failure = "$failure; rollback: $($_.Exception.Message). Recovery paths: $backupRoot" }
    }
    if (Test-Path -LiteralPath $runRoot) {
        @{ restored=$restored; error=$failure; run=$runRoot } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'result.json') -Encoding UTF8
    }
    Pop-Location
}
Write-Host "Cold-start artifacts: $runRoot"
if ($failure) { [Console]::Error.WriteLine($failure); exit 1 }
if (-not $restored) { [Console]::Error.WriteLine('No verified rollback.'); exit 1 }
Write-Host 'Cold start passed; working index restored with matching SHA-256 manifest.'

[CmdletBinding()]
param([switch]$Force)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateFile = Join-Path $projectRoot '.artifacts\stage15\server-state.json'
try {
    if (-not (Test-Path -LiteralPath $stateFile)) { Write-Host 'No launcher-owned server recorded.'; exit 0 }
    $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    $expectedExe = Join-Path $projectRoot 'tools\llama\llama-server.exe'
    if ([int]$state.Pid -le 0 -or $state.Path -ne $expectedExe) { throw 'Invalid server identity.' }
    $serverProcess = Get-Process -Id ([int]$state.Pid) -ErrorAction SilentlyContinue
    if ($null -eq $serverProcess) { Write-Host 'Recorded server has already exited.'; exit 0 }
    $recordedStart = ([DateTime]$state.StartTime).ToUniversalTime()
    if ($serverProcess.Path -ne $expectedExe -or $serverProcess.StartTime.ToUniversalTime().Ticks -ne $recordedStart.Ticks) { throw 'Process identity changed; refusing to close it.' }
    if (-not $serverProcess.CloseMainWindow() -or -not $serverProcess.WaitForExit(5000)) {
        if (-not $Force) { throw "No graceful window shutdown available. Server PID $($serverProcess.Id) remains running; use -Force to terminate this verified process." }
        Write-Host "Force stopping verified server PID $($serverProcess.Id)."
        $serverProcess.Kill()
        if (-not $serverProcess.WaitForExit(10000)) { throw 'Server did not exit.' }
    }
    Write-Host 'Server stopped. Indexes retained.'
} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }

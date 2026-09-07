[CmdletBinding()]
param(
    [string]$Query, [string]$Preset,
    [ValidateSet('golden','holdout')][string]$Dataset = 'golden',
    [string]$Export, [switch]$IndexOnly, [switch]$NoColor, [switch]$Cli,
    [ValidateRange(1024,65535)][int]$WebPort = 8501,
    [ValidateRange(10,600)][int]$TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $projectRoot '.artifacts\stage15'
$launchTimer = [Diagnostics.Stopwatch]::StartNew()
$serverStarted = $false
function Test-Ready {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 3
        $models = Invoke-RestMethod 'http://127.0.0.1:8080/v1/models' -TimeoutSec 3
        return ($health.status -eq 'ok' -and @($models.data).Count -gt 0)
    } catch { return $false }
}
function Test-ExistingWeb {
    $webListeners = @(Get-NetTCPConnection -LocalPort $WebPort -State Listen -ErrorAction SilentlyContinue)
    if ($webListeners.Count -eq 0) { return $false }
    foreach ($listener in $webListeners) {
        $webProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
        if ($null -eq $webProcess -or $webProcess.CommandLine -notmatch '-m\s+streamlit\s+run\s+') {
            throw "Мы не запускаем второй интерфейс: порт $WebPort занят другим процессом."
        }
        $appPath = Join-Path $projectRoot 'app.py'
        $ourProcess = $webProcess.CommandLine.Contains($appPath)
        # Мы узнаём также прежний запуск с относительным app.py по родителю launcher.
        $ancestor = $webProcess
        for ($depth = 0; -not $ourProcess -and $depth -lt 4; $depth++) {
            $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId = $($ancestor.ParentProcessId)"
            if ($null -eq $ancestor) { break }
            $ourProcess = $null -ne $ancestor.CommandLine -and $ancestor.CommandLine.Contains($PSCommandPath)
        }
        if (-not $ourProcess) { throw "Мы не распознали наш интерфейс на порту $WebPort; оставляем процесс без изменений." }
    }
    try {
        $webHealth = Invoke-WebRequest "http://127.0.0.1:$WebPort/_stcore/health" -UseBasicParsing -TimeoutSec 5
        if ($webHealth.StatusCode -eq 200 -and $webHealth.Content.Trim() -eq 'ok') { return $true }
    } catch { }
    throw "Наш Web-процесс уже работает, но пока не отвечает на проверку готовности. Мы не открываем второй индекс."
}
Push-Location $projectRoot
try {
    if ($Query -and $Preset) { throw 'Query and Preset are mutually exclusive.' }
    if ($IndexOnly -and ($Query -or $Preset -or $Export)) { throw 'IndexOnly cannot include a query or export.' }
    if ($Export -and [IO.Path]::GetExtension($Export) -ne '.json') { throw 'Export must end in .json.' }
    $pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Missing .venv Python.' }
    $webMode = -not ($Cli -or $IndexOnly -or $Query -or $Preset -or $Export)
    $webAlreadyRunning = $false
    if ($webMode) { $webAlreadyRunning = Test-ExistingWeb }
    if ($webMode -and -not $webAlreadyRunning) {
        & $pythonExe -c "import streamlit"
        if ($LASTEXITCODE -ne 0) { throw 'Мы не нашли Streamlit. Установим зависимости: python -m pip install -e .[web]' }
    }
    New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
    if (-not (Test-Ready)) {
        $listeners = @(Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue)
        $startedServer = $null
        if ($listeners.Count -eq 0) {
            $serverExe = Join-Path $projectRoot 'tools\llama\llama-server.exe'
            $modelFile = Join-Path $projectRoot 'models\llm\Qwen2.5-14B-Instruct-Q4_K_M.gguf'
            if (-not (Test-Path -LiteralPath $serverExe) -or -not (Test-Path -LiteralPath $modelFile)) { throw 'Missing local server or GGUF.' }
            $serverArgs = @('-m', ('"' + $modelFile + '"'), '-ngl','35','-c','4096','-b','128','-ub','64','-np','1','--port','8080','--host','127.0.0.1')
            $startedServer = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $artifactRoot 'llama.stdout.log') -RedirectStandardError (Join-Path $artifactRoot 'llama.stderr.log')
            $serverStarted = $true
            Write-Host "Cold server start: PID $($startedServer.Id); -ngl 35 -c 4096 -b 128 -ub 64 -np 1"
            @{ Pid=$startedServer.Id; StartTime=$startedServer.StartTime.ToUniversalTime().ToString('o'); Path=$serverExe } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $artifactRoot 'server-state.json') -Encoding UTF8
        }
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        do {
            if ($null -ne $startedServer -and $startedServer.HasExited) { throw 'Server exited; inspect .artifacts/stage15/llama.stderr.log.' }
            if (Test-Ready) { break }
            Start-Sleep -Milliseconds 500
        } while ([DateTime]::UtcNow -lt $deadline)
        if (-not (Test-Ready)) { throw 'Endpoint not ready. Inspect listener and logs; any started server remains recorded in server-state.json.' }
    }
    Write-Host 'LLM ready: http://127.0.0.1:8080/v1'
    @{ server_started=$serverStarted; readiness_seconds=$launchTimer.Elapsed.TotalSeconds; ready=$true } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $artifactRoot 'launch.json') -Encoding UTF8
    if ($webAlreadyRunning) {
        Write-Host "Наш интерфейс уже работает: http://127.0.0.1:$WebPort — мы используем открытый индекс."
        $resultCode = 0
    } else {
    $demoArgs = @('-X','utf8','demo.py','--ensure-index','--dataset',$Dataset)
    if ($Query) { $demoArgs += @('--query',$Query) }
    if ($Preset) { $demoArgs += @('--preset',$Preset) }
    if ($Export) { $demoArgs += @('--export',$Export) }
    if ($IndexOnly -or $webMode) { $demoArgs += '--index-only' }
    if ($NoColor) { $demoArgs += '--no-color' }
    & $pythonExe @demoArgs
    $resultCode = $LASTEXITCODE
    if ($resultCode -eq 0 -and $webMode) {
        Write-Host "Мы открываем наш интерфейс: http://127.0.0.1:$WebPort"
        & $pythonExe -m streamlit run (Join-Path $projectRoot 'app.py') --server.address 127.0.0.1 --server.port $WebPort --server.headless true --browser.gatherUsageStats false
        $resultCode = $LASTEXITCODE
    }
    }
} catch { [Console]::Error.WriteLine($_.Exception.Message); $resultCode = 1 }
finally { Pop-Location }
exit $resultCode

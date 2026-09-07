"""Мы проверяем повторный запуск launcher без серверов, моделей и рабочего индекса."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Мы проверяем Windows PowerShell 5.1")
@pytest.mark.parametrize(
    "scenario,expected", [("ours", 0), ("free", 0), ("foreign", 2), ("unready", 2)]
)
def test_existing_web_detection(scenario: str, expected: int) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "start_rag.ps1"
    assert script.read_bytes().startswith(b"\xef\xbb\xbf")
    # Мы извлекаем функцию из настоящего скрипта, а внешние процессы/HTTP подменяем.
    command = r"""
$projectRoot = '__ROOT__'
$WebPort = 8501
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('__SCRIPT__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { exit 10 }
$fn = $ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-ExistingWeb'}, $true)
Invoke-Expression $fn.Extent.Text
function Get-NetTCPConnection { if ('__CASE__' -ne 'free') { [pscustomobject]@{OwningProcess=42} } }
function Get-CimInstance {
    $cmd = if ('__CASE__' -eq 'foreign') { 'another-service' } else { '-m streamlit run ' + (Join-Path $projectRoot 'app.py') }
    [pscustomobject]@{CommandLine=$cmd; ParentProcessId=0}
}
function Invoke-WebRequest {
    if ('__CASE__' -eq 'unready') { throw 'Our simulated HTTP failure' }
    [pscustomobject]@{StatusCode=200; Content='ok'}
}
try {
    $result = Test-ExistingWeb
    $wanted = '__CASE__' -ne 'free'
    if ($result -ne $wanted) { exit 11 }
    exit 0
} catch { exit 2 }
"""
    command = command.replace("__ROOT__", str(root).replace("'", "''"))
    command = command.replace("__SCRIPT__", str(script).replace("'", "''"))
    command = command.replace("__CASE__", scenario)
    shell = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run(
        [str(shell), "-NoProfile", "-Command", command], capture_output=True, timeout=20
    )
    assert result.returncode == expected, result.stderr

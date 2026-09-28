[CmdletBinding()]
param(
    [string]$Environment = '.venv-rfdetr',
    [string]$PythonLauncher = 'py'
)

$ErrorActionPreference = 'Stop'
if (Test-Path $Environment) {
    throw "Environment already exists: $Environment. Delete it explicitly only if you want a clean reinstall."
}
& $PythonLauncher -3.11 -m venv $Environment
if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.11 environment.' }
$EnvironmentPython = Join-Path $Environment 'Scripts\python.exe'
& $EnvironmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
& $EnvironmentPython -m pip install -r training\rfdetr\requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install RF-DETR dependencies.' }
Write-Host "RF-DETR environment ready: $EnvironmentPython"

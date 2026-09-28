param([switch]$Cameras)
$ErrorActionPreference = 'Stop'
$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ThisDir '..\..')).Path
$IsaacRoot = (Resolve-Path (Join-Path $ProjectRoot '..\..\..\..')).Path
Push-Location $ProjectRoot
try {
    $PreviewArgs = @((Join-Path $ProjectRoot 'scripts\p4.py'), 'tool', 'preview_env')
    if ($Cameras) { $PreviewArgs += '--cameras' }
    & (Join-Path $IsaacRoot '_isaac_sim\python.bat') @PreviewArgs
} finally { Pop-Location }

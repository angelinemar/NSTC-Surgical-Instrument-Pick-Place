param([switch]$Cameras)
$ErrorActionPreference = 'Stop'
$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IsaacRoot = (Resolve-Path (Join-Path $ThisDir '..\..\..\..')).Path
Push-Location $ThisDir
try {
    $PreviewArgs = @((Join-Path $ThisDir 'preview_env.py'))
    if ($Cameras) { $PreviewArgs += '--cameras' }
    & (Join-Path $IsaacRoot '_isaac_sim\python.bat') @PreviewArgs
} finally { Pop-Location }

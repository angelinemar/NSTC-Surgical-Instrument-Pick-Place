[CmdletBinding()]
param([ValidateSet('panel','record')][string]$Mode='panel', [string]$Output,
      [int]$Scenes=100, [int]$MinObjects=8, [int]$MaxObjects=18,
      [int]$RingCameras=8, [int]$Seed=42, [switch]$Headless, [switch]$Resume)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'panel') { & python detection\panel.py }
    else {
        if (-not $Output) { throw '-Output is required for record mode.' }
        $arguments=@('detection\record.py','--output',$Output,'--scenes',$Scenes,
            '--min-objects',$MinObjects,'--max-objects',$MaxObjects,'--ring-cameras',$RingCameras,'--seed',$Seed)
        if ($Headless) { $arguments+='--headless' }
        if ($Resume) { $arguments+='--resume' }
        & C:\IsaacLab\_isaac_sim\python.bat @arguments
    }
    $result=$LASTEXITCODE
} finally { Pop-Location }
exit $result

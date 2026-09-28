param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('scalpel','scissor','love_retractor','kelly','scalpel_type2')]
    [string]$Object,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RecorderArgs
)

$ErrorActionPreference = 'Stop'
$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IsaacRoot = (Resolve-Path (Join-Path $ThisDir '..\..\..\..')).Path
$PythonBat = Join-Path $IsaacRoot '_isaac_sim\python.bat'
& $PythonBat (Join-Path $ThisDir 'record.py') --object $Object @RecorderArgs
exit $LASTEXITCODE


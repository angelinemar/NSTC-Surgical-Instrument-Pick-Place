[CmdletBinding()]
param(
    [ValidateSet('check', 'objects', 'record-dry-run', 'panel', 'validate', 'export', 'train-smoke')]
    [string]$Mode = 'check',
    [string]$Object,
    [string]$RunDirectory,
    [string]$Source,
    [string]$Output,
    [string]$Dataset,
    [ValidateSet('pick', 'place')]
    [string]$Skill
)

$ErrorActionPreference = 'Stop'
$IsaacPython = 'C:\IsaacLab\_isaac_sim\python.bat'

function Require-Value([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '^<.*>$') {
        throw "-$Name is required. Replace the matching README placeholder with a real value."
    }
}

switch ($Mode) {
    'check' { & python scripts\check_structure.py }
    'objects' { & python scripts\p4.py record --list-objects }
    'record-dry-run' { Require-Value Object $Object; & python scripts\p4.py record --object $Object --dry-run }
    'panel' { & python scripts\p4.py panel }
    'validate' { Require-Value RunDirectory $RunDirectory; & $IsaacPython scripts\p4.py tool validate_feedback_dataset $RunDirectory }
    'export' { Require-Value Source $Source; Require-Value Output $Output; & $IsaacPython training\export_sensor_only.py --source $Source --output $Output --validation-cells 1 8 }
    'train-smoke' { Require-Value Dataset $Dataset; Require-Value Skill $Skill; $run = Join-Path 'training\runs' ("{0}_smoke" -f $Skill); & $IsaacPython training\train_sensor_policy.py --dataset $Dataset --output $run --skill $Skill --steps 3 --batch-size 1 --smoke }
}

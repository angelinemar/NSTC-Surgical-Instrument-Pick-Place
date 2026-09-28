[CmdletBinding()]
param(
    [ValidateSet('check', 'objects', 'record-dry-run', 'panel', 'validate', 'export', 'train-smoke', 'rfdetr-setup', 'rfdetr-check', 'rfdetr-train')]
    [string]$Mode = 'check',
    [string]$Object,
    [string]$RunDirectory,
    [string]$Source,
    [string]$Output,
    [string]$Dataset,
    [ValidateSet('pick', 'place')]
    [string]$Skill,
    [ValidateSet('nano', 'small', 'medium', 'large')]
    [string]$RFDetrModel = 'small',
    [int]$Epochs = 50,
    [int]$BatchSize = 4,
    [int]$GradAccumSteps = 4,
    [string]$RFDetrEnvironment = '.venv-rfdetr'
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
    'export' { Require-Value Source $Source; Require-Value Output $Output; & $IsaacPython training\export_recordings.py --source $Source --output $Output --purpose both }
    'train-smoke' { Require-Value Dataset $Dataset; Require-Value Skill $Skill; $run = Join-Path 'training\runs' ("{0}_smoke" -f $Skill); & $IsaacPython training\train_sensor_policy.py --dataset $Dataset --output $run --skill $Skill --steps 3 --batch-size 1 --smoke }
    'rfdetr-setup' { & scripts\launchers\setup_rfdetr.ps1 -Environment $RFDetrEnvironment }
    'rfdetr-check' { Require-Value Dataset $Dataset; & python training\rfdetr_pipeline.py --dataset $Dataset --check-only }
    'rfdetr-train' {
        Require-Value Dataset $Dataset
        Require-Value Output $Output
        $RFDetrPython = Join-Path $RFDetrEnvironment 'Scripts\python.exe'
        if (-not (Test-Path $RFDetrPython)) { throw "RF-DETR environment not found. Run: .\RUNME.ps1 -Mode rfdetr-setup" }
        & $RFDetrPython training\rfdetr_pipeline.py --dataset $Dataset --output $Output --model $RFDetrModel --epochs $Epochs --batch-size $BatchSize --grad-accum-steps $GradAccumSteps
    }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

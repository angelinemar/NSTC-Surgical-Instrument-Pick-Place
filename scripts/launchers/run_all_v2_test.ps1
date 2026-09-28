param(
    [ValidateRange(1, 1000000)]
    [int]$Episodes = 1,
    [ValidateRange(0, 1000000)]
    [int]$MaxAttempts = 0,
    [switch]$Gui,
    [ValidateSet('random','empty','full')]
    [string]$TrayOccupancy = 'random',
    [switch]$SkipGif,
    [ValidateRange(1, 1000)]
    [int]$GifStride = 3,
    [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss')
)

$ErrorActionPreference = 'Stop'
$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ThisDir '..\..')).Path
$IsaacRoot = (Resolve-Path (Join-Path $ProjectRoot '..\..\..\..')).Path
$PythonBat = Join-Path $IsaacRoot '_isaac_sim\python.bat'
$EntryPoint = Join-Path $ProjectRoot 'record.py'
$GifMaker = Join-Path $ProjectRoot 'scripts\tools\make_all_camera_gif.py'
$SemanticGifMaker = Join-Path $ProjectRoot 'scripts\tools\make_all_camera_semantic_gif.py'
$RunRoot = Join-Path $ProjectRoot "debug\test_runs\$RunName"
$LogDir = Join-Path $RunRoot 'logs'
$DataDir = Join-Path $RunRoot 'datasets'
$GifDir = Join-Path $RunRoot 'gifs'
$SemanticGifDir = Join-Path $RunRoot 'gifs_semantic'

if (-not (Test-Path -LiteralPath $PythonBat)) {
    throw "Isaac Sim Python launcher not found: $PythonBat"
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $GifDir -Force | Out-Null
New-Item -ItemType Directory -Path $SemanticGifDir -Force | Out-Null

$Objects = @('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
$Results = @()

foreach ($Name in $Objects) {
    $OutputDir = Join-Path $DataDir $Name
    $LogFile = Join-Path $LogDir "$Name.log"

    Write-Host ''
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host " P4 TEST: $Name" -ForegroundColor Cyan
    Write-Host " OUTPUT : $OutputDir" -ForegroundColor DarkCyan
    Write-Host '============================================================' -ForegroundColor Cyan

    $Arguments = @(
        $EntryPoint,
        '--object', $Name,
        '--episodes', "$Episodes",
        '--max-attempts', "$MaxAttempts",
        '--tray-occupancy', $TrayOccupancy,
        '--enable_cameras',
        '--out_dir', $OutputDir
    )
    if (-not $Gui) {
        $Arguments += '--headless'
    }

    $Started = Get-Date
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
        $PreviousNativeErrorAction = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    & $PythonBat @Arguments 2>&1 | Tee-Object -FilePath $LogFile
    $ExitCode = $LASTEXITCODE
    if (Test-Path variable:PreviousNativeErrorAction) {
        $PSNativeCommandUseErrorActionPreference = $PreviousNativeErrorAction
    }
    $ErrorActionPreference = $PreviousErrorAction
    $Elapsed = [math]::Round(((Get-Date) - $Started).TotalMinutes, 2)

    $H5Count = @(Get-ChildItem -LiteralPath $OutputDir -Recurse -Filter '*.h5' -File -ErrorAction SilentlyContinue).Count
    $MetricsPath = Join-Path $OutputDir 'run_metrics.json'
    $Metrics = $null
    if (Test-Path -LiteralPath $MetricsPath) {
        try { $Metrics = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json }
        catch { Write-Warning "Cannot read run metrics: $MetricsPath" }
    }
    $Passed = ($ExitCode -eq 0) -and ($H5Count -eq (2 * $Episodes)) -and
              ($null -ne $Metrics) -and ($Metrics.success_count -eq $Episodes)
    $GifCount = 0
    $SemanticGifCount = 0

    if ($Passed -and (-not $SkipGif)) {
        for ($Episode = 0; $Episode -lt $Episodes; $Episode++) {
            $GifPath = Join-Path $GifDir ("{0}_episode_{1:D6}_all_cameras.gif" -f $Name, $Episode)
            # Each recorder uses dt=0.02 s. Match GIF time to frame sampling,
            # including when the caller requests stride 1 instead of 3.
            & $PythonBat $GifMaker $OutputDir --object $Name --episode $Episode --stride $GifStride --duration-ms (20 * $GifStride) --panel-width 224 --output $GifPath
            if (($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath $GifPath)) {
                $GifCount++
            }
            $SemanticGifPath = Join-Path $SemanticGifDir ("{0}_episode_{1:D6}_all_cameras_semantic.gif" -f $Name, $Episode)
            & $PythonBat $SemanticGifMaker $OutputDir --object $Name --episode $Episode --stride $GifStride --duration-ms (20 * $GifStride) --panel-width 224 --output $SemanticGifPath
            if (($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath $SemanticGifPath)) {
                $SemanticGifCount++
            }
        }
    }

    $Results += [pscustomobject]@{
        recorder = $Name
        status = if ($Passed) { 'PASS' } else { 'FAIL' }
        exit_code = $ExitCode
        attempts = if ($null -ne $Metrics) { $Metrics.total_attempts } else { $null }
        successes = if ($null -ne $Metrics) { $Metrics.success_count } else { $null }
        failures = if ($null -ne $Metrics) { $Metrics.failure_count } else { $null }
        spawn_fail = if ($null -ne $Metrics) { $Metrics.spawn_fail_count } else { $null }
        pick_fail = if ($null -ne $Metrics) { $Metrics.pick_fail_count } else { $null }
        place_fail = if ($null -ne $Metrics) { $Metrics.place_fail_count } else { $null }
        other_fail = if ($null -ne $Metrics) { $Metrics.other_fail_count } else { $null }
        h5_files = $H5Count
        gifs = $GifCount
        semantic_gifs = $SemanticGifCount
        minutes = $Elapsed
        log = "logs/$Name.log"
        dataset = "datasets/$Name"
    }
}

$CsvPath = Join-Path $RunRoot 'summary.csv'
$MdPath = Join-Path $RunRoot 'SUMMARY.md'
$Results | Export-Csv -LiteralPath $CsvPath -NoTypeInformation

$Md = @(
    '# P4 five-recorder test',
    '',
    "Run: ``$RunName``  ",
    "Episodes per recorder: ``$Episodes``  ",
    "Mode: ``$(if ($Gui) { 'GUI' } else { 'headless' })``  ",
    "GIF stride: ``$GifStride``",
    '',
    '| Recorder | Status | Attempts | Success | Fail | Exit | H5 | Minutes |',
    '|---|---:|---:|---:|---:|---:|---:|---:|'
)
foreach ($Result in $Results) {
    $Md += "| $($Result.recorder) | $($Result.status) | $($Result.attempts) | $($Result.successes) | $($Result.failures) | $($Result.exit_code) | $($Result.h5_files) | $($Result.minutes) |"
}
$Md += ''
$Md += '| Recorder | Spawn fail | Pick fail | Place fail | Other fail | RGB GIF | Semantic GIF |'
$Md += '|---|---:|---:|---:|---:|---:|---:|'
foreach ($Result in $Results) {
    $Md += "| $($Result.recorder) | $($Result.spawn_fail) | $($Result.pick_fail) | $($Result.place_fail) | $($Result.other_fail) | $($Result.gifs) | $($Result.semantic_gifs) |"
}
$Md += ''
$Md += 'Logs, datasets, and GIF reports are stored in their respective subfolders.'
$Md += 'PASS means the requested successful H5 episodes were saved. It does not mean zero failed attempts. Minutes exclude GIF generation.'
$Md | Set-Content -LiteralPath $MdPath -Encoding UTF8

Write-Host ''
Write-Host '================ P4 RECORDING SUMMARY ================' -ForegroundColor Yellow
$Results | Format-Table recorder, status, attempts, successes, failures, minutes -AutoSize
Write-Host '---------------- Failure categories ------------------' -ForegroundColor Cyan
$Results | Format-Table recorder, spawn_fail, pick_fail, place_fail, other_fail -AutoSize
Write-Host '---------------- Saved outputs -----------------------' -ForegroundColor Cyan
$Results | Format-Table recorder, exit_code, h5_files, gifs, semantic_gifs -AutoSize
Write-Host 'PASS = requested saved successes; check failures separately.' -ForegroundColor Yellow
Write-Host "Run folder : $RunRoot"
Write-Host "Summary    : $MdPath"
Write-Host "GIF folder : $GifDir"
Write-Host "Semantic GIF folder : $SemanticGifDir"

if (@($Results | Where-Object status -ne 'PASS').Count -gt 0) {
    exit 1
}
exit 0

# =============================================================================
# batch_preprocess.ps1
# Batch execution of SNAP GPT preprocessing for Sentinel-1 GRD on Windows
# Output projection: EPSG:2180 (PL-1992) - hardcoded in XML graph
#
# Usage:
#   .\batch_preprocess.ps1                          # uses default paths below
#   .\batch_preprocess.ps1 -InputDir C:\data\asc -OutputDir C:\data\processed\asc
#   .\batch_preprocess.ps1 -Threads 8
#   .\batch_preprocess.ps1 -Orbit ASC              # process ascending only
#   .\batch_preprocess.ps1 -Orbit DESC             # process descending only
#   .\batch_preprocess.ps1 -Orbit BOTH             # process both (default)
#
# Prerequisites:
#   - SNAP installed at C:\Program Files\snap\
#   - S1_preprocessing_levee.xml in the same directory as this script
#   - PowerShell execution policy allows scripts:
#       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# =============================================================================

param(
    [string]$GptPath    = "C:\Program Files\snap\bin\gpt.exe",
    [string]$GraphPath  = "$PSScriptRoot\S1_preprocessing_levee.xml",
    [string]$RawDir     = "C:\data\raw",
    [string]$OutputBase = "C:\data\processed",
    [int]   $Threads    = 4,
    [ValidateSet("ASC", "DESC", "BOTH")]
    [string]$Orbit      = "BOTH",
    [switch]$DryRun
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Header {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  SNAP GPT Batch Preprocessing - Sentinel-1 GRD" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  GPT:        $GptPath"
    Write-Host "  Graph:      $GraphPath"
    Write-Host "  Raw dir:    $RawDir"
    Write-Host "  Output:     $OutputBase"
    Write-Host "  Projection: EPSG:2180 (PL-1992)"
    Write-Host "  Threads:    $Threads"
    Write-Host "  Orbit:      $Orbit"
    if ($DryRun) {
        Write-Host "  MODE:       DRY RUN (no files will be processed)" -ForegroundColor Yellow
    }
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success { param($msg) Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Skip    { param($msg) Write-Host "  [SKIP]  $msg" -ForegroundColor DarkGray }
function Write-Fail    { param($msg) Write-Host "  [FAIL]  $msg" -ForegroundColor Red }
function Write-Info    { param($msg) Write-Host "  [INFO]  $msg" -ForegroundColor White }

function Process-Directory {
    param(
        [string]$InputDir,
        [string]$OutputDir,
        [string]$Label
    )

    Write-Host ""
    Write-Host "  --- $Label ---" -ForegroundColor Yellow

    if (-not (Test-Path $InputDir)) {
        Write-Host "  Input directory not found, skipping: $InputDir" -ForegroundColor DarkYellow
        return @{Total=0; Success=0; Failed=0; Skipped=0}
    }

    # Create output directory and logs subdirectory
    $LogDir = Join-Path $OutputDir "logs"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
        New-Item -ItemType Directory -Force -Path $LogDir    | Out-Null
    }

    $files = Get-ChildItem -Path $InputDir -Filter "*.zip" -ErrorAction SilentlyContinue
    if ($files.Count -eq 0) {
        Write-Info "No .zip files found in $InputDir"
        return @{Total=0; Success=0; Failed=0; Skipped=0}
    }

    $total   = 0
    $success = 0
    $failed  = 0
    $skipped = 0

    foreach ($file in $files) {
        $total++
        $baseName  = $file.BaseName
        $outputFile = Join-Path $OutputDir "${baseName}_sigma0"
        $logFile    = Join-Path $LogDir "${baseName}.log"

        # Skip if output already exists
        if (Test-Path "${outputFile}.tif") {
            Write-Skip "$($file.Name)"
            $skipped++
            continue
        }

        Write-Info "[$total/$($files.Count)] $($file.Name)"
        Write-Info "  Started: $(Get-Date -Format 'HH:mm:ss')"

        if ($DryRun) {
            Write-Host "  [DRY-RUN] Would run: gpt ... -Pinput=$($file.FullName) -Poutput=$outputFile" -ForegroundColor DarkYellow
            continue
        }

        # Run GPT
        $startTime = Get-Date
        $process = Start-Process -FilePath $GptPath `
            -ArgumentList @(
                "`"$GraphPath`"",
                "-Pinput=`"$($file.FullName)`"",
                "-Poutput=`"$outputFile`"",
                "-q", $Threads
            ) `
            -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $logFile `
            -RedirectStandardError  "${logFile}.err"

        $elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)

        if ($process.ExitCode -eq 0) {
            Write-Success "$($file.Name) - ${elapsed} min"
            $success++
        } else {
            Write-Fail "$($file.Name) - exit code $($process.ExitCode)"
            Write-Fail "  See log: $logFile"
            $failed++
        }
    }

    return @{Total=$total; Success=$success; Failed=$failed; Skipped=$skipped}
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

Write-Header

# Check GPT exists
if (-not (Test-Path $GptPath)) {
    Write-Host "ERROR: GPT not found at: $GptPath" -ForegroundColor Red
    Write-Host "       Check SNAP installation path and update -GptPath parameter." -ForegroundColor Red
    exit 1
}

# Check graph exists
if (-not (Test-Path $GraphPath)) {
    Write-Host "ERROR: Graph XML not found at: $GraphPath" -ForegroundColor Red
    Write-Host "       Make sure S1_preprocessing_levee.xml is in the same folder as this script." -ForegroundColor Red
    exit 1
}

Write-Success "GPT found: $GptPath"
Write-Success "Graph found: $GraphPath"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

$totalStats = @{Total=0; Success=0; Failed=0; Skipped=0}

if ($Orbit -eq "ASC" -or $Orbit -eq "BOTH") {
    $stats = Process-Directory `
        -InputDir  (Join-Path $RawDir "ascending") `
        -OutputDir (Join-Path $OutputBase "ascending") `
        -Label     "ASCENDING"

    $totalStats.Total   += $stats.Total
    $totalStats.Success += $stats.Success
    $totalStats.Failed  += $stats.Failed
    $totalStats.Skipped += $stats.Skipped
}

if ($Orbit -eq "DESC" -or $Orbit -eq "BOTH") {
    $stats = Process-Directory `
        -InputDir  (Join-Path $RawDir "descending") `
        -OutputDir (Join-Path $OutputBase "descending") `
        -Label     "DESCENDING"

    $totalStats.Total   += $stats.Total
    $totalStats.Success += $stats.Success
    $totalStats.Failed  += $stats.Failed
    $totalStats.Skipped += $stats.Skipped
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Total scenes : $($totalStats.Total)"
Write-Host "  Successful   : $($totalStats.Success)" -ForegroundColor Green
Write-Host "  Skipped      : $($totalStats.Skipped)" -ForegroundColor DarkGray
if ($totalStats.Failed -gt 0) {
    Write-Host "  Failed       : $($totalStats.Failed)" -ForegroundColor Red
    Write-Host "  Check logs in: $OutputBase\ascending\logs and $OutputBase\descending\logs" -ForegroundColor Red
}
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
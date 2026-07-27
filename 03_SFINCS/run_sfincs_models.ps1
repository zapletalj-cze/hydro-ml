$ErrorActionPreference = "Stop"

$exePath = "D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS\SFINCS_2026_01_release\SFINCS_v2.4.0_Galibier_release_exe\sfincs.exe"
$modelRoots = @(
    "D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_baseline",
    "D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_levees"
)

if (-not (Test-Path $exePath)) {
    throw "SFINCS executable not found: $exePath"
}

foreach ($root in $modelRoots) {
    if (-not (Test-Path $root)) {
        throw "Model folder not found: $root"
    }

    Push-Location $root
    try {
        Write-Host "Running $(Split-Path $root -Leaf) ..."
        & $exePath *> "sfincs_log.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "SFINCS failed in $root with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
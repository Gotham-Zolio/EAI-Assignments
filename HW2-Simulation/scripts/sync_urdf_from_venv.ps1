<#
. SYNCHRONIZE URDF ASSETS FROM .venv TO REPO
.
# Usage: run this script from anywhere. It detects the HW2 root based on the script location.
# In PowerShell (pwsh):
#   pwsh ./scripts/sync_urdf_from_venv.ps1
#
# What it does:
#  - Finds the genesis assets/urdf directory inside the project's .venv
#  - Creates ./urdf/ in the HW2 root and copies available folders (go2, plane, panda_bullet, g1, ...)
#  - Copies panda_bullet content into ./assets/Robots/panda/ for convenience
#  - Reports which URDF folders were found and which were missing
#
# Notes:
#  - This script only copies files; it does not modify any code.
#  - If you want to rename or transform files (e.g. xacro -> urdf), do that separately.
#  - The script is idempotent and uses -Force on Copy-Item to overwrite existing files.
#>

Set-StrictMode -Version Latest

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    # HW2 root is parent of scripts/
    $hw2Root = Split-Path -Parent $scriptDir

    Write-Host "HW2 root detected: $hw2Root"

    # possible venv paths (case differences on Windows)
    $cand1 = Join-Path $hw2Root ".venv\Lib\site-packages\genesis\assets\urdf"
    $cand2 = Join-Path $hw2Root ".venv\lib\python3.12\site-packages\genesis\assets\urdf"

    if (Test-Path $cand1) {
        $venvUrdf = $cand1
    } elseif (Test-Path $cand2) {
        $venvUrdf = $cand2
    } else {
        Write-Error "Could not find genesis assets/urdf in .venv. Checked:`n $cand1`n $cand2`
Please ensure .venv is installed and contains genesis assets.";
        exit 2
    }

    Write-Host "Found genesis urdf assets at: $venvUrdf"

    $dest = Join-Path $hw2Root "urdf"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null

    # list of common asset folders to copy (adjust as needed)
    $foldersToCopy = @("go2", "plane", "panda_bullet", "g1", "anymal_c", "kuka_iiwa", "shadow_hand", "drones", "wheel", "3763", "simple")

    $found = @()
    $missing = @()

    foreach ($f in $foldersToCopy) {
        $src = Join-Path $venvUrdf $f
        if (Test-Path $src) {
            $dst = Join-Path $dest $f
            Write-Host "Copying $f -> $dst"
            Copy-Item -Path $src -Destination $dst -Recurse -Force
            $found += $f
        } else {
            Write-Host "Not found in venv assets: $f"
            $missing += $f
        }
    }

    # Additionally, if panda_bullet was copied, also place copies into assets/Robots/panda
    $pandaSrc = Join-Path $venvUrdf "panda_bullet"
    if (Test-Path $pandaSrc) {
        $pandaDest = Join-Path $hw2Root "assets\Robots\panda"
        New-Item -ItemType Directory -Path $pandaDest -Force | Out-Null
        Write-Host "Copying panda_bullet content -> $pandaDest"
        Copy-Item -Path (Join-Path $pandaSrc "*") -Destination $pandaDest -Recurse -Force
    }

    Write-Host "\nSummary:" -ForegroundColor Cyan
    Write-Host "Found and copied folders: $($found -join ', ')"
    if ($missing.Count -gt 0) {
        Write-Host "Missing (not present in venv): $($missing -join ', ')" -ForegroundColor Yellow
    } else {
        Write-Host "No missing folders from the expected list." -ForegroundColor Green
    }

    Write-Host "\nFinished. Check the $dest directory for copied URDF assets.\n"

} catch {
    Write-Error "Error during sync: $_"
    exit 1
}

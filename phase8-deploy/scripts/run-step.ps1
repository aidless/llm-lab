# run-step.ps1
# Wrapper to run a .sh script via Git Bash from Windows PowerShell.
# Replaces run-step.bat (which broke on paths with spaces).
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run-step.ps1 <script-name-without-sh> [args...]
#   e.g. powershell -ExecutionPolicy Bypass -File scripts\run-step.ps1 01_ecs_snapshot
#        powershell -ExecutionPolicy Bypass -File scripts\run-step.ps1 05_rollback B
#
# Returns the underlying bash script's exit code via $LASTEXITCODE.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# Locate Git for Windows bash directly. Do NOT use Get-Command bash:
# on Windows 10/11 it resolves to C:\Windows\System32\bash.exe (WSL),
# which does not accept Windows paths.
$GitBashPaths = @(
    "C:\Program Files\Git\bin\bash.exe"
    "C:\Program Files (x86)\Git\bin\bash.exe"
)
$bash = $null
foreach ($p in $GitBashPaths) {
    if (Test-Path $p) { $bash = $p; break }
}
if (-not $bash) {
    Write-Error "Git for Windows not found. Searched:`n  $($GitBashPaths -join "`n  ")`nInstall: https://git-scm.com/download/win"
    exit 127
}

# Resolve .sh path (absolute) and convert to MSYS-style for Git Bash
$shPath = Join-Path $PSScriptRoot "$Script.sh"
if (-not (Test-Path $shPath)) {
    Write-Error ".sh not found: $shPath"
    exit 2
}
# C:\Program Files\Git\bin\bash.exe accepts both Windows and POSIX paths,
# but POSIX is safer across all Git Bash versions.
$msysPath = (($shPath -replace '\\', '/') -replace '^([A-Z]):', '/$1')

Write-Host "=== run-step.ps1: $Script.sh via Git Bash ==="
Write-Host "=== BASH:    $bash"
Write-Host "=== Script:  $msysPath"
Write-Host "=== Args:    $($ExtraArgs -join ' ')"
Write-Host ""

# Run via call operator (&)
& $bash $msysPath @ExtraArgs
$rc = $LASTEXITCODE

Write-Host ""
Write-Host "=== $Script.sh exited with code $rc ==="
exit $rc
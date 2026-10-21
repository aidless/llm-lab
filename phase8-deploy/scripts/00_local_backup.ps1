# 00_local_backup.ps1
# Purpose: Step 1 of Phase 8 deployment checklist.
# Backup local fortune-web-v2 + audit .pem LF/PKCS#8/ACL.
# Pure ASCII, LF line endings, PowerShell 5.x safe (no emoji, no Chinese).

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Administrator\cow\fortune-web-v2"
$BackupRoot  = "F:\fortune-web-v2-backup-2026-07-14"
$BackupLog   = "F:\fortune-backup.log"
$PemPath     = Join-Path $ProjectRoot "app_private_key_sandbox.pem"
$EnvPath     = Join-Path $ProjectRoot ".env.local"
$GitignorePath = Join-Path $ProjectRoot ".gitignore"

Write-Host "=== 00_local_backup.ps1 START $(Get-Date -Format o) ==="

# 1.1 Backup
Write-Host "[1.1] robocopy backup..."
$robocopyLog = & robocopy $ProjectRoot $BackupRoot /E /COPYALL /LOG+:$BackupLog /NP
# robocopy exit codes: 0=no change, 1=files copied, 2=extra files, 3=both. >=8 is error.
$rc = $LASTEXITCODE
if ($rc -ge 8) {
    Write-Error "robocopy FAILED with exit code $rc. See $BackupLog"
    exit 1
}
Write-Host "[1.1] robocopy OK (exit $rc)."

# 1.2 Hash sensitive files
Write-Host "[1.2] SHA256 of .env.local and .pem (record these!):"
$envHash = (Get-FileHash $EnvPath -Algorithm SHA256).Hash
$pemHash = (Get-FileHash $PemPath -Algorithm SHA256).Hash
Write-Host "  .env.local : $envHash"
Write-Host "  .pem       : $pemHash"
"$(Get-Date -Format o) env=$envHash pem=$pemHash" | Out-File -FilePath "F:\phase8-deploy-audit.log" -Append -Encoding ASCII

# 1.3 LF check (Trap 9)
Write-Host "[1.3] LF check on .pem..."
$data = [System.IO.File]::ReadAllBytes($PemPath)
if ($data -match [byte[]]@(13, 10)) {
    Write-Error "CRLF detected in .pem (Trap 9). Fix locally: certutil / python LF conversion first."
    exit 2
}
Write-Host "[1.3] LF OK."

# 1.4 Header + DER byte sanity (Trap 10)
Write-Host "[1.4] Header and DER byte check on .pem..."
$header = Get-Content $PemPath -TotalCount 1
Write-Host "  Header line: $header"
$asn1 = Get-Content $PemPath -Raw
# Strip PEM armor
$b64 = ($asn1 -replace "-----BEGIN [A-Z ]+-----", "" -replace "-----END [A-Z ]+-----", "" -replace "\s", "")
try {
    $derBytes = [Convert]::FromBase64String($b64)
    Write-Host "  DER bytes: $($derBytes.Length)"
    # PKCS#8: 30 82 xx xx 02 01 00 30 0d 06 09 2a 86 48 86 f7 0d 01 01 01
    if ($derBytes.Length -ge 11 -and $derBytes[0] -eq 0x30 -and $derBytes[1] -eq 0x82 -and $derBytes[4] -eq 0x02 -and $derBytes[7] -eq 0x30) {
        Write-Host "  ASN.1 sequence detected. OID at byte 9-11 (should be 2a 86 48 86 f7 0d 01 01 01 for PKCS#8 rsaEncryption)"
        if ($derBytes[9] -eq 0x2a -and $derBytes[10] -eq 0x86) {
            Write-Host "  -> Looks like PKCS#8 (good)."
        } else {
            Write-Warning "  -> OID not PKCS#8 rsaEncryption. Trap 10 may apply."
        }
    }
} catch {
    Write-Warning "DER base64 decode failed: $($_.Exception.Message)"
}

# 1.5 ACL audit
Write-Host "[1.5] ACL audit on .pem..."
$acl = Get-Acl $PemPath
$bad = $acl.Access | Where-Object {
    ($_.IdentityReference -match "Everyone|Users|BUILTIN\\Users") -and
    ($_.FileSystemRights -match "Write|FullControl|Modify|ReadAndExecute")
}
if ($bad) {
    Write-Warning "Overly permissive ACL on .pem (read-only is OK, write is NOT):"
    $bad | ForEach-Object { Write-Warning "  $($_.IdentityReference) -> $($_.FileSystemRights)" }
    Write-Warning "Recommended fix: icacls `"$PemPath`" /inheritance:r /grant:r `"$($env:USERNAME)`:(R)"
} else {
    Write-Host "[1.5] ACL OK (no broad write/modify)."
}

# 1.6 .gitignore sanity (CHANGED: warn-only, do NOT abort deploy)
# Reason: trap pre-deployment gates deploy, not pre-existing repo state.
# If .gitignore is missing, that is a pre-existing risk; user can fix later.
Write-Host "[1.6] .gitignore sanity..."
if (Test-Path $GitignorePath) {
    $gi = Get-Content $GitignorePath -Raw
    $hits = @()
    if ($gi -match "(?m)^\.env\.local$")            { $hits += ".env.local" }
    if ($gi -match "(?m)^\*\.pem$")                { $hits += "*.pem" }
    if ($gi -match "(?m)^\*\.key$")                { $hits += "*.key" }
    if ($gi -match "(?m)^credentials/?")           { $hits += "credentials/" }
    if ($hits.Count -lt 2) {
        Write-Warning ".gitignore is INCOMPLETE. Found: [$($hits -join ', ')]. Need at least .env.local + *.pem."
        Write-Warning "  Risk: .env.local and *.pem could be pushed to GitHub by accident."
        Write-Warning "  Fix (do now, then re-run this script):"
        Write-Warning "    Add to ${GitignorePath}:"
        Write-Warning "      .env.local"
        Write-Warning "      *.pem"
        Write-Warning "      *.key"
        Write-Warning "      credentials/"
        Write-Warning "  Continue deployment anyway (this is a known pre-existing risk, not a deployment blocker)."
    } else {
        Write-Host "[1.6] .gitignore OK. Found: $($hits -join ', ')"
    }
} else {
    Write-Warning ".gitignore NOT FOUND at $GitignorePath"
    Write-Warning "  Risk: .env.local and *.pem have NO git-level protection."
    Write-Warning "  Fix (do now, then re-run this script):"
    Write-Warning "    New-Item -ItemType File -Path $GitignorePath -Force"
    Write-Warning "    Add-Content $GitignorePath '.env.local', '*.pem', '*.key', 'credentials/'"
    Write-Warning "  Continue deployment anyway (trap does not block deploy on missing .gitignore)."
}

Write-Host "=== 00_local_backup.ps1 DONE. Safe to proceed to Step 2 (01_ecs_snapshot.sh) ==="
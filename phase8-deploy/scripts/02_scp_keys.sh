#!/usr/bin/env bash
# 02_scp_keys.sh
# Purpose: Step 4a of Phase 8 deployment checklist.
# SCP .env.local + app_private_key_sandbox.pem to ECS with LF-fix (Trap 9) and size verify.
# Trap 11: NEVER scp .pem or .env.local to GitHub, only to ECS.

set -eu

ECS_IP="116.62.69.83"
ECS_USER="root"
ECS_PROJECT="/var/www/taixuan"
LOCAL_PROJECT="C:\\Users\\Administrator\\cow\\fortune-web-v2"
# Git Bash path mapping:
LOCAL_PROJECT_BASH="/c/Users/Administrator/cow/fortune-web-v2"

ENV_LOCAL_SRC="${LOCAL_PROJECT_BASH}/.env.local"
PEM_SRC="${LOCAL_PROJECT_BASH}/app_private_key_sandbox.pem"

ENV_LOCAL_DST="${ECS_USER}@${ECS_IP}:${ECS_PROJECT}/.env.local"
PEM_DST="${ECS_USER}@${ECS_IP}:${ECS_PROJECT}/app_private_key_sandbox.pem"

EXPECTED_PEM_SIZE_MIN=1620  # PEM armor overhead varies; RSA-2048 ~1670, RSA-4096 ~3300
EXPECTED_PEM_SIZE_MAX=1800  # Conservative range; will alert outside this band

echo "============================================="
echo " SCP keys to ECS @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Source: $LOCAL_PROJECT"
echo " Target: ${ECS_USER}@${ECS_IP}:${ECS_PROJECT}"
echo "============================================="

# Pre-flight: local files exist and are LF + PKCS#8
echo "[pre] local file checks..."
[ -f "$ENV_LOCAL_SRC" ] || { echo "MISSING: $ENV_LOCAL_SRC"; exit 1; }
[ -f "$PEM_SRC" ]       || { echo "MISSING: $PEM_SRC";       exit 1; }

if grep -q $'\r' "$PEM_SRC"; then
    echo "ERROR: CRLF detected in $PEM_SRC (Trap 9). Abort."
    echo "Fix: python -c \"d=open('$PEM_SRC','rb').read(); open('$PEM_SRC','wb').write(d.replace(b'\\r\\n',b'\\n'))\""
    exit 2
fi

HEADER=$(head -1 "$PEM_SRC")
if [[ "$HEADER" == *"BEGIN RSA PRIVATE KEY"* ]]; then
    echo "WARNING: PKCS#1-style header detected (Trap 10). Phase 8 note: alipay sandbox displays PKCS#1 but content is usually PKCS#8. Will verify on ECS."
fi

# SCP
echo "[1] scp .env.local..."
scp -o StrictHostKeyChecking=accept-new "$ENV_LOCAL_SRC" "$ENV_LOCAL_DST"
echo "[2] scp .pem..."
scp -o StrictHostKeyChecking=accept-new "$PEM_SRC" "$PEM_DST"

# On ECS: defensive LF fix + size verify + header normalize + chmod 600
echo "[3] ECS-side normalization + verify..."
ssh -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} bash -s -- "$ECS_PROJECT" "$EXPECTED_PEM_SIZE_MIN" "$EXPECTED_PEM_SIZE_MAX" <<'REMOTE_EOF'
set -eu
PROJECT="$1"
SIZE_MIN="$2"
SIZE_MAX="$3"
PEM="${PROJECT}/app_private_key_sandbox.pem"

cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 3; }

# Defensive LF fix (idempotent if already LF)
if file "$PEM" 2>/dev/null | grep -q CRLF; then
    sed -i 's/\r$//' "$PEM"
    echo "[3a] CRLF->LF fixed on ECS side."
else
    echo "[3a] LF OK on ECS side (file reports no CRLF)."
fi

# Size verify (range check; specific size varies by key generation)
ACTUAL_SIZE=$(stat -c%s "$PEM")
echo "[3b] .pem size: $ACTUAL_SIZE bytes (acceptable range: $SIZE_MIN - $SIZE_MAX)"
if [ "$ACTUAL_SIZE" -lt "$SIZE_MIN" ] || [ "$ACTUAL_SIZE" -gt "$SIZE_MAX" ]; then
    echo "ERROR: size out of range. Got $ACTUAL_SIZE, allowed [$SIZE_MIN, $SIZE_MAX]."
    echo "Possible causes: key truncated in transit, or wrong file (e.g. EC key ~250 bytes vs RSA-2048 ~1670 bytes)."
    exit 4
fi

# Header normalize (Trap 10): if header is PKCS#1 but DER is PKCS#8, fix header
HEADER=$(head -1 "$PEM")
if [[ "$HEADER" == *"BEGIN RSA PRIVATE KEY"* ]]; then
    # Check DER bytes
    b64=$(grep -v '^-' "$PEM" | tr -d '\n')
    der=$(echo "$b64" | base64 -d 2>/dev/null | head -c 11 | od -An -tx1 | tr -d ' \n')
    # PKCS#8 starts: 30 82 xx xx 02 01 00 30 0d 06 09 2a
    if [[ "$der" == "3082"* ]] && [[ "$der" =~ 300d06092a86 ]]; then
        echo "[3c] Header is PKCS#1 but DER is PKCS#8 -> normalizing header to PKCS#8."
        sed -i 's/-----BEGIN RSA PRIVATE KEY-----/-----BEGIN PRIVATE KEY-----/' "$PEM"
        sed -i 's/-----END RSA PRIVATE KEY-----/-----END PRIVATE KEY-----/' "$PEM"
    else
        echo "[3c] Header is PKCS#1 AND DER is PKCS#1 -> real PKCS#1, alipay SDK may fail. Flag this."
    fi
fi

# Chmod 600 (Trap 11)
chmod 600 "$PEM"
chmod 600 "${PROJECT}/.env.local"
echo "[3d] chmod 600 applied to .pem and .env.local."

# Final verification
echo "[3e] final header + size:"
head -1 "$PEM"
stat -c'%n %s bytes mode %a' "$PEM"
stat -c'%n %s bytes mode %a' "${PROJECT}/.env.local"

# grep out real-mode env vars to confirm .env.local content loaded
echo "[3f] env vars in .env.local (REDACTED values):"
grep -E '^TAIXUAN_' "${PROJECT}/.env.local" | sed -E 's/=.*/=<REDACTED>/'
REMOTE_EOF

echo ""
echo "============================================="
echo " SCP + normalize + verify complete."
echo " If [3b] size mismatch OR [3c] real PKCS#1 detected -> STOP and inspect."
echo " Else proceed to scripts/03_deploy_apply.sh (Step 4b-d: git pull + restart + health check)."
echo "============================================="
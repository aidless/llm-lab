#!/usr/bin/env bash
# 03_deploy_apply.sh
# Purpose: Step 4b-d of Phase 8 deployment checklist.
# Code-validate via /healthz (no git pull) + pip install + supervisor restart + health check.
# Run AFTER scripts/02_scp_keys.sh confirms .env.local + .pem are on ECS.

set -eu

ECS_IP="116.62.69.83"
ECS_USER="root"
ECS_PROJECT="/var/www/taixuan"
SERVICE_NAME="taixuan"

# Phase 8 sentinel: git_commit field embedded by build, served via /healthz.
# ECS deployment is rsync/cp, NOT git pull, so we verify via /healthz instead.
EXPECTED_GIT_COMMIT_PREFIX="c376ac3"  # /healthz currently reports c376ac3

echo "============================================="
echo " Deploy apply @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="

run_remote() {
    local desc="$1"
    local cmd="$2"
    echo ""
    echo "--- $desc ---"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} "$cmd"
}

# 4b. Verify code version via /healthz (NOT git pull; /var/www/taixuan is not a git repo)
echo "[4b] verify code version via /healthz..."
HZ_BODY=$(ssh ${ECS_USER}@${ECS_IP} "curl -s http://localhost/healthz" 2>/dev/null || echo "")
if [ -z "$HZ_BODY" ]; then
    echo "ERROR: /healthz unreachable. Is service up?"
    exit 4
fi
ACTUAL_COMMIT=$(echo "$HZ_BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('git_commit', ''))" 2>/dev/null || echo "")
echo "  /healthz reports git_commit: $ACTUAL_COMMIT (expected prefix: $EXPECTED_GIT_COMMIT_PREFIX)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_GIT_COMMIT_PREFIX"* ]]; then
    echo "ERROR: /healthz git_commit does not match Phase 8 sentinel."
    echo "  Got:      $ACTUAL_COMMIT"
    echo "  Expected: $EXPECTED_GIT_COMMIT_PREFIX*"
    echo "  Action: re-deploy code via rsync/cp from git clone (manual)."
    exit 5
fi
echo "[4b-verify] code version OK (Phase 8 deployed)."

# 4c. pip install (idempotent; pip skips already-installed)
echo ""
echo "[4c] pip install -r requirements.txt..."
run_remote "pip install" "cd ${ECS_PROJECT} && pip install -r requirements.txt 2>&1 | tail -10"

# Quick check alipay SDK importable
echo ""
echo "[4c-verify] python can import alipay..."
run_remote "import alipay check" "cd ${ECS_PROJECT} && python3 -c 'from alipay import AliPay; print(\"alipay SDK importable, version:\", __import__(\"alipay\").__version__ if hasattr(__import__(\"alipay\"), \"__version__\") else \"unknown\")' 2>&1"

# 4d. supervisor restart + status + health check
echo ""
echo "[4d] supervisor restart + status..."
run_remote "restart" "supervisorctl restart ${SERVICE_NAME} 2>&1 && sleep 3 && supervisorctl status ${SERVICE_NAME}"

# Health check
echo ""
echo "[4d-verify] health checks (3 attempts, 10s interval, threshold 3 fails)..."
ATTEMPTS=0
MAX_ATTEMPTS=3
HEALTH_OK=0
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "  attempt $ATTEMPTS/$MAX_ATTEMPTS..."

    # Check /healthz
    HZ=$(ssh ${ECS_USER}@${ECS_IP} "curl -s -o /dev/null -w '%{http_code}' http://localhost/healthz" 2>/dev/null || echo "000")
    echo "    /healthz HTTP: $HZ"

    # Check subscribe/plans mode
    MODE=$(ssh ${ECS_USER}@${ECS_IP} "curl -s http://localhost/api/v2/subscribe/plans 2>/dev/null | python3 -c 'import sys, json; d=json.load(sys.stdin); print(d.get(\"mode\", \"UNKNOWN\"))'" 2>/dev/null || echo "FETCH_FAILED")
    echo "    subscribe/plans mode: $MODE"

    if [ "$HZ" = "200" ] && [ "$MODE" = "alipay" ]; then
        HEALTH_OK=1
        break
    fi
    if [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; then
        sleep 10
    fi
done

echo ""
if [ "$HEALTH_OK" = "1" ]; then
    echo "============================================="
    echo " HEALTH CHECK PASSED"
    echo "   /healthz: 200"
    echo "   mode: alipay"
    echo " Proceed to scripts/04_smoke_test.sh (Step 5)"
    echo "============================================="
    exit 0
else
    echo "============================================="
    echo " HEALTH CHECK FAILED after $MAX_ATTEMPTS attempts"
    echo ""
    echo " DIAGNOSIS HINTS:"
    echo "  1. .env.local present at ${ECS_PROJECT}/.env.local?"
    echo "     -> ssh ${ECS_USER}@${ECS_IP} 'ls -la ${ECS_PROJECT}/.env.local'"
    echo "  2. .env.local loaded?"
    echo "     -> ssh ${ECS_USER}@${ECS_IP} 'cd ${ECS_PROJECT} && python3 -c \"from dotenv import load_dotenv; load_dotenv(\\\".env.local\\\"); import os; print(os.environ.get(\\\"TAIXUAN_ALIPAY_APP_ID\\\"))\"'"
    echo "  3. .pem PKCS#8 and readable?"
    echo "     -> ssh ${ECS_USER}@${ECS_IP} 'head -1 ${ECS_PROJECT}/app_private_key_sandbox.pem'"
    echo "  4. supervisor log?"
    echo "     -> ssh ${ECS_USER}@${ECS_IP} 'supervisorctl tail ${SERVICE_NAME} stderr' or 'journalctl -u supervisor'"
    echo ""
    echo " If unresolvable -> run scripts/05_rollback.sh (Step 6) class B (env-only rollback)"
    echo "============================================="
    exit 6
fi
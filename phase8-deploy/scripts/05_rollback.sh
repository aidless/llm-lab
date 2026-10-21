#!/usr/bin/env bash
# 05_rollback.sh
# Purpose: Step 6 of Phase 8 deployment checklist.
# Three rollback classes: code (A), env-only (B), service (C).
# Default: B is safest. Pick class as argument: A / B / C

set -eu

ECS_IP="116.62.69.83"
ECS_USER="root"
ECS_PROJECT="/var/www/taixuan"
SERVICE_NAME="taixuan"

CLASS="${1:-B}"  # default to B (env-only)

case "$CLASS" in
    A) DESC="code rollback (git checkout HEAD~3 for phase 8 files)" ;;
    B) DESC="env-only rollback (delete .env.local + .pem, revert to mock mode)" ;;
    C) DESC="service rollback (stop supervisor, full 502)" ;;
    *) echo "Usage: $0 [A|B|C]"; echo "  A = code rollback"; echo "  B = env-only rollback (DEFAULT, safest)"; echo "  C = service stop (emergency)"; exit 1 ;;
esac

echo "============================================="
echo " ROLLBACK CLASS $CLASS: $DESC"
echo " Target: ${ECS_USER}@${ECS_IP}:${ECS_PROJECT}"
echo " Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo ""
echo " This will make PERMANENT changes to ECS. Confirm in 5 seconds..."
sleep 5

case "$CLASS" in
    A)
        ssh -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} bash -s -- "${ECS_PROJECT}" "${SERVICE_NAME}" <<'REMOTE_EOF'
set -eu
PROJECT="$1"
SERVICE="$2"
cd "$PROJECT"
echo "[A1] Snapshot pre-rollback commit:"
git log -1 --oneline
echo "[A2] Reverting Phase 8 files to HEAD~3..."
git checkout HEAD~3 -- subscriptions_routes.py alipay_mock.py requirements.txt 2>&1 || echo "checkout failed; trying specific commits"
echo "[A3] Confirm reverted content lacks Phase 8 markers:"
grep -l "TAIXUAN_ALIPAY" subscriptions_routes.py alipay_mock.py 2>&1 | head -3 || echo "no Phase 8 markers remain (good)"
echo "[A4] Restart service..."
supervisorctl restart "$SERVICE"
sleep 3
supervisorctl status "$SERVICE"
echo "[A5] Verify mode is back to mock:"
curl -s http://localhost/api/v2/subscribe/plans | python3 -c "import sys,json; d=json.load(sys.stdin); print('mode:', d.get('mode'))" 2>&1
REMOTE_EOF
        ;;

    B)
        ssh -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} bash -s -- "${ECS_PROJECT}" "${SERVICE_NAME}" <<'REMOTE_EOF'
set -eu
PROJECT="$1"
SERVICE="$2"
cd "$PROJECT"
echo "[B1] Snapshot files about to be deleted:"
ls -la .env.local app_private_key_sandbox.pem 2>&1 | head -5
echo "[B2] Saving backup to /tmp/ in case quick-recovery needed..."
mkdir -p /tmp/phase8-rollback-$(date +%Y%m%d-%H%M%S)
cp .env.local app_private_key_sandbox.pem /tmp/phase8-rollback-$(date +%Y%m%d-%H%M%S)/ 2>/dev/null || true
echo "[B3] Deleting env files (env-only rollback, code untouched)..."
rm -f .env.local app_private_key_sandbox.pem
echo "[B4] Restart service..."
supervisorctl restart "$SERVICE"
sleep 3
supervisorctl status "$SERVICE"
echo "[B5] Verify mode is back to mock (USE_REAL_ALIPAY should be False):"
curl -s http://localhost/api/v2/subscribe/plans | python3 -c "import sys,json; d=json.load(sys.stdin); print('mode:', d.get('mode'))" 2>&1
echo "[B6] Backup location (for recovery if needed):"
ls -d /tmp/phase8-rollback-*
REMOTE_EOF
        ;;

    C)
        ssh -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} bash -s -- "${ECS_PROJECT}" "${SERVICE_NAME}" <<'REMOTE_EOF'
set -eu
SERVICE="$1"
echo "[C1] Stopping supervisor service..."
supervisorctl stop "$SERVICE"
sleep 2
supervisorctl status "$SERVICE"
echo "[C2] /healthz should now fail..."
curl -s -o /dev/null -w 'HTTP %{http_code}\n' -m 3 http://localhost/healthz || echo 'healthz unreachable (expected)'
echo "[C3] To recover: ssh to ECS and run 'supervisorctl start $SERVICE'"
REMOTE_EOF
        ;;
esac

echo ""
echo "============================================="
echo " ROLLBACK CLASS $CLASS complete."
echo " Verify user-facing app: ${ECS_IP}"
echo " If app is broken -> check supervisor logs: ssh ${ECS_USER}@${ECS_IP} 'supervisorctl tail ${SERVICE_NAME} stderr'"
echo "============================================="
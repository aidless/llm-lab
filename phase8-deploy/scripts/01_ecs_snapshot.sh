#!/usr/bin/env bash
# 01_ecs_snapshot.sh
# Purpose: Step 2 of Phase 8 deployment checklist.
# Read-only ECS baseline capture. NO writes. NO restart. NO git pull.
# Run from Windows host (Git Bash) with: bash scripts/01_ecs_snapshot.sh
#
# Targets: ECS 116.62.69.83 (root), project /var/www/taixuan/
# All commands are idempotent and safe to re-run.

set -u  # do NOT use -e: we want to capture all output even if some checks fail

ECS_IP="116.62.69.83"
ECS_USER="root"
ECS_PROJECT="/var/www/taixuan"
SERVICE_NAME="taixuan"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "============================================="
echo " ECS snapshot @ $TS"
echo " Target: ${ECS_USER}@${ECS_IP}:${ECS_PROJECT}"
echo "============================================="

run_remote() {
    local desc="$1"
    local cmd="$2"
    echo ""
    echo "--- $desc ---"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} "$cmd" 2>&1
}

# A. Confirm path actually exists (Phase 8 note: "verify with pwd")
run_remote "A. pwd actual project path" \
    "cd ${ECS_PROJECT} 2>/dev/null && pwd && ls -la .env.local 2>/dev/null; readlink -f \$(pwd)"

# B. Git state
run_remote "B. git head + status" \
    "cd ${ECS_PROJECT} && git log -1 --oneline && echo '---' && git status --short && echo '---branches---' && git branch -vv"

# C. Disk
run_remote "C. disk usage" \
    "df -h /var / 2>/dev/null | head -10"

# D. Supervisor
run_remote "D. supervisor status" \
    "supervisorctl status ${SERVICE_NAME} 2>&1"

# E. Env var presence (REDACTED values)
run_remote "E. env var presence (redacted)" \
    "if [ -f ${ECS_PROJECT}/.env.local ]; then grep -E '^TAIXUAN_(ALIPAY|PUBLIC)_' ${ECS_PROJECT}/.env.local | sed -E 's/=.*/=<REDACTED>/'; else echo '.env.local NOT YET PRESENT'; fi"

# F. Pem file presence
run_remote "F. pem file presence + size + first line" \
    "if [ -f ${ECS_PROJECT}/app_private_key_sandbox.pem ]; then stat -c '%n %s bytes' ${ECS_PROJECT}/app_private_key_sandbox.pem && echo 'header:' && head -1 ${ECS_PROJECT}/app_private_key_sandbox.pem; else echo '.pem NOT YET PRESENT'; fi"

# G. Process list
run_remote "G. python/gunicorn processes" \
    "ps -ef | grep -E 'gunicorn|python.*app\\.py' | grep -v grep || echo 'no python/gunicorn processes'"

# H. Route map (import app to verify routes)
run_remote "H. Flask route map (via python)" \
    "cd ${ECS_PROJECT} && python3 -c 'import app; print(chr(10).join(sorted(str(r) for r in app.app.url_map.iter_rules())))' 2>&1 | grep -E 'subscribe|auth|favorites' || echo 'route import failed'"

# I. Local healthz
run_remote "I. local /healthz" \
    "curl -s http://localhost/healthz 2>&1 || echo 'healthz unreachable'"

# J. Subscribe plans (mock vs real)
run_remote "J. subscribe/plans mode" \
    "curl -s http://localhost/api/v2/subscribe/plans 2>&1 | head -50"

# K. Internet sanity (alipay sandbox reachable from ECS)
run_remote "K. alipay sandbox reachability" \
    "curl -sI -m 5 https://openapi-sandbox.dl.alipaydev.com/gateway.do 2>&1 | head -3 || echo 'unreachable'"

echo ""
echo "============================================="
echo " Snapshot complete @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo " GO/NO-GO CHECKLIST:"
echo "  [ ] A: pwd matches /var/www/taixuan"
echo "  [ ] B: git HEAD has 73a3ec09 + d0f42ef0 + fd116b72 (Phase 8 commits)"
echo "  [ ] B: working tree clean"
echo "  [ ] C: disk free >= 1GB"
echo "  [ ] D: ${SERVICE_NAME} RUNNING"
echo "  [ ] E: 0 TAIXAN_ALIPAY_* present (real mode NOT yet active)"
echo "  [ ] F: .pem NOT YET PRESENT (or size matches 1686 if already staged)"
echo "  [ ] G: gunicorn/app.py process visible"
echo "  [ ] H: /api/v2/subscribe/{plans,checkout,webhook,return,...} routes present"
echo "  [ ] I: /healthz returns 200 with version + git_commit"
echo "  [ ] J: subscribe/plans mode is mock (no real-mode before .env.local)"
echo "  [ ] K: alipay sandbox gateway reachable (TLS handshake OK)"
echo "============================================="
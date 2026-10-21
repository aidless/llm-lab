#!/usr/bin/env bash
# 04_smoke_test.sh
# Purpose: Step 5 of Phase 8 deployment checklist.
# 5-scenario smoke test for real Alipay mode on ECS.
# Requires: ECS deployed with .env.local + .pem, app in real mode.

set -u  # do NOT use -e: we want to count passes even if some scenarios fail

ECS_IP="116.62.69.83"
ECS_USER="root"
ECS_PROJECT="/var/www/taixuan"
ECS_BASE="http://${ECS_IP}"
LOCAL_BASE="http://localhost"  # when SSHing in

# Results accumulator
PASS=0
FAIL=0
RESULTS=()

check() {
    local name="$1"
    local cond="$2"  # 0=pass, nonzero=fail
    if [ "$cond" = "0" ]; then
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $name")
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $name")
    fi
}

run_remote() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${ECS_USER}@${ECS_IP} "$1" 2>/dev/null
}

echo "============================================="
echo " 5-scenario smoke test @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Target: ${ECS_BASE}"
echo "============================================="

# ------ Scenario 1: GET /subscribe/plans returns mode=alipay ------
echo ""
echo "[Scenario 1] GET /api/v2/subscribe/plans (mode check)..."
PLANS=$(curl -s "${ECS_BASE}/api/v2/subscribe/plans")
echo "  Response: $(echo "$PLANS" | head -c 300)..."
MODE=$(echo "$PLANS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('mode',''))" 2>/dev/null || echo "")
if [ "$MODE" = "alipay" ]; then
    check "1. subscribe/plans mode=alipay" 0
else
    check "1. subscribe/plans mode=alipay (got '$MODE')" 1
fi

# ------ Scenario 2: POST /subscribe/checkout returns alipay URL ------
echo ""
echo "[Scenario 2] POST /api/v2/subscribe/checkout (alipay URL generation)..."
# First, need a logged-in user. Use smoke test account.
LOGIN_RES=$(curl -s -X POST "${ECS_BASE}/api/v2/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email":"smoke@test.com","password":"SmokeTest123!"}')
TOKEN=$(echo "$LOGIN_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")
if [ -z "$TOKEN" ]; then
    # Try register first
    curl -s -X POST "${ECS_BASE}/api/v2/auth/register" \
        -H "Content-Type: application/json" \
        -d '{"email":"smoke@test.com","password":"SmokeTest123!","nickname":"smoke"}' > /dev/null
    LOGIN_RES=$(curl -s -X POST "${ECS_BASE}/api/v2/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"email":"smoke@test.com","password":"SmokeTest123!"}')
    TOKEN=$(echo "$LOGIN_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")
fi

if [ -z "$TOKEN" ]; then
    check "2. login + obtain JWT" 1
    echo "  Cannot continue scenarios 2-4 without auth token."
else
    check "2. login + obtain JWT" 0
    echo "  Token obtained (length: ${#TOKEN})"

    CHECKOUT=$(curl -s -X POST "${ECS_BASE}/api/v2/subscribe/checkout" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d '{"plan":"monthly"}')
    echo "  Checkout response: $(echo "$CHECKOUT" | head -c 400)..."
    CHECKOUT_URL=$(echo "$CHECKOUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('url',''))" 2>/dev/null || echo "")
    CHECKOUT_MODE=$(echo "$CHECKOUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode',''))" 2>/dev/null || echo "")
    if [[ "$CHECKOUT_URL" == *"openapi-sandbox.dl.alipaydev.com"* ]] && [ "$CHECKOUT_MODE" = "alipay" ]; then
        check "3. checkout returns alipay sandbox URL" 0
    else
        check "3. checkout returns alipay sandbox URL (url=$CHECKOUT_URL, mode=$CHECKOUT_MODE)" 1
    fi

    # ------ Scenario 4: mock_confirm should 405 (POST-only, Trap 8) ------
    echo ""
    echo "[Scenario 4] GET /api/v2/subscribe/mock_confirm (should 405 in real mode)..."
    MOCK_HTTP=$(curl -s -o /dev/null -w "%{http_code}" "${ECS_BASE}/api/v2/subscribe/mock_confirm?plan=monthly&user_id=1" 2>/dev/null || echo "000")
    # In real mode the endpoint either:
    #   - returns 405 (POST only, GET not allowed)
    #   - returns 404 (route removed in real-only build)
    # Both indicate Trap 8 mitigation: frontend cannot GET this URL.
    if [ "$MOCK_HTTP" = "405" ] || [ "$MOCK_HTTP" = "404" ]; then
        check "4. mock_confirm GET rejected ($MOCK_HTTP)" 0
    else
        check "4. mock_confirm GET rejected (got HTTP $MOCK_HTTP)" 1
    fi
fi

# ------ Scenario 5: /me?payment_returned=1 does not 500 ------
echo ""
echo "[Scenario 5] GET /me?payment_returned=1 (return URL smoke)..."
if [ -n "$TOKEN" ]; then
    ME_HTTP=$(curl -s -o /dev/null -w "%{http_code}" "${ECS_BASE}/me?payment_returned=1" \
        -H "Authorization: Bearer ${TOKEN}" 2>/dev/null || echo "000")
else
    ME_HTTP=$(curl -s -o /dev/null -w "%{http_code}" "${ECS_BASE}/me?payment_returned=1" 2>/dev/null || echo "000")
fi
if [ "$ME_HTTP" = "200" ] || [ "$ME_HTTP" = "302" ]; then
    check "5. /me?payment_returned=1 returns OK (HTTP $ME_HTTP)" 0
else
    check "5. /me?payment_returned=1 returns OK (got HTTP $ME_HTTP)" 1
fi

# ------ Summary ------
echo ""
echo "============================================="
echo " 5-Scenario Results:"
echo "============================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  PASS: $PASS / 5"
echo "  FAIL: $FAIL / 5"
echo ""
echo "============================================="
echo " MANUAL SCENARIOS (NOT auto-tested, must do by hand):"
echo "  [M1] Open ${ECS_BASE}/subscribe in browser"
echo "  [M2] Click 'Subscribe Monthly'"
echo "  [M3] Scan QR with Alipay sandbox app"
echo "       Buyer account: alipay_test_buyer@alipay.com / 111111 (from https://open.alipay.com/develop/sandbox/account)"
echo "  [M4] Confirm payment (pay password 111111)"
echo "  [M5] Verify return to /me with subscription status = active"
echo "  [M6] Check ECS supervisor log for /webhook POST and signature verification"
echo "============================================="

# Exit code: 0 if >=3 pass, else 1
if [ "$PASS" -ge 3 ]; then
    echo ""
    echo " OVERALL: PASS (>= 3 scenarios OK)"
    exit 0
else
    echo ""
    echo " OVERALL: FAIL (< 3 scenarios OK)"
    exit 1
fi
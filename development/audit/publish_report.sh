#!/usr/bin/env bash
set -e
BASE="${TENSOR_API_BASE:-http://127.0.0.1:8001/api/v1}"
USER_="${TENSOR_AGENT_USER:?}"
PASS="${TENSOR_AGENT_PASS:?}"
BODY=$(curl -s -X POST "$BASE/auth/login" -H 'Content-Type: application/json' \
  -d "{\"username\":\"$USER_\",\"password\":\"$PASS\"}")
TOKEN=$(printf '%s' "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token') or d.get('access_token') or '')")
if [ -z "$TOKEN" ]; then echo "LOGIN FAILED: $BODY" | head -c 300; exit 1; fi
echo "login ok (token len ${#TOKEN})"
curl -s -X POST "$BASE/tensor/reports" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d @/tmp/report.json | head -c 600
echo

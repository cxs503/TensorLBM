#!/bin/bash
# Login
TOKEN=$(curl -s -X POST $TENSOR_API_BASE/login -H 'Content-Type: application/json' -d "{\"username\": \"$TENSOR_AGENT_USER\", \"password\": \"$TENSOR_AGENT_PASS\"}" | jq -r '.access_token')
echo "Token: $TOKEN"

# Post report
curl -s -X POST $TENSOR_API_BASE/tensor/reports -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d @/root/tp-agent/work/m26711/tmp/report.json

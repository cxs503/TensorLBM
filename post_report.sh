#!/bin/bash
curl -s -X POST http://127.0.0.1:8010/api/v1/tensor/reports \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlY29fYWRtaW4iLCJ1c2VyX2lkIjoxLCJyb2xlIjoiQURNSU4iLCJleHAiOjE3OTU0ODA4NTQsInR5cGUiOiJhY2Nlc3MiLCJpYXQiOjE3OTU0NzkwNTR9.EqsROgpiRvOvs-vQGgrSNgVaEGcoL_V6f9DeOXRzHHw" \
  -H 'Content-Type: application/json' \
  -d @/tmp/report.json

import json
import os
import requests

# Get token
login_url = "http://127.0.0.1:8010/api/v1/tensor/login"
login_data = {
    "username": os.environ["TENSORPLATFORM_REPORT_USERNAME"],
    "password": os.environ["TENSORPLATFORM_REPORT_PASSWORD"],
}
response = requests.post(login_url, json=login_data)
print(f"Login Response: {response.text}")
token = response.json().get('token', '')
print(f"Token: {token}")

# Send report
report_url = "http://127.0.0.1:8010/api/v1/tensor/reports"
with open('/root/tp-agent/work/m26856/report.json', 'r') as f:
    report_data = json.load(f)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
response = requests.post(report_url, json=report_data, headers=headers)
print(f"Report Response: {response.text}")

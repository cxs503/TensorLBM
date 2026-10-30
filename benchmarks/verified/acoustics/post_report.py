"""POST the report to the API."""

import json
import os
import requests

# API configuration
API_BASE = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1')
USERNAME = os.environ.get('TENSOR_AGENT_USER', 'eco_admin')
PASSWORD = os.environ.get('TENSOR_AGENT_PASS', 'EcoTest!2026pass')

print(f"API Base: {API_BASE}")
print(f"Username: {USERNAME}")

# Step 1: Login to get token
print("\n[1/2] Logging in...")
login_url = f"{API_BASE}/auth/login"
login_data = {
    "username": USERNAME,
    "password": PASSWORD
}

try:
    response = requests.post(login_url, json=login_data, timeout=10)
    print(f"Login response status: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get('token') or token_data.get('access_token')
        print(f"Token obtained: {token[:20]}..." if token else "No token in response")
        
        if not token:
            print(f"Full response: {json.dumps(token_data, indent=2)}")
            exit(1)
    else:
        print(f"Login failed: {response.text}")
        exit(1)
except Exception as e:
    print(f"Login error: {e}")
    exit(1)

# Step 2: POST the report
print("\n[2/2] Posting report...")
report_url = f"{API_BASE}/tensor/reports"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

with open('/tmp/report.json', 'r') as f:
    report_data = json.load(f)

print(f"Report title: {report_data['title']}")
print(f"Report figures: {len(report_data.get('figures', []))}")
print(f"Markdown length: {len(report_data.get('markdown', ''))} chars")

try:
    response = requests.post(report_url, headers=headers, json=report_data, timeout=30)
    print(f"\nPost response status: {response.status_code}")
    
    if response.status_code in [200, 201]:
        print("✅ Report posted successfully!")
        try:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        except:
            print(f"Response text: {response.text[:500]}")
    else:
        print(f"❌ Post failed: {response.text}")
except Exception as e:
    print(f"Post error: {e}")
    exit(1)

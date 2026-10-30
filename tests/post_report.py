#!/usr/bin/env python3
"""Post report using Python requests."""

import json
import requests
from pathlib import Path
import os

# Get environment variables
api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
username = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
password = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")

print(f"API Base: {api_base}")
print(f"Username: {username}")

# Login
print("\nLogging in...")
login_url = f"{api_base}/auth/login"
login_data = {"username": username, "password": password}

try:
    response = requests.post(login_url, json=login_data, timeout=10)
    print(f"Login response: {response.status_code}")
    
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"Token obtained: {len(token)} chars")
        
        # Post report
        print("\nPosting report...")
        report = json.loads(Path("/tmp/report.json").read_text())
        
        report_url = f"{api_base}/tensor/reports"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(report_url, json=report, headers=headers, timeout=30)
        print(f"Report response: {response.status_code}")
        
        if response.status_code in [200, 201]:
            print("✓ Report posted successfully!")
            print(f"Response: {response.json()}")
        else:
            print(f"✗ Report posting failed: {response.text}")
    else:
        print(f"✗ Login failed: {response.text}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

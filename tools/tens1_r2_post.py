#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1 R2 报告 POST（m26894）：登录 /auth/login 拿 access_token → POST /tensor/reports。"""
import json
import os
import urllib.request


def main():
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    token = json.load(urllib.request.urlopen(req, timeout=30))["access_token"]
    rep = open("/tmp/report.json", "rb").read()
    req = urllib.request.Request(base + "/tensor/reports", data=rep,
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=60) as r:
        print("POST", r.status, r.read().decode()[:400])


if __name__ == "__main__":
    main()

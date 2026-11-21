#!/bin/bash
# TENS1 R2 后台轮询器（m26894）：每 ~60s 轮询双臂，追加到 /tmp/tens1_poll.log
cd /root/tp-agent/work/m26894
while true; do
  echo "=== $(date -u +%H:%M:%S)" >> /tmp/tens1_poll.log
  timeout 50 python3 tools/sdaa_fs_probe_split.py poll --arm ctrl --tree /tmp/tens1_ctrl_tree >> /tmp/tens1_poll.log 2>&1
  timeout 50 python3 tools/sdaa_fs_probe_split.py poll --arm csf_smooth --tree /root/tp-agent/work/m26894 >> /tmp/tens1_poll.log 2>&1
  sleep 50
done

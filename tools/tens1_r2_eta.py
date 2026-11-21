#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1 R2 满窗 ETA 监视（m26894）：从 wrapper 轮询日志算 pace/ETA，对照 7200s 硬顶。
用法：python3 tools/tens1_r2_eta.py <launch.log> [...]
"""
import re
import sys


def pace(log):
    pts = []
    for m in re.finditer(r"polling (\d+)s: step (\d+)", open(log).read()):
        pts.append((int(m.group(1)), int(m.group(2))))
    if len(pts) < 2:
        return None
    (t0, s0), (t1, s1) = pts[0], pts[-1]
    p = (t1 - t0) / max(1, s1 - s0)
    return p, t1, s1, len(pts)


def main():
    for log in sys.argv[1:]:
        r = pace(log)
        if not r:
            print(log, "no data")
            continue
        p, t, s, n = r
        eta = (10000 - s) * p
        ok = "ON-TRACK" if t + eta < 7200 else "AT-RISK"
        print("%s: step=%d t=%ds pace=%.3fs/step ETA_total=%.0fs [%s] (cap 7200s, slack=%.0fs)"
              % (log.split("/")[-1], s, t, p, t + eta, ok, 7200 - (t + eta)))


if __name__ == "__main__":
    main()

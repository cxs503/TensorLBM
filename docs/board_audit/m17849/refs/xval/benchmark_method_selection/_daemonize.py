#!/usr/bin/env python3
"""Detach a command into its own session so it survives the caller's timeout.

Usage: python xval/benchmark_method_selection/_daemonize.py <logfile> <argv...>
"""
import os
import subprocess
import sys

log = sys.argv[1]
argv = sys.argv[2:]

# double fork -> the child is reparented to init and leaves the caller's session
if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)

with open(log, "ab", buffering=0) as fh:
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)
    (open(log + ".pid", "w").write(str(p.pid)))
sys.exit(0)
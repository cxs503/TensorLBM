import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim"))
from tensorlbm.invariance.independent import runner  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(runner.main(["shard"] + sys.argv[1:]))

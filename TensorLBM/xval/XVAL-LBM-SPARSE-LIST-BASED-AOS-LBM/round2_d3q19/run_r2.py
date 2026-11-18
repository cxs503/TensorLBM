"""Run tests for Sparse List AOS LBM."""

import subprocess
import sys

def main():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_xval_sparse_list_aos_3d.py", "-v"],
        cwd="/root/tp-agent/work/m24708/TensorLBM",
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

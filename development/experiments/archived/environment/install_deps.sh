#!/bin/bash
cd /srv/tensor/operator-b/TensorLBM
uv pip install fastapi uvicorn pydantic --python .venv/bin/python3
echo "DONE"

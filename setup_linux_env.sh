#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade "pip<27" "setuptools<70" wheel packaging ninja
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt

python -m pip install \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install \
  "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python - <<'PY'
import causal_conv1d
import mamba_ssm
import torch

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("causal-conv1d:", causal_conv1d.__version__)
print("mamba-ssm:", mamba_ssm.__version__)
PY

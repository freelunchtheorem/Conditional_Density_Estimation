#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cde-pytorch"
PYTHON_VERSION="3.12"
REPO_PATH="/Users/fabioferreira/.cursor/worktrees/Conditional_Density_Estimation/nnb"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; install Miniconda/Anaconda first."
  exit 1
fi

echo "Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "Sourcing conda to enable activation..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing PyTorch 2.9.1 and essentials (CPU only)..."
conda install -y -c pytorch pytorch=2.9.1 torchvision torchaudio cpuonly
echo "Installing compatible NumPy/SciPy builds..."
conda install -y -c conda-forge numpy=1.26.4 scipy=1.11.3
echo "Installing multidict/aiohttp dependencies via conda-forge..."
conda install -y -c conda-forge multidict aiohttp yarl

echo "Installing repository in editable mode..."
pip install -e "${REPO_PATH}"

echo "Done. Activate the environment with: conda activate ${ENV_NAME}"


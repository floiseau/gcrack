#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gcrack-local-test}"
PYTHON_VERSION="${2:-3.14}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CHANNEL="file://${SCRIPT_DIR}/build-artifacts"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found in PATH." >&2
  exit 1
fi

echo "Creating test environment '${ENV_NAME}' and installing gcrack from local channel"
conda create -n "${ENV_NAME}" -c "${LOCAL_CHANNEL}" -c conda-forge python="${PYTHON_VERSION}" gcrack -y

echo
echo "To validate in the new environment, run:"
echo "  conda activate ${ENV_NAME}"
echo "  conda list | grep -E 'gcrack|fenics-dolfinx|fenics-ufl|python-gmsh|jax|numpy'"
echo "  python -c \"import gcrack; print(gcrack.__name__)\""

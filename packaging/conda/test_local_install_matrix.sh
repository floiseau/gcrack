#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${1:-gcrack-local-test}"

PYTHON_VERSIONS=(3.10 3.11 3.12 3.13 3.14)

echo "Testing local conda install matrix for: ${PYTHON_VERSIONS[*]}"

for pyver in "${PYTHON_VERSIONS[@]}"; do
  env_name="${PREFIX}-${pyver//./}"
  echo
  echo "=== [${pyver}] Creating environment: ${env_name} ==="
  bash "${SCRIPT_DIR}/test_local_install.sh" "${env_name}" "${pyver}"

  echo "=== [${pyver}] Running smoke checks ==="
  conda run -n "${env_name}" python -m pip check
  conda run -n "${env_name}" python -c "import gcrack; print(gcrack.__name__)"
done

echo
echo "Matrix test completed successfully."

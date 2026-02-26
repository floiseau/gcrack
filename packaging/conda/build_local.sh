#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/build-artifacts"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found in PATH." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Building gcrack conda package from ${REPO_ROOT}"
conda build "${SCRIPT_DIR}" --override-channels -c conda-forge --output-folder "${OUTPUT_DIR}" "$@"

echo
echo "Build completed. Artifacts are in:"
echo "  ${OUTPUT_DIR}"

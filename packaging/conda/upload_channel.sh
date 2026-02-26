#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <anaconda-channel> [label]" >&2
  echo "Example: $0 floiseau" >&2
  echo "Example: $0 floiseau dev" >&2
  exit 1
fi

CHANNEL="$1"
LABEL="${2:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_GLOB="${SCRIPT_DIR}/build-artifacts/*/gcrack-*.conda"

shopt -s nullglob
artifacts=( ${ARTIFACT_GLOB} )
shopt -u nullglob

if [[ ${#artifacts[@]} -eq 0 ]]; then
  echo "Error: no package artifact found under ${SCRIPT_DIR}/build-artifacts" >&2
  echo "Run: bash packaging/conda/build_local.sh" >&2
  exit 1
fi

if ! command -v anaconda >/dev/null 2>&1; then
  echo "Error: anaconda client not found. Install with: conda install -c conda-forge anaconda-client" >&2
  exit 1
fi

if [[ -n "${ANACONDA_API_TOKEN:-}" ]]; then
  anaconda -t "${ANACONDA_API_TOKEN}" whoami >/dev/null
  AUTH_ARGS=( -t "${ANACONDA_API_TOKEN}" )
else
  echo "ANACONDA_API_TOKEN not set. Falling back to interactive login (anaconda login)."
  anaconda login
  AUTH_ARGS=()
fi

echo "Uploading ${#artifacts[@]} artifact(s) to channel '${CHANNEL}' with label '${LABEL}'"
anaconda "${AUTH_ARGS[@]}" upload --user "${CHANNEL}" --label "${LABEL}" "${artifacts[@]}"

echo "Upload complete."

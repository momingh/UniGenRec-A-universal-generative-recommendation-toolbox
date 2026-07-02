#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rqvae}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CHECK_ONLY=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

usage() {
  cat <<EOF
Usage: ./setup_env.sh [--check]

Creates or updates the conda environment used by this project.

Defaults:
  ENV_NAME=${ENV_NAME}
  PYTHON_VERSION=${PYTHON_VERSION}

Options:
  --check      Validate conda/env settings without creating or installing packages.
  -h, --help   Show this help message.

Environment overrides:
  ENV_NAME=myenv PYTHON_VERSION=3.10 ./setup_env.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH." >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Error: requirements.txt not found at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

env_python_version() {
  conda run -n "${ENV_NAME}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

if env_exists; then
  current_python="$(env_python_version)"
  if [[ "${current_python}" != "${PYTHON_VERSION}" ]]; then
    echo "Error: conda env '${ENV_NAME}' already exists with Python ${current_python}, expected ${PYTHON_VERSION}." >&2
    echo "Use a different ENV_NAME or recreate the existing environment." >&2
    exit 1
  fi
  echo "Using existing conda env '${ENV_NAME}' with Python ${current_python}."
else
  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "Conda env '${ENV_NAME}' does not exist. It would be created with Python ${PYTHON_VERSION}."
    exit 0
  fi
  echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Check passed. No packages were installed."
  exit 0
fi

echo "Upgrading pip tooling..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements..."
conda run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"

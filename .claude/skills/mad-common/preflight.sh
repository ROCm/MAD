#!/usr/bin/env bash
# Shared MAD pre-flight: ensure madengine is installed and we're at the repo root.
# Referenced by every GPU-touching mad-* skill via dynamic context injection so
# the check is inlined into the skill prompt before the work begins.
set -u

if ! command -v madengine &>/dev/null; then
  if [ -f requirements.txt ] && grep -q madengine requirements.txt; then
    echo "[pre-flight] madengine not found. Installing from requirements.txt..."
    pip install -r requirements.txt
  else
    echo "[pre-flight] madengine not found and requirements.txt is missing."
    echo "  Install:  pip install git+https://github.com/ROCm/madengine.git@main"
    echo "  Or clone MAD and run from its root (which has requirements.txt)."
    exit 1
  fi
fi

if [ ! -f models.json ]; then
  echo "[pre-flight] Warning: models.json not found — run from the MAD repo root."
fi

echo "[pre-flight] OK: madengine=$(command -v madengine), cwd=$(pwd)"

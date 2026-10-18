#!/usr/bin/env bash
# Shared MAD pre-flight: check madengine is installed and we're at the repo root.
# It never installs anything; it prints the install command and fails.
# Referenced by every GPU-touching mad-* skill via dynamic context injection
# (`bash .claude/skills/mad-common/preflight.sh`, relative to the repo root) so
# the check is inlined into the skill prompt before the work begins.
#
# The root models.json is the repo-root marker. Models are registered per
# directory (scripts/<dir>/models.json), so it is normally just [], but
# madengine still requires it to exist.
set -euo pipefail

if ! command -v madengine &>/dev/null; then
  # Never install into whatever Python environment happens to be active; say
  # how instead and let the user choose the environment.
  echo "[pre-flight] madengine is not installed (or not on PATH)."
  echo "  Install the version MAD pins, in the environment you run from:"
  echo "    pip install -r requirements.txt   # from the MAD repo root"
  echo "  then retry."
  exit 1
fi

if [ ! -f models.json ]; then
  root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -n "$root" ] && [ -f "$root/models.json" ]; then
    echo "[pre-flight] Not at the MAD repo root. cd ${root} and retry."
  else
    echo "[pre-flight] models.json not found — run from the MAD repo root (madengine requires the root models.json, even when it is [])."
  fi
  exit 1
fi

echo "[pre-flight] OK: madengine=$(command -v madengine), cwd=$(pwd)"

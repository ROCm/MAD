#!/usr/bin/env bash
# Regression: MAD jax-maxtext launches Primus via primus-cli, not examples/run_pretrain.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/../run.sh"

assert_file_contains() {
  local path="$1"
  local pattern="$2"
  if ! grep -qE -- "$pattern" "$path"; then
    echo "Expected $path to match /$pattern/" >&2
    return 1
  fi
}

assert_file_not_contains() {
  local path="$1"
  local pattern="$2"
  if grep -qE -- "$pattern" "$path"; then
    echo "Expected $path not to match /$pattern/" >&2
    return 1
  fi
}

if grep -vE '^\s*#' "$RUN_SH" | grep -qE -- 'examples/run_pretrain.sh'; then
  echo "$RUN_SH must not invoke examples/run_pretrain.sh" >&2
  exit 1
fi
assert_file_contains "$RUN_SH" '\$PRIMUS_ROOT/primus-cli'
assert_file_contains "$RUN_SH" 'train pretrain --config'
assert_file_contains "$RUN_SH" 'PRIMUS_SKIP_PIP'
assert_file_contains "$RUN_SH" 'MAXTEXT_PATH'

DOCKERFILE="$SCRIPT_DIR/../../../docker/primus_maxtext.ubuntu.amd.Dockerfile"
assert_file_contains "$DOCKERFILE" '/workspace/Primus/primus-cli'
assert_file_not_contains "$DOCKERFILE" 'examples/run_pretrain.sh'

echo "jax-maxtext run.sh uses primus-cli; image pip install stays skippable."

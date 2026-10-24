#!/bin/bash
# Entry point for sglang configs generated from InferenceX e2e scenarios by
# tools/infx2mad.py. The body is shared by every engine; see
# scripts/common/infx_run_lib.sh for why.
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common/infx_run_lib.sh" 2>/dev/null \
  || source ../scripts/common/infx_run_lib.sh
infx_main run_sglang.py "$@"

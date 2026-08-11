# shellcheck shell=bash
# Shared ROCm CLI resolution for MAD benchmark scripts.
# UTD / Primus images set ROCM_PATH to e.g. .../site-packages/_rocm_sdk_devel
# with tools under ${ROCM_PATH}/bin; classic ROCm uses /opt/rocm/bin.

ROCM_BIN="/opt/rocm/bin"
if [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocminfo" ]]; then
  ROCM_BIN="${ROCM_PATH}/bin"
elif [[ -x /opt/rocm/bin/rocminfo ]]; then
  ROCM_BIN="/opt/rocm/bin"
else
  shopt -s nullglob
  for _d in /opt/venv/lib/python*/site-packages/_rocm_sdk_devel/bin; do
    if [[ -x "${_d}/rocminfo" ]]; then
      ROCM_BIN="${_d}"
      break
    fi
  done
  shopt -u nullglob
fi
if [[ ! -x "${ROCM_BIN}/rocminfo" ]]; then
  _rocm_which="$(command -v rocminfo 2>/dev/null || true)"
  if [[ -n "${_rocm_which}" && -x "${_rocm_which}" ]]; then
    ROCM_BIN="$(dirname "${_rocm_which}")"
  fi
fi

export ROCM_BIN
export PATH="${ROCM_BIN}:${PATH}"

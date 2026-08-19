#!/usr/bin/env bash
# Step 0 preflight for mad-slurm-multinode.
# Prints a PASS/WARN/FAIL table of prerequisites. Exits non-zero if any HARD
# requirement (docker daemon, SLURM client) is missing.
set -u

pass=0; warn=0; fail=0
row() { printf '  %-6s %-22s %s\n' "$1" "$2" "$3"; }
ok()   { row "PASS" "$1" "$2"; pass=$((pass+1)); }
wn()   { row "WARN" "$1" "$2"; warn=$((warn+1)); }
bad()  { row "FAIL" "$1" "$2"; fail=$((fail+1)); }

echo "== mad-slurm-multinode preflight =="

# Python >= 3.10 (soft: a conda env is created later anyway)
if command -v python3 >/dev/null 2>&1; then
  pv=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
  if python3 -c 'import sys;exit(0 if sys.version_info[:2]>=(3,10) else 1)' 2>/dev/null; then
    ok "python3" "$pv (>= 3.10)"
  else
    wn "python3" "$pv (< 3.10; conda env 3.12 will be created)"
  fi
else
  wn "python3" "not found (conda env 3.12 will be created)"
fi

# Docker (HARD): present + daemon reachable
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "docker" "daemon reachable"
  else
    bad "docker" "present but 'docker info' failed (daemon/perms?)"
  fi
else
  bad "docker" "not found"
fi

# SLURM client (HARD)
if command -v sbatch >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1; then
  ok "slurm" "$(sinfo --version 2>/dev/null | head -1)"
else
  bad "slurm" "sbatch/sinfo not found"
fi

# conda / miniforge (soft: installed in Step 2 if missing)
if command -v conda >/dev/null 2>&1; then
  ok "conda" "$(conda --version 2>/dev/null)"
else
  # conda is not on PATH; an existing install may just need sourcing
  cphit=""
  for cp in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" \
            "/opt/conda" "${CONDA_PREFIX:-}" "${MAMBA_ROOT_PREFIX:-}"; do
    if [ -n "$cp" ] && [ -f "$cp/etc/profile.d/conda.sh" ]; then cphit="$cp"; break; fi
  done
  if [ -n "$cphit" ]; then
    wn "conda" "found at $cphit, not on PATH — source $cphit/etc/profile.d/conda.sh"
  else
    wn "conda" "not found (miniforge installed in Step 2)"
  fi
fi

# GPU SMI (soft: used for arch detection)
if command -v rocm-smi >/dev/null 2>&1; then
  ok "gpu-smi" "rocm-smi (AMD)"
elif command -v nvidia-smi >/dev/null 2>&1; then
  ok "gpu-smi" "nvidia-smi (NVIDIA)"
else
  wn "gpu-smi" "neither rocm-smi nor nvidia-smi found"
fi

# git (HARD: needed to clone/switch repos)
if command -v git >/dev/null 2>&1; then
  ok "git" "$(git --version 2>/dev/null)"
else
  bad "git" "not found"
fi

# HF token file (soft: required at run time for gated Llama-3.1)
if [ -s "$HOME/.huggingface/token" ]; then
  ok "hf-token" "~/.huggingface/token present"
else
  wn "hf-token" "~/.huggingface/token missing (needed before launch)"
fi

echo "== summary: $pass PASS, $warn WARN, $fail FAIL =="
if [ "$fail" -gt 0 ]; then
  echo "Hard requirements failed. This node is not ready for madengine SLURM runs."
  exit 1
fi
exit 0

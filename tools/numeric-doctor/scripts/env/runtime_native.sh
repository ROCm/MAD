# RUNTIME=native -- sourced by lib.sh; contract in runtime_docker.sh.
#
# No container: the command runs on the host, with the model's Python environment active.
# Profile knobs: VENV (a venv: bin/activate is sourced), CONDA_ENV (a conda env name, with
# conda on PATH), neither = the python3 on PATH.  For a workload that runs from a host venv.

rt_init() { ND_SCRATCH_RT=${NODE_SCRATCH:-/tmp}; }
rt_path() { printf '%s\n' "$1"; }

rt_prepare() {
  if [ -n "${VENV:-}" ]; then
    [ -r "$VENV/bin/activate" ] || nd_die "VENV=$VENV has no bin/activate"
  elif [ -n "${CONDA_ENV:-}" ]; then
    command -v conda >/dev/null || nd_die "CONDA_ENV=$CONDA_ENV but no conda on PATH"
  fi
}

rt_exec() {
  local name=$1 gpus=$2 workdir=$3 cmd=$4 act=""
  if [ -n "${VENV:-}" ]; then act="source '$VENV/bin/activate' && "
  elif [ -n "${CONDA_ENV:-}" ]; then act="eval \"\$(conda shell.bash hook)\" && conda activate '$CONDA_ENV' && "
  fi
  # its own process group, so rt_stop's TERM reaches the command and everything it starts.
  # The trap makes the wrapper shell wait for the command and exit with ITS rc, skipping the
  # rest of a compound command; an untrapped bash dies at once and nd returned while the
  # command was still stopping.
  nd_exec_bg "$name" setsid -w env "${ND_RUN_ENV[@]}" bash -c "trap 'exit \$?' TERM INT; cd '$workdir' && $act$cmd"
}

rt_stop() { kill -TERM -- "-$2" 2>/dev/null || kill -TERM "$2" 2>/dev/null || true; }
rt_kill() { :; }

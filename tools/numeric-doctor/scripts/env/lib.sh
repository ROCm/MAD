# numeric-doctor execution environment: shared library.  SOURCE this file (scripts/nd does).
#
# Two independent layers, each chosen by the profile (ND_ENV):
#   RUNTIME = docker | podman | apptainer | native   -> env/runtime_<RUNTIME>.sh  (nd run)
#   SCHED   = slurm | local                          -> env/sched_<SCHED>.sh      (nd submit/status/wait/stop)
#
# Paths.  On the host: ND_REPO_HOST (the checkout), ND_ART_HOST (the investigation's
# artifact dir).  Inside a runtime, every command sees ND_REPO and ND_ART -- the same
# directories as that runtime mounts them (host paths for docker/podman/native, /mnt/...
# for apptainer, whose read-only SIF cannot mount a path that does not exist in it).
# Commands are written with those variables, never with host paths:
#     nd run -- 'python3 $ND_REPO/scripts/replay_nan_op.py $ND_ART/capsules/t0/x.pt'
# `nd path <host path>` translates one host path for a command built on the host.

ND_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ND_REPO_HOST=${ND_REPO_HOST:-$(cd -- "$ND_LIB_DIR/../.." && pwd -P)}
export ND_REPO_HOST

# Where the Python tools live, relative to the repo.  Phase 3 of the restructure
# (infra_restructure_ledge.md) flattens them into scripts/: then this is "scripts".
# A step-6 package already uses the flat layout, so "scripts" is searched too.
ND_PYDIRS=${ND_PYDIRS:-scripts}

nd_die() { echo "nd: $*" >&2; exit 2; }

# The command of `nd run -- <command>` / `exec.sh ... -- <command>` as ONE bash -c string.  One
# argument is the command as written (the documented form: '...' with $ND_REPO / $ND_ART).  Several
# arguments are words: one with whitespace or a shell character is double-quoted, so
# `--contend-with '<cmd with spaces>'` stays one word and '[%s]\n' keeps its backslash ($ND_REPO /
# $ND_ART in a word still expand inside the runtime).  A plain "$*" re-split
# it: replay_nan_op printed its usage and the job ended EXEC_RC=126 (2026-11-20).
nd_join_cmd() {
  [ $# = 1 ] && { printf '%s' "$1"; return; }
  local a q out=""
  for a in "$@"; do
    if [[ ! $a =~ ^[][A-Za-z0-9_@%+=:,./{}\$-]+$ ]]; then
      q=${a//\\/\\\\}; q=${q//\"/\\\"}; q=${q//\`/\\\`}; a="\"$q\""
    fi
    out+="${out:+ }$a"
  done
  printf '%s' "$out"
}
nd_say() { echo "[nd $(date +%T)] $*" >&2; }

# What the banners print as the image: the one the runtime really uses.
nd_image_desc() {
  case ${RUNTIME:-} in
    apptainer) echo "${SIF:-${IMAGE:-?}}" ;;
    native) echo "${VENV:-${CONDA_ENV:-host}}" ;;
    *) echo "${IMAGE:-?}" ;;
  esac
}

# ---- profile ------------------------------------------------------------------
# A profile is a shell file of VAR=VALUE lines (scripts/env/profiles/*.env, or the
# investigation's own <ARTIFACT_DIR>/env.sh written by `nd profile`).  Variables already
# set in the caller's environment win over the file, so one submit can override a value
# (IMAGE=... nd run ...) without editing it.  Within the file a later line wins: `nd profile`
# appends its K=V overrides after the site profile's lines [phase 4, bug A].
nd_load_profile() {
  local f=${ND_ENV:-}
  if [ -z "$f" ]; then
    nd_say "ND_ENV unset: SCHED=${SCHED:-local}, RUNTIME=${RUNTIME:-auto}"
  else
    [ -r "$f" ] || nd_die "ND_ENV=$f is not readable"
    f=$(cd -- "$(dirname -- "$f")" && pwd -P)/$(basename -- "$f")
    export ND_ENV=$f
    local line key
    local -A caller=()
    while IFS= read -r line || [ -n "$line" ]; do        # the caller's keys, before the file sets any
      [[ ${line%%#*} =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
      key=${BASH_REMATCH[1]}
      [ -n "${!key+x}" ] && caller[$key]=1
    done < "$f"
    while IFS= read -r line || [ -n "$line" ]; do
      line=${line%%#*}
      [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
      key=${BASH_REMATCH[1]}
      [ -n "${caller[$key]:-}" ] && continue             # caller's value wins
      eval "export $line"                                # values may use ${VAR} of earlier lines
    done < "$f"
  fi
  SCHED=${SCHED:-local}
  if [ -z "${RUNTIME:-}" ] || [ "$RUNTIME" = auto ]; then
    RUNTIME=native
    local r; for r in docker podman apptainer; do command -v "$r" >/dev/null && { RUNTIME=$r; break; }; done
  fi
  export SCHED RUNTIME
  [ -f "$ND_LIB_DIR/runtime_$RUNTIME.sh" ] || nd_die "unknown RUNTIME=$RUNTIME (docker|podman|apptainer|native)"
  [ -f "$ND_LIB_DIR/sched_$SCHED.sh" ] || nd_die "unknown SCHED=$SCHED (slurm|local)"
  # The artifact dir: explicit, else the directory of an investigation's env.sh.
  if [ -z "${ND_ART_HOST:-}" ] && [ -n "${ND_ENV:-}" ] && [ "$(basename "$ND_ENV")" = env.sh ]; then
    ND_ART_HOST=$(dirname "$ND_ENV")
  fi
  [ -n "${ND_ART_HOST:-}" ] && ND_ART_HOST=$(cd -- "$ND_ART_HOST" 2>/dev/null && pwd -P || echo "$ND_ART_HOST")
  export ND_ART_HOST
  # shellcheck disable=SC1090
  source "$ND_LIB_DIR/runtime_$RUNTIME.sh"
  # shellcheck disable=SC1090
  source "$ND_LIB_DIR/sched_$SCHED.sh"
}

# ---- helpers shared by the runtimes ---------------------------------------------
nd_slug() { tr '/:@' '___' <<< "$1"; }

# Port per first-listed GPU so concurrent single-GPU runs do not all claim 29500.
# (A function: GPUS=all breaks $((29500 + ${GPUS%%,*})) under set -u.)
nd_port() { local g=${1%%,*}; case $g in ''|*[!0-9]*) echo 29500 ;; *) echo $((29500 + g)) ;; esac; }

# A lock that works on NFS and Lustre (flock over NFS has wedged jobs): mkdir is
# atomic everywhere.  A lock older than ND_LOCK_STALE seconds (default 3 h) is broken.
nd_lock() {
  local d=$1.lock i=0
  # a missing parent made mkdir fail forever, read as "held"
  mkdir -p "$(dirname -- "$d")" || nd_die "cannot create $(dirname -- "$d") for lock $d"
  while ! mkdir "$d" 2>/dev/null; do
    [ -d "$d" ] || nd_die "cannot create lock $d"
    if [ -d "$d" ] && [ $(( $(date +%s) - $(stat -c %Y "$d" 2>/dev/null || date +%s) )) -gt "${ND_LOCK_STALE:-10800}" ]; then
      nd_say "breaking stale lock $d"; rmdir "$d" 2>/dev/null; continue
    fi
    [ $((i % 60)) = 0 ] && nd_say "waiting for lock $d"
    i=$((i + 1)); sleep 5
  done
  echo "$$ $(hostname)" > "$d/owner"
}
nd_unlock() { rm -rf "$1.lock"; }

# Decompress an image archive to stdout (.zst with zstd, else the ctypes fallback for
# hosts with libzstd but no zstd binary; .gz; plain .tar).
nd_unpack() {
  case $1 in
    *.zst|*.zstd)
      if command -v zstd >/dev/null; then zstd -dcq "$1"
      else python3 "$ND_LIB_DIR/zstdcat.py" "$1"; fi ;;
    *.gz) zcat "$1" ;;
    *) cat "$1" ;;
  esac
}

# Values of --env may name $ND_REPO / $ND_ART; the runtimes pass env literally, so
# substitute the in-runtime paths here.
nd_subst() {
  local v=$1
  v=${v//\$\{ND_REPO\}/$ND_REPO_RT}; v=${v//\$ND_REPO/$ND_REPO_RT}
  if [ -n "${ND_ART_RT:-}" ]; then v=${v//\$\{ND_ART\}/$ND_ART_RT}; v=${v//\$ND_ART/$ND_ART_RT}; fi
  printf '%s\n' "$v"
}

# The environment every command gets, whatever the runtime: fills ND_RUN_ENV (an array
# of VAR=VALUE) from the defaults, the profile's EXTRA_ENV and the caller's --env list.
#   nd_build_env <name> <gpus> [VAR=VALUE ...]
nd_build_env() {
  local name=$1 gpus=$2 kv pyp="" d extra_pp=""; shift 2
  ND_REPO_RT=$(rt_path "$ND_REPO_HOST")
  ND_ART_RT=""; [ -n "${ND_ART_HOST:-}" ] && ND_ART_RT=$(rt_path "$ND_ART_HOST")
  for d in $ND_PYDIRS; do pyp=${pyp:+$pyp:}$ND_REPO_RT/$d; done
  ND_RUN_ENV=(
    "ND_REPO=$ND_REPO_RT" "NUMERIC_DOCTOR_ROOT=$ND_REPO_RT" "PYTHONUNBUFFERED=1"
    "USER=$(id -un)" "LOGNAME=$(id -un)"
    "TORCHINDUCTOR_CACHE_DIR=${ND_SCRATCH_RT:-/tmp}/inductor-$name"
    "MASTER_PORT=$(nd_port "$gpus")"
  )
  [ -n "$ND_ART_RT" ] && ND_RUN_ENV+=("ND_ART=$ND_ART_RT")
  # the job id for names and out dirs inside the runtime (SLURM_* itself stays out: launchers
  # that see SLURM_NTASKS & co. switch to Slurm-driven distributed init)
  [ -n "${SLURM_JOB_ID:-}" ] && ND_RUN_ENV+=("ND_JOB_ID=$SLURM_JOB_ID")
  [ "$gpus" != all ] && ND_RUN_ENV+=("HIP_VISIBLE_DEVICES=$gpus")
  [ -n "${GPU_IDS:-}" ] && ND_RUN_ENV+=("ROCR_VISIBLE_DEVICES=$GPU_IDS")
  # A multi-node run (nd_multinode: one `nd run` per node under srun): the node count, this
  # node's rank and the rendezvous host, for the command's torchrun --nnodes / --node_rank /
  # --master_addr.  The rank is this host's place in the launch's own host list (ND_NODE_HOSTS):
  # a group of nodes from an offset (rate.sh) must count from 0 whatever SLURM_NODEID says.
  if [ -n "${ND_NNODES:-}" ]; then
    local r=${SLURM_NODEID:-0} i=0 h me; me=$(hostname -s)
    for h in ${ND_NODE_HOSTS//,/ }; do [ "$h" = "$me" ] && { r=$i; break; }; i=$((i + 1)); done
    ND_RUN_ENV+=("ND_NNODES=$ND_NNODES" "ND_NODE_RANK=$r" "ND_MASTER_ADDR=${ND_MASTER_ADDR:-localhost}")
    # Cross-node sockets on the interface that routes to the other nodes. A container sees no RDMA device
    # (no /dev/infiniband), so RCCL falls back to TCP on every interface it finds -- here first the backend
    # RDMA NICs (benic*, 192.168.x), which do not carry TCP between nodes: a 2-node vLLM PP group hung in
    # ncclSocketPollConnect "Connection timed out" (2026-11-24). The hostnames resolve to the routable network:
    # the interface the kernel uses for a peer's address is the one. EXTRA_ENV / the profile may set them.
    if [[ " ${EXTRA_ENV:-} " != *" NCCL_SOCKET_IFNAME="* ]] && [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
      local peer="" ip dev
      for h in ${ND_NODE_HOSTS//,/ }; do [ "$h" != "$me" ] && { peer=$h; break; }; done
      ip=$(getent hosts "${peer:-$ND_MASTER_ADDR}" 2>/dev/null | awk '{print $1; exit}')
      dev=$([ -n "$ip" ] && ip -o route get "$ip" 2>/dev/null | grep -oP '\bdev \K\S+')
      [ -n "$dev" ] && [ "$dev" != lo ] && ND_RUN_ENV+=("NCCL_SOCKET_IFNAME=$dev" "GLOO_SOCKET_IFNAME=$dev")
    fi
    # This node's address on that network: frameworks that advertise their own IP to peers pick it themselves and
    # can pick an unroutable one -- vLLM's headless workers did (the head's remote message queues stayed None,
    # 2026-11-24); VLLM_HOST_IP is vLLM's override. ND_NODE_IP for any other framework.
    local mip; mip=$(getent hosts "$me" 2>/dev/null | awk '{print $1; exit}')
    if [ -n "$mip" ]; then
      ND_RUN_ENV+=("ND_NODE_IP=$mip")
      [[ " ${EXTRA_ENV:-} " != *" VLLM_HOST_IP="* ]] && [ -z "${VLLM_HOST_IP:-}" ] && ND_RUN_ENV+=("VLLM_HOST_IP=$mip")
    fi
  fi
  # shellcheck disable=SC2086
  for kv in ${EXTRA_ENV:-} "$@"; do
    [ -n "$kv" ] || continue
    kv=$(nd_subst "$kv")
    case $kv in
      PYTHONPATH=*) extra_pp=${extra_pp:+$extra_pp:}${kv#PYTHONPATH=} ;;   # prepended, never replaces
      *) ND_RUN_ENV+=("$kv") ;;
    esac
  done
  ND_RUN_ENV+=("PYTHONPATH=${extra_pp:+$extra_pp:}$pyp")
  # one value per variable, the last one given: a caller's --env MASTER_PORT=... (rate.sh's
  # copies side by side) replaces the default instead of riding along as a duplicate
  local -A seen=(); local out=() i
  for ((i = ${#ND_RUN_ENV[@]} - 1; i >= 0; i--)); do
    kv=${ND_RUN_ENV[$i]}; [ -n "${seen[${kv%%=*}]:-}" ] && continue
    seen[${kv%%=*}]=1; out=("$kv" "${out[@]}")
  done
  ND_RUN_ENV=("${out[@]}")
}

# ---- artifact-dir job records (nd submit / status / wait / stop) -----------------
nd_jobs_dir() {
  [ -n "${ND_ART_HOST:-}" ] || nd_die "no artifact dir: set ND_ART_HOST or use an investigation's env.sh as ND_ENV"
  mkdir -p "$ND_ART_HOST/jobs" && echo "$ND_ART_HOST/jobs"
}
# ---- nodes that already hold IMAGE (docker/podman: images are node-local) ---------------------
# A new node pulls the image before the first command: ~2 min for a 20 GB vLLM image, 4 of the 18
# minutes to a reproducer on 2026-11-19 (three jobs, three fresh nodes).  rt_prepare's success
# records "<host> <IMAGE>" in <ART>/jobs/image_nodes; nd submit pins an idle one (nd_pick_image_node).
nd_image_nodes_file() { echo "$ND_ART_HOST/jobs/image_nodes"; }
nd_note_image_node() {
  case $RUNTIME in docker|podman) ;; *) return 0 ;; esac
  [ -n "${ND_ART_HOST:-}" ] && [ -d "$ND_ART_HOST/jobs" ] || return 0
  local f h l; f=$(nd_image_nodes_file); h=$(hostname -s); l="$h $IMAGE${IMAGE_ID:+ ${IMAGE_ID#sha256:}}"
  grep -qxF "$l" "$f" 2>/dev/null || echo "$l" >>"$f" 2>/dev/null
  return 0
}
# nd_image_stale <id on this node>: IMAGE_ID (build_image.sh) names the build this investigation
# made; the same tag on a node can hold another investigation's build of the same Dockerfile, and
# it was used as is (2026-11-20 sglang: pinned to a node holding the 11-19 build, the ID never
# checked).  Stale only where IMAGE_TAR can replace it -- off-site a package builds its own.
nd_image_stale() {
  [ -n "${IMAGE_ID:-}" ] && [ -r "${IMAGE_TAR:-}" ] && [ "${1#sha256:}" != "${IMAGE_ID#sha256:}" ]
}
# nd_pick_image_node: an idle node that holds IMAGE -- this investigation's nodes newest first, then
# the sibling investigations' (<ART>/../*/jobs/image_nodes: which node holds an image is a cluster
# fact; a pruned image is just pulled again; with IMAGE_ID set, only lines with that build's ID) -- not excluded, and not claimed by another submit in the
# last 120 s: two jobs submitted together must not queue for the same node.
nd_pick_image_node() {
  local f c now h img iid t
  f=$(nd_image_nodes_file); c=$f.claims; now=$(date +%s)
  mkdir -p "$(dirname "$f")"     # a fresh artifact dir has no jobs/ yet: the claim's append failed (2026-11-20)
  while read -r h img iid; do
    [ "$img" = "$IMAGE" ] || continue
    # a built image: the tag is not the build -- only a node recorded with this build's ID
    [ -z "${IMAGE_ID:-}" ] || [ "$iid" = "${IMAGE_ID#sha256:}" ] || continue
    case ",${ND_EXCLUDE:-}," in *",$h,"*) continue ;; esac
    t=$(awk -v h="$h" '$1 == h {t = $2} END {print t + 0}' "$c" 2>/dev/null)
    [ $((now - ${t:-0})) -ge 120 ] || continue
    [ "$(sch_node_state "$h")" = idle ] || continue
    echo "$h $now" >>"$c"; echo "$h"; return 0
  done < <(tac "$f" 2>/dev/null; ls -t "$(dirname "$ND_ART_HOST")"/*/jobs/image_nodes 2>/dev/null | grep -vxF "$f" |
           xargs -r cat 2>/dev/null)
  return 1
}

# A job is named by NAME (its record <ART>/jobs/NAME.job), by <sched>:<id>, or by the bare
# scheduler id the ledger records (`job=144302`): a record with that id, else the scheduler's own
# log path -- so a resume from a ledger line, or a job sbatch'ed by hand (a step-6/8 package's
# run.sh / validate.sh), can be waited on too (2026-11-20: `nd status 144302` -> no job record).
nd_job_record() {
  local ref=$1 f jd id
  jd=$(nd_jobs_dir); f=$jd/$ref.job
  if [ -r "$f" ]; then cat "$f"; return; fi
  case $ref in
    *:*) id=$ref ;;
    *[!0-9]*|"") nd_die "no job record $f (a name from $jd, a scheduler id, or nd adopt NAME ID)" ;;
    *) id=$SCHED:$ref ;;
  esac
  f=$(grep -lxF "id=$id" "$jd"/*.job 2>/dev/null | head -1)
  if [ -n "$f" ]; then cat "$f"; return; fi
  printf 'id=%s\nlog=%s\n' "$id" "$(sch_job_log "$id" 2>/dev/null)"
}

# Run a runtime's command line in the background and wait for it, so this shell can still
# act on a signal (Slurm's --signal=B:TERM@60, nd stop, the wave's early stop) and pass it
# on gracefully through rt_stop.  The process is the runtime itself, not a pipe's tail, so
# its exit code is the command's (a `... | tee &` would report tee's).
#   nd_exec_bg <name> <argv...>
nd_exec_bg() {
  local name=$1 bg rc stopping=0; shift
  "$@" &
  bg=$!
  trap '[ "$stopping" = 1 ] || { stopping=1; rt_stop "'"$name"'" '"$bg"'; }' TERM INT
  nd_wait_pid "$bg"; rc=$?
  [ "$stopping" = 1 ] && [ "$rc" = 127 ] && rc=143
  trap - TERM INT
  return "$rc"
}

# Wait for a child until it has really ended and return its exit code.  A trapped signal
# makes `wait` return early (>128) with the child still running; a second signal (local
# `nd stop` plus a job's own forwarding trap) used to cut the second `wait` short too, so a job
# reported the rc and ended while its command was still shutting down.
nd_wait_pid() {
  local rc
  while :; do
    wait "$1" 2>/dev/null; rc=$?
    kill -0 "$1" 2>/dev/null || return "$rc"
  done
}

# In-container wrapper for runtimes that signal only the top process (apptainer forwards TERM
# to its child, podman's catatonit to its child): `bash -c "$ND_TERM_WRAP" nd-wrap "<cmd>"`.
# A plain `bash -c "<cmd>; echo"` died on TERM and python never saw it; a bash trap alone waits
# for the foreground child without passing TERM on.  So the command runs in its own process
# group, TERM goes to that whole group, and the inner trap makes a compound command wait for
# the running step and exit with its rc, skipping the rest.
# shellcheck disable=SC2016
ND_TERM_WRAP='c=; trap '\''[ -n "$c" ] && kill -TERM -- -"$c" 2>/dev/null'\'' TERM INT
command -v setsid >/dev/null 2>&1 || exec bash -c "trap '\''exit \$?'\'' TERM INT; $1"
setsid bash -c "trap '\''exit \$?'\'' TERM INT; $1" & c=$!
while :; do wait "$c"; rc=$?; kill -0 "$c" 2>/dev/null || break; done; exit "$rc"'

# ---- helpers for job scripts (scripts/jobs/*.sh source this file) -----------------
# A tool's HOST path, wherever the restructure has put it (ND_PYDIRS, then scripts/).
nd_tool() {
  local d
  for d in $ND_PYDIRS scripts scripts/env; do
    [ -e "$ND_REPO_HOST/$d/$1" ] && { echo "$ND_REPO_HOST/$d/$1"; return 0; }
  done
  nd_die "tool $1 not found under $ND_REPO_HOST ($ND_PYDIRS scripts)"
}
# The inverse of rt_path, for paths a runtime wrote into a file (groups.json records them as
# the container saw them).  Identity unless the runtime overrides it (apptainer).
rt_hostpath() { printf '%s\n' "$1"; }
# GPU_IDS (profile or env): the host's physical GPUs this run may use, e.g. 1,2,3,4,5,6,7 when
# GPU 0 belongs to someone else. The runtime sees only those (ROCR_VISIBLE_DEVICES), so logical
# ids 0..n-1 mean the listed devices in order, everywhere ("GPU 0" in a log is the first one).
nd_gpu_ids_count() { echo "$GPU_IDS" | tr ',' '\n' | grep -c .; }
# GPUs on this node: GPU_IDS, else NGPUS, else Slurm's count, else /sys/class/kfd GPU nodes, else 8.
nd_ngpus() {
  if [ -n "${GPU_IDS:-}" ]; then nd_gpu_ids_count; return; fi
  if [ -n "${NGPUS:-}" ]; then echo "$NGPUS"; return; fi
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then echo "$SLURM_GPUS_ON_NODE"; return; fi
  local n=0 p
  for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    grep -qE '^gfx_target_version [1-9]' "$p" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && echo "$n" || echo 8
}
# One multi-rank run over <nodes> of the job's nodes, from node <offset> (default 0) on (wave.sh,
# model_ab.sh, rate.sh): fills ND_LAUNCH with the prefix that runs a following
# `"$ND" run --gpus all ...` once per node -- srun on those nodes, or nothing more than the env on
# this node alone -- and ND_HOSTS.  The prefix ends in `env ND_NNODES=<n> ND_MASTER_ADDR=<first
# node>`, which nd_build_env hands every `nd run` with ND_NODE_RANK: the command's torchrun
# --nnodes / --node_rank / --master_addr.  Nothing is exported, so groups of nodes running side
# by side (rate.sh) do not share them.
#   nd_multinode <nodes> [offset];  "${ND_LAUNCH[@]}" "$ND" run --gpus all ... -- '<cmd>'
nd_multinode() {
  local n=$1 off=${2:-0} hosts=() g="" addr=localhost
  [[ $n =~ ^[1-9][0-9]*$ ]] || nd_die "nodes=$n is not a positive integer"
  [[ $off =~ ^[0-9]+$ ]] || nd_die "node offset $off is not a number"
  ND_LAUNCH=(); ND_HOSTS=$(hostname -s)
  if [ "$n" -gt 1 ] || [ "$off" -gt 0 ]; then
    [ -n "${SLURM_JOB_NODELIST:-}" ] || nd_die "$n nodes from node $off need a Slurm allocation (nd submit --nodes ...)"
    mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    [ "${#hosts[@]}" -ge $((off + n)) ] || nd_die "nodes $off..$((off + n - 1)) asked, job ${SLURM_JOB_ID:-?} has ${#hosts[@]}"
    addr=${hosts[$off]}
    ND_HOSTS=$(IFS=,; echo "${hosts[*]:$off:$n}")
    [ -n "${SLURM_GPUS_FLAG:-}" ] && g=${SLURM_GPUS_FLAG//\{n\}/$(nd_ngpus)}
    # --wait, not --kill-on-bad-exit: when the capture stops a job, one node's torchrun exits 1
    # first and an immediate kill cut the others off mid-log (TRIAL_RC=143: read as DIED, external
    # kill -- node-count ladder test).  ND_NODE_GRACE seconds (60) after the first node
    # ends, srun ends the rest: a coordinated stop finishes, a real crash does not hang the others
    # in RCCL.  --overlap: copies of a job on the same nodes (wave.sh's k `all` lines, rate.sh
    # COPIES) each ask for every GPU; without it Slurm 26.05 queued the second step behind the
    # first (probe: 40 s vs 20 s for two 20-s steps; both see all 8 GPUs either way).
    # shellcheck disable=SC2206
    ND_LAUNCH=(srun --nodes="$n" --ntasks-per-node=1 --nodelist="$ND_HOSTS"
               --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" $g --kill-on-bad-exit=0 --overlap
               --wait="${ND_NODE_GRACE:-60}" --export=ALL)
  fi
  ND_LAUNCH+=(env "ND_NNODES=$n" "ND_MASTER_ADDR=$addr" "ND_NODE_HOSTS=$ND_HOSTS")
}
# Copies of a multi-rank job side by side on the same GPUs (rate.sh COPIES, wave.sh's k `all`
# lines) under apptainer --pid: every container numbers its processes from 1, so the copies' ranks
# share PIDs and RCCL's intra-node P2P setup fails -- `hipIpcGetMemHandle failed : invalid
# argument`, HIP FATAL, within seconds (a probe; the same 2 copies without --pid ran and
# NaN'd).  Drops --pid from APPTAINER_FLAGS for this job's `nd run`s (the caller's value
# wins over the profile's); returns 0 if it did.  Keep declaring NAN_CAPTURE_CONTAINERS_PER_GPU=k:
# without --pid the capture's scan still recorded gpu_procs=1 per rank (2 copies), so
# a declared 1 would replay one process per GPU where the capture had k.
nd_drop_pid_ns() {
  [ "${RUNTIME:-}" = apptainer ] || return 1
  case " ${APPTAINER_FLAGS:-} " in *" --pid "*) ;; *) return 1 ;; esac
  APPTAINER_FLAGS=$(printf '%s\n' "$APPTAINER_FLAGS" | tr ' ' '\n' | grep -vx -- '--pid' | paste -sd' ')
  export APPTAINER_FLAGS
}
# CAPTURE_ENV's NAN_CAPTURE_SUSPEND must be "pkg.module:Qual.name" entries: checked before any GPU
# work, because a wrong form ("Class.attr", no module) only fails inside every rank at its first
# step, as a bare ModuleNotFoundError that ends the wave as an ERROR trial and NO_CAPSULE.
nd_check_capture_env() {
  local kv v e
  # shellcheck disable=SC2086
  for kv in ${CAPTURE_ENV:-}; do
    case $kv in NAN_CAPTURE_SUSPEND=*) v=${kv#*=} ;; *) continue ;; esac
    for e in ${v//,/ }; do
      [[ $e =~ ^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$ ]] ||
        nd_die "CAPTURE_ENV NAN_CAPTURE_SUSPEND entry '$e': expected pkg.module:Qual.name (module, a colon, then the attribute; e.g. torchtitan.models.common.attention:FlexAttention.compiled_flex_attn)"
    done
  done
}
# Load the library in a job script: profile, runtime, scheduler, and $ND (the nd command).
nd_job_init() {
  nd_load_profile
  rt_init
  # One tag per job, inherited by the job scripts it runs (capture_to_repro -> wave, ladder):
  # their files are named by it and found by it.  $$ differs per script, so locally (no
  # SLURM_JOB_ID) the parent never found the ladder's summary.txt.
  ND_JOB_TAG=${ND_JOB_TAG:-${SLURM_JOB_ID:-$$}}
  export ND_JOB_TAG
  ND=$ND_REPO_HOST/scripts/nd
  export ND
}

# Run one trial in the background with its output in LOG and a last line TRIAL_RC=<rc>.
# A plain `( cmd; echo TRIAL_RC=$? ) &` does not pass SIGTERM on: the subshell dies and its
# foreground `nd run` keeps running, orphaned.  This one forwards it, then records the real rc.
#   nd_bg_trial <log> <argv...>      ($! is the trial's handle: kill -TERM it to stop gracefully)
nd_bg_trial() {
  local log=$1; shift
  ( c=""; trap '[ -n "$c" ] && kill -TERM "$c" 2>/dev/null' TERM INT
    "$@" & c=$!
    nd_wait_pid "$c"; rc=$?
    echo "TRIAL_RC=$rc" ) >"$log" 2>&1 &
}

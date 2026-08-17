#!/bin/bash
###############################################################################
# launch_disagg_skyriver.sh — non-SLURM driver for MAD vllm_dissag on skyRiver.
#
# Replicates what run_xPyD_models.slurm does (node pick, IP resolve, per-node
# `docker run` env plumb) but over our SSH+docker mesh — skyRiver has no SLURM.
# Reuses vllm_disagg.sh + connectors/moriio.sh UNCHANGED (they are env-driven).
#
# Usage:
#   PREFILL="skyriver04" DECODE="skyriver07" \
#   MODEL_NAME=GLM-5.2-FP8 MODEL_PATH=/models/GLM-5.2-FP8 \
#   ./launch_disagg_skyriver.sh
#
#   PREFILL="skyriver04,skyriver05" DECODE="skyriver06,skyriver07" ...   # 2P/2D
#
# Env:
#   PREFILL / DECODE   comma-lists of node hostnames (order = NODE_RANK order:
#                      prefill nodes first [ranks 0..xP-1], then decode [xP..]).
#   MODEL_NAME         key in models.yaml (recipe). MODEL_PATH: weights dir (uniform /models path).
#   CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori   (defaults here = MoRI-EP wideEP).
#   IMAGE              docker image (default localhost/rocmshared/mori-wideep-glm:v027).
#   FABRIC_NET         fabric subnet prefix for MASTER_ADDR/IPADDRS (default 192.168.200).
#   FABRIC_PROFILE     per-fabric connector env overlay, loaded AFTER connectors/
#                      <CONNECTOR>.env (default thor2 -> connectors/moriio.thor2.env).
#                      Set to "-" to use the base connector .env alone.
#   MASTER_PORT/PROXY_PORT  defaults 23731 / 8000.
#   DRY_RUN=1          print the per-node docker run commands, do not execute.
#   EXTRA_MOUNTS       extra `docker run` args, injected verbatim (expanded LOCALLY,
#                      so quote accordingly). Intended for hot-patching a file into
#                      the image without a rebuild, e.g. overlaying a single .py:
#                        EXTRA_MOUNTS="-v /models/common/patches/rocm_aiter_mla.py:\
#                        /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/\
#                        backends/mla/rocm_aiter_mla.py:ro"
#                      The file must already exist at that path on EVERY node, else
#                      docker creates a DIRECTORY there and the import fails.
###############################################################################
set -u

PREFILL="${PREFILL:?set PREFILL=comma-list of prefill node hostnames}"
DECODE="${DECODE:?set DECODE=comma-list of decode node hostnames}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME (models.yaml key)}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH (weights dir, e.g. /models/GLM-5.2-FP8)}"
CONNECTOR="${CONNECTOR:-moriio}"
WIDE_EP="${WIDE_EP:-1}"
EP_BACKEND="${EP_BACKEND:-mori}"
# v027-bnxt238, NOT plain v027. MoRI-EP uses RdmaBackendType::DirectVerbs
# (context.cpp:190), whose RdmaDeviceFactory vendor-switches on Broadcom and calls
# BnxtDvApi::Available() -> dlopen libbnxt_re.so + dlsym 8 bnxt_re_dv_* symbols
# (dv_loader.hpp:132-147). Stock rdma-core ships the *ibverbs driver* but NOT those
# direct-verbs extensions: in plain v027 libbnxt_re-rdmav59.so loads yet all 8
# symbols are MISSING (verified 2026-08-15), so the factory returns nullptr for
# every NIC and MoRI-EP's device list is EMPTY.
# This is SILENT at EP8: intra-node EP uses kernel_type=IntraNode and posts no
# verbs, so the run serves fine while logging 64x "BNXT device detected but
# libbnxt_re.so not available at runtime" (8 devices x 8 ranks). It is FATAL at
# EP16, where kernel_type=InterNode needs real QPs -> "no rdma device found".
# MoRIIO is unaffected either way: it builds RdmaContext(IBVerbs)
# (backend_impl.cpp:1502), and the IBVerbs branch returns IBVerbsDevice with no
# vendor check, so the cross-node KV path never needed libbnxt_re.
IMAGE="${IMAGE:-localhost/rocmshared/mori-wideep-glm:v027-bnxt238}"
FABRIC_NET="${FABRIC_NET:-192.168.200}"
MASTER_PORT="${MASTER_PORT:-23731}"
PROXY_PORT="${PROXY_PORT:-8000}"
COOKBOOK_IN_CTR="/workspace/vllm_dissag"          # where we mount the scripts inside the container
HOST_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"   # this dir on the host (per node via /models/common)
JOB_ID="skyriver_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="/models/common/logs/${JOB_ID}"

IFS=',' read -ra P_NODES <<< "$PREFILL"
IFS=',' read -ra D_NODES <<< "$DECODE"
ALL_NODES=( "${P_NODES[@]}" "${D_NODES[@]}" )
xP=${#P_NODES[@]}
yD=${#D_NODES[@]}
NNODES=$(( xP + yD ))

# Resolve each node's fabric IP LIVE (the address it holds on the FABRIC_NET rail),
# rather than from a hostname->octet table: a table goes stale the moment a node is
# re-addressed or a different set of machines is used, and it silently hands the wrong
# IP to MASTER_ADDR/IPADDRS. OCT below is only a fallback for a node we cannot reach.
declare -A OCT=( [skyriver04]=55 [skyriver05]=52 [skyriver06]=105 [skyriver07]=61 )
IPS=()
for n in "${ALL_NODES[@]}"; do
  ip4="$(ssh -n -o ConnectTimeout=5 "$n" \
        "ip -br -4 addr show | awk '\$3 ~ /^${FABRIC_NET//./\\.}\./ {print \$3; exit}' | cut -d/ -f1" 2>/dev/null)"
  if [ -z "$ip4" ]; then
    o="${OCT[$n]:-}"
    [ -z "$o" ] && { echo "ERROR: cannot resolve ${n}'s ${FABRIC_NET}.x address (ssh failed?) and it is not in the OCT fallback map"; exit 1; }
    ip4="${FABRIC_NET}.${o}"
    echo "    WARN: ${n}: live fabric-IP lookup failed, using fallback ${ip4}"
  fi
  IPS+=( "$ip4" )
done
IPADDRS=$(IFS=,; echo "${IPS[*]}")
MASTER_ADDR="${IPS[0]}"

echo "=================================================================="
echo " skyRiver disagg launch: ${MODEL_NAME}  (${xP}P/${yD}D, ${CONNECTOR}/WIDE_EP=${WIDE_EP}/${EP_BACKEND})"
echo "   prefill nodes: ${P_NODES[*]}   decode nodes: ${D_NODES[*]}"
echo "   IPADDRS=${IPADDRS}  MASTER_ADDR=${MASTER_ADDR}  image=${IMAGE}"
echo "   logs: ${LOG_PATH} (per node)"
echo "=================================================================="

# Source connector .env -> -e KEY=VALUE args, then layer the per-fabric profile
# ${CONNECTOR}.${FABRIC_PROFILE}.env on top (thor2 = Broadcom bnxt_re; that is what
# skyRiver is, hence the default). Same two rules as run_xPyD_models.slurm, and they
# matter: `-e KEY=` given twice means the LAST one wins in docker run, so the profile
# overrides the base file -- while `${!_k:-$_v}` means an export in THIS shell beats
# both. A raw `-e ${line}` passthrough (what this script used to do) silently ignored
# such an export, so a one-off `MORI_RDMA_TC=104 ./launch...` did nothing.
CONNECTOR_ENV_ARGS=""

# models.yaml env precedence, identical to run_xPyD_models.slurm's block: capture
# which recipe knobs the USER set in THIS shell, forward both the values and the
# protect-list into the container, and let vllm_disagg.sh apply
#   image-baked ENV  <  models.yaml env:  <  submit-time export.
# Without this the driver falls into its no-protect-list branch (any pre-existing
# env wins over the recipe) AND the values never cross the docker boundary at all,
# so e.g. `export GPU_MEMORY_UTILIZATION=0.72` for EP16 is silently dropped.
# Captured BEFORE the connector .env files are loaded, so it reflects user intent
# only -- a value that merely came from a connector .env must not be "protected".
_RECIPE_ENV_KEYS="VLLM_USE_V1 VLLM_USE_LAYERNAME VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_RMSNORM VLLM_ROCM_USE_AITER_MLA KV_BLOCK_SIZE KV_CACHE_DTYPE KV_CACHE_MEMORY_BYTES GPU_MEMORY_UTILIZATION VLLM_CUDAGRAPH_MODE PREFILL_CUDAGRAPH_MODE DECODE_CUDAGRAPH_MODE CUDAGRAPH_CAPTURE_SIZES VLLM_ALL2ALL_BACKEND PREFILL_MORI_BACKEND DECODE_MORI_BACKEND MORI_SHMEM_HEAP_SIZE"
MODELS_YAML_PROTECT=""
RECIPE_ENV_ARGS=""
# Values are single-quoted: the whole `docker run` is assembled into ONE string and
# re-parsed by the remote shell over ssh, so a multi-word value (e.g.
# CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 ...", or the space-separated protect-list itself)
# would otherwise split into extra argv words and corrupt the command line.
for _k in $_RECIPE_ENV_KEYS; do
  if [ -n "${!_k+x}" ]; then
    MODELS_YAML_PROTECT="${MODELS_YAML_PROTECT} ${_k}"
    RECIPE_ENV_ARGS+=" -e ${_k}='${!_k}'"
  fi
done
MODELS_YAML_PROTECT="${MODELS_YAML_PROTECT# }"
RECIPE_ENV_ARGS+=" -e MODELS_YAML_PROTECT='${MODELS_YAML_PROTECT}'"
echo "models.yaml protect-list (submit-time overrides): '${MODELS_YAML_PROTECT}'"

# Per-run recipe knobs that models.yaml exposes as ${VAR:-default} (expanded by
# vllm_disagg.sh). Not in the protect-list -- they are flag-string substitutions,
# not env: keys -- but they still have to cross into the container to take effect.
for _k in GLM_MAX_MODEL_LEN GLM_PREFILL_BATCHED_TOKENS GLM_DECODE_BATCHED_TOKENS; do
  [ -n "${!_k+x}" ] && RECIPE_ENV_ARGS+=" -e ${_k}='${!_k}'"
done

# Benchmark selection and its knobs. vllm_disagg.sh already honours
# `${BENCHMARK_SCRIPT_FILE:-benchmark_xPyD.sh}` when it invokes the benchmark, so the
# selector exists -- it just had no way to cross the container boundary from the host.
# Without this loop the NIAH harness can only be driven by hand (docker exec after the
# fact), which is how a scored run ends up measuring a different configuration than the
# one it claims to. Quoted for the same reason as the protect-list above: NIAH_WORDS and
# BENCHMARK_COMBINATIONS carry commas and spaces that would otherwise split into extra
# argv words when this command line is re-parsed by the remote shell over ssh.
# All are unset by default, so behaviour is unchanged unless a caller opts in.
for _k in BENCHMARK_SCRIPT_FILE BENCHMARK_PORT BENCHMARK_CON BENCHMARK_COMBINATIONS \
          BENCHMARK_ITR NIAH_WORDS NIAH_SEEDS NIAH_MAXTOK NIAH_TIMEOUT NIAH_WARMUP; do
  [ -n "${!_k+x}" ] && RECIPE_ENV_ARGS+=" -e ${_k}='${!_k}'"
done

_load_connector_env() {   # $1 = env file
  local _line _k _v
  while IFS= read -r _line; do
    [[ "$_line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${_line// }" ]] && continue
    _k="${_line%%=*}"; _v="${_line#*=}"
    CONNECTOR_ENV_ARGS+=" -e ${_k}=${!_k:-$_v}"
  done < "$1"
}
ENVF="${HOST_SCRIPTS}/connectors/${CONNECTOR}.env"
if [ -f "$ENVF" ]; then
  echo "Loading connector platform env: $ENVF"
  _load_connector_env "$ENVF"
else
  echo "WARN: connector env file not found: $ENVF" >&2
fi
FABRIC_PROFILE="${FABRIC_PROFILE:-thor2}"
if [ "$FABRIC_PROFILE" != "-" ]; then
  FENVF="${HOST_SCRIPTS}/connectors/${CONNECTOR}.${FABRIC_PROFILE}.env"
  if [ -f "$FENVF" ]; then
    echo "Loading connector platform env: $FENVF (FABRIC_PROFILE=${FABRIC_PROFILE})"
    _load_connector_env "$FENVF"
  else
    echo "ERROR: FABRIC_PROFILE=${FABRIC_PROFILE} but ${FENVF} not found." >&2
    echo "       Set FABRIC_PROFILE=- to run with the base ${CONNECTOR}.env only." >&2
    exit 1
  fi
fi

# NO host RDMA userspace mounts. Deliberate — do not re-add (verified 2026-08-15):
#   * The container's rdma-core is 62 (libibverbs.so.1.16.62.0 = IBVERBS_PRIVATE_59).
#     It loads providers as lib<vendor>-rdmav59.so from /usr/lib/x86_64-linux-gnu/libibverbs/
#     and NEVER consults /usr/local/lib -> the old libbnxt_re mounts were INERT.
#   * skyriver hosts run MLNX OFED (rdma-core-58mlnx43, libibverbs.so.1.14.43.0 =
#     IBVERBS_PRIVATE_34) and ship NO bnxt provider. Mounting the host libibverbs over
#     the container's would DOWNGRADE ABI 59 -> 34 and break every MoRI/vLLM .so
#     (libmori_io/libmori_shmem/libmori_pybinds all link the ABI-59 libibverbs).
# Stock image verbs were verified working: 8 devices, open_device/alloc_pd/create_cq/reg_mr OK.
# (The old "0/56 pairs unreachable" signal was an artifact of the image's ibv_*/perftest
#  binaries coming from rdma-core 39 -> they fail to even start; the verbs path is fine.)
RDMA_MOUNTS=""

# --- Cross-rail reachability (REQUIRED for EP>8; added 2026-08-15) -----------------
# MoRI builds a full QP mesh, so any multi-node EP forms QPs BETWEEN rails, e.g.
# 192.168.200.52 -> 192.168.205.105. Those need (a) a per-source policy route via the
# rail gateway .254 and (b) rp_filter=2 on the receiver, else the QP dies with
# "resolve gid dmac: -110" / bnxt.cpp:417 ModifyInit2Rtr assert.
# This state is RUNTIME ONLY -- the ifcfg-bond* files carry no GATEWAY= and there are no
# route-* files, so every reboot wipes it. Re-applying here is idempotent and cheap.
# Set SKIP_RAIL_ROUTES=1 to bypass (e.g. when deliberately testing the broken state).
# DRY_RUN=1 must NOT touch the hosts: this changes routing tables and a sysctl on every
# node, and a "dry run" that silently reconfigures the fabric is exactly the surprise a
# dry run exists to prevent. Under DRY_RUN it only reports what it would do.
if [ "${SKIP_RAIL_ROUTES:-0}" != "1" ]; then
  RR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/diag/rail_routes.sh"
  if [ ! -x "$RR" ]; then
    echo "    WARN: $RR not found/executable; cross-rail QPs will fail at EP>8"
  elif [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--- [DRY_RUN] would apply cross-rail routes + rp_filter on ${ALL_NODES[*]} ---"
    echo "    $RR add ${ALL_NODES[*]}"
  else
    echo "--- applying cross-rail routes + rp_filter on ${ALL_NODES[*]} ---"
    "$RR" add "${ALL_NODES[@]}" 2>&1 | sed 's/^/    /'
  fi
fi

# --- Clear ORPHANED aiter JIT locks (2026-08-16) -------------------------------
# aiter's FileBaton (aiter/jit/utils/file_baton.py) is O_CREAT|O_EXCL + a bare
#   while os.path.exists(lock): sleep(0.2)
# spin. No flock, no PID, no timeout. If the holder dies between try_acquire()
# and release() -- e.g. the OOM-killed 2P/2D run of 2026-08-15 -- the lock file
# survives and EVERY other worker on that node waits on it FOREVER. Symptom: the
# node's log freezes on "waiting for baton release at /jit_cache/aiter/build/
# lock_module_<x>", workers stay at ~130% CPU (spin), GPU use 0%, and nothing
# ever times out. Cost 25 min of silent hang before it was spotted.
# Because /models/common/jit_cache is a persistent host mount, the orphan also
# survives container removal and poisons every later run on that node.
# A lock is safe to delete iff no live process is compiling: FileBaton keeps NO
# flock, so we test for real compiler processes instead. Only ever run this
# BEFORE the containers start, never against a running job.
# Skipped under DRY_RUN=1 -- it deletes files on every node, which a dry run must not do.
if [ "${SKIP_JIT_LOCK_CLEAN:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "--- clearing orphaned aiter JIT locks ---"
  for n in "${ALL_NODES[@]}"; do
    ssh -n "$n" '
      # aiter uses TWO lock shapes and an early version of this sweep only matched the
      # first, which let a nested orphan hang skyriver06 a second time (2026-08-16):
      #   .../aiter/build/lock_module_<name>          <- outer, taken by build_module
      #   .../aiter/build/module_<name>/build/lock    <- inner, taken during the compile
      # Match both with find, not a glob.
      locks=$(find /models/common/jit_cache/aiter \( -name "lock_module_*" -o -name "lock" \) 2>/dev/null)
      [ -z "$locks" ] && { echo "  '"$n"': none"; exit 0; }
      # NB: pgrep -f / ps|grep both MATCH THIS SCRIPT ITSELF -- the pattern string is in
      # our own argv, so the naive check always reported "compiler ACTIVE" and silently
      # skipped the sweep (cost one wasted 2P/2D launch, 2026-08-16). Exclude our own PID
      # and our parent, and match on the binary basename via ps -eo comm,args.
      ncomp=$(ps -eo pid=,comm=,args= | awk -v me=$$ -v pp=$PPID \
        "\$1!=me && \$1!=pp && (\$2 ~ /^(clang|clang\+\+|hipcc|ld\.lld)$/)" | wc -l)
      if [ "$ncomp" -gt 0 ]; then
        echo "  '"$n"': compiler ACTIVE ($ncomp procs) -- leaving locks alone"; exit 0
      fi
      for L in $locks; do rm -f "$L" && echo "  '"$n"': removed ${L#/models/common/jit_cache/aiter/}"; done
      # A lock death mid-compile also leaves a HALF-BUILT module dir (hundreds of .o, no
      # .so). aiter does not detect that: the next builder re-enters the same dir, and any
      # worker that merely waited on the lock falls straight through to get_module() and
      # dies with ModuleNotFoundError because the .so never appeared (mp_lock passes no
      # WaitFunc at core.py:1029, so a waiter never rebuilds). Wipe the partial build so
      # the next run compiles from scratch.
      for d in /models/common/jit_cache/aiter/build/module_*/; do
        m=$(basename "$d")
        [ -d "$d/build" ] || continue
        ls "$d"/build/*.o >/dev/null 2>&1 || continue
        [ -f "/models/common/jit_cache/aiter/$m.so" ] && continue
        rm -rf "$d" && echo "  '"$n"': wiped partial build $m (no .so)"
      done
    ' 2>&1
  done
fi

rank=0
for n in "${ALL_NODES[@]}"; do
  role="prefill"; [ "$rank" -ge "$xP" ] && role="decode"
  CNAME="disagg_${MODEL_NAME}_${JOB_ID}"
  echo "--- NODE_RANK=$rank  $n  ($role) ---"
  # /models/<name> is a SYMLINK to a per-node real dir; only /models is mounted, so the
  # symlink dangles inside the container. Resolve the real target ON THIS NODE and
  # bind-mount it at the same real path so MODEL_PATH resolves. (real path differs per node.)
  REAL_MODEL=$(ssh "$n" "readlink -f ${MODEL_PATH}" 2>/dev/null)
  MODEL_MOUNT=""
  [ -n "$REAL_MODEL" ] && MODEL_MOUNT="-v ${REAL_MODEL}:${REAL_MODEL}:ro"
  # gloo/NCCL TCP socket: vLLM DP resolves mq_connect_ip on the MGMT NIC (10.67.x), and
  # gloo fails (nfds=-1) on the bonded fabric iface. Use each node's mgmt iface for the
  # TCP control plane; RDMA data stays on the bnxt fabric (MORI_RDMA_DEVICES). mgmt iface
  # name differs per node, so resolve it here. Override via SOCKET_IFNAME_OVERRIDE.
  MGMT_IFACE="${SOCKET_IFNAME_OVERRIDE:-$(ssh "$n" "ip -br -4 addr | awk '/10\\.67\\./{print \$1; exit}'" 2>/dev/null)}"
  SOCK_ENV=""
  [ -n "$MGMT_IFACE" ] && SOCK_ENV="-e GLOO_SOCKET_IFNAME=${MGMT_IFACE} -e NCCL_SOCKET_IFNAME=${MGMT_IFACE} -e MORI_SOCKET_IFNAME=${MGMT_IFACE}"

  ###########################################################################
  # RDMA DEVICE ORDER — must be BY FABRIC RAIL (subnet), not by device name.
  #
  # THE FABRIC IS ROUTED, not isolated. Each 192.168.20X.0/24 rail has a live
  # gateway .254 (all eight SVIs answer with one MAC, 98:4a:6b:6c:e8:9a). Cross-rail
  # used to die at ibv_modify_qp INIT->RTR errno 110 because of TWO host-side faults,
  # both now fixed by diag/rail_routes.sh (applied automatically above):
  #   (1) sender had no route off its own rail (scope-link entries only, no policy tables)
  #   (2) receiver dropped every arrival: net.ipv4.conf.all.rp_filter=1 and the kernel
  #       takes max(all, per-dev), so strict RPF applied on every bond.
  # After both: 64/64 rail pairs 0% loss @0.22 ms, EP16 NDEV=8 probe ALL OK, QP110=0.
  # (An earlier note here claimed the rails were isolated L2 with no gateway. That was
  # WRONG. It came from a bad test: `ping -I bond0 <dst>` is SO_BINDTODEVICE, which
  # bypasses the policy rule and ARPs for the destination on-link. Always test
  # cross-rail with the SOURCE-ADDRESS form: `ping -I 192.168.200.52 <dst>`.)
  #
  # Rail-ordering the device list is STILL REQUIRED — not for reachability now, but
  # because MoRI pairs peers positionally (below) and NCCL cannot coordinate rails
  # across nodes. Ordering is a correctness/perf property, independent of the fix.
  #
  # The bnxt_re_bond<N> name<->rail mapping is SCRAMBLED, DIFFERENT per node, AND IT
  # MOVES ACROSS REBOOTS. e.g. rail .200 was bnxt_re_bond4 on skyriver04 and
  # bnxt_re_bond1 on skyriver07; after a reboot on 2026-08-16 those became bond6 and
  # bond4. The device name also does NOT match the bondN interface name. Never
  # hardcode it, never carry it over from a doc. Only sysfs is authoritative:
  #   /sys/class/infiniband/<dev>/device/net/<slave> -> bond master -> IPv4.
  # (`show_gids` agrees but is easy to mis-parse; sysfs is the ground truth.)
  #
  # Two consumers, two different requirements:
  #  * MoRI pairs peers POSITIONALLY (rank i's i-th device talks to rank j's i-th
  #    device), so it needs the FULL list rail-ordered — then position i is the
  #    same physical rail on every node. A name-sorted list crosses rails and
  #    dies at bnxt.cpp:417 ("DV Modify QP v2 error 110").
  #  * NCCL/RCCL picks its device per node by PCI locality and does NOT coordinate
  #    across nodes. Handing it the same multi-device list lets it choose .200 on
  #    one node and .203 on another -> cross-rail -> ibvwrap.cc:302
  #    "ibv_modify_qp failed with 110" -> ncclCommInitRank fails in the DP group
  #    ("unhandled system error" / "remote process exited"). This is why 1P/1D
  #    (DP group intra-node) worked and 2P/2D (DP group spans nodes) did not.
  #    So NCCL gets exactly ONE rail, NCCL_RAIL (default .200), pinned per node.
  #    Verified 2026-08-15 by diag/rccl_probe.py: unpinned = fail, pinned = ALL_OK.
  #
  # Resolved live per node because the mapping changes across driver reloads
  # / reboots. Override with MORI_RDMA_DEVICES_OVERRIDE (or set it to "-" to
  # fall back to whatever connectors/moriio.env exports).
  ###########################################################################
  # Print "<rail-octet> <device>" per RDMA device, sorted by rail. Sysfs-authoritative.
  RAILMAP_CMD='for d in /sys/class/infiniband/*; do dv=$(basename $d);
    nd=$(ls $d/device/net 2>/dev/null | head -1); [ -n "$nd" ] || continue;
    m=$(basename $(readlink /sys/class/net/$nd/master 2>/dev/null) 2>/dev/null);
    ip=$(ip -br -4 addr show ${m:-$nd} 2>/dev/null | awk "{print \$3}" | cut -d/ -f1);
    case $ip in 192.168.20*) echo "$(echo $ip | cut -d. -f3) $dv";; esac;
  done | sort -n'
  RDMA_DEV_ENV=""
  if [ "${MORI_RDMA_DEVICES_OVERRIDE:-}" = "-" ]; then
    :
  elif [ -n "${MORI_RDMA_DEVICES_OVERRIDE:-}" ]; then
    RDMA_DEV_ENV="-e MORI_RDMA_DEVICES=${MORI_RDMA_DEVICES_OVERRIDE}"
  else
    # MORI_RDMA_NDEV=N keeps only the first N rail-ordered devices. DEFAULT 8 = all bonded
    # NICs (resolved live; devices move across reboots).
    #
    # HISTORY (do not re-derive): NDEV>1 used to fail the EP16 probe with 8 errors/node --
    #   bnxt_re_resolve_eth_dmac: Failed to resolve gid dmac: -110
    #   -> mori bnxt.cpp:417 ModifyInit2Rtr: Assertion `!status' failed (DV Modify QP v2 error 110)
    # SOLVED 2026-08-15. It was never the NIC count. Cross-rail needed TWO host-side fixes:
    #   (1) sender had no route to other rails (only on-link scope-link entries), and
    #   (2) receiver dropped the arrivals -- net.ipv4.conf.all.rp_filter=1, and the kernel
    #       takes max(all, per-dev), so strict RPF applied despite every per-dev knob = 0.
    #       Proven: nstat TcpExtIPReversePathFilter on the receiver ticked 1:1 with pings.
    # Both are applied by diag/rail_routes.sh (see its header for the full derivation), which
    # this launcher runs automatically below. After the fix: 64/64 rail-pairs ping 0% loss and
    # the EP16 probe reports ALL OK at NDEV=8, QP110_count=0 on both nodes.
    # Two earlier theories were WRONG and must not be revived: "rails are isolated L2" and
    # "the .254 SVIs answer but do not forward". The switch routes fine.
    DEVS=$(ssh "$n" "$RAILMAP_CMD" 2>/dev/null | awk '{print $2}' | head -n "${MORI_RDMA_NDEV:-8}" | paste -sd,)
    if [ -n "$DEVS" ]; then
      RDMA_DEV_ENV="-e MORI_RDMA_DEVICES=${DEVS}"
      echo "    MoRI RDMA devices (rail-ordered): ${DEVS}"
    else
      echo "    WARN: could not derive rail-ordered RDMA devices on $n; using moriio.env defaults"
    fi
  fi
  # NCCL gets ONE rail, same rail on every node. NCCL_RAIL=- disables the pinning.
  if [ "${NCCL_RAIL:-200}" != "-" ]; then
    NDEV=$(ssh "$n" "$RAILMAP_CMD" 2>/dev/null | awk -v r="${NCCL_RAIL:-200}" '$1==r {print $2; exit}')
    if [ -n "$NDEV" ]; then
      RDMA_DEV_ENV="${RDMA_DEV_ENV} -e NCCL_IB_HCA=${NDEV}"
      echo "    NCCL rail .${NCCL_RAIL:-200} device: ${NDEV}"
    else
      echo "    WARN: no NIC on rail .${NCCL_RAIL:-200} on $n; leaving NCCL_IB_HCA unset"
    fi
  fi
  # --- RDMA provenance preflight (added 2026-08-15) -------------------------------
  # The 2026-08-15 1P/1D run was un-auditable after the fact: neither the image tag nor
  # any MORI_*/NCCL_IB_* value was written anywhere, so "which NICs did MoRI actually
  # bind" could not be answered from the logs. Worse, MoRI-EP's DirectVerbs backend had
  # ZERO devices for the whole run (missing bnxt_re_dv_* symbols) and said so only via 64
  # buried [error] lines while the server came up healthy. Both are fixed here: dump the
  # full RDMA env to the log dir, and hard-fail if the image cannot do DirectVerbs on
  # Broadcom. Set SKIP_RDMA_PREFLIGHT=1 to bypass (e.g. a deliberately IBVerbs-only run).
  if [ "${SKIP_RDMA_PREFLIGHT:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    DVOK=$(ssh "$n" "docker run --rm --entrypoint python3 ${IMAGE} -c '
import ctypes,sys
for n in [\"libbnxt_re.so\",\"libbnxt_re-rdmav59.so\",\"libbnxt_re-rdmav34.so\"]:
    try: h=ctypes.CDLL(n)
    except Exception: continue
    if all(hasattr(h,s) for s in (\"bnxt_re_dv_umem_reg\",\"bnxt_re_dv_umem_dereg\",
        \"bnxt_re_dv_create_cq\",\"bnxt_re_dv_destroy_cq\",\"bnxt_re_dv_init_obj\",
        \"bnxt_re_dv_create_qp\",\"bnxt_re_dv_destroy_qp\",\"bnxt_re_dv_modify_qp\")):
        print(\"OK \"+n); sys.exit(0)
print(\"MISSING\")' 2>/dev/null" 2>/dev/null)
    case "$DVOK" in
      OK*) echo "    RDMA DirectVerbs preflight: ${DVOK} (MoRI-EP InterNode capable)" ;;
      *)   echo "    ERROR: image ${IMAGE} has NO usable Broadcom direct-verbs provider on $n."
           echo "           MoRI-EP RdmaDeviceFactory will return nullptr for all 8 NICs."
           echo "           EP8 would still 'work' (IntraNode, no verbs) but EP16 cannot."
           echo "           Use IMAGE=localhost/rocmshared/mori-wideep-glm:v027-bnxt238,"
           echo "           or set SKIP_RDMA_PREFLIGHT=1 to proceed anyway."
           exit 1 ;;
    esac
  fi
  # Record exactly what this node was launched with, so a post-mortem never has to guess.
  if [ "${DRY_RUN:-0}" != "1" ]; then
    ssh "$n" "mkdir -p ${LOG_PATH}; { echo \"node=$n rank=${rank} role=${role}\"; \
      echo \"image=${IMAGE}\"; echo \"image_id=\$(docker image inspect -f '{{.Id}}' ${IMAGE} 2>/dev/null)\"; \
      echo \"directverbs_preflight=${DVOK:-skipped}\"; \
      echo \"rdma_env=${RDMA_DEV_ENV}\"; echo \"sock_env=${SOCK_ENV}\"; \
      echo \"nnodes=${NNODES} xP=${xP} yD=${yD} master=${MASTER_ADDR} ipaddrs=${IPADDRS}\"; \
      echo '--- rail map (sysfs) ---'; $RAILMAP_CMD; \
      echo \"fabric_profile=${FABRIC_PROFILE}\"; \
      echo '--- connector env MORI_/NCCL_ (base, then profile) ---'; \
      grep -hE '^(MORI_|NCCL_)' ${HOST_SCRIPTS}/connectors/${CONNECTOR}.env ${FENVF:-/dev/null} 2>/dev/null; \
      echo '--- resolved CONNECTOR_ENV_ARGS ---'; echo '${CONNECTOR_ENV_ARGS}'; \
      } > ${LOG_PATH}/rdma_provenance_NODE${rank}.txt 2>&1" 2>/dev/null
  fi
  # jit_cache must pre-exist on EVERY node: docker refuses to auto-create a bind source
  # ("Error: statfs /models/common/jit_cache: no such file or directory") and the node
  # silently never launches. /models/common is local per node, so mkdir per node.
  # Kill stale disagg containers from PREVIOUS jobs too, not just this name. A failed run
  # leaves its container up holding all 8 GPUs; the next run then dies deep inside NCCL
  # with the unhelpful "unhandled cuda error" during ncclCommInitRank. Set KEEP_STALE=1 to
  # leave older jobs alone (e.g. when deliberately running two topologies side by side).
  STALE_CLEAN="for c in \$(docker ps -aq --filter name=^disagg_ 2>/dev/null); do docker rm -f \$c >/dev/null 2>&1; done;"
  [ "${KEEP_STALE:-0}" = "1" ] && STALE_CLEAN=""
  RUN="${STALE_CLEAN} docker rm -f $CNAME 2>/dev/null; mkdir -p ${LOG_PATH} /models/common/jit_cache/{aiter,triton,vllm,comgr,mori}; \
docker run -d --name $CNAME \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --network host --ipc host --privileged \
  --ulimit memlock=-1:-1 --ulimit nproc=100000:100000 --ulimit nofile=524288:524288 --pids-limit=-1 \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /models:/models ${MODEL_MOUNT} -v ${LOG_PATH}:/run_logs/${JOB_ID} \
  -v /models/common/jit_cache:/jit_cache \
  -e AITER_JIT_DIR=/jit_cache/aiter -e TRITON_CACHE_DIR=/jit_cache/triton \
  -e VLLM_CACHE_ROOT=/jit_cache/vllm -e COMGR_CACHE_DIR=/jit_cache/comgr \
  -e MORI_JIT_CACHE_DIR=/jit_cache/mori \
  -v /models/common/code/MAD/scripts/vllm_dissag:${COOKBOOK_IN_CTR} \
  -v /sys:/sys -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
  -v /sys/kernel/config:/sys/kernel/config -v /sys/kernel/debug:/sys/kernel/debug \
  ${RDMA_MOUNTS} ${EXTRA_MOUNTS:-} \
  -e SLURM_JOB_ID=${JOB_ID} -e NNODES=${NNODES} -e NODE_RANK=${rank} \
  -e MASTER_ADDR=${MASTER_ADDR} -e MASTER_PORT=${MASTER_PORT} \
  -e IPADDRS=${IPADDRS} -e xP=${xP} -e yD=${yD} \
  -e FABRIC_SUBNET=${FABRIC_NET}. \
  -e MODEL_NAME=${MODEL_NAME} -e MODEL_PATH=${MODEL_PATH} \
  -e CONNECTOR=${CONNECTOR} -e WIDE_EP=${WIDE_EP} -e EP_BACKEND=${EP_BACKEND} \
  -e PROXY_PORT=${PROXY_PORT} -e NIXL_COOKBOOK_PATH=${COOKBOOK_IN_CTR} \
  ${CONNECTOR_ENV_ARGS} ${SOCK_ENV} ${RDMA_DEV_ENV} ${RECIPE_ENV_ARGS} \
  ${IMAGE} bash -c 'mkdir -p /run_logs/${JOB_ID}; bash ${COOKBOOK_IN_CTR}/vllm_disagg.sh 2>&1 | tee /run_logs/${JOB_ID}/pd_NODE${rank}.log'"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$RUN" | sed 's/  */ /g'
  else
    ssh "$n" "$RUN" 2>&1 | tail -2
  fi
  rank=$((rank+1))
done

###############################################################################
# CROSS-NODE LOG RELAY (required — the upstream driver assumes a shared FS).
#
# connectors/moriio.sh:connector_wait_workers_ready() runs ON RANK 0 and blocks
# until it greps "Application startup complete." out of BOTH
#   /run_logs/$JOB/prefill_NODE0.log      (local to rank0 — fine)
#   /run_logs/$JOB/decode_NODE${xP}.log   (written on the DECODE node — NOT fine)
# On SLURM clusters those live on one shared filesystem. On skyRiver
# /models/common is a LOCAL disk per node (verified: /dev/md127 on skyriver04 vs
# /dev/nvme3n1p1 on skyriver07), so rank0 can never see the decode log, never
# starts the proxy/router, and both sides hang forever ("Waiting for prefill &
# decode servers to be ready..." on rank0, "Waiting for nodes. . ." on rank1).
#
# Fix: poll each non-rank0 node's log and copy it into rank0's log dir (which is
# bind-mounted to /run_logs/$JOB inside rank0's container). Runs in the
# background; self-terminates once every relayed log shows the startup signal,
# or after RELAY_TIMEOUT_SECONDS. Set LOG_RELAY=0 to disable (e.g. if you ever
# put ${LOG_PATH} on a genuinely shared mount).
###############################################################################
if [ "${DRY_RUN:-0}" != "1" ] && [ "${LOG_RELAY:-1}" = "1" ] && [ "$NNODES" -gt 1 ]; then
  (
    RELAY_TIMEOUT_SECONDS="${RELAY_TIMEOUT_SECONDS:-4200}"
    SIGNAL="Application startup complete."
    deadline=$(( $(date +%s) + RELAY_TIMEOUT_SECONDS ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      done_all=1
      r=1
      while [ "$r" -lt "$NNODES" ]; do
        src="${ALL_NODES[$r]}"
        rrole="prefill"; [ "$r" -ge "$xP" ] && rrole="decode"
        f="${rrole}_NODE${r}.log"
        # -F on the source so a missing/short file is not fatal; overwrite whole file.
        if ssh -o BatchMode=yes "$src" "cat ${LOG_PATH}/${f} 2>/dev/null" \
             | ssh -o BatchMode=yes "${ALL_NODES[0]}" "cat > ${LOG_PATH}/${f}.relay && mv -f ${LOG_PATH}/${f}.relay ${LOG_PATH}/${f}" 2>/dev/null; then
          ssh -o BatchMode=yes "${ALL_NODES[0]}" "grep -qF '${SIGNAL}' ${LOG_PATH}/${f} 2>/dev/null" || done_all=0
        else
          done_all=0
        fi
        r=$((r+1))
      done
      [ "$done_all" = "1" ] && { echo "[relay] all remote logs show '${SIGNAL}' — relay done."; break; }
      sleep 5
    done
  ) > "/tmp/${JOB_ID}.relay.log" 2>&1 &
  echo "Log relay started (pid $!) -> ${ALL_NODES[0]}:${LOG_PATH}/  [log: /tmp/${JOB_ID}.relay.log]"
fi

echo "=================================================================="
echo "Launched. Watch: ssh <node> 'docker logs -f disagg_${MODEL_NAME}_${JOB_ID}'"
echo "Logs on each node under ${LOG_PATH}/"
echo "Prefill master + proxy on NODE_RANK 0 = ${P_NODES[0]} (${MASTER_ADDR}:${PROXY_PORT})"

#!/bin/bash
# WideEP submit — derive every dependent knob from BENCH + MODEL + TOPO.
# =============================================================================
# Set DOCKER_IMAGE_NAME (or --image) to an image built from the model's
# Dockerfile; for DSV4 that is docker/vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile.
#
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:dsv4
#   PARTITION=<partition> ACCOUNT=<account> ./run_wideep_bench.sh validate dsv4fls ep32
#   ./run_wideep_bench.sh validate dsv4pro ep32
#   ./run_wideep_bench.sh smoke dsv4fls 1p1d
#   ./run_wideep_bench.sh niah dsv4pro 2p2d --time 04:00:00
#   ./run_wideep_bench.sh niah glm 4p4d --image <your-registry>/vllm-disagg:glmv5.1
#   ./run_wideep_bench.sh validate dsv4fls ep32 --dry-run
#
# Site knobs: PARTITION, ACCOUNT, QOS, EXCLUDE_NODES (or --exclude), NODELIST
# (or --nodelist), ASSOC_MAX_WALL (clamps the computed walltime default).
#
# Changing EP16 → EP24/EP32 is TOPO only. Do not set xP/yD/-N by hand.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Empty + --nodelist means "only that pool".
DEFAULT_EXCLUDE=''
# GLM-5.2 NVMe pool (/mnt/m2m_nobackup/models_blog/GLM-5.2-FP8). glm52 defaults here.
GLM52_STAGED='useocpm2m-097-[017,023,026,028,030,033,038,039,040,045,051,069,077,078,080,082,083,084,089,094,095,100,114,115,122,123,124,125,132,135,137,139,140,142,144,148,151,153,155]'

usage() {
    cat <<'EOF'
Usage: ./run_wideep_bench.sh BENCH MODEL TOPO [flags]

  BENCH   niah | smoke | validate | accuracy
  MODEL   glm | glm51 | glm52 | dsv3 | dsv4fls | dsv4pro | hy3 | hy3p | GLM-5.1-FP8 | ...
  TOPO    1p1d | 2p2d | 3p3d | 4p4d | ep8 | ep16 | ep24 | ep32
          asymmetric: 1p2d | 2p1d | 2p3d | 3p2d  (driver derives per-role DP)

Flags:
  --dry-run              print sbatch, do not submit
  --image TAG            image to run (overrides DOCKER_IMAGE_NAME; one of the two is required)
  --exclude LIST         slurm --exclude (ignored if --nodelist is set)
  --nodelist LIST|glm52  pin to staged NVMe pool (no --exclude)
  --name NAME            slurm job name override
  --time HH:MM:SS        walltime override
  --after JOBID          sbatch --dependency=afterany:JOBID (queue behind a live job)

Env still wins for rare knobs (NIAH_SEEDS, PARTITION, DSV4_EAGER, ...).
  DSV4_EAGER=1  one-shot: decode cudagraph NONE +quant_fp8 (never --enforce-eager).
  HOLD_AFTER_READY=1  skip proxy/router and bench; keep P+D up for a manual proxy.
EOF
    exit 2
}

BENCH="${1:-}"
MODEL_IN="${2:-}"
TOPO_IN="${3:-}"
[[ -n "$BENCH" && -n "$MODEL_IN" && -n "$TOPO_IN" ]] || usage
shift 3

DRY_RUN=0
IMAGE_ARG=""
EXCLUDE_ARG="${EXCLUDE_NODES:-$DEFAULT_EXCLUDE}"
EXCLUDE_EXPLICIT=0
NODELIST_ARG="${NODELIST:-}"
JOB_NAME_ARG=""
TIME_ARG=""
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --image) IMAGE_ARG="${2:-}"; shift 2 ;;
        --exclude) EXCLUDE_ARG="${2:-}"; EXCLUDE_EXPLICIT=1; shift 2 ;;
        --nodelist) NODELIST_ARG="${2:-}"; shift 2 ;;
        --name) JOB_NAME_ARG="${2:-}"; shift 2 ;;
        --time) TIME_ARG="${2:-}"; shift 2 ;;
        --after) AFTER_JOB="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Error: unknown flag '$1'" >&2; usage ;;
    esac
done

# --- MODEL ---
case "$MODEL_IN" in
    glm|glm51|GLM-5.1-FP8) MODEL_NAME="GLM-5.1-FP8"; SHORT=glm ;;
    glm52|GLM-5.2-FP8)     MODEL_NAME="GLM-5.2-FP8"; SHORT=glm52 ;;
    dsv3|DeepSeek-V3)      MODEL_NAME="DeepSeek-V3"; SHORT=dsv3 ;;
    dsv4fls|ds4fls|dsv4|dsv4flash|DeepSeek-V4-Flash-FP8) MODEL_NAME="DeepSeek-V4-Flash-FP8"; SHORT=dsv4fls ;;
    dsv4pro|dsv4-pro|DeepSeek-V4-Pro-FP8) MODEL_NAME="DeepSeek-V4-Pro-FP8"; SHORT=dsv4pro ;;
    hy3p|Hy3-preview)      MODEL_NAME="Hy3-preview"; SHORT=hy3p ;;
    hy3|Hy3)               MODEL_NAME="Hy3"; SHORT=hy3 ;;
    *) echo "Error: unknown MODEL '$MODEL_IN'" >&2; usage ;;
esac

# --- TOPO → xP yD N EP ---
# Asymmetric P/D is supported by the driver: vllm_disagg.sh computes
# PREFILL_DP_SIZE=xP*8 and DECODE_DP_SIZE=yD*8 independently. Only this case
# list gated it.
case "$TOPO_IN" in
    1p1d|ep8)  xP=1; yD=1; TOPO=1p1d ;;
    2p2d|ep16) xP=2; yD=2; TOPO=2p2d ;;
    3p3d|ep24) xP=3; yD=3; TOPO=3p3d ;;
    4p4d|ep32) xP=4; yD=4; TOPO=4p4d ;;
    2p3d)      xP=2; yD=3; TOPO=2p3d ;;
    3p2d)      xP=3; yD=2; TOPO=3p2d ;;
    1p2d)      xP=1; yD=2; TOPO=1p2d ;;
    2p1d)      xP=2; yD=1; TOPO=2p1d ;;
    *) echo "Error: unknown TOPO '$TOPO_IN' (2p2d/3p3d/4p4d, asymmetric 1p2d/2p1d/2p3d/3p2d, or ep8/ep16/ep24/ep32)" >&2; usage ;;
esac
N=$((xP + yD))
# Per-role EP, display only — the real DP world is derived in vllm_disagg.sh.
# 2P/2D=16, 4P/4D=32. Not N*8. Asymmetric reports both roles.
EP_P=$((xP * 8))
EP_D=$((yD * 8))
if [[ "$EP_P" == "$EP_D" ]]; then EP="$EP_P"; else EP="P${EP_P}/D${EP_D}"; fi

# --- BENCH ---
case "$BENCH" in
    niah)
        BENCHMARK_SCRIPT=niah
        # Always the full ladder. Rungs are sequential: a 400/500 on 8k
        # is the failure point. Do not cap Pro to 2000 (that forced reruns).
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS=""
        ;;
    smoke)
        BENCHMARK_SCRIPT=sweep
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        NIAH_WORDS=""
        # con=64+ stalls with 32 decode slots (8 DP ranks ×
        # max-num-seqs=4). Stop the DSV4 smoke ladder there until slots grow.
        # Pro 4P/4D: skip con=1 — eight serial 1024-token decodes at
        # 2.66 s/tok plus ~2.5 h load overrun the 8 h assoc wall.
        if [[ "$SHORT" == "dsv4pro" && "$TOPO" == "4p4d" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-8 16 32}"
            STEP_SEC_PER_TOK="${STEP_SEC_PER_TOK:-3}"
            STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"
        elif [[ "$SHORT" == "dsv4fls" || "$SHORT" == "dsv4pro" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-1 8 16 32}"
        fi
        ;;
    validate)
        # One allocation: curl (moriio.sh) then NIAH then smoke. Do not halt
        # NIAH on a failed rung — smoke still has to run.
        BENCHMARK_SCRIPT=validate
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        if [[ "$SHORT" == "dsv4pro" && "$TOPO" == "4p4d" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-8 16 32}"
            STEP_SEC_PER_TOK="${STEP_SEC_PER_TOK:-3}"
            STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"
        elif [[ "$SHORT" == "dsv4fls" || "$SHORT" == "dsv4pro" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-1 8 16 32}"
        fi
        ;;
    accuracy)
        # 20-prompt correctness suite (benchmark_accuracy.py). curl still runs
        # first from moriio.sh; no NIAH, no sweep.
        BENCHMARK_SCRIPT=accuracy
        ACC_THINKING="${ACC_THINKING:-chat}"
        ACC_TIMEOUT="${ACC_TIMEOUT:-900}"
        NIAH_WORDS=""
        BENCHMARK_COMBINATIONS=""
        unset NIAH_LIST_PRIME || true
        ;;
    *) echo "Error: unknown BENCH '$BENCH' (niah|smoke|validate|accuracy)" >&2; usage ;;
esac

# Walltime scales with node count (AITER JIT). validate uses niah's budget
# (load once, then niah + smoke).
# Asking for more than the association's MaxWall does NOT fail at submit --
# slurm queues the job forever as AssocMaxWallDurationPerJobLimit with no
# error to grep. Set ASSOC_MAX_WALL to your site's cap.
if [[ "${QOS:-}" == "low" ]]; then
    ASSOC_MAX_WALL="${ASSOC_MAX_WALL:-04:00:00}"
else
    ASSOC_MAX_WALL="${ASSOC_MAX_WALL:-08:00:00}"
fi
WT_BENCH="$BENCH"
[[ "$BENCH" == "validate" ]] && WT_BENCH=niah
if [[ -z "$TIME_ARG" ]]; then
    case "$WT_BENCH-$TOPO" in
        niah-1p1d)  TIME_ARG=06:00:00 ;;
        niah-2p2d)  TIME_ARG=08:00:00 ;;
        niah-3p3d)  TIME_ARG=10:00:00 ;;
        niah-4p4d)  TIME_ARG=12:00:00 ;;
        accuracy-1p1d)  TIME_ARG=04:00:00 ;;
        accuracy-2p2d)  TIME_ARG=04:00:00 ;;
        accuracy-3p3d)  TIME_ARG=06:00:00 ;;
        accuracy-4p4d)  TIME_ARG=08:00:00 ;;
        smoke-1p1d) TIME_ARG=04:00:00 ;;
        smoke-2p2d) TIME_ARG=06:00:00 ;;
        smoke-3p3d) TIME_ARG=07:00:00 ;;
        smoke-4p4d) TIME_ARG=08:00:00 ;;
    esac
    # Flash ~69k shards; 4h is tight for first load + JIT. Pro is 865 GB.
    # 2P/2D is an EP16 MoRI decode probe. Not an ITL score.
    if [[ "$SHORT" == "dsv4fls" ]]; then
        case "$WT_BENCH-$TOPO" in
            smoke-1p1d) TIME_ARG=06:00:00 ;;
            smoke-2p2d) TIME_ARG=08:00:00 ;;
            niah-1p1d|niah-2p2d) TIME_ARG=12:00:00 ;;
            accuracy-1p1d|accuracy-2p2d) TIME_ARG=04:00:00 ;;
        esac
    fi
    if [[ "$SHORT" == "dsv4pro" ]]; then
        case "$WT_BENCH-$TOPO" in
            smoke-1p1d) TIME_ARG=08:00:00 ;;
            smoke-2p2d) TIME_ARG=10:00:00 ;;
            niah-1p1d|niah-2p2d|niah-4p4d) TIME_ARG=12:00:00 ;;
            accuracy-1p1d|accuracy-2p2d|accuracy-4p4d) TIME_ARG=04:00:00 ;;
        esac
    fi
    # Asymmetric topologies match no case arm above and would leave TIME_ARG
    # empty, which sbatch takes as no limit request. Fall back on node count.
    if [[ -z "$TIME_ARG" ]]; then
        case "$WT_BENCH" in
            niah)  TIME_ARG=$(printf '%02d:00:00' $(( N < 6 ? 6 : 8 )) ) ;;
            accuracy) TIME_ARG=04:00:00 ;;
            smoke) TIME_ARG=$(printf '%02d:00:00' $(( N < 6 ? 4 : 6 )) ) ;;
        esac
        echo "note: $BENCH-$TOPO has no walltime default; using $TIME_ARG for N=$N" >&2
    fi
    # The Pro/Flash bumps above ask for 10-12h. Clamp the computed default to
    # ASSOC_MAX_WALL (check with sacctmgr show assoc); --time still wins.
    if [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
        echo "note: default TIME=$TIME_ARG exceeds assoc MaxWall; clamping to $ASSOC_MAX_WALL" >&2
        TIME_ARG="$ASSOC_MAX_WALL"
    fi
elif [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
    echo "WARN: --time $TIME_ARG exceeds assoc MaxWall $ASSOC_MAX_WALL; slurm will hold this job as AssocMaxWallDurationPerJobLimit" >&2
fi

# vllm_router 2P2D KV-notify wants --moriio-dp-size when the binary has it.
# ROUTER_SKIP_MORIIO_DP_SIZE=1 omits the flag (toy-proxy default).
if [[ "${PROXY_TYPE:-vllm_router}" == "vllm_router" ]]; then
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-0}"
else
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-1}"
fi
# Router discovery port. Keep it below ip_local_port_range (32768+): prefill
# starts first on the router node and its outgoing connections can take an
# ephemeral port as their source, so the router then fails to bind.
MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-31367}"

# --- IMAGE ---
DOCKER_IMAGE_NAME="${IMAGE_ARG:-${DOCKER_IMAGE_NAME:-}}"
if [[ -z "$DOCKER_IMAGE_NAME" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        DOCKER_IMAGE_NAME="<unset>"
    else
        echo "Error: set DOCKER_IMAGE_NAME or pass --image (build from docker/vllm_disagg_inference.*.Dockerfile)." >&2
        exit 1
    fi
fi

JOB_NAME="${JOB_NAME_ARG:-${SHORT}-${BENCH}-${TOPO}}"
# Empty = the site's default partition/account (sbatch gets no -p / -A).
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"

# DSV3: fp8 KV, AITER MLA off.
# GLM persist-gate stays unset (moriio.sh default GATE=0).
# DSV3 EP32: a 16 GiB static heap was full when dispatch_combine asked
# 7.5 GiB, so 32 GiB. Do NOT set AITER_MLA=0 for DSV4 (V4 sparse needs it).
# NVIDIA --kv-cache-dtype fp8 remaps to fp8_ds_mla, which ROCm MLA backends
# reject; DSV4 uses fp8_e4m3. models.yaml must match.
unset GLM_SKIP_PATCHERS || true
EXTRA_ENV=()
if [[ "$MODEL_NAME" == "DeepSeek-V3" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8 VLLM_ROCM_USE_AITER_MLA=0)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=34359738368)
fi
if [[ "$MODEL_NAME" == "DeepSeek-V4-Flash-FP8" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8_e4m3 VLLM_ROCM_USE_AITER_MLA=1)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=34359738368)
    # HMA is always 0 (moriio.sh also forces it). .attn (group-0
    # sliding_window=None) was never WRITTEN, leaving decode a 128-token
    # window; the HMA=0 arm must WRITE it.
    EXTRA_ENV+=(DSV4_ENABLE_HMA=0)
    EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-1}")
    EXTRA_ENV+=(DSV4_STORAGE_UPSTREAM_SPAN="${DSV4_STORAGE_UPSTREAM_SPAN:-0}")
    # yaml decode is FULL_DECODE_ONLY. DSV4_EAGER=1 splits CUDA-graph
    # capture vs compress_ratio: DECODE_CUDAGRAPH_MODE=NONE with +quant_fp8.
    # NEVER --enforce-eager (AITER aiter_tensor_t crash). Submit-time wins yaml.
    EXTRA_ENV+=(DSV4_EAGER="${DSV4_EAGER:-0}")
    if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE=NONE)
    elif [[ -n "${DECODE_CUDAGRAPH_MODE:-}" ]]; then
        # Reaches the container via --export=ALL + the slurm's -e either way, but
        # listing it makes the knob visible in the plan/EXTRA line instead of
        # silently absent (the same invisibility NIAH_TERSE once had).
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE}")
    fi
    EXTRA_ENV+=(VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}")
    EXTRA_ENV+=(CURL_SUITE=short)
    EXTRA_ENV+=(MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT}")
    # Product NIAH: original 6-rung sizes, 64-token answers, bare prompt
    # (no <User>/<Assistant> wrap). Do not inherit NIAH_DS_WRAP from the
    # submit environment — that silently wrapped two runs.
    if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
        # 27 Aug: hybrid is the standard method — PR-176's system+user through
        # the DeepSeek template PLUS a restatement after the haystack. Faithful
        # pr176 puts the ask before the haystack only, which collapses on DSV4
        # at 8k+ (Pro: 0/10 x3, 512 tokens of filler); keep it available
        # for cross-track comparison, not for scoring. method=product restores
        # the bare 64-token stem, comparable ONLY to the old product rows
        # (GLM / DSV3 / Hy3).
        # MAXTOK 512 rather than 2048 because DSV4 runs ~1 s/tok and
        # 2048 x 18 requests overruns an 8h walltime — a cap, not a target.
        NIAH_METHOD="${NIAH_METHOD:-hybrid}"
        if [[ "$NIAH_METHOD" == "product" ]]; then
            NIAH_MAXTOK="${NIAH_MAXTOK:-64}"
        else
            NIAH_MAXTOK="${NIAH_MAXTOK:-512}"
        fi
        NIAH_WARMUP=0
        NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}"
        # validate must finish the ladder so smoke still has a full NIAH log.
        if [[ "$BENCH" == "validate" ]]; then
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        else
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-1}"
        fi
        EXTRA_ENV+=(NIAH_METHOD="$NIAH_METHOD")
        EXTRA_ENV+=(NIAH_DS_WRAP=0)
        EXTRA_ENV+=(NIAH_TERSE="${NIAH_TERSE:-0}")
        EXTRA_ENV+=(NIAH_STOP_BLANK="${NIAH_STOP_BLANK:-0}")
        EXTRA_ENV+=(NIAH_MIN_TOKENS="${NIAH_MIN_TOKENS:-0}")
        # Flash NIAH: prime the stem with "1." so token 1 is given, not sampled.
        # Unprimed Flash answers `0` / `0 Animals found...`.
        # NIAH_LIST_PRIME=off restores the unprimed stem. A non-empty value wins.
        case "${NIAH_LIST_PRIME:-}" in
            off|OFF|0) NIAH_LIST_PRIME="" ;;
            "") NIAH_LIST_PRIME="1." ;;
        esac
        EXTRA_ENV+=(NIAH_LIST_PRIME="${NIAH_LIST_PRIME}")
        EXTRA_ENV+=(NIAH_STOP="${NIAH_STOP:-}")
        EXTRA_ENV+=(NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}")
    fi
fi
# Pro weights 125 GiB/GPU; 32 GiB heap left ~8 GiB for a 9 GiB
# profiling dummy (seqs*len*hidden). 16 GiB heap. yaml has the rest.
if [[ "$MODEL_NAME" == "DeepSeek-V4-Pro-FP8" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8_e4m3 VLLM_ROCM_USE_AITER_MLA=1)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=17179869184)
    EXTRA_ENV+=(DSV4_ENABLE_HMA=0)
    EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-1}")
    EXTRA_ENV+=(DSV4_STORAGE_UPSTREAM_SPAN="${DSV4_STORAGE_UPSTREAM_SPAN:-0}")
    # yaml decode is FULL_DECODE_ONLY. DSV4_EAGER=1 splits CUDA-graph
    # capture vs compress_ratio: DECODE_CUDAGRAPH_MODE=NONE with +quant_fp8.
    # NEVER --enforce-eager (AITER aiter_tensor_t crash). Submit-time wins yaml.
    EXTRA_ENV+=(DSV4_EAGER="${DSV4_EAGER:-0}")
    if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE=NONE)
    elif [[ -n "${DECODE_CUDAGRAPH_MODE:-}" ]]; then
        # Reaches the container via --export=ALL + the slurm's -e either way, but
        # listing it makes the knob visible in the plan/EXTRA line instead of
        # silently absent (the same invisibility NIAH_TERSE once had).
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE}")
    fi
    EXTRA_ENV+=(VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}")
    EXTRA_ENV+=(CURL_SUITE=short)
    EXTRA_ENV+=(MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT}")
    if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
        # 27 Aug: hybrid is the standard method — PR-176's system+user through
        # the DeepSeek template PLUS a restatement after the haystack. Faithful
        # pr176 puts the ask before the haystack only, which collapses on DSV4
        # at 8k+ (Pro: 0/10 x3, 512 tokens of filler); keep it available
        # for cross-track comparison, not for scoring. method=product restores
        # the bare 64-token stem, comparable ONLY to the old product rows
        # (GLM / DSV3 / Hy3).
        # MAXTOK 512 rather than 2048 because DSV4 runs ~1 s/tok and
        # 2048 x 18 requests overruns an 8h walltime — a cap, not a target.
        NIAH_METHOD="${NIAH_METHOD:-hybrid}"
        if [[ "$NIAH_METHOD" == "product" ]]; then
            NIAH_MAXTOK="${NIAH_MAXTOK:-64}"
        else
            NIAH_MAXTOK="${NIAH_MAXTOK:-512}"
        fi
        NIAH_WARMUP=0
        NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}"
        # validate must finish the ladder so smoke still has a full NIAH log.
        if [[ "$BENCH" == "validate" ]]; then
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        else
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-1}"
        fi
        EXTRA_ENV+=(NIAH_METHOD="$NIAH_METHOD")
        EXTRA_ENV+=(NIAH_DS_WRAP=0)
        EXTRA_ENV+=(NIAH_TERSE="${NIAH_TERSE:-0}")
        EXTRA_ENV+=(NIAH_STOP_BLANK="${NIAH_STOP_BLANK:-0}")
        EXTRA_ENV+=(NIAH_MIN_TOKENS="${NIAH_MIN_TOKENS:-0}")
        # Pro NIAH: never prime. It retrieves on the bare stem (prime
        # is a wash). Force empty so a leftover Flash `1.` from --export=ALL
        # cannot leak into this cell.
        NIAH_LIST_PRIME=""
        EXTRA_ENV+=(NIAH_LIST_PRIME="${NIAH_LIST_PRIME}")
        EXTRA_ENV+=(NIAH_STOP="${NIAH_STOP:-}")
        EXTRA_ENV+=(NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}")
    fi
fi

# HOLD_AFTER_READY=1: skip proxy/router + bench; keep P+D up for a manual proxy.
if [[ "${HOLD_AFTER_READY:-0}" == "1" ]]; then
    EXTRA_ENV+=(HOLD_AFTER_READY=1)
fi

# The DSV4 image sets ENV SKIP_RUNTIME_PATCH=1; slurm forwards
# ${SKIP_RUNTIME_PATCH:-1}. Forward only an explicit override.
if [[ "$MODEL_NAME" == "DeepSeek-V4-Flash-FP8" || "$MODEL_NAME" == "DeepSeek-V4-Pro-FP8" ]]; then
    [[ -n "${SKIP_RUNTIME_PATCH:-}" ]] && EXTRA_ENV+=(SKIP_RUNTIME_PATCH="$SKIP_RUNTIME_PATCH")
    # The DSV4 images bake /usr/local/bin/vllm-router. Do not default
    # ROUTER_BOOT_INSTALL=git — that re-clones main at every NODE0 boot.
    # Forward only when the operator overrides (old digest, or a newer SHA).
    if [[ "${PROXY_TYPE:-vllm_router}" == "vllm_router" ]]; then
        [[ -n "${ROUTER_BOOT_INSTALL:-}" ]] && EXTRA_ENV+=(ROUTER_BOOT_INSTALL="$ROUTER_BOOT_INSTALL")
        [[ -n "${ROUTER_REPO:-}" ]] && EXTRA_ENV+=(ROUTER_REPO="$ROUTER_REPO")
        [[ -n "${ROUTER_REF:-}" ]] && EXTRA_ENV+=(ROUTER_REF="$ROUTER_REF")
        [[ -n "${RUST_TOOLCHAIN:-}" ]] && EXTRA_ENV+=(RUST_TOOLCHAIN="$RUST_TOOLCHAIN")
    fi
fi

if [[ "$BENCH" == "accuracy" ]]; then
    EXTRA_ENV+=(ACC_THINKING="$ACC_THINKING" ACC_TIMEOUT="$ACC_TIMEOUT")
    [[ -n "${ACC_MAXTOK:-}" ]] && EXTRA_ENV+=(ACC_MAXTOK="$ACC_MAXTOK")
fi

# EP16 NFS+JIT boot exceeds a 4000s gate. --export=ALL can leak a smaller
# LOG_WAIT_TIMEOUT_SECONDS from the submit shell; force unless overridden.
LOG_WAIT_TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-10800}"
EXTRA_ENV+=(LOG_WAIT_TIMEOUT_SECONDS="$LOG_WAIT_TIMEOUT_SECONDS")

# glm52: default to NVMe staged pool unless operator passed --nodelist/--exclude.
if [[ "$SHORT" == "glm52" && -z "$NODELIST_ARG" && "$EXCLUDE_EXPLICIT" == "0" ]]; then
    NODELIST_ARG="$GLM52_STAGED"
fi
case "$NODELIST_ARG" in
    glm52|GLM52) NODELIST_ARG="$GLM52_STAGED" ;;
esac

# --nodelist (staged weights) replaces --exclude so the pool is not filtered out.
SBATCH_LOC=()
if [[ -n "$NODELIST_ARG" ]]; then
    SBATCH_LOC+=(--nodelist="$NODELIST_ARG")
elif [[ -n "$EXCLUDE_ARG" ]]; then
    SBATCH_LOC+=(--exclude="$EXCLUDE_ARG")
fi

if [[ ! -f run_xPyD_models.slurm ]]; then
    echo "Error: run_xPyD_models.slurm missing in $SCRIPT_DIR" >&2
    exit 1
fi

unset SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST || true

echo "=== WideEP bench plan ==="
echo "BENCH=$BENCH  MODEL=$MODEL_NAME  TOPO=$TOPO  EP=$EP  xP=$xP yD=$yD  N=$N"
echo "IMAGE=$DOCKER_IMAGE_NAME"
echo "PROXY_TYPE=${PROXY_TYPE:-vllm_router}  SKIP_MORIIO_DP=${ROUTER_SKIP_MORIIO_DP_SIZE}  PING=${MORI_PROXY_PING_PORT}  WIDE_EP=1"
[[ "${PROXY_TYPE:-vllm_router}" == "vllm_router" ]] && echo "ROUTER_BOOT=${ROUTER_BOOT_INSTALL:-baked}  REPO=${ROUTER_REPO:-image}  REF=${ROUTER_REF:-image}"
echo "TIME=$TIME_ARG"
[[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]] && echo "SMOKE CON=${BENCHMARK_CON:-default}  COMBOS=$BENCHMARK_COMBINATIONS  STEP_SEC_PER_TOK=${STEP_SEC_PER_TOK:-}  STEP_TIMEOUT=${STEP_TIMEOUT:-}"
[[ "$BENCH" == "niah" || "$BENCH" == "validate" ]] && echo "NIAH_METHOD=${NIAH_METHOD:-}  NIAH_WORDS=$NIAH_WORDS  NIAH_SEEDS=$NIAH_SEEDS  HALT=$NIAH_HALT_ON_FAIL  MAXTOK=${NIAH_MAXTOK:-}  WARMUP=${NIAH_WARMUP:-}  TIMEOUT=${NIAH_TIMEOUT:-}  WRAP=0  TERSE=${NIAH_TERSE:-0}  STOP_BLANK=${NIAH_STOP_BLANK:-0}  MINTOK=${NIAH_MIN_TOKENS:-0}  PRIME=${NIAH_LIST_PRIME:-}  STOP=${NIAH_STOP:-}  LOGPROBS=${NIAH_LOGPROBS:-0}"
[[ "$BENCH" == "accuracy" ]] && echo "ACCURACY prompts=20  THINKING=${ACC_THINKING}  MAXTOK=${ACC_MAXTOK:-auto}  TIMEOUT=${ACC_TIMEOUT}"
[[ ${#EXTRA_ENV[@]} -gt 0 ]] && echo "EXTRA ${EXTRA_ENV[*]}"
echo "JOB_NAME=$JOB_NAME  ACCOUNT=${ACCOUNT:-default}  PARTITION=${PARTITION:-default}  QOS=${QOS:-default}  ${SBATCH_LOC[*]:-any-node}"
[[ -n "$AFTER_JOB" ]] && echo "DEPENDENCY=afterany:${AFTER_JOB}"
echo "========================="

SBATCH_EXTRA=()
if [[ -n "$AFTER_JOB" ]]; then
    SBATCH_EXTRA+=(--dependency="afterany:${AFTER_JOB}")
fi

SBATCH_CMD=(sbatch)
[[ -n "$ACCOUNT" ]] && SBATCH_CMD+=(-A "$ACCOUNT")
[[ -n "$PARTITION" ]] && SBATCH_CMD+=(-p "$PARTITION")
[[ -n "$QOS" ]] && SBATCH_CMD+=(--qos="$QOS")
SBATCH_CMD+=(-N "$N" -n "$N" --ntasks-per-node=1 --gres=gpu:8 --time="$TIME_ARG"
    "${SBATCH_LOC[@]}" "${SBATCH_EXTRA[@]}"
    --job-name="$JOB_NAME" --export=ALL
    run_xPyD_models.slurm)
echo "SBATCH: ${SBATCH_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] not submitting"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: SLURM_JOB_ID is set (${SLURM_JOB_ID}). Submit from outside a job." >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch not found." >&2
    exit 1
fi

export DOCKER_IMAGE_NAME PYTHONUNBUFFERED=1
export PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
export CONNECTOR=moriio WIDE_EP=1
export xP yD MODEL_NAME BENCHMARK_SCRIPT
export ROUTER_SKIP_MORIIO_DP_SIZE
[[ -n "${ROUTER_BOOT_INSTALL:-}" ]] && export ROUTER_BOOT_INSTALL
[[ -n "${ROUTER_REPO:-}" ]] && export ROUTER_REPO
[[ -n "${ROUTER_REF:-}" ]] && export ROUTER_REF
[[ -n "${RUST_TOOLCHAIN:-}" ]] && export RUST_TOOLCHAIN
export MORI_PROXY_PING_PORT
if [[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]]; then
    export BENCHMARK_COMBINATIONS
    [[ -n "${BENCHMARK_CON:-}" ]] && export BENCHMARK_CON
    [[ -n "${STEP_SEC_PER_TOK:-}" ]] && export STEP_SEC_PER_TOK
    [[ -n "${STEP_TIMEOUT:-}" ]] && export STEP_TIMEOUT
fi
if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
    export NIAH_WORDS NIAH_SEEDS NIAH_HALT_ON_FAIL
    export NIAH_PAIR_SLEEP_S NIAH_WARMUP_SCORE_SLEEP_S
    [[ -n "${NIAH_METHOD:-}" ]] && export NIAH_METHOD
    [[ -n "${NIAH_MAXTOK:-}" ]] && export NIAH_MAXTOK
    [[ -n "${NIAH_WARMUP:-}" ]] && export NIAH_WARMUP
    [[ -n "${NIAH_TIMEOUT:-}" ]] && export NIAH_TIMEOUT
    [[ -n "${NIAH_TERSE:-}" ]] && export NIAH_TERSE
    [[ -n "${NIAH_STOP_BLANK:-}" ]] && export NIAH_STOP_BLANK
    [[ -n "${NIAH_MIN_TOKENS:-}" ]] && export NIAH_MIN_TOKENS
    # Always export: empty is a real value (byte-identical product stem).
    # EXTRA_ENV export below also sets them; this covers niah on non-DSV4 models.
    export NIAH_LIST_PRIME="${NIAH_LIST_PRIME:-}"
    export NIAH_STOP="${NIAH_STOP:-}"
    export NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}"
fi
if [[ "$BENCH" == "accuracy" ]]; then
    export ACC_THINKING ACC_TIMEOUT
    [[ -n "${ACC_MAXTOK:-}" ]] && export ACC_MAXTOK
fi
if [[ ${#EXTRA_ENV[@]} -gt 0 ]]; then
    export "${EXTRA_ENV[@]}"
fi

"${SBATCH_CMD[@]}"

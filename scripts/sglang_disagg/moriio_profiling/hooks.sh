#!/usr/bin/env bash
# Scheduler-first rocprof finalization hooks for the SGLang entrypoint.

_rocprof_prefix() {
    local role="$1"
    if [[ "${ROCPROF:-0}" != "1" ]]; then echo ""; return 0; fi
    local _rocprof_base="${ROCPROF_DIR_BASE:-/run_logs}"
    local rpdir="${_rocprof_base}/${SLURM_JOB_ID:-0}/rocprof_${role}_NODE${NODE_RANK}"
    mkdir -p "$rpdir"
    local flags="${ROCPROF_FLAGS:-"--kernel-trace --marker-trace --hip-trace --hsa-trace"}"
    echo "rocprofv3 ${flags} --output-format csv json -d ${rpdir} -o %hostname%_%pid% -- "
}

roctx_finalize_workers() (
    # SIGINT exact scheduler workers so rocprofiler can finalize, then stop the main.
    # Serialization can spend minutes without changing an output file. Never
    # re-signal a worker after finalization starts; wait up to the bounded timeout.
    getpids(){ ps -eo pid,args | awk '$2 ~ /^sglang::scheduler_(DP|TP)/ {print $1}'; }
    count_alive(){ getpids | awk 'END { print NR + 0 }'; }

    FINALIZE_TIMEOUT="${ROCPROF_FINALIZE_TIMEOUT:-1800}"
    STALL_LIMIT="${ROCPROF_STALL_LIMIT:-45}"
    HOST=$(hostname -s)

    WORKERS="$(getpids)"
    NW=$(printf '%s\n' $WORKERS | grep -c . 2>/dev/null || echo 0)
    echo "workers: $(echo $WORKERS | tr '\n' ' ')  (count=$NW)"

    # Discover the current job's -d path from one process snapshot. With JOBID set,
    # never accept unscoped matches because stale jobs can cause false completion.
    ALL_RPDIRS=$(ps -eo args 2>/dev/null | grep -oE '\-d /run_logs/[0-9]+/rocprof_[a-z]+_NODE[0-9]+' | awk '{print $2}')
    RPDIR=""
    if [ -n "${JOBID:-}" ]; then
      RPDIR=$(printf '%s\n' "$ALL_RPDIRS" | grep -F "/run_logs/${JOBID}/" | head -1)
      # If JOBID is known, reject unscoped paths and use job-scoped fallbacks.
      if [ -z "$RPDIR" ]; then
        echo "[roctx_finalize_workers] WARN: JOBID=$JOBID set but no matching rocprofv3 -d arg found for it (candidates: $(printf '%s' "$ALL_RPDIRS" | tr '\n' ' ')); this job's rocprofv3 for this role likely never started/wrapped cleanly (e.g. engine hang) -- skipping unscoped ps match (would only find OTHER jobs' stale dirs) and going straight to mtime-based fallback" >&2
      fi
    else
      RPDIR=$(printf '%s\n' "$ALL_RPDIRS" | head -1)
    fi

    # Topology-derived paths work before output files appear.
    if [ -z "$RPDIR" ] && [ -n "${JOBID:-}" ]; then
      idx=0
      for n in ${PREFILL_NODES:-}; do
        ns=${n%%.*}
        if [ "$n" = "$HOST" ] || [ "$ns" = "$HOST" ]; then
          RPDIR="/run_logs/${JOBID}/rocprof_prefill_NODE${idx}"
          break
        fi
        idx=$((idx + 1))
      done
      if [ -z "$RPDIR" ]; then
        idx=0; base_idx=${XP:-0}
        for n in ${DECODE_NODES:-}; do
          ns=${n%%.*}
          if [ "$n" = "$HOST" ] || [ "$ns" = "$HOST" ]; then
            RPDIR="/run_logs/${JOBID}/rocprof_decode_NODE$((base_idx + idx))"
            break
          fi
          idx=$((idx + 1))
        done
      fi
      [ -n "$RPDIR" ] && echo "[roctx_finalize_workers] $HOST topology-derived rocprof dir = $RPDIR"
    fi

    for p in $WORKERS; do kill -INT "$p" 2>/dev/null; done

    # Keep the fallback job-scoped whenever JOBID is available.
    if [ -z "$RPDIR" ]; then
      for _ in $(seq 1 20); do
        f=""
        if [ -n "${JOBID:-}" ]; then
          f=$(find "/run_logs/${JOBID}" -maxdepth 2 -path '*/rocprof_*' -name "${HOST}_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        else
          f=$(find /run_logs -maxdepth 3 -path '*/rocprof_*' -name "${HOST}_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        fi
        [ -n "$f" ] && { RPDIR=$(dirname "$f"); break; }
        sleep 1
      done
    fi

    echo "[roctx_finalize_workers] $HOST rocprof dir = ${RPDIR:-<not found>}  (finalize_timeout=${FINALIZE_TIMEOUT}s stall_limit=${STALL_LIMIT}s expect=${NW} workers)"

    count_files(){ ls "$RPDIR"/${HOST}_*"$1" 2>/dev/null | grep -c . ; }
    newest_mtime(){ find "$RPDIR" -maxdepth 1 -name "${HOST}_*" -printf '%T@\n' 2>/dev/null | sort -n | tail -1; }

    if [ -n "$RPDIR" ] && [ "$NW" -gt 0 ]; then
      start=$(date +%s)
      last_progress=$start
      last_res=0; last_mt=0
      stall_warned=0
      while :; do
        res=$(count_files _results.json)
        mrk=$(count_files _marker_api_trace.csv)
        alive=$(count_alive)
        now=$(date +%s); el=$(( now - start ))
        if [ "$res" -ge "$NW" ] && [ "$mrk" -ge "$NW" ]; then
          echo "[roctx_finalize_workers] $HOST: all $NW workers finalized COMPLETE (results.json=$res marker=$mrk) after ${el}s"
          break
        fi
        if [ "$el" -ge "$FINALIZE_TIMEOUT" ]; then
          echo "[roctx_finalize_workers] WARN: $HOST finalize timeout ${FINALIZE_TIMEOUT}s (results.json=$res/$NW marker=$mrk/$NW, $alive still alive)" >&2
          break
        fi
        mt=$(newest_mtime); mt=${mt%.*}; [ -z "$mt" ] && mt=0
        if [ "$res" -gt "$last_res" ] || [ "${mt:-0}" -gt "${last_mt:-0}" ]; then
          last_res=$res; last_mt=$mt; last_progress=$now
        fi
        if [ "$(( now - last_progress ))" -ge "$STALL_LIMIT" ] && [ "$stall_warned" -eq 0 ]; then
          echo "[roctx_finalize_workers] INFO: $HOST no visible output progress for ${STALL_LIMIT}s (results.json=$res/$NW marker=$mrk/$NW); serialization still running" >&2
          stall_warned=1
        fi
        sleep 3
      done
    else
      # fallback (no rocprof dir found / ROCPROF off): preserve a bounded settle window
      echo "[roctx_finalize_workers] $HOST: no rocprof dir located -- falling back to bounded 90s settle"
      sleep 90
      for p in $(getpids); do kill -INT "$p" 2>/dev/null; done
      sleep 20
    fi
    echo "workers remaining: $(count_alive)"

    mp=$(ps -eo pid,args | awk '/[p]ython3 -m sglang.launch_server/ {print $1; exit}')
    echo "main pid: $mp"
    if [ -z "$mp" ]; then
      echo "[roctx_finalize_workers] WARN: no launch_server main process found on this node -- it was ALREADY GONE before this flush ran." >&2
      echo "[roctx_finalize_workers] WARN: (topology: xP=${XP:-<unset>} yD=${YD:-<unset>}) if xP>1 or yD>1, this is the known cross-node" >&2
      echo "[roctx_finalize_workers] WARN: DP_MODE=1 collective-cascade symptom (fixed by parallel per-node finalization dispatch, 2026-07-06)." >&2
      echo "[roctx_finalize_workers] WARN: rocprofv3 likely did NOT flush cleanly on this node -- check for a 0-byte rocprof_*_NODE<i> dir." >&2
    fi
    [ -n "$mp" ] && kill -INT "$mp" 2>/dev/null
    sleep 15
    echo "main remaining: $(ps -eo args | grep -c '[p]ython3 -m sglang.launch_server')"
)

finish_server() {
    local role="$1" pipeline_pid="$2"
    local rc=0 suffix count
    JOBID="${SLURM_JOB_ID:-0}" XP="$xP" YD="$yD" \
        roctx_finalize_workers || rc=1

    local leftover
    leftover=$(ps -eo pid,args | awk '/[p]ython3 -m sglang\.launch_server/ {print $1}')
    if [[ -n "$leftover" ]]; then
        echo "[profile] WARN: ${role} worker traces finalized; stopping launch_server parent" >&2
        kill -TERM $leftover 2>/dev/null || true
        sleep 5
        leftover=$(ps -eo pid,args | awk '/[p]ython3 -m sglang\.launch_server/ {print $1}')
        [[ -z "$leftover" ]] || kill -KILL $leftover 2>/dev/null || true
    fi
    if kill -0 "$pipeline_pid" 2>/dev/null; then
        echo "[profile] WARN: ${role} rocprof wrapper still running; stopping it" >&2
        kill -TERM "$pipeline_pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pipeline_pid" 2>/dev/null || true
    fi
    wait "$pipeline_pid" 2>/dev/null || true

    local rpdir="${ROCPROF_DIR_BASE:-/run_logs}/${SLURM_JOB_ID:-0}/rocprof_${role}_NODE${NODE_RANK}"
    for suffix in kernel_trace.csv marker_api_trace.csv results.json; do
        count=$(find "$rpdir" -maxdepth 1 -name "*_${suffix}" 2>/dev/null | wc -l)
        if (( count < GPUS_PER_NODE )); then
            echo "[profile] ERROR: ${role} NODE${NODE_RANK} has ${count}/${GPUS_PER_NODE} ${suffix} files" >&2
            rc=1
        fi
    done
    # Keep rank 0 alive until every node has finalized its worker outputs. This
    # prevents the master task from looking complete while decode serialization is still active on another node.
    local done_dir="${ROCPROF_DIR_BASE:-/run_logs}/${SLURM_JOB_ID:-0}"
    local done_file="$done_dir/.profile_done_NODE${NODE_RANK}"
    touch "$done_file" || rc=1
    if [[ "$NODE_RANK" -eq 0 ]]; then
        local expected_nodes=$((xP + yD))
        local barrier_deadline=$((SECONDS + ${ROCPROF_NODE_BARRIER_TIMEOUT:-2100}))
        local done_nodes=0
        while (( SECONDS < barrier_deadline )); do
            done_nodes=$(find "$done_dir" -maxdepth 1 -name '.profile_done_NODE*' 2>/dev/null | wc -l)
            (( done_nodes >= expected_nodes )) && break
            sleep 5
        done
        if (( done_nodes < expected_nodes )); then
            echo "[profile] ERROR: node finalization barrier has $done_nodes/$expected_nodes nodes" >&2
            rc=1
        fi
    fi
    return "$rc"
}

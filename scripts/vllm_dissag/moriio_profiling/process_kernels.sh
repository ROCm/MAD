#!/usr/bin/env bash

# Post-processing continues after failures, preserves raw traces, and returns failure.

_moriio_profiling_setup_kernel_analysis() (
    set -euo pipefail

    local here ext_dir tracelens_repo venv
    local -a venv_extra_deps
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit
    ext_dir="$here/external_copies"
    tracelens_repo="${TRACELENS_REPO:-https://github.com/AMD-AGI/TraceLens.git}"
    venv="${VENV:-$ext_dir/venv}"
    venv_extra_deps=(pandas plotly matplotlib openpyxl ijson perfetto numpy)

    mkdir -p "$ext_dir" || exit

    echo "=== [1/4] tracelens (clone) ==="
    if [ -d "$ext_dir/tracelens/.git" ]; then
        echo "already cloned at $ext_dir/tracelens -- skipping (no git network calls)"
    else
        git clone "$tracelens_repo" "$ext_dir/tracelens" || exit
    fi

    echo "=== [2/4] venv ($venv) ==="
    if [ ! -f "$venv/bin/activate" ]; then
        python3 -m venv "$venv" || exit
    fi
    # shellcheck disable=SC1091
    source "$venv/bin/activate" || exit
    python3 -V || exit

    echo "=== [3/4] pip install (into venv, NOT --user) ==="
    pip install -e "$ext_dir/tracelens" || exit
    pip install "${venv_extra_deps[@]}" || exit

    echo "=== [4/4] traceconv (vendored check) ==="
    if [ -f "$ext_dir/traceconv_bin/traceconv" ]; then
        echo "traceconv present: $ext_dir/traceconv_bin/traceconv"
    else
        echo "WARN: traceconv missing at $ext_dir/traceconv_bin/traceconv -- .pftrace analysis will fail until vendored" >&2
    fi

    echo "=== verify (imports must resolve from the venv alone) ==="
    python3 -c "import TraceLens, pandas, plotly, matplotlib, ijson; print('OK TraceLens import:', TraceLens.__file__)" || exit
    echo "=== kernel-analysis setup DONE -> $ext_dir ==="
)

_moriio_profiling_analyze_job() (
    set -euo pipefail

    local here jobdir output_name out d label node_rc
    local found=0 failures=0
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    jobdir="${1:?usage: _moriio_profiling_analyze_job /path/to/jobdir [output-name]}"
    output_name="${2:-${KERNEL_ANALYSIS_OUTPUT_NAME:-kernel_analysis_v2}}"

    if [[ ! -d "$jobdir" ]]; then
        echo "ERROR: job dir not found: $jobdir" >&2
        exit 2
    fi
    if [[ ! "$output_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: invalid analysis output name: $output_name" >&2
        exit 2
    fi
    out="$jobdir/$output_name"
    if [[ -e "$out" ]]; then
        echo "ERROR: analysis output already exists; choose a new versioned name: $out" >&2
        exit 2
    fi
    mkdir -p "$out"

    python3 "$here/trace_tools.py" rank-manifest "$jobdir" --out "$out/rank_manifest.json"
    python3 "$here/trace_tools.py" window-manifest "$jobdir" \
        --rank-manifest "$out/rank_manifest.json" --out "$out/window_manifest.json"
    printf 'role_node\tstatus\texit_code\n' > "$out/node_status.tsv"

    shopt -s nullglob
    for d in "$jobdir"/rocprof_prefill_NODE* "$jobdir"/rocprof_decode_NODE*; do
        [[ -d "$d" ]] || continue
        found=$((found + 1))
        label="${d##*/}"
        label="${label#rocprof_}"
        echo "===== analyze_job: $label ($d) ====="
        if python3 "$here/trace_tools.py" analyze "$d" "$out/$label" "$label" \
            --rank-manifest "$out/rank_manifest.json" \
            --window-manifest "$out/window_manifest.json"; then
            printf '%s\tsuccess\t0\n' "$label" >> "$out/node_status.tsv"
        else
            node_rc=$?
            failures=$((failures + 1))
            printf '%s\tfailed\t%s\n' "$label" "$node_rc" >> "$out/node_status.tsv"
            echo "[analyze_job] ERROR: $label failed with status $node_rc; continuing for diagnostics" >&2
        fi
    done
    shopt -u nullglob

    if (( found == 0 )); then
        echo "ERROR: no rocprof_{prefill,decode}_NODE* dirs found under $jobdir" >&2
        exit 2
    fi
    if (( failures != 0 )); then
        echo "[analyze_job] ERROR: $failures of $found node analyses failed; see $out/node_status.tsv" >&2
        exit 1
    fi
    echo "[analyze_job] all $found node analyses succeeded -> $out"
)


moriio_profiling_run_posthoc() {
    local job_id="$1" log_path="$2" master_node="$3" repo_dir="$4" cookbook_path="$5" docker_image="$6"
    local rc=0

    [[ "${ROCPROF:-0}" == "1" ]] || return 0

    echo "[moriio_profiling] ROCPROF=1 -> combining rocprofv3 traces for job ${job_id}"
    srun --nodes=1 --ntasks=1 --nodelist="${master_node}" bash -c "
        docker run --rm \
            -v ${log_path}:/run_logs \
            -v ${repo_dir}:${cookbook_path} \
            --entrypoint /bin/bash \
            ${docker_image} -c '
                python3 ${cookbook_path}/moriio_profiling/trace_tools.py combine /run_logs/${job_id}
            '
    " || { echo "[moriio_profiling] ERROR: trace_tools.py combine failed; raw per-rank output is untouched" >&2; rc=1; }

    if [[ "${MORIIO_REQID_MAP:-0}" == "1" ]]; then
        echo "[moriio_profiling] MORIIO_REQID_MAP=1 -> extracting reqid map for job ${job_id}"
        srun --nodes=1 --ntasks=1 --nodelist="${master_node}" bash -c "
            docker run --rm \
                -v ${log_path}:/run_logs \
                -v ${repo_dir}:${cookbook_path} \
                --entrypoint /bin/bash \
                ${docker_image} -c '
                    shopt -s nullglob
                    logs=(/run_logs/${job_id}/prefill_NODE*.log)
                    if [ \${#logs[@]} -eq 0 ]; then
                        echo \"[trace_tools extract-reqid] WARNING: no prefill_NODE*.log files found under /run_logs/${job_id} -- skipping\" >&2
                        exit 0
                    fi
                    python3 ${cookbook_path}/moriio_profiling/trace_tools.py extract-reqid \"\${logs[@]}\" -o /run_logs/${job_id}/reqid_map.csv
                '
        " || { echo "[moriio_profiling] ERROR: trace_tools.py extract-reqid failed; prefill logs are untouched" >&2; rc=1; }
    else
        echo "[moriio_profiling] MORIIO_REQID_MAP!=1 -> skipping reqid map extraction for job ${job_id}"
    fi

    [[ "${ANALYZE_KERNELS:-1}" == "1" ]] || return "$rc"

    echo "[moriio_profiling] ANALYZE_KERNELS=1 -> running kernel analysis for job ${job_id}"
    srun --nodes=1 --ntasks=1 --nodelist="${master_node}" bash -c "
        docker run --rm \
            -v ${log_path}:/run_logs \
            -v ${repo_dir}:${cookbook_path} \
            --entrypoint /bin/bash \
            ${docker_image} -c '
                source ${cookbook_path}/moriio_profiling/process_kernels.sh &&
                _moriio_profiling_setup_kernel_analysis &&
                _moriio_profiling_analyze_job /run_logs/${job_id}
            '
    " || { echo "[moriio_profiling] ERROR: kernel analysis failed; raw output and combined traces are untouched" >&2; rc=1; }
    return "$rc"
}

_moriio_profiling_kernels_cli() (
    set -euo pipefail

    if [[ $# -ne 2 || "$1" != "--kernels" || -z "$2" ]]; then
        echo "usage: $0 --kernels <job-id|job-dir>" >&2
        exit 2
    fi

    local arg="$2"
    local here repo_dir user jobdir job_root job_name cookbook_path image base
    local -a srun_args cmd
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_dir="$(cd "$here/.." && pwd)"
    user="${USER:-$(id -un)}"

    if [[ -d "$arg" ]]; then
        jobdir="$(realpath "$arg")"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        jobdir=""
        for base in "${LOG_PATH:-}" "/shared_inference/${user}/model_blog_logs" "/shared_inference/${user}/moriio_prof_runs"; do
            [[ -n "$base" && -d "$base/$arg" ]] || continue
            jobdir="$(realpath "$base/$arg")"
            break
        done
        if [[ -z "$jobdir" ]]; then
            echo "ERROR: job $arg not found under LOG_PATH, model_blog_logs, or moriio_prof_runs" >&2
            exit 2
        fi
    else
        echo "ERROR: not a job ID or directory: $arg" >&2
        exit 2
    fi

    job_root="$(dirname "$jobdir")"
    job_name="$(basename "$jobdir")"
    cookbook_path="${NIXL_COOKBOOK_PATH:-/opt/nixl-vllm-cookbook}"
    image="${DOCKER_IMAGE_NAME:-rocmshared/pytorch-private:aarai_vllm_moriio_srcinstrumented_20260722_r1}"

    srun_args=(--nodes=1 --ntasks=1)
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        srun_args+=(--partition="${SBATCH_PARTITION:-amd-rccl}" --gres="${SBATCH_GRES:-gpu:1}" --time="${SBATCH_TIME:-02:00:00}")
    fi
    cmd=(srun "${srun_args[@]}" docker run --rm
        -v "$job_root:/run_logs"
        -v "$repo_dir:$cookbook_path"
        --entrypoint /bin/bash "$image" -lc
        "source '$cookbook_path/moriio_profiling/process_kernels.sh' && _moriio_profiling_setup_kernel_analysis && _moriio_profiling_analyze_job '/run_logs/$job_name' '${KERNEL_ANALYSIS_OUTPUT_NAME:-kernel_analysis_v2}'")

    echo "[process_kernels.sh --kernels] job: $jobdir"
    echo "[process_kernels.sh --kernels] image: $image"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[process_kernels.sh --kernels] DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        exit 0
    fi
    exec "${cmd[@]}"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _moriio_profiling_kernels_cli "$@"
fi

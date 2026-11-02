#!/bin/bash
# Shared parts of a node program (scripts/<launcher>/node.sh): what every launcher
# does the same way on each node before and around its docker run. Sourced by
# node.sh, which runs on the compute node in its own shell; cluster.sh is not
# sourced there.

# Site storage, bound at the same path inside: shared storage, local NVMe, and
# MODEL_DIR when it is outside both. All three come from cluster.sh or the caller.
# Only paths present on this node: docker creates a missing bind source as an empty
# directory, which would shadow the weights. Evaluated here, on each node.
# Sets _SITE_MOUNTS (-v flags).
node_site_mounts() {
    local _p _m _covered
    _SITE_MOUNTS=""
    for _p in ${SHARED_MOUNT:-/shared_inference} ${NVME_ROOT:-/mnt/m2m_nobackup} ${MODEL_DIR:-}; do
        [ -d "$_p" ] || continue
        _covered=0
        for _m in $_SITE_MOUNTS; do case "$_p/" in "${_m%%:*}"/*) _covered=1 ;; esac; done
        [ "$_covered" = 1 ] || _SITE_MOUNTS="$_SITE_MOUNTS $_p:$_p"
    done
    _SITE_MOUNTS="$(for _m in $_SITE_MOUNTS; do printf -- " -v %s" "$_m"; done)"
    echo "[mounts] site storage:${_SITE_MOUNTS:- none}"
}

# Host RDMA userspace libs must match the host kernel driver; mount them over the
# image copies. Sets _RDMA_MOUNTS (-v flags, read-only).
#
# -f (a regular file), not -e: docker creates a DIRECTORY at a bind-mount source that
# does not exist, so on a node lacking one of these libs the first run leaves an empty
# directory behind, and every later run mounts a directory over a file inside the
# image, which docker rejects ("not a directory"). Self-propagating and per-node, so
# it presents as an intermittent failure: the container starts on a clean node and
# dies on a poisoned one, leaving the survivors waiting at the barrier.
#
# The SONAMEs first, then the versioned files they point at. Both are needed:
# libibverbs.so.1 is the name every consumer links against, and it is a symlink.
# Mounting only the versioned file lands the host copy at a path the image symlink
# does not point to, so libibverbs.so.1 still resolves to the image version while
# libmlx5.so.1 -- matched below and therefore the HOST copy -- demands a symbol only
# the host libibverbs exports:
#     OSError: libibverbs.so.1: version `IBVERBS_PRIVATE_57` not found
# vLLM loads its plugins during startup, so this kills the server about ten seconds
# in, with a traceback that names a quantization plugin and not the mount.
node_rdma_mounts() {
    local _lib _vlib _pattern _libdir=/usr/lib/x86_64-linux-gnu
    _RDMA_MOUNTS=""
    for _lib in libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1; do
        [ -f "$_libdir/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_libdir/$_lib:$_libdir/$_lib:ro"
    done
    for _vlib in $_libdir/libibverbs.so.1.* $_libdir/librdmacm.so.1.*; do
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
    for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
        for _vlib in $_libdir/${_pattern}; do
            [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
        done
    done
    [ -d "$_libdir/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_libdir/libibverbs:$_libdir/libibverbs:ro"
    [ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"
    return 0
}

# Persistent JIT cache mount:  node_jit_cache WRITER [SUBDIR]
# The image points AITER_JIT_DIR / TRITON_CACHE_DIR / VLLM_CACHE_ROOT / COMGR_CACHE_DIR
# at /opt/vllm_cache. A host directory mounted there lets AITER CK kernels compile once
# and be reused across runs: cold compile is ~15 min for a MoE/FP8 GEMM kernel set, a
# warm boot ~1 min, and per node, so a node compiling cold while its peer does not is
# the start-up skew the cross-node barrier cannot absorb. Keyed by image ID so a new
# image starts a fresh cache rather than reusing stale-ABI shared objects; SUBDIR
# separates caches whose kernels differ (the disagg launcher's prefill / decode roles).
# Host directory on local NVMe. Override JIT_CACHE_HOST, or set JIT_CACHE_PERSIST=0 to
# use the image's empty in-container cache. WRITER names the launcher in README.ci.
# Sets _JIT_CACHE_MOUNT (empty when off) and _JIT_CACHE_HOST.
node_jit_cache() {
    local writer="$1" subdir="${2:-}" _img_raw _img_key _base
    _JIT_CACHE_MOUNT=""
    [ "${JIT_CACHE_PERSIST:-1}" = "1" ] || return 0
    _img_raw=$(docker image inspect --format "{{.Id}}" "$DOCKER_IMAGE_NAME" 2>/dev/null)
    _img_key=$(printf "%s" "$_img_raw" | tr -cd "a-f0-9" | cut -c1-12)
    _img_key="${_img_key:-noimg}"
    _base="${JIT_CACHE_HOST:-${NVME_ROOT:-/mnt/m2m_nobackup}/${USER}/vllm_jit_cache/${_img_key}}"
    _JIT_CACHE_HOST="${_base}${subdir:+/$subdir}"
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
    # Inline, not cluster_mark_dir: cluster.sh is not sourced on the compute node.
    if [ ! -e "$_JIT_CACHE_HOST/README.ci" ]; then
        printf "Signature: 8a477f597d28d172789f06886806bc55\n# Cache directory created by MAD CI.\n" \
            > "$_JIT_CACHE_HOST/CACHEDIR.TAG" 2>/dev/null || true
        printf "%s\n%s\n\n%s\n" \
            "What: JIT/Triton/AITER kernel cache for one docker image" \
            "Written by: MAD CI ${writer} launcher" \
            "Safe to delete when no job is running; it is rebuilt on demand." \
            > "$_JIT_CACHE_HOST/README.ci" 2>/dev/null || true
    fi
    # Stale AITER build locks in this cache are cleared inside the container, which
    # owns them; JIT_CACHE_HOST / JIT_LOCK_STALE_MIN go in.
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache"
    echo "[jit-cache] persistent image ${_img_key}: ${_JIT_CACHE_HOST} to /opt/vllm_cache"
}

# Run this node's container and exit with its status:  node_run_container RUN_FN [STOP_HOOK]
# RUN_FN runs `docker run --name $DOCKER_CONT_NAME ...` in the foreground; STOP_HOOK, if
# given, runs when the container ends or the task is signalled (e.g. stop a sampler).
#
# The container belongs to the docker daemon and its PID 1 is a plain bash that ignores
# SIGTERM, so scancel used to leave it running and holding every GPU: one cancelled job
# left its container holding 1535 GB on a node. docker run therefore runs in the
# background, and a trap stops and removes the container as soon as a signal lands.
node_run_container() {
    local run_fn="$1" hook="${2:-}" _pid _rc
    _node_stop_container() {
        [ -n "$hook" ] && "$hook"
        docker stop -t 10 "$DOCKER_CONT_NAME" >/dev/null 2>&1
        docker rm -f "$DOCKER_CONT_NAME" >/dev/null 2>&1
    }
    trap "_node_stop_container; exit 143" TERM INT
    "$run_fn" &
    _pid=$!
    wait "$_pid"
    _rc=$?
    [ -n "$hook" ] && "$hook"
    trap - TERM INT
    exit "$_rc"
}

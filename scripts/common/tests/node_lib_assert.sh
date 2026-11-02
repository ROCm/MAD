#!/bin/bash
# Offline check of scripts/common/node/node_lib.sh, the parts every node.sh shares.
# Runs the functions for real against a stand-in docker and fake directories.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$DIR/common/node/node_lib.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

B="$TMP/bin"; mkdir -p "$B"
cat > "$B/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >> "$DOCKER_LOG"
[ "$1 $2" = "image inspect" ] && echo "sha256:abcdef0123456789abcdef"
exit 0
EOF
chmod +x "$B/docker"
_lib() { env -i PATH="$B:$PATH" HOME="$TMP" USER=t DOCKER_LOG="$TMP/docker.log" "$@"; }

echo "=== node_site_mounts ==="
mkdir -p "$TMP/shared/models" "$TMP/nvme" "$TMP/elsewhere"
_mounts() { _lib "$@" bash -c '. "$0"; node_site_mounts >/dev/null; echo "$_SITE_MOUNTS"' "$LIB"; }
_is "$(_mounts SHARED_MOUNT="$TMP/shared" NVME_ROOT="$TMP/nvme")" " -v $TMP/shared:$TMP/shared -v $TMP/nvme:$TMP/nvme" \
    "shared storage and NVMe, at the same path"
_is "$(_mounts SHARED_MOUNT="$TMP/shared" NVME_ROOT="$TMP/nvme" MODEL_DIR="$TMP/shared/models")" \
    " -v $TMP/shared:$TMP/shared -v $TMP/nvme:$TMP/nvme" "MODEL_DIR inside shared storage adds no mount"
_is "$(_mounts SHARED_MOUNT="$TMP/shared" NVME_ROOT="$TMP/nvme" MODEL_DIR="$TMP/elsewhere")" \
    " -v $TMP/shared:$TMP/shared -v $TMP/nvme:$TMP/nvme -v $TMP/elsewhere:$TMP/elsewhere" "MODEL_DIR elsewhere is mounted"
_is "$(_mounts SHARED_MOUNT="$TMP/absent" NVME_ROOT="$TMP/nvme")" " -v $TMP/nvme:$TMP/nvme" \
    "a path missing on this node is not mounted (docker would create it empty)"

echo ""
echo "=== node_jit_cache ==="
_jit() { _lib DOCKER_IMAGE_NAME=img "$@" bash -c '. "$0"; node_jit_cache "$1" "$2" >/dev/null; echo "$_JIT_CACHE_MOUNT"' "$LIB" "${W:-disagg}" "${SUB:-}"; }
# The key is the first 12 hex characters of `docker image inspect`'s "sha256:<id>",
# which include the "a256" of the prefix: a256 + 8 characters of the id. Pinned as is;
# changing it moves every existing cache.
_is "$(NVME_ROOT="$TMP/nvme" _jit NVME_ROOT="$TMP/nvme")" "-v $TMP/nvme/t/vllm_jit_cache/a256abcdef01:/opt/vllm_cache" \
    "keyed by image id under NVME_ROOT/USER"
_is "$(SUB=prefill _jit JIT_CACHE_HOST="$TMP/jit")" "-v $TMP/jit/prefill:/opt/vllm_cache" "a role subdirectory under JIT_CACHE_HOST"
_has "$(cat "$TMP/jit/prefill/README.ci")" "Written by: MAD CI disagg launcher" "README.ci names the launcher"
_is "$(_jit JIT_CACHE_HOST="$TMP/jit2" JIT_CACHE_PERSIST=0)" "" "JIT_CACHE_PERSIST=0: no mount"
_is "$([ -e "$TMP/jit2" ] && echo created || echo none)" "none" "...and nothing created"

echo ""
echo "=== node_run_container ==="
# RUN_FN stands in for docker run: sleeps, then exits with the given status.
_run() { # status sleep_s [signal_after_s]
    rm -f "$TMP/docker.log" "$TMP/hook.log"
    # Started directly, not through _lib: env execs bash, so $! is the process the
    # signal must reach (a function call would put a subshell in between).
    env -i PATH="$B:$PATH" HOME="$TMP" USER=t DOCKER_LOG="$TMP/docker.log" \
        DOCKER_CONT_NAME=c1 HOOK_LOG="$TMP/hook.log" bash -c '
        . "$0"
        _st="$1" _sl="$2"
        _fake_run() { sleep "$_sl"; return "$_st"; }
        _hook() { echo hook >> "$HOOK_LOG"; }
        node_run_container _fake_run _hook' "$LIB" "$1" "$2" &
    local pid=$!
    if [ -n "${3:-}" ]; then sleep "$3"; kill -TERM "$pid"; fi
    wait "$pid"; echo "rc=$?"
}
_is "$(_run 0 0)" "rc=0" "a container that exits 0: the node exits 0"
_is "$(_run 3 0)" "rc=3" "a container that exits 3: the node exits 3"
_is "$(grep -c hook "$TMP/hook.log")" "1" "the hook runs once when the container ends"
_is "$(_run 0 30 1)" "rc=143" "TERM while the container runs: the node exits 143"
_has "$(cat "$TMP/docker.log" 2>/dev/null)" "docker stop -t 10 c1" "...the container is stopped"
_has "$(cat "$TMP/docker.log" 2>/dev/null)" "docker rm -f c1" "...and removed"
_has "$(cat "$TMP/hook.log" 2>/dev/null)" "hook" "...and the hook runs"
pkill -f "sleep 30" 2>/dev/null

echo "======================================================"
echo "  node_lib_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]

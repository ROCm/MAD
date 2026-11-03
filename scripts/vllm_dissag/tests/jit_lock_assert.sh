#!/bin/bash
# Offline check of the stale AITER build-lock cleanup (_clear_stale_jit_locks) that both
# container scripts run before any server starts: vllm_dissag/vllm_disagg.sh and
# vllm_multinode/serve_colocated.sh. It runs in the container because the locks belong
# to the container's root user; the batch scripts, running as the job's user, got
# "Permission denied". Runs the scripts' own function against a fake cache.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_has()    { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot() { grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }
_func() { sed -n '/^_clear_stale_jit_locks() {/,/^}/p' "$1"; }

_cache() { # <aiter_jit dir>: stale locks (file, dir, the inner build/<module>/build/lock),
           # fresh locks (outer and inner), built modules, a source file named lock elsewhere
    mkdir -p "$1/build/module_gemm_a8w8_blockscale" "$1/build/lock_module_moe_dir" \
             "$1/build/module_rmsnorm/build" "$1/build/module_fresh/build" "$1/build/module_rmsnorm/blob"
    touch "$1/build/module_gemm_a8w8_blockscale/module.so" "$1/module_rmsnorm.so" "$1/build/module_rmsnorm/blob/lock"
    touch -d '3 hours ago' "$1/build/lock_module_gemm_a8w8_blockscale" "$1/build/lock_module_moe_dir" \
             "$1/build/module_rmsnorm/build/lock" "$1/build/module_rmsnorm/blob/lock"
    touch "$1/build/lock_module_fresh" "$1/build/module_fresh/build/lock"
}
_left() { (cd "$1" && find . -type f -o -type d -name 'lock_*' | sort | tr '\n' ' '); }

D="$DIR/vllm_disagg.sh"; C="$DIR/../vllm_multinode/serve_colocated.sh"
for S in "$D" "$C"; do
    n=$(basename "$S"); F="$(_func "$S")"
    echo "=== $n ==="
    _has "$(grep -c '^ *_clear_stale_jit_locks$' "$S")" "1" "$n: calls the cleanup once"
    _cache "$TMP/$n/default"
    O="$(env -i PATH="$PATH" AITER_JIT_DIR="$TMP/$n/default" bash -c "$F
_clear_stale_jit_locks")"
    _has "$O" "removing stale AITER build lock $TMP/$n/default/build/lock_module_gemm_a8w8_blockscale" "default cache: says which lock it removed"
    _has "$O" "removing stale AITER build lock $TMP/$n/default/build/module_rmsnorm/build/lock" "default cache: removes the inner build/<module>/build/lock too"
    _has "$(_left "$TMP/$n/default")" "./build/module_gemm_a8w8_blockscale/module.so ./build/module_rmsnorm/blob/lock ./module_rmsnorm.so " \
         "default cache: every lock gone, built modules and other files named lock kept"
    _cache "$TMP/$n/shared"
    env -i PATH="$PATH" AITER_JIT_DIR="$TMP/$n/shared" JIT_CACHE_HOST=/somewhere bash -c "$F
_clear_stale_jit_locks" >/dev/null
    L="$(_left "$TMP/$n/shared")"
    _has    "$L" "./build/lock_module_fresh " "caller-set (maybe shared) cache: a fresh lock is left for its holder"
    _has    "$L" "./build/module_fresh/build/lock " "caller-set cache: a fresh inner lock is left too"
    _hasnot "$L" "./build/module_rmsnorm/build/lock " "caller-set cache: a 3-hour-old inner lock is removed"
    _hasnot "$L" "lock_module_gemm_a8w8_blockscale" "caller-set cache: a 3-hour-old lock is removed"
    O="$(env -i PATH="$PATH" AITER_JIT_DIR="$TMP/$n/none" bash -c "$F
_clear_stale_jit_locks; echo rc=\$?")"
    _has "$O" "rc=0" "no cache directory: nothing to do, no error"
done
_has "$(diff <(_func "$D") <(_func "$C") >/dev/null && echo same || echo differs)" "same" "both container scripts carry the same cleanup"
_has "$(awk '/^    _clear_stale_jit_locks$/{c=NR} /^    _check_gpus_clean$/{g=NR} END{print (c && g && c<g) ? "first" : "later"}' "$D")" "first" \
     "vllm_disagg.sh clears locks before anything else starts"
for B in "$DIR/run_xPyD_models.slurm" "$DIR/../vllm_multinode/run_multinode.slurm"; do
    _hasnot "$(grep -v '^ *#' "$B")" "rm -rf \"\$_l\"" "$(basename "$B"): no host-side removal (it cannot delete root's files)"
    _has "$(grep -c 'JIT_CACHE_HOST:+-e JIT_CACHE_HOST' "$B")" "1" "$(basename "$B"): passes JIT_CACHE_HOST into the container"
done

echo "======================================================"
echo "  jit_lock_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]

#!/usr/bin/env python3
"""Overlay the fixed AITER sampling kernel (ROCm/aiter #3658 + hang cap) into the image.

DEFECT 2 (the 8k prefill/decode crash): the AITER TopP/TopK sampling kernel
(csrc/cpp_itfs/sampling/sampling.cuh) has two bugs on the released aiter post3
that this image ships:

  1. HSA OUT-OF-BOUNDS (ROCm/aiter #3658): SamplingTempStorage::last_valid_id is
     never initialized. When a probs row is all-zero / NaN (more likely at long
     input, e.g. 8k), the guarded write-back (max_valid != -1) is skipped, and the
     fallback `sampled_id = temp_storage.last_valid_id` reads UNINITIALIZED shared
     memory -> garbage index -> `probs[row*d + sampled_id]` dereferences OOB and
     HSA page-faults ("Memory access fault by GPU node-N"). Deterministic under
     CUDA graph (shared-mem residue is stable across replays). This is the silent
     worker death at 8k that collapses the disagg DP group -> 503.
     Fix: init `last_valid_id = 0` at top of each loop iter + defensive clamp on
     the loaded sampled_id before it indexes probs.

  2. REJECTION-SAMPLING HANG: the bisection `do { ... } while(low < high)` can
     spin forever when the [low,high] interval stagnates in float precision on a
     degenerate (near-uniform) row -> never-completing HSA signal / hang (the
     "sampler hang" that forced the skip-warmup workaround). Fix: cap the loop at
     kMaxSamplingRounds=32 (float32 mantissa is exhausted well within 32 rounds,
     so healthy distributions always converge via break long before the cap).

Both fixes land in sampling.cuh. #3658 is MERGED upstream but NOT in the released
aiter post3 (this image). Source: A/B-tested by Shiksha (shikpate); staged fixed
tree at SAMPLING_FIX_DIR.

METHOD (from Shiksha's validated in-container overlay): copy the whole patched
sampling source dir (.cuh + .py + .jinja) over the container's aiter, then purge
any compiled sampling JIT objects so the kernel recompiles from the fixed source
on next use.

Idempotent: skips if the fix markers are already present. Model-agnostic at the
kernel level, but invoked from the GLM patch hook. Safe no-op if the staged fix
dir or the target aiter dir is absent.

Usage: apply_glm_aiter_sampling_oob_fix.py <vllm_install_dir>
       (vllm_install_dir arg is accepted for hook uniformity but not required;
        the aiter dir is resolved via `import aiter`.)
"""
import os
import shutil
import subprocess
import sys

FIX_DIR = os.environ.get(
    "SAMPLING_FIX_DIR",
    "/shared_inference/ravgupta/aiter_sampling_fix_3658/sampling_patched",
)
MARKERS = ("last_valid_id = 0", "kMaxSamplingRounds")


def _aiter_sampling_dir():
    """Locate the installed aiter sampling source dir (aiter_meta/csrc/...)."""
    try:
        import aiter  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[sampling-fix] aiter not importable ({e}); skipping.")
        return None
    # The kernel source lives under aiter_meta (sibling of aiter), path is stable.
    candidates = []
    try:
        import aiter_meta  # type: ignore

        candidates.append(
            os.path.join(os.path.dirname(aiter_meta.__file__),
                         "csrc", "cpp_itfs", "sampling")
        )
    except Exception:  # noqa: BLE001
        pass
    # Fallback: search site-packages.
    import aiter
    sp = os.path.dirname(os.path.dirname(aiter.__file__))
    candidates.append(os.path.join(sp, "aiter_meta", "csrc", "cpp_itfs", "sampling"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    print(f"[sampling-fix] could not locate aiter sampling dir (tried {candidates}); skipping.")
    return None


def main() -> int:
    tgt = _aiter_sampling_dir()
    if tgt is None:
        return 0  # safe no-op

    tgt_cuh = os.path.join(tgt, "sampling.cuh")
    if os.path.isfile(tgt_cuh):
        cur = open(tgt_cuh, errors="ignore").read()
        if all(m in cur for m in MARKERS):
            print(f"[sampling-fix] already applied (markers present) in {tgt_cuh}.")
            return 0

    if not os.path.isdir(FIX_DIR):
        print(f"[sampling-fix] WARN: staged fix dir {FIX_DIR} not found; leaving image kernel unpatched.", file=sys.stderr)
        return 0

    src_cuh = os.path.join(FIX_DIR, "sampling.cuh")
    if not os.path.isfile(src_cuh) or not all(m in open(src_cuh, errors="ignore").read() for m in MARKERS):
        print(f"[sampling-fix] WARN: staged {src_cuh} missing/lacks fix markers; skipping.", file=sys.stderr)
        return 0

    # Overlay the whole sampling source dir (.cuh + .py + .jinja), per Shiksha's method.
    copied = []
    for fn in os.listdir(FIX_DIR):
        s = os.path.join(FIX_DIR, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(tgt, fn))
            copied.append(fn)
    print(f"[sampling-fix] overlaid #3658 + hang-cap into {tgt}: {', '.join(sorted(copied))}")

    # Verify.
    cur = open(tgt_cuh, errors="ignore").read()
    if not all(m in cur for m in MARKERS):
        print(f"[sampling-fix] ERROR: markers still absent after overlay in {tgt_cuh}.", file=sys.stderr)
        return 1

    # Purge any compiled sampling JIT objects so the kernel recompiles from source.
    purged = 0
    for base in (
        os.path.expanduser("~/.aiter"), "/root/.aiter", "/tmp/aiter",
        "/opt/vllm_cache/aiter_jit", os.path.join(os.path.dirname(tgt), "..", "..", "jit"),
    ):
        if base and os.path.isdir(base):
            try:
                out = subprocess.run(
                    ["find", base, "-maxdepth", "6", "-iname", "*sampling_from_probs*"],
                    capture_output=True, text=True, timeout=60,
                )
                for p in out.stdout.split():
                    try:
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            os.remove(p)
                        purged += 1
                    except OSError:
                        pass
            except Exception:  # noqa: BLE001
                pass
    print(f"[sampling-fix] purged {purged} stale sampling JIT object(s); kernel will recompile from fixed source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

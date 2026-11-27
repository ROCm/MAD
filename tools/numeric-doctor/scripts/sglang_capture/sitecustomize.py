# debug-numeric step 4: the per-op capture inside SGLang's scheduler process -- where sglang.Engine runs the model.
#
#   PYTHONPATH=<REPO_ROOT>/scripts/sglang_capture  (+ NAN_CAPTURE_DIR, set by wave.sh on capture lines)
#     python3 serve.py ...          (the user's script, unedited: an offline sglang.Engine request loop)
#
# sglang.Engine spawns a scheduler subprocess that owns the GPU and runs every forward; the calling process only
# sends requests, so a capture wired into its request loop sees no model op (`NAN_NO_CAPSULE`). Loaded at
# interpreter start in every process (spawned children inherit PYTHONPATH), this file waits for
# sglang.srt.managers.scheduler to be imported and patches Scheduler.run_batch: the first call builds the capture
# (capture_from_env: after model load and HIP-graph capture, so the engine's start-up is never watched), every call
# starts with cap.step() -- one step per batch (a prefill, or one decode step of the running batch). run_batch
# launches the forward on the scheduler's own thread (the overlap scheduler only switches streams), so the
# thread-local dispatch mode sees every eager op: the prefill, and aiter / sgl_kernel custom ops as one op each.
# Decode steps replayed from HIP graphs are not seen.
#
# Nothing happens without NAN_CAPTURE_DIR. Prints `[sglang-capture] ...` lines. Tensor / pipeline parallel > 1: one
# scheduler per rank, one capture each; their coordinated stop runs over the replica's default group (every rank
# runs every batch). Data parallel > 1 (replicas, not DP attention): one capture dir per replica, <dir>/dp<i>.
# Validated with SGLang 0.5.23 (lmsysorg/sglang:v0.5.23-rocm10-mi35x, 2026-11-20); re-check the patch point after an
# upgrade.
# A sitecustomize later on sys.path (the image's own) is still run.
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "sglang.srt.managers.scheduler"


def _say(msg):
    print("[sglang-capture] " + msg, flush=True)


def _patch(mod):
    sched = mod.Scheduler
    orig = sched.run_batch
    state = {}

    def run_batch(self, *a, **kw):
        cap = state.get("cap")
        if cap is None:
            sa = getattr(self, "server_args", None)
            tp = int(getattr(sa, "tp_size", None) or getattr(self, "tp_size", 1) or 1)
            pp = int(getattr(sa, "pp_size", 1) or 1)
            dp = int(getattr(sa, "dp_size", 1) or 1)
            dpa = bool(getattr(sa, "enable_dp_attention", False))
            tp_rank = int(getattr(self, "tp_rank", 0) or 0)
            # the stop all-reduce runs over the default group: in a scheduler process that is this replica's
            # tp x pp ranks, which all run every batch (they share each forward) -- lockstep by construction.
            # Their siblings sit on other GPUs: not contention (op_nan_capture._sibling_ranks_on_other_gpus)
            os.environ.setdefault("LOCAL_WORLD_SIZE", str(tp * pp))
            os.environ.setdefault("LOCAL_RANK", str(tp_rank))
            if dp > 1 and not dpa and os.environ.get("NAN_CAPTURE_DIR"):
                # data-parallel replicas run independent forwards, each its own torch.distributed world: one
                # capture dir per replica, or the replicas' rank-0 manifests overwrite each other
                gpu = int(getattr(self, "gpu_id", 0) or 0)
                rep = (gpu - int(getattr(sa, "base_gpu_id", 0) or 0)) // max(1, tp * pp)
                os.environ["NAN_CAPTURE_DIR"] = os.path.join(os.environ["NAN_CAPTURE_DIR"], "dp%d" % rep)
                os.makedirs(os.environ["NAN_CAPTURE_DIR"], exist_ok=True)   # the capture asserts its dir exists
                os.chmod(os.environ["NAN_CAPTURE_DIR"], 0o777)
            sys.path.insert(0, os.path.join(os.environ["NUMERIC_DOCTOR_ROOT"], "scripts"))
            from op_nan_capture import capture_from_env
            cap = state["cap"] = capture_from_env()
            _say("capture built in the scheduler, pid %d, tp %d (rank %d) pp %d dp %d%s -> %s"
                 % (os.getpid(), tp, tp_rank, pp, dp, " (dp attention: one world)" if dpa else "",
                    os.environ.get("NAN_CAPTURE_DIR", "-")))
        cap.step()
        return orig(self, *a, **kw)

    sched.run_batch = run_batch
    _say("patched %s.Scheduler.run_batch in pid %d" % (_TARGET, os.getpid()))


class _Finder(importlib.abc.MetaPathFinder):
    """Patches the scheduler module right after it executes, wherever it is first imported."""

    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _chain():
    """Run the next sitecustomize on sys.path -- one AFTER this directory: this one shadows it, and an
    earlier one (another of ours on the same PYTHONPATH) already ran and chained here."""
    here = os.path.dirname(os.path.abspath(__file__))
    seen = False
    for d in sys.path:
        if os.path.abspath(d or ".") == here:
            seen = True
            continue
        f = os.path.join(d or ".", "sitecustomize.py")
        if seen and os.path.isfile(f):
            # registered in sys.modules: torch.compile re-imports the module of any function it traces
            # (a patch defined there), and an unregistered one is ModuleNotFoundError (2026-11-22)
            name = "_sitecustomize_next_%x" % (hash(os.path.abspath(f)) & 0xffffffff)
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return


if os.environ.get("NAN_CAPTURE_DIR"):
    sys.meta_path.insert(0, _Finder())
_chain()

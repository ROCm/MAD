# debug-numeric step 4: the per-op capture inside vLLM's model workers -- where vLLM runs the model at tensor parallel > 1.
#
#   PYTHONPATH=<REPO_ROOT>/scripts/vllm_capture  (+ NAN_CAPTURE_DIR, set by wave.sh on capture lines)
#     python3 serve.py --tp 2 ...    (the user's script, unedited: an offline vllm.LLM request loop)
#
# At tensor parallel > 1 vLLM V1 runs the model in one worker process per rank (the multiproc executor); the
# calling process only schedules, so a capture wired into its request loop sees no model op. Loaded at
# interpreter start in every process (spawned workers inherit PYTHONPATH), this file waits for
# vllm.v1.worker.gpu_worker to be imported and patches Worker.execute_model: the first call builds the capture
# (capture_from_env), and vLLM's warm-up / dummy forwards and HIP-graph capture run with the capture suspended
# (_WARMUPS: they do not always end before the first execute_model), so start-up is never watched; every call
# starts with cap.step() -- one step per scheduler step. Every TP rank runs execute_model every step, so the
# capture's coordinated stop (sync_stop: an all-reduce on the default group, which is the TP group on one node)
# stays in lockstep; at a capture all ranks stop together and the engine ends with the capture's RuntimeError.
#
# Seen: eager ops and vLLM's custom ops as one op each -- in the default compiled mode too, where Inductor calls
# them as extern ops (vllm.all_reduce, vllm.unified_attention_with_output, vllm.rocm_unquantized_gemm); a NaN
# from the all-reduce itself is captured at vllm.all_reduce on every rank (op_nan_capture.py, synchronous
# collectives). Not seen: Triton kernels inside compiled pieces, and HIP-graph replays (decode at <= 512 tokens).
#
# The engine core must be in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) only for TP 1; at TP > 1 the workers are
# the model's processes either way. Pipeline parallel > 1 and multi-node (vLLM's own --nnodes / --node-rank, no Ray):
# every worker still gets every execute_model, so the stop all-reduce on the default group stays in lockstep; each
# rank's manifest entry carries its stage (NAN_CAPTURE_RANK_TAG=pp<i>/tp<j>). Validated on one node (PP 2, TP 2 x PP 2,
# 2026-11-24) and over two nodes (TP 2 x PP 2, 2026-11-25; each node must see only its tp*pp/nnodes GPUs, step1-ask.md).
# Nothing happens without NAN_CAPTURE_DIR. Prints `[vllm-capture] ...` lines.
# Validated with vLLM 0.33.0 (vllm/vllm-openai-rocm:v0.33.0-rocm72, 2026-11-21); re-check the patch point after
# an upgrade. A sitecustomize later on sys.path (the image's own) is still run.
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm.v1.worker.gpu_worker"


def _say(msg):
    print("[vllm-capture] " + msg, flush=True)


def _patch(mod):
    worker = mod.Worker
    orig = worker.execute_model
    state = {}

    def execute_model(self, *a, **kw):
        cap = state.get("cap")
        if cap is None:
            root = os.environ.get("NUMERIC_DOCTOR_ROOT")
            if not root:
                raise RuntimeError("[vllm-capture] NUMERIC_DOCTOR_ROOT is not set (nd sets it in every runtime)")
            sys.path.insert(0, os.path.join(root, "scripts"))
            try:
                from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                                              get_tensor_model_parallel_world_size)
                import torch.distributed as dist
                tp, pp = get_tensor_model_parallel_world_size(), get_pp_group().world_size
                world = dist.get_world_size() if dist.is_initialized() else tp * pp
                rank = dist.get_rank() if dist.is_initialized() else get_tensor_model_parallel_rank()
                nnodes = max(1, int(getattr(getattr(self, "parallel_config", None), "nnodes", 1) or 1))
                local = max(1, world // nnodes)
                # torchrun's variables, which the capture reads: the sibling workers sit on OTHER GPUs, so
                # they are not contention (op_nan_capture._sibling_ranks_on_other_gpus)
                os.environ.setdefault("LOCAL_WORLD_SIZE", str(local))
                os.environ.setdefault("LOCAL_RANK", str(rank % local))
                # the manifest names each rank's stage: a NaN made in stage 0 reaches stage 1 through the pipeline's
                # send / recv, not through a collective the capture tracks -- the stage tells the origin apart
                os.environ["NAN_CAPTURE_RANK_TAG"] = "pp%d/tp%d" % (get_pp_group().rank_in_group,
                                                                  get_tensor_model_parallel_rank())
                # PP > 1: every worker of every stage gets every execute_model (the executor broadcasts it), so the
                # stop all-reduce on the default group stays in lockstep; a later stage waits for the earlier
                # one's stop check before its forward -- slower, no deadlock (tests/coverage G1, 2026-11-24)
            except Exception as e:                      # noqa: BLE001 - older/newer vLLM: carry on
                tp = pp = None
                _say("TP / PP size unknown (%s)" % e)
            from op_nan_capture import capture_from_env
            cap = state["cap"] = capture_from_env()
            _say("capture built in the worker, pid %d, tp %s pp %s %s" % (os.getpid(), tp, pp,
                                                                         os.environ.get("NAN_CAPTURE_RANK_TAG", "")))
        cap.step()
        return orig(self, *a, **kw)

    worker.execute_model = execute_model
    _say("patched %s.Worker.execute_model in pid %d" % (_TARGET, os.getpid()))
    _quiet_warmups()


# vLLM's warm-up / dummy forwards and its HIP-graph capture: no model output, nothing to judge. They are not
# always over before the first execute_model: vLLM 0.33 ran its graph-size warm-ups (_dummy_run at 112, 120,
# ... rows) after the capture was armed, and the capture stopped at a warm-up all-reduce's padded rows -- at
# "request 0", while the model's real overflow came at request 9 (2026-11-23/24). Suspended quietly (their
# Triton / Inductor launches unwatched too); a name a vLLM version lacks is skipped.
_WARMUPS = ("vllm.v1.worker.gpu_worker:Worker.compile_or_warm_up_model",
            "vllm.v1.worker.gpu_worker:Worker.execute_dummy_batch",
            "vllm.v1.worker.gpu_model_runner:GPUModelRunner._dummy_run",
            "vllm.v1.worker.gpu_model_runner:GPUModelRunner._dummy_sampler_run",
            "vllm.v1.worker.gpu_model_runner:GPUModelRunner.profile_run",
            "vllm.v1.worker.gpu_model_runner:GPUModelRunner.capture_model",
            "vllm.v1.worker.gpu.model_runner:GPUModelRunner._dummy_run",
            "vllm.v1.worker.gpu.model_runner:GPUModelRunner._dummy_sampler_run",
            "vllm.v1.worker.gpu.model_runner:GPUModelRunner.profile_run",
            "vllm.v1.worker.gpu.model_runner:GPUModelRunner.capture_model")


def _quiet_warmups():
    root = os.environ.get("NUMERIC_DOCTOR_ROOT")
    if not root:
        return
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import importlib
        from op_nan_capture import suspend_capture_in
    except Exception as e:                              # noqa: BLE001
        _say("warm-up suspension unavailable (%s)" % e)
        return
    done = []
    for path in _WARMUPS:
        mod, _, qual = path.partition(":")
        try:
            owner = importlib.import_module(mod)
            for part in qual.split("."):
                owner = getattr(owner, part)
        except Exception:                               # noqa: BLE001 - not in this vLLM version
            continue
        try:
            done += suspend_capture_in([path], quiet=True)
        except Exception as e:                          # noqa: BLE001
            _say("cannot suspend the capture in %s (%s)" % (path, e))
    _say("capture suspended in %d vLLM warm-up / graph-capture method(s): %s"
         % (len(done), ", ".join(p.split(":")[1] for p in done)))


class _Finder(importlib.abc.MetaPathFinder):
    """Patches the worker module right after it executes, wherever it is first imported."""

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

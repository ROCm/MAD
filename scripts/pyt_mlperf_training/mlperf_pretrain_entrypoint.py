###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
###############################################################################
"""MLPerf Llama-3.1 in-allocation torchrun entrypoint.

Upstream ``pretrain_llama31.py`` always goes through a NeMo-Run ``Experiment``
with either ``LocalExecutor`` (single node) or ``SlurmExecutor`` (submits a
fresh sbatch). Neither path fits the MAD scenario where ``madengine`` already
holds a multi-node Slurm allocation and just needs the training to run inside
it via ``torchrun``.

This shim reuses the exact model/data/callback construction from upstream
(``get_pretrain``, ``get_data``, ``callbacks.*``) and invokes the resulting
``run.Partial`` in-process via ``fdl.build(...)()``. Under a torchrun-launched
process group, NeMo's ``MegatronStrategy`` / Lightning pick up the rank/world
env vars that torchrun exports, so no executor is needed.

Intended to be executed as:

    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
             --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             mlperf_pretrain_entrypoint.py [args]

The script must live inside ``small_llm_pretraining/nemo/`` at run time so that
``pretrain_llama31`` and ``callbacks`` are importable as siblings.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
from pathlib import Path

import fiddle as fdl
import nemo_run as run
from nemo.collections.llm.gpt.data import build_pretraining_datamodule

# Upstream helpers: define model recipe and dataset config.
from pretrain_llama31 import get_pretrain, get_data  # type: ignore  # noqa: E402
from callbacks import PreemptiveStop, MLPerfCallback, MetricsLogger  # type: ignore  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="8b")
    parser.add_argument("--gbs", type=int, required=True)
    parser.add_argument("--mbs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--eval_every", type=int, required=True)
    parser.add_argument("--start_eval_at", type=int, default=0)
    parser.add_argument("--eval_tokens", type=int, default=1024)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--continual_ckpt_path", required=True)
    parser.add_argument("--target_log_ppl", type=float, default=3.3)
    parser.add_argument("--step_time_atol", type=int, default=18000)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--use_full_dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--data_index_sentinel",
        default="/mlperf-outputs/.data_index_done",
        help="File-based rank-0 barrier marker for data index build completion.",
    )
    parser.add_argument(
        "--data_index_timeout",
        type=int,
        default=3600,
        help="Seconds non-zero ranks wait for rank 0 to finish data index build.",
    )
    return parser.parse_args()


def _set_mlperf_env_defaults() -> None:
    # Match ``local_executor`` defaults from upstream pretrain_llama31.py, plus
    # the TE/NVTE knobs used by ``slurm_executor``.
    defaults = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _data_index_store(rank: int, world_size: int, timeout: int):
    """Rendezvous used to publish the data index build, on its own port.

    A file sentinel cannot distinguish a fresh publication from one left behind
    by a previous job, and ranks whose node started late must still be able to
    join, so the handshake goes through a store instead. It lives on
    MASTER_PORT+1 to keep the store of the training process group untouched.
    """
    import datetime

    import torch.distributed as dist

    return dist.TCPStore(
        host_name=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        port=int(os.environ.get("MASTER_PORT", "29500")) + 1,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=timeout),
        wait_for_workers=False,
    )


def _build_data_index_rank0_sync(
    sentinel: str, timeout: int, rank: int, world_size: int, build_partial
) -> None:
    import datetime

    sentinel_path = Path(sentinel)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    store = _data_index_store(rank, world_size, timeout) if world_size > 1 else None
    done_key = "mlperf_data_index_done"
    if rank == 0:
        if sentinel_path.exists():
            sentinel_path.unlink()
        print(f"[mlperf-entrypoint] rank=0 building Megatron data index", flush=True)
        # The build initializes a private world_size=1 process group through
        # env://. Under torchrun that lands in the launcher agent's store, which
        # every rank shares, so the NCCL bootstrap address this single-rank
        # communicator publishes under /default_pg/0 is what ranks 1..N-1 read
        # for the training group — and by then it no longer listens
        # ("Connection refused"). Rendezvous on a private port instead, with
        # TORCHELASTIC_USE_AGENT_STORE off so this rank serves that store itself.
        index_env = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "TORCHELASTIC_USE_AGENT_STORE": "False",
        }
        saved = {key: os.environ.get(key) for key in index_env}
        os.environ.update(index_env)
        try:
            fdl.build(build_partial)()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        sentinel_path.write_text("done\n")
        if store is not None:
            store.set(done_key, "done")
        print(f"[mlperf-entrypoint] rank=0 data index build complete", flush=True)
        return

    print(
        f"[mlperf-entrypoint] rank={rank} waiting for the rank-0 data index "
        f"(timeout {timeout}s)",
        flush=True,
    )
    store.wait([done_key], datetime.timedelta(seconds=timeout))


def main() -> None:
    args = _parse_args()

    # torchrun is the only authoritative source for distributed topology inside
    # the launched workers. NNODES / NPROC_PER_NODE inherited from the parent
    # shell may be set by madengine using a different convention (e.g. counting
    # tasks across all nodes), so we ignore them for the geometry check and
    # derive nnodes / gpus_per_node from WORLD_SIZE / LOCAL_WORLD_SIZE.
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as exc:
        raise RuntimeError(
            "Missing torchrun env var; this entrypoint must be launched via "
            "`python -m torch.distributed.run`/`torchrun`."
        ) from exc

    if local_world_size <= 0 or world_size % local_world_size != 0:
        raise RuntimeError(
            f"Inconsistent torchrun env: WORLD_SIZE={world_size}, "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )
    nnodes = world_size // local_world_size
    gpus_per_node = local_world_size

    print(
        "[mlperf-entrypoint] env: "
        f"WORLD_SIZE={world_size} LOCAL_WORLD_SIZE={local_world_size} "
        f"RANK={rank} LOCAL_RANK={local_rank} "
        f"NNODES(env)={os.environ.get('NNODES')} "
        f"NPROC_PER_NODE(env)={os.environ.get('NPROC_PER_NODE')} "
        f"derived nnodes={nnodes} gpus_per_node={gpus_per_node}",
        flush=True,
    )

    # --tensor_parallel_size exists to mirror the reference argument list, but
    # neither the reference nor this entrypoint feeds it to the trainer strategy:
    # it is only read back for the MLLOG DP figure. Reject anything but 1 rather
    # than let the flag silently mean nothing.
    if args.tensor_parallel_size not in (None, 1):
        raise RuntimeError(
            f"--tensor_parallel_size={args.tensor_parallel_size} is not applied to "
            "the trainer; the 8B recipe runs tp=pp=cp=1. Drop the flag."
        )

    # GBS / DP / MBS divisibility check: matches what NeMo's MegatronStrategy
    # asserts later, but produces a much clearer error before we touch CUDA.
    dp = world_size  # tp=pp=cp=1 for the 8B recipe
    if args.gbs % dp != 0:
        raise RuntimeError(
            f"GBS={args.gbs} is not divisible by DP={dp} "
            f"(world_size={world_size}). Pick GBS as a multiple of {dp}."
        )
    mini_batch = args.gbs // dp
    if mini_batch % args.mbs != 0:
        raise RuntimeError(
            f"mini_batch=GBS/DP={mini_batch} is not divisible by MBS={args.mbs}. "
            f"For world_size={world_size}, pick MBS that divides {mini_batch} "
            f"(e.g. MBS<={mini_batch})."
        )

    _set_mlperf_env_defaults()
    # pretrain_llama31.get_pretrain reads GBS via getenv, ensure it is visible.
    os.environ["GBS"] = str(args.gbs)
    os.environ.setdefault("PREPROCESSED_PATH", "/preproc_data")

    seq_length = 8192

    data = get_data(
        gbs=args.gbs,
        mbs=args.mbs,
        seq_length=seq_length,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        use_full_dataset=args.use_full_dataset,
    )

    eval_every_n_batches = math.ceil(args.eval_every / args.gbs)
    eval_batches = math.ceil(args.eval_tokens / args.gbs)
    # Reads as inverted, and is: a non-zero --start_eval_at is discarded in favour
    # of eval_every_n_batches, while zero divides zero. Kept verbatim from the
    # reference (mlcommons/training, llama31/pretrain_llama31.py: the
    # `if args.start_eval_at == 0` block) — the evaluation schedule is part of
    # the benchmark, so it must not diverge here.
    start_eval_at = (
        math.ceil(args.start_eval_at / args.gbs) if args.start_eval_at == 0 else eval_every_n_batches
    )

    exp_prefix, pretrain = get_pretrain(
        max_lr=args.max_lr,
        size=args.size,
        nnodes=nnodes,
        ngpus_per_node=gpus_per_node,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        data_module=data,
        eval_every=eval_every_n_batches,
        start_eval_at=start_eval_at,
        eval_batches=eval_batches,
    )

    from mlperf_logging.mllog import constants  # noqa: E402

    tp = args.tensor_parallel_size or pretrain.trainer.strategy.tensor_model_parallel_size
    pp = pretrain.trainer.strategy.pipeline_model_parallel_size
    cp = pretrain.trainer.strategy.context_parallel_size
    dp = (pretrain.trainer.num_nodes * pretrain.trainer.devices) // (tp * pp * cp)
    mini_batch_size = args.gbs // dp
    grad_accumulation_steps = mini_batch_size // args.mbs

    configs = {
        constants.GLOBAL_BATCH_SIZE: args.gbs,
        constants.GRADIENT_ACCUMULATION_STEPS: grad_accumulation_steps,
        constants.MAX_SEQUENCE_LENGTH: seq_length,
        constants.EVAL_SAMPLES: args.eval_tokens,
        constants.OPT_NAME: "adamw",
        constants.OPT_BASE_LR: pretrain.optim.config.lr,
        constants.OPT_ADAMW_BETA_1: pretrain.optim.config.adam_beta1,
        constants.OPT_ADAMW_BETA_2: pretrain.optim.config.adam_beta2,
        constants.OPT_ADAMW_EPSILON: pretrain.optim.config.adam_eps,
        constants.OPT_ADAMW_WEIGHT_DECAY: pretrain.optim.config.weight_decay,
        constants.OPT_GRADIENT_CLIP_NORM: pretrain.optim.config.clip_grad,
        constants.OPT_END_LR: pretrain.optim.lr_scheduler.min_lr,
        constants.OPT_LR_WARMUP_STEPS: pretrain.optim.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_STEPS: pretrain.trainer.max_steps - pretrain.optim.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        constants.SEED: args.seed,
    }

    pretrain.trainer.num_sanity_val_steps = 0
    pretrain.data.num_train_samples = pretrain.trainer.max_steps * pretrain.data.global_batch_size
    # No checkpoint writes in the initial real run; keep symmetry with upstream
    # dryrun path that also sets this to False.
    pretrain.trainer.enable_checkpointing = False
    pretrain.log.tensorboard = None
    pretrain.log.ckpt.every_n_train_steps = None
    pretrain.log.ckpt.save_top_k = 0
    pretrain.log.ckpt.save_last = False
    pretrain.log.ckpt.always_save_context = False

    original_callbacks = list(pretrain.trainer.callbacks or [])
    pretrain.trainer.callbacks = original_callbacks + [
        run.Config(PreemptiveStop, stop_on_step=args.max_steps),
        run.Config(
            MLPerfCallback,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            sequence_length=seq_length,
            eval_every=eval_every_n_batches,
            init_global_step=0,
            configs=configs,
        ),
    ]

    pretrain.log.extra_loggers = [
        run.Config(
            MetricsLogger,
            init_global_step=0,
            global_batch_size=args.gbs,
            seq_length=seq_length,
            target_log_ppl=args.target_log_ppl,
            train_step_time_atol=args.step_time_atol,
        ),
    ]

    # Data index build mirrors upstream ``data_index_executor`` phase: run once
    # before training. File-based barrier avoids racing with Megatron's own
    # torch.distributed init inside Lightning.
    datamodule = pretrain.data.clone()
    datamodule.num_dataset_builder_threads = 64
    build_data_index = run.Partial(
        build_pretraining_datamodule,
        datamodule=datamodule,
        trainer_max_steps=pretrain.trainer.max_steps,
        trainer_val_check_interval=pretrain.trainer.val_check_interval,
        trainer_limit_val_batches=pretrain.trainer.limit_val_batches,
        trainer_limit_test_batches=pretrain.trainer.limit_test_batches,
    )
    _build_data_index_rank0_sync(
        sentinel=args.data_index_sentinel,
        timeout=args.data_index_timeout,
        rank=rank,
        world_size=world_size,
        build_partial=build_data_index,
    )

    # Distributed setup is left to MegatronStrategy. It must find
    # torch.distributed uninitialized: ``setup_distributed`` returns early on an
    # already-initialized process group and thereby skips
    # ``init_model_parallel``, which later shows up as "data parallel group with
    # context parallel combined is not initialized" from
    # ProcessGroupCollection.use_mpu_process_groups().
    #
    # NeMo's ``build_pretraining_datamodule`` above does leave a private
    # single-rank group behind on rank 0, so that one has to go.
    import torch
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() != world_size:
        stale_ws = dist.get_world_size()
        dist.destroy_process_group()
        print(
            f"[mlperf-entrypoint] rank={rank} tore down stale PG "
            f"(world_size={stale_ws}) left by NeMo data index build",
            flush=True,
        )

    if torch.cuda.is_available():
        # Pin to LOCAL_RANK so NCCL's bootstrap uses the correct device for
        # this process; otherwise NCCL may pick GPU 0 on every local rank and
        # later fail on collective ops when devices don't match the PG.
        torch.cuda.set_device(local_rank)

    print(f"[mlperf-entrypoint] rank={rank} starting pretrain fit", flush=True)
    try:
        fdl.build(pretrain)()
    finally:
        # Deliberate clean shutdown:
        #   1) barrier so that rank 0 does not race ahead and tear down the
        #      MASTER TCPStore while other ranks are still in late teardown
        #      hooks (callbacks, MLPerfCallback.on_train_end, etc.). Without
        #      this we observed rank 5 hanging 30 min on a gloo sub-PG
        #      rendezvous after rank 0 had already exited.
        #   2) explicit destroy_process_group() to avoid the "was not called
        #      before program exit" warning and NCCL proxy abort spam.
        try:
            import torch.distributed as _dist

            if _dist.is_available() and _dist.is_initialized():
                _dist.barrier()
                _dist.destroy_process_group()
                print(
                    f"[mlperf-entrypoint] rank={rank} process group destroyed",
                    flush=True,
                )
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(
                f"[mlperf-entrypoint] rank={rank} clean shutdown failed: {exc!r}",
                flush=True,
            )
    print(f"[mlperf-entrypoint] rank={rank} pretrain fit returned", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

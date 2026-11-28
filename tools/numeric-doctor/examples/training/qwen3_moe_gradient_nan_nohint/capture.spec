# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for
# qwen3_moe_gradient_nan_nohint/run.sh: ONE 8-rank job (FSDP 4 x TP 2 x EP 4, 1 node), GPU field `all`,
# scripts/torchtitan_capture_train.py in place of `-m torchtitan.train` (step4-distributed.md); flags =
# run.sh's train_args, 100 steps.  No wired copy needed: the capture is patched into torchtitan's
# Trainer.train_step at run time.  Needs run.sh's seed checkpoint (./run.sh build) and its image.
# Validated on Slurm + docker 2026-11-13.  Setup only: what it found is not in this header --
# a PRIORS=none run reads it (step1-ask.md Q7).
#   SPEC=examples/training/qwen3_moe_gradient_nan_nohint/capture.spec CONTROLS=none \
#   NAN_SIGNAL='Loss is not finite|grad_norm: +([1-9][0-9]{2,}|nan|inf)' \
#   CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1 NAN_CAPTURE_SUSPEND=torchtitan.models.common.attention:FlexAttention.compiled_flex_attn NAN_CAPTURE_BOUNDS=1 NAN_CAPTURE_VERIFY_EVERY=1' \
#     ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh
# (image qwen3-moe:<sha256 of ./Dockerfile, first 12> -- run.sh's DIMAGE; build it once, IMAGE_TAR for other nodes)
all|cap|MODULE=qwen3 CONFIG=qwen3_moe_debug PYTORCH_ALLOC_CONF=expandable_segments:True TORCHFT_LIGHTHOUSE=http://localhost:29510 LOG_RANK=0|cd $ND_REPO/examples/training/qwen3_moe_gradient_nan_nohint/src/torchtitan && export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} && D=$ND_ART/run/$ND_JOB_ID && mkdir -p $D/checkpoint $D/ranklogs && ln -sfn $ND_REPO/examples/training/qwen3_moe_gradient_nan_nohint/model/seed_checkpoint/step-0 $D/checkpoint/step-0 && torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --log-dir $D/ranklogs --redirects 3 --tee 3 --local-ranks-filter 0 --role rank $ND_REPO/scripts/torchtitan_capture_train.py --module qwen3 --config qwen3_moe_debug --dump_folder=$D --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4 --debug.deterministic --debug.seed=42 --metrics.enable_tensorboard --metrics.log_freq=1 --training.steps=100 --checkpoint.enable --checkpoint.export_dtype=bfloat16 --checkpoint.load_only

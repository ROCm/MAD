# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for qwen_vllm_serve_nan
# (run.sh's configuration): one vLLM engine per GPU, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), vLLM's defaults
# (torch.compile + HIP graphs). Laid out per step4-run.md *vLLM with graphs*: 3 capture lines with graph replay off
# (scripts/vllm_graphs, ND_VLLM_CUDAGRAPH_MODE=NONE; torch.compile kept) + 2 controls as the user runs it (graphs on:
# the folded step-2 rate) + the graph check `g` + the compiled track's C0 (--enforce-eager 1) and C1 (Inductor
# nan_asserts + the kernel-capsule hook, graphs off through scripts/vllm_graphs -- no hand-made copy needed);
# 75 requests = 1.5 x the NaN at request 49.
# Before submitting, in <ART>:
#   mkdir -p <ART>/model <ART>/capsules_inductor <ART>/capsules_graph && chmod 777 <ART>/capsules_inductor <ART>/capsules_graph
#   cp examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py <ART>/model/ &&
#   python3 scripts/wire_capture.py apply <ART>/model/qwen_vllm_serve.py --backup-dir <ART>
# Profile (the Dockerfile only sets ENV + copies the script: its base image is used; nd clears its entrypoint):
#   scripts/nd profile <site> --art <ART> \
#     IMAGE=docker.io/vllm/vllm-openai-rocm@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba \
#     EXTRA_ENV="HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0"
# Validated on Slurm + docker 2026-11-23 (1m55s).  Setup only: what it found is not in this header --
# a PRIORS=none run reads it (step1-ask.md Q7).
#   SPEC=examples/inference/qwen_vllm_serve_nan/capture.spec NAN_SIGNAL='Logprobs are not finite' \
#   WAIT_CONTROLS=1 WAIT_ARMS='ctl g c0 c1' CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1' \
#     ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive --time 00:45:00 --step step4 -- capture_to_repro.sh
0|cap|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_VLLM_CUDAGRAPH_MODE=NONE|python3 $ND_ART/model/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 0
1|cap|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_VLLM_CUDAGRAPH_MODE=NONE|python3 $ND_ART/model/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 1
2|cap|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_VLLM_CUDAGRAPH_MODE=NONE|python3 $ND_ART/model/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 2
3|ctl||python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 3
4|ctl||python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 4
5|g|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_GRAPH_CHECK=1 ND_GRAPH_CHECK_DIR=$ND_ART/capsules_graph|python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 5
6|c0||python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 1 --seed 6
7|c1|PYTHONPATH=$ND_REPO/scripts/vllm_graphs:$ND_REPO/scripts/inductor_capsule ND_VLLM_CUDAGRAPH_MODE=NONE TORCHINDUCTOR_NAN_ASSERTS=1 TORCH_LOGS=output_code TORCH_LOGS_OUT=$ND_ART/output_code.c1.log INDUCTOR_CAPSULE_DIR=$ND_ART/capsules_inductor|python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --enforce-eager 0 --seed 7

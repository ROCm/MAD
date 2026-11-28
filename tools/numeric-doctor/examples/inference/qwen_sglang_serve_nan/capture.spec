# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for qwen_sglang_serve_nan,
# arm serve: one sglang.Engine per GPU, SGLang's defaults (eager prefill, HIP-graph decode).
# The model runs in SGLang's scheduler subprocess: capture lines run the ORIGINAL script with
# PYTHONPATH=$ND_REPO/scripts/sglang_capture, whose sitecustomize installs the capture at Scheduler.run_batch
# (step1-ask.md *SGLang*). 6 capture + 2 control (step 2 folded: the control arm is the rate); 75 requests =
# 1.5 x the NaN at request 49.
# Validated on Slurm + docker 2026-11-19 and 2026-11-20.  Setup only: what it
# found is not in this header -- a PRIORS=none run reads it (step1-ask.md Q7).
# Profile (the Dockerfile prebuilds SGLang's Rust radix tree: build the image once, run.sh pull or env-runtime.md
# *Images*; aiter needs root in the container):
#   scripts/nd profile <site> --art <ART> IMAGE=qwen-sglang-serve:v0.5.23 IMAGE_TAR=<saved archive> \
#     DOCKER_AS_USER=0 EXTRA_ENV="HF_HUB_OFFLINE=1 SGLANG_CACHE_DIR=/opt/sglang-cache"
# Submit:
#   SPEC=<this file> NAN_SIGNAL='Logprobs are not finite' CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1' \
#     WAIT_CONTROLS=1 ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh
0|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 0
1|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 1
2|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 2
3|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 3
4|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 4
5|cap|PYTHONPATH=$ND_REPO/scripts/sglang_capture|python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 5
6|ctl||python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 6
7|ctl||python3 $ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py --requests 75 --prompt-len 256 --new-tokens 8 --log-every 25 --seed 7

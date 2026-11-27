# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for
# qwen_train_nan, translated from run.sh: one container per GPU running --trials 3 (3 mp.spawn processes share
# the GPU and are counted by the capture itself under apptainer --pid; wave.sh declares 1 container per GPU),
# 300 steps, seq 1280, bs 1, a seed per GPU; capture arm on GPUs 0-5, control arm on GPUs 6 7.
# Capture lines run a WIRED COPY at $ND_ART/model/qwen_train.py (wire_capture.py apply).
# Setup only: what it found is not in this header (step1-ask.md Q7).
# The run script uses --cleanenv: the investigation's env.sh needs APPTAINER_FLAGS="--pid --cleanenv".
#   SPEC=examples/training/qwen_train_nan/capture.spec NAN_SIGNAL='nan_step=[0-9]' \
#     ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive -- capture_to_repro.sh
0|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 4242 --log-every 50
1|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 5242 --log-every 50
2|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 6242 --log-every 50
3|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 7242 --log-every 50
4|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 8242 --log-every 50
5|cap|NAN_PROGRESS_EVERY=25|python3 $ND_ART/model/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 9242 --log-every 50
6|ctl|NAN_PROGRESS_EVERY=25|python3 $ND_REPO/examples/training/qwen_train_nan/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 10242 --log-every 50
7|ctl|NAN_PROGRESS_EVERY=25|python3 $ND_REPO/examples/training/qwen_train_nan/qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3 --seed 11242 --log-every 50

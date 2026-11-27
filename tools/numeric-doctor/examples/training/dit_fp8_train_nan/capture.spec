# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for
# dit_fp8_train_nan, translated from run.sh: one trainer per GPU, 30 steps, a seed per GPU, the training
# environment of run.sh (PyTorch TunableOp on, one results file per trainer); capture arm on GPUs 0-5, control
# arm on GPUs 6 7.
# Capture lines run a WIRED COPY of dit_fp8_train.py at $ND_ART/model/dit_fp8_train.py:
#   mkdir -p <ART>/model && cp examples/training/dit_fp8_train_nan/dit_fp8_train.py <ART>/model/ &&
#   python3 scripts/wire_capture.py apply <ART>/model/dit_fp8_train.py --backup-dir <ART>
# Not validated yet.  Setup only: what it finds is not in this header (step1-ask.md Q7).
#   SPEC=examples/training/dit_fp8_train_nan/capture.spec NAN_SIGNAL='Gradient norm is not finite|Loss is not finite' \
#     ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive -- capture_to_repro.sh
# (image dit-fp8-train-nan:primus-v26.3, built by run.sh from ./Dockerfile)
0|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g0_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 4242 --log-every 5
1|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g1_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 5242 --log-every 5
2|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g2_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 6242 --log-every 5
3|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g3_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 7242 --log-every 5
4|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g4_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 8242 --log-every 5
5|cap|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g5_tunableop_%d.csv|python3 $ND_ART/model/dit_fp8_train.py --steps 30 --seed 9242 --log-every 5
6|ctl|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g6_tunableop_%d.csv|python3 $ND_REPO/examples/training/dit_fp8_train_nan/dit_fp8_train.py --steps 30 --seed 10242 --log-every 5
7|ctl|PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=$ND_ART/g7_tunableop_%d.csv|python3 $ND_REPO/examples/training/dit_fp8_train_nan/dit_fp8_train.py --steps 30 --seed 11242 --log-every 5

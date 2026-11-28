# debug-numeric capture spec (scripts/jobs/wave.sh format: <gpu>|<arm>|<env>|<command>) for
# softmoe_train_nan: layers 3-18 compiled, one trainer per GPU, 6 capture + 2 control.
# Capture lines run a WIRED COPY of train.py at $ND_ART/model/train.py:
#   mkdir -p <ART>/model && cp examples/training/softmoe_train_nan/train.py <ART>/model/ &&
#   python3 scripts/wire_capture.py apply <ART>/model/train.py --backup-dir <ART>
# Validated on Slurm + docker 2026-11-11.  Setup only: what it found is not in this header --
# a PRIORS=none run reads it (step1-ask.md Q7).
#   SPEC=examples/training/softmoe_train_nan/capture.spec NAN_SIGNAL='Loss is not finite' LOCATE=1 \
#   CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1 NAN_CAPTURE_SUSPEND=torch._dynamo.eval_frame:OptimizedModule.__call__' \
#     ND_ENV=<ART>/env.sh scripts/nd submit --name c2r --gpus 8 --exclusive -- capture_to_repro.sh
# (image docker.io/rocm/pytorch@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4)
0|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
1|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
2|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
3|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
4|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
5|cap||python3 $ND_ART/model/train.py --steps 100 --compile-layers 3-18
6|ctl||python3 $ND_REPO/examples/training/softmoe_train_nan/train.py --steps 100 --compile-layers 3-18
7|ctl||python3 $ND_REPO/examples/training/softmoe_train_nan/train.py --steps 100 --compile-layers 3-18

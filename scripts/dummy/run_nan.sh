#!/bin/bash
#
# dummy_nan: a tiny PyTorch training loop that produces a NaN on purpose.
#
# At step --plant_step (default 20) one activation goes through sqrt() of a
# negative number, so aten.sqrt creates the NaN from finite inputs; the loss,
# the gradients and then the weights turn NaN. It exists to validate the NaN
# debugging path (madengine's nan_doctor tool, /mad-debug-nan, mad-nan-hunt)
# end to end on one GPU. --plant_step -1 trains cleanly.
#
# Prints `step <n> loss: <value>` per step. It keeps training through the first
# NaN step (so the NaN reaches the weights, as in a real run), then stops with
# `NaN loss at step <n>` and exits 1. A clean run prints the performance line.

set -e
PLANT_STEP=20
STEPS=60
while [ $# -gt 0 ]; do
    case "$1" in
        --plant_step) PLANT_STEP="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

python3 - "$PLANT_STEP" "$STEPS" <<'PY'
import sys
import time

import torch

plant, steps = int(sys.argv[1]), int(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1)
).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
batch = 64
start = time.time()
for step in range(steps):
    x = torch.randn(batch, 256, device=dev)
    h = model[1](model[0](x))
    if step == plant:
        h = torch.sqrt(h - 1e3)  # negative under the root: the planted NaN
    loss = model[2](h).pow(2).mean()
    print(f"step {step} loss: {loss.item():.6f}", flush=True)
    loss.backward()
    opt.step()  # a NaN loss still updates: the weights turn NaN, as in a real run
    opt.zero_grad()
    if not torch.isfinite(loss) and step > plant:
        print(f"NaN loss at step {step}", flush=True)
        sys.exit(1)
if dev == "cuda":
    torch.cuda.synchronize()
print(f"performance: {steps * batch / (time.time() - start):.2f} samples_per_second", flush=True)
PY

# vLLM native data-parallel (--data-parallel-size) is BROKEN for Qwen3-Next-80B FP8

## Finding (2026-06-28)
vLLM's native DP fails to start for the FP8 model, at ANY TP including TP1:
```
ValueError: The output_size of gate's and up's weight = 64 is not divisible by
weight quantization block_n = 128.
```
- `--data-parallel-size 8 --tensor-parallel-size 1` → FAILS (Worker_DP1/DP4 ...)
- `--data-parallel-size 2 --tensor-parallel-size 4` → FAILS (Worker_DP0_TP3 ...)
- Plain `--tensor-parallel-size 4` (DP1, single replica) → WORKS (validated Phase B).

## Why
vLLM's DP path appears to apply an extra split to the MoE gate/up projection that produces a
64-wide shard, violating the FP8 128-block quant (same class as the TP8 issue, but triggered by
the DP machinery even at TP1). It is NOT just a TP-sharding effect — DP8/TP1 has no TP split yet
still hits 64.

## Decision
Do NOT use engine-native DP for the FP8 sweep. Instead use the original v2 approach:
**N independent single-server replicas** (each a plain TP-k server on its own GPU group), launched
by run_engine.sh placement. Full-node = (8/TP) independent replicas. Aggregate is measured by
fanning the benchmark across the replica ports (or a router), NOT by engine DP.

## Implication for "inbuilt router"
The engines DO expose DP flags, but vLLM's DP is unusable for this FP8 checkpoint. So the
manual-replica + external aggregation path is required for FP8. (SGLang/ATOM DP untested for FP8
given vLLM's failure; assume manual replicas for uniformity.)

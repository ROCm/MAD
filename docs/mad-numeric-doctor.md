# NaN-Doctor in MAD: user manual

This manual covers the current integration: the `mad-debug-numeric` skill in MAD, the
`madengine nan` commands and the `nan_doctor` tool in madengine, and the container
runtime (`nan_doctor_rt`) that madengine ships. The nan-doctor sources they were
ported from are upstream commit `6ce054c`.

Every command below was run on one 8× AMD Instinct MI300X (gfx942) node, with madengine on branch
`nan-doctor-integration`. Where an example did not reproduce, the section says so
(1.4 and 3.4).

## 1. What is NaN-Doctor?

A skill for an LLM harness (Claude Code, Cursor, Codex) that debugs numeric failures in
the AI models you run with madengine. A numeric failure is a NaN, an Inf, a loss or
gradient spike, or a finite value that is wrong. The harness drives the madengine
commands; every command in this manual can also be typed by hand.

Expected usage:

- Numeric failures in models or system libraries.
- PyTorch-based frameworks and models.
- AMD Instinct GPUs (MI300X, MI350X/MI355X).
- Training and inference. vLLM and SGLang are supported by their own capture modes.
- Multi-GPU in one node. Multi-node runs launched by madengine's SLURM deployment capture
  per node; models with their own multi-node launcher (`slurm_multi`) get no capture.

**What you get out.** An artifact directory with a **capsule** (the saved record of the
first op that went bad, with its inputs), a **group** (capsules that share one cause),
a **replay verdict** (`REPRODUCED`, `NOT_REPRODUCED` or `NOT_SET_UP`, with the rate), a
standalone **reproducer** `repro-G<n>.tar.gz` that needs no model, and a report. With
root-cause and fix switched on it also gives a root cause and a proposed fix. The fix
is proposed, never applied.

| Step | What happens | Command behind it |
|---|---|---|
| 1 | You say how to run the model | `madengine discover --tags <tag>` |
| 2 | Run it and measure the failure | `madengine run` |
| 3 | Raise the failure rate (optional) | |
| 4 | Capture a capsule | `madengine run` with the `nan_doctor` tool |
| 5a, 5b | Group capsules; replay the largest group | `madengine nan group`, `madengine nan replay` |
| 5N | Native fallback when no capsule replays | `madengine nan handoff` |
| 6 | Package a reproducer | `madengine nan package` |
| 7, 8 | Root-cause and fix (budgeted, experimental) | |
| 9 | Report | `madengine nan report` |

For a model that prints no NaN and no error but returns wrong answers, step 2 uses
**differential mode** (section 4): run the model under several configurations and
compare what it returned.

## 2. Where to get it

MAD holds the skill, the agent and the workflow. madengine holds the commands, the tool and
the runtime. The `nan` command group is on madengine branch `nan-doctor-integration`; it is
not on `main`, so `pip install -r requirements.txt` in MAD does not give you `madengine nan`.

```bash
git clone https://github.com/ROCm/madengine.git
cd madengine && git checkout nan-doctor-integration && pip install -e .
madengine nan --help        # must list group, replay, package, arms, diff, ...
```

In MAD (branch `nan-doctor-integration`):

| Piece | Path |
|---|---|
| Skill (all harnesses; `/mad-debug-numeric`) | `.claude/skills/mad-debug-numeric/` |
| Agent | `.claude/agents/mad-numeric-doctor.md` (Codex: `.codex/agents/mad_numeric_doctor.toml`) |
| Workflow (Claude Code only, plans by default) | `.claude/workflows/mad-nan-hunt.js`, `docs/mad-nan-hunt.md` |
| GPU run helper | `.claude/skills/mad-common/mad-run.sh` |
| Two validation models | `scripts/dummy` (`dummy_nan`), `scripts/torchtitan_qwen3_moe_nan` |

Not ported from nan-doctor: its own launcher `nd`, the Slurm and Apptainer site profiles.
`madengine run` and `mad-run.sh` take their place.

## 3. Example: debug a planted NaN (`dummy_nan`)

`dummy_nan` is a small training loop on one GPU. At step 20 it takes the square root of a
negative activation, so `aten.sqrt` makes a NaN from finite inputs and the weights turn NaN
a step later. The whole investigation takes about two minutes.

### 3.1 Let the harness do it

From the MAD repo root, in Claude Code or Cursor:

```text
/mad-debug-numeric dummy_nan
```

(Codex: `$mad-debug-numeric dummy_nan`.) The skill prints one line saying what it will do and
asks how to run the model; the tag is that answer. Everything else takes its default unless you
say *ask me the options*:

| Question | Default |
|---|---|
| Run the model once before capture (step 2) | Skip |
| Root-cause and fix (steps 7 and 8) | Yes, within a time budget the skill states |
| Capture geometry | The model's own |
| Capsule contents | Shapes only, unless you set `NAN_CAPTURE_DATA=1` |

The workflow form plans without touching a GPU and runs only with `--execute`:

```text
/mad-nan-hunt --tag dummy_nan
/mad-nan-hunt --tag dummy_nan --skip-detect --execute
```

The rest of this section is what the skill runs, so you can reproduce it by hand. Every
GPU command goes through `mad-run.sh`, which holds one host GPU lock and survives an
agent's 10-minute tool-call limit.

### 3.2 Capture (step 4)

```bash
bash .claude/skills/mad-common/mad-run.sh start cap -- madengine run --tags dummy_nan --live-output \
  --additional-context '{"tools": [{"name": "nan_doctor"}], "docker_env_vars": {"NAN_CAPTURE_DATA": "1", "NAN_CAPTURE_ARM_AFTER_STEPS": "1"}}'
bash .claude/skills/mad-common/mad-run.sh wait cap      # repeat until succeeded or failed
```

The model exits 1 (the planted NaN), so the run is `failed`; the signal is in
`.mad-runs/cap/log`, not the exit code. The log shows:

```text
✓ nan_doctor: capture armed (site /usr/local/lib/python3.12/dist-packages, capsules -> /myworkspace/nan_doctor_output/capsules)
[op-nan] nan-capture armed at step() 1: ... gpu_procs=1 containers_per_gpu=1
[op-nan][rank 0] pending: aten.sqrt.default @ <stdin>:19 (return) at step 18, out=(64, 512) nan=32768 inf=0; origin: math (fp64 re-run, of 32768 non-finite: 32768 also non-finite in fp64, ...)
[op-nan][rank 0] captured -> /myworkspace/nan_doctor_output/capsules/rank0_<hash>.pt
```

`nan_doctor_output/summary.txt` says `capsules=1`. A run under the tool is not a
performance measurement. Copy the capsules out before the next run: the next capture
moves a non-empty `capsules/` aside to `capsules.prev-<time>/`.

```bash
mkdir -p nan_hunts/dn && chmod 777 nan_hunts/dn
cp -r nan_doctor_output/capsules nan_hunts/dn/capsules
```

The tool needs no change to the model; its pre-script arms a hook on the optimizer step.
Loops with no optimizer step, and Trainer loops such as torchtitan's, are armed by the
model's own entry point (see 3.4).

### 3.3 Group, replay, package (steps 5 and 6)

```bash
C=nan_hunts/dn/capsules/rank0_<hash>.pt
bash .claude/skills/mad-common/mad-run.sh start grp -- madengine nan group nan_hunts/dn/capsules \
  --manifest build_manifest.json --out-dir nan_hunts/dn
mkdir -p nan_hunts/dn/replay/G1 && chmod 777 nan_hunts/dn/replay nan_hunts/dn/replay/G1
bash .claude/skills/mad-common/mad-run.sh start rep -- madengine nan replay $C \
  --manifest build_manifest.json --out-dir nan_hunts/dn/replay/G1
bash .claude/skills/mad-common/mad-run.sh wait rep
```

`group` found one group, `G1 1 aten.sqrt.default`. The replay ends:

```text
walk: ... aten.sub.Tensor (finite) -> aten.sqrt.default NO (non-finite; fp64 reference NON-FINITE)
-> first op to go wrong: aten.sqrt.default (call #880, NON_FINITE)
proposed verdict: FP32_OVERFLOW? (terminal fp64 ref is NON-FINITE ... not a kernel bug)
deterministic probe: REPRODUCED 100/100 -> REPRODUCED at concurrency 1
LADDER_RESULT=REPRODUCED at concurrency 1 -- deterministic: 100/100 solo runs on GPU 0 non-finite
```

The reproduced NaN is the model's own arithmetic (the true result is not finite either), so
the verdict is a **model-numerics** reproducer, not a kernel bug. It is still shipped: steps 7 and 8
find why the model hands the op those values. Package it, using the attempt directory the
replay wrote (`replay/G1/c1det.<pid>.<host>`; `c<N>.` for a ladder attempt):

```bash
bash .claude/skills/mad-common/mad-run.sh start pkg -- madengine nan package --group G1 \
  --artifact-dir nan_hunts/dn --image <docker_image in build_manifest.json> --capsule $C \
  --attempt nan_hunts/dn/replay/G1/c1det.<pid>.<host> --groups nan_hunts/dn/groups.json --pack
```

It prints `PACKAGE .../repro-G1`, `MODEL_NUMERICS ...` and the tarball's sha256. The
mini-reproducer is `nan_hunts/dn/repro-G1.tar.gz` (about 1.7 MB: the scripts, a copy of the
replay package, and the capsule).

### 3.4 Verify the reproducer the way a vendor would

Extract it somewhere clean and run it. It needs docker, python3 and the image, not madengine.

```bash
mkdir -p nan_hunts/dn/verify-G1 && tar -xzf nan_hunts/dn/repro-G1.tar.gz -C nan_hunts/dn/verify-G1
bash .claude/skills/mad-common/mad-run.sh start ver -- bash nan_hunts/dn/verify-G1/repro-G1/run.sh
bash .claude/skills/mad-common/mad-run.sh wait ver
# ... LADDER_RESULT=REPRODUCED at concurrency 1 -- deterministic: 100/100 solo runs on GPU 0 non-finite
```

The pass criterion is `n/N > 0` in that log, not the exit code. (`CAPSULE_ITER_COUNT=2000
bash run.sh` shortens it.) The extracted copy also ran with an empty environment
(`env -i ...`) and gave the same 100/100.

### 3.5 Report (step 9)

```bash
madengine nan report nan_hunts/dn
```

writes `report.md` and `result.json` from the ledger and the artifacts, with numbered
`TODO(agent)` lines for the judgement parts (verdict paragraph, GPU line, root cause). The
harness fills them (`madengine nan report <dir> --fill <answers.md>`). The result card is
checked with `madengine nan exec --manifest build_manifest.json -- python3 -m
nan_doctor_rt.result_card check <dir>`.

## 4. Example: no NaN, wrong answers (differential mode, GPT-2)

Some failures print nothing. Differential mode runs the model once per **arm** and compares what
it returned:

- `A`: your configuration. `A2`: the same again (the noise floor).
- `REF`: the conservative configuration: every switch (torch.compile, TF32, the BLAS
  library, fused attention) on its safe side.
- `flip_<switch>`: yours with one switch flipped.

The tap (`nan_doctor_rt/diff_tap`) records the training loss per step (or, for an inference model,
generated tokens and logprobs) into `outputs.jsonl`. The switches are environment-only; the model is
not edited.

```bash
madengine nan arms detect -- bash scripts/huggingface_gpt2/run.sh     # FRAMEWORK=torch
madengine nan arms arms --framework torch --format context --out-dir nan_hunts/diff-gpt2 > arms.tsv
# arms: A, A2, REF, flip_compile, flip_tf32, flip_blas, flip_sdpa
for arm in A A2 REF flip_compile flip_tf32 flip_blas flip_sdpa; do
  ctx=$(awk -F'\t' -v a=$arm '$1==a{print $2}' arms.tsv)
  bash .claude/skills/mad-common/mad-run.sh start gpt2-$arm -- \
    madengine run --tags pyt_huggingface_gpt2:max_steps=30 --live-output -c "$ctx"
done
```

(The runs queue behind the GPU lock, about five minutes each.) Each context is
`{"docker_env_vars": {"ND_DIFF_OUT": "/myworkspace/nan_hunts/diff-gpt2/<arm>", "PYTHONPATH":
"/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt/diff_tap", <the arm's switches>}}`. Then:

```bash
D=nan_hunts/diff-gpt2
madengine nan diff --a $D/A/outputs.jsonl --a2 $D/A2/outputs.jsonl --ref $D/REF/outputs.jsonl \
  --arm flip_compile=$D/flip_compile/outputs.jsonl --arm flip_tf32=$D/flip_tf32/outputs.jsonl \
  --arm flip_blas=$D/flip_blas/outputs.jsonl --arm flip_sdpa=$D/flip_sdpa/outputs.jsonl --json $D/diff.json
```

```text
ND_DIFF_ARM name=flip_compile verdict=DIVERGE records=30 diverge=1 tie_flips=0 worst_gap=0.6966
ND_DIFF_ARM name=flip_tf32    verdict=DIVERGE records=30 diverge=1 tie_flips=0 worst_gap=0.6966
ND_DIFF_ARM name=flip_blas    verdict=DIVERGE records=30 diverge=1 tie_flips=0 worst_gap=0.6966
ND_DIFF_ARM name=flip_sdpa    verdict=AGREE  records=30 diverge=0 tie_flips=0 worst_gap=0.0002662
ND_DIFF_SWITCH=flip_sdpa
ND_DIFF verdict=DIVERGE suspect=A records=30 diverge=1 tie_flips=0 first=step:6 worst_gap=0.6966 ...
```

Reading it: A and A2 agree (the run is deterministic, so the gap is not noise); A and REF
diverge first at step 6 by a loss gap of 0.70 against a threshold of 0.02; flipping only the SDPA
switch (the fused attention kernels replaced by the math backend) brings A back to the
reference. `ND_DIFF_SWITCH=` names the switch, and the harness continues from there. The verdicts are
`AGREE`, `DIVERGE`, `NONDETERMINISTIC` (A differs from A2) and `INPUTS_DIFFER` (pin the inputs first).
`--modules` (with `ND_DIFF_MODULES` set on a run) names the first module whose output
diverges. Check mode (`ND_DIFF_REF=<reference outputs.jsonl>` on a later run) prints `ND_DIFF_HIT` lines,
so a capture wave can use a model that prints no signal itself.

Notes: GPT-2 downloads its weights; `scripts/huggingface_gpt2/run.sh` sets
`HF_HUB_DISABLE_XET=1` so the download does not fail on the XET token request.

## 5. Wrong finite values: twin references

With `NAN_CAPTURE_TWIN_EVERY=N` (in `docker_env_vars`), the capture recomputes every Nth call of an
op that has a reference (the framework's own PyTorch path, an alternative kernel, or an fp64
recompute of an Inductor kernel) and captures the first call that is off by more than the dtype's
floor. The capsule's origin is `wrong_value`; replay counts a launch when the op fails its
twin again. `NAN_CAPTURE_TWIN_OPS=<op,...>` limits the ops; the check is off unless the variable is set.

Reproduce it on one GPU with the planted wrong-value op that ships in madengine's tests:

```bash
cd <madengine>   # any PyTorch image with the repo mounted, one GPU
python3 tests/unit/nan_doctor/gpu/gpu_twins_capture.py /tmp/twinwork
# 16 checks: hit, default off, filtered, in-place, NaN with a twin, late bug, sampled,
# clean census, Inductor kernels, replay 20/20 -> TWIN_CAPTURE=PASS 16 0
```

## 6. Example: the Qwen3-MoE gradient NaN (not reproduced on this node)

The original manual's example is torchtitan `qwen3_moe_debug` on 8 GPUs (FSDP 4 × TP 2 × EP 4, bf16,
seed 42), whose gradient NaN comes from a hipBLASLt stream-K GEMM leaving output tiles unwritten
(solution 13813; workaround `TENSILE_FIXED_WGMXCC=1` or `TENSILE_STREAMK_DYNAMIC_GRID=0`). It is a MAD model:

```bash
madengine build --tags pyt_torchtitan_qwen3_moe_nan
bash .claude/skills/mad-common/mad-run.sh start qwen -- madengine run --tags pyt_torchtitan_qwen3_moe_nan --live-output \
  --additional-context '{"tools": [{"name": "nan_doctor"}], "docker_env_vars": {"NAN_CAPTURE_DATA": "1", "NAN_CAPTURE_ARM_AFTER_STEPS": "1", "NAN_CAPTURE_OP_CLASSES": "gemm"}}'
```

or `/mad-debug-numeric pyt_torchtitan_qwen3_moe_nan`. With the tool on, `run.sh` starts training
through `nan_doctor_rt.torchtitan_capture`, which hooks `Trainer.train_step`.

**On this MI300X node the job never finishes a step**, with or without the tool: the image builds
and the seed checkpoint is created, then, after `Applied FSDP`, either a rank crashes (exit -11 in one
run) or an FSDP / expert-parallel collective stalls until the run's timeout (an earlier attempt logged
a NCCL timeout, and another 1800 s of a queued all-gather with no kernel launched). That is the training job, not the capture. No capsule was produced, so the
root cause (solution 13813) and the fix were not reproduced here. Run it on a node where
`qwen3_moe_debug` completes steps; the skill, tool and runtime path for it is the one in section 3.

## 7. Root cause and fix (steps 7 and 8, experimental)

On by default, within a time budget the skill states (roughly 1 h when the reproducer hits every run, 4 h when it
does not). Step 7 root-causes from the reproducer: a mutation sweep of the kernel
(`replay_nan_op --mutate`, `nan_doctor_rt.native_sweep`), the replay under different environments, and a
"why was this not prevented" pass. Step 8 looks for a fix or workaround and validates it on the
reproducer and on the model. A clean result on another node is a node-bound reproducer, not a fix.
The skill reports what it found, or says it found nothing; `/mad-nan-hunt` enters step 7 only with
`--rootcause` and never runs step 8. Neither step was run for `dummy_nan` (the NaN is the model's own
arithmetic) or for the Qwen3-MoE example (section 6); the commands behind them are in
`.claude/skills/mad-debug-numeric/reference/step7-rootcause.md` and `step8-fix.md`.

## 8. Reference

### `madengine nan` commands

| Command | Step | Use |
|---|---|---|
| `exec` | | Run a command in the model's image on the GPUs (`--image` or `--manifest`; `nan_doctor_rt` is on `PYTHONPATH`) |
| `group` | 5a | Group capsules; order the groups for replay |
| `replay`, `summary` | 5b | Replay a capsule under load (the concurrency ladder); summarise an attempt |
| `package`, `readme` | 6 | Build, describe and pack a reproducer |
| `report` | 9 | Draft `report.md` and `result.json` |
| `arms`, `diff` | 2 | Differential mode (section 4) |
| `intake`, `ledger`, `verdict`, `waves`, `handoff`, `power`, `kernel-match`, `digest` | | Readers, gates and bookkeeping |
| `framework-check`, `gpu-check`, `preflight`, `capture-points`, `wire` | 0-4 | Gates before capture; `wire` is the source-wiring fallback |

`madengine nan <command> --help` lists the options.

### Capture knobs (`docker_env_vars`)

`NAN_CAPTURE_DATA=1` (save operand values; without it the capsule holds shapes only and a replay needs
`--synthesize`), `NAN_CAPTURE_ARM_AFTER_STEPS=<n>`, `NAN_CAPTURE_OP_CLASSES=gemm`, `NAN_CAPTURE_TWIN_EVERY=<n>`,
`NAN_CAPTURE_TWIN_OPS`. The tool itself sets `NAN_CAPTURE_DIR`, `NAN_CAPTURE_AUTO=optimizer` and
`NAN_CAPTURE_CONTAINERS_PER_GPU=1`.

### Other capture modes

Selected by `PYTHONPATH` in `docker_env_vars`, as an absolute path inside the container:
`/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt/<dir>`.

| `<dir>` | Mode |
|---|---|
| `diff_tap` | Differential mode (section 4) |
| `vllm_capture`, `vllm_graphs` | vLLM tensor-parallel workers; vLLM CUDA graphs |
| `sglang_capture` | SGLang |
| `inductor_capsule` | `torch.compile` kernels (replay: `python3 -m nan_doctor_rt.replay_inductor_kernel`) |

### How this differs from the original NaN-Doctor

| | Original | MAD + madengine |
|---|---|---|
| Entry | A path to the job's `run.sh` | A MAD model tag; `madengine run` builds the image and starts the container |
| Launch | `nd` (Slurm or local, Apptainer or Docker, site profiles) | `mad-run.sh`: one GPU job at a time under a host lock |
| Capture | Source wiring or a launcher hook | The `nan_doctor` tool arms an optimizer-step hook, no source edit |
| Artifacts | `artifacts/<date>-<name>/` | `nan_hunts/<hunt>/` (gitignored) |
| Skill name | `debug-numeric` | `mad-debug-numeric` (agent `mad-numeric-doctor`); the command group, tool and variables keep the `nan` names |

## 9. Verification record

| Check | Result |
|---|---|
| `dummy_nan`: capture, group, replay, package, extracted reproducer (section 3) | Passed. Capsule of `aten.sqrt.default`, 100/100 non-finite on replay, extracted reproducer 100/100, also with an empty environment |
| GPT-2 differential mode (section 4) | Passed. Names the SDPA switch |
| Twin references (section 5) | Passed, 16 checks |
| GPT-2 on 8 GPUs, capture and replay, `--learning_rate 1 --max_steps 20` | Captured `aten.mm.default` in the backward on all 8 ranks, one group; replay 160/160 non-finite (`FP32_OVERFLOW?`: model numerics) |
| Qwen3-MoE (section 6) | Not reproduced: the job does not finish a step on this node |
| vLLM and SGLang capture, Inductor capsule, vLLM graphs | Ported, not run on this node |
| Root cause and fix (section 7) | Not run |
| Tests | madengine `tests/unit/nan_doctor`: 92 pass on the host, plus the 21 torch tests in the GPT-2 image; MAD `tools/tests`: 26 Python (1 skipped) and 65 Node tests pass |

Two fixes came out of this run and are in madengine: `package` now copies every module the replay
imports (`math_refs`, `twins`, `inductor_ref`, `failure_signature`, `capsule_operands`) into the reproducer,
and accepts the `c1det.<pid>.<host>` attempt directory that the deterministic probe writes. Before them
the extracted `run.sh` stopped with `LADDER_RESULT=ERROR` ("`--capture-procs` printed no `CAPTURE_PROCS` line").

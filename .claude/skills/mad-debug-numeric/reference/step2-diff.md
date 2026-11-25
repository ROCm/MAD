# Step 2, no signal: differential mode -- make the model's wrong outputs visible

Read when there is no `<NAN_SIGNAL>`: the model prints nothing when it is wrong, and the user suspects
silently wrong outputs ("is my model silently wrong", "the outputs look off", "validate this upgrade /
config change"), or asks to compare configurations. Not for a NaN, a spike or a parity line the model already
prints: those are `<NAN_SIGNAL>`s (`step1-ask.md`). Never added to a NaN investigation by default: it costs
at least three model runs.

**What it does.** The same model runs under the user's configuration (A), A again (A2: the noise
floor), the conservative **reference** (REF: every catalogue switch on its safe side -- eager, no graphs,
native ops, no aiter, no TF32, math attention) and A with **one switch flipped** per switch (`flip_<s>`: the
bisection), one `madengine run` per arm. `nan_doctor_rt/diff_tap` records what each run returned -- no model
edit; `madengine nan diff` compares. The switches: `madengine nan arms list [--framework F]` (vLLM, SGLang,
plain PyTorch / HF / training; environment-only).

## Run

Inputs pinned first (fixed seed, fixed data order, the same prompts): a run whose inputs differ is no
comparison. `<DIR>` = `<ARTIFACT_DIR>/diff/<wave>` (relative to `<MAD_ROOT>`, which is `/myworkspace` in
the container, so arm `<a>` writes `<MAD_ROOT>/<DIR>/<a>/outputs.jsonl`).

```bash
madengine nan arms detect --root <the model's scripts dir> <the model's command>   # FRAMEWORK=<vllm|sglang|torch>
madengine nan arms arms --framework <F> [--only a,b] [--drop a,b] [--extra fp32] [--set VAR=VALUE] --format context --out-dir <DIR>
```

The second prints one line per arm, `<name><TAB><additional-context JSON>` (A, A2, REF, `flip_<switch>`...;
`docker_env_vars` with the arm's switches, `ND_DIFF_OUT` and the diff_tap `PYTHONPATH`). Run them one after
another, A first (`waves.md`), each with the intake's context merged in (its `docker_env_vars` beside the
arm's), never with `tools`:

```bash
bash .claude/skills/mad-common/mad-run.sh start diff-<wave>-<arm> -- madengine run --tags <tag> --live-output --additional-context '<JSON>'
```

then `mad-run.sh wait` each. Compare (no GPU):

```bash
madengine nan diff --a <DIR>/A/outputs.jsonl --a2 <DIR>/A2/outputs.jsonl --ref <DIR>/REF/outputs.jsonl \
  --arm flip_<s>=<DIR>/flip_<s>/outputs.jsonl ... [--dtype D] --json <DIR>/diff.json
# ND_DIFF verdict=<AGREE|DIVERGE|NONDETERMINISTIC|INPUTS_DIFFER> ...   ND_DIFF_SWITCH=<s>
```

Knobs: `--only` / `--drop` / `--extra fp32` (switches), `--set ND_VLLM_ATTENTION_BACKEND=TRITON_MLA` (an MLA
model's reference attention), `ND_DIFF_TAP` / `ND_DIFF_FUNC` in an arm's `docker_env_vars` (below), a subset of
arms (run only those lines). Ledger the `ND_DIFF` and `ND_DIFF_SWITCH` lines.

**What the tap sees** (`ND_DIFF_OUT` per arm): vLLM `LLM.generate`, SGLang `Engine.generate`,
transformers `generate` -- per request the tokens, the chosen token's logprob and the top-1 / top-2 margin
(logprobs raised to >= 2 on a copy of the sampling params: the engine computes two more numbers, the math is
unchanged); training -- the loss per `backward()` and `clip_grad_norm_`'s result. A hand-written loop:
`ND_DIFF_TAP=fwd` (the outermost `nn.Module`'s output) or `ND_DIFF_FUNC=<function>` (the return value of a
function of the script, e.g. `generate`: tokens only, no logprobs). A server (`vllm serve` /
`sglang.launch_server`): each arm's run is `$NAN_DOCTOR_RT/bin/serve_run.sh --server "<command with {port}>"
--requests <file>` (`step1-ask.md` *vLLM*) -- the client records the outputs (and runs check mode); the
switches act in the server's processes.

## Read the verdict

| `ND_DIFF verdict=` | meaning | next |
|---|---|---|
| `AGREE` | no request / step beyond the threshold under any switch (`tie_flips` = near-tie tokens resolved differently: benign) | report it: no silent divergence among these switches on these inputs. More requests, `--extra fp32`, or the user's real prompts widen it |
| `DIVERGE`, `ND_DIFF_SWITCH=<s>` | A's outputs differ from the reference beyond max(dtype floor, 5 x A-vs-A2), and flipping `<s>` alone removes it | the signal: *Check mode* below, then step 4 with twins on the component `<s>` names |
| `DIVERGE`, `ND_DIFF_SWITCH=NONE_SINGLE` | no single switch removes it: two switches together, or the reference itself | a run with two switches set together; or *Locate* |
| `NONDETERMINISTIC` | A disagrees with A2: the configuration does not reproduce itself (inference: tokens differ between identical runs; training: one run's loss or grad norm is the odd one out at a step -- `ND_DIFF_SELF step=<t>`) | inference: *Check mode* against A2 (`ND_DIFF_REF=.../A2/outputs.jsonl`, `ND_DIFF_SAME_CONFIG=1`) -- the requests that come out differently are hits. Training: the signal is the magnitude the `ND_DIFF_SELF` line names (`step4-explode.md`, `--signal-above loss=...` / grad norm) |
| `INPUTS_DIFFER` | the runs did not get the same prompts / batches | pin the inputs (seed, data order) and rerun |

An arm with no or an empty `outputs.jsonl`: the tap saw nothing -- read that run's log; `ND_DIFF_TAP=fwd` or
`ND_DIFF_FUNC=<function>`.

The thresholds are calibrated on clean runs: a trained bf16 model's chosen-token logprob differs by <= 0.19
between correct implementations (a 0.5B model, vLLM and SGLang, every switch; floor 0.25), and a differing
token counts as a near-tie while both runs' top-1 / top-2 margins are <= 2 x the threshold; training is judged
three ways, because two runs of one configuration drift apart by themselves after ~10 steps (backward atomics,
amplified): a run counts as the odd one out only where its gaps to both others exceed 5 x the closest pair's,
and the reference only on a **jump** (> 5 x its gap over the previous 5 steps) that persists 3 steps -- two
correct implementations drift smoothly (~3x every few steps), a bug jumps.

## Check mode: the signal for steps 2-5

With a finished comparison, any later run of the model prints a line per request that disagrees with the
reference -- `<NAN_SIGNAL>` = `ND_DIFF_HIT`, and steps 2-5 run unchanged (`madengine nan waves --signal
ND_DIFF_HIT`, the clean runs and the capture run). Add to the run's `docker_env_vars`:

```
"PYTHONPATH": "/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt/diff_tap",
"ND_DIFF_REF": "/myworkspace/<DIR>/REF/outputs.jsonl", "ND_DIFF_THR": "/myworkspace/<DIR>/diff.json"
```

Each run ends with `ND_DIFF_CHECK records=<n> hits=<h> tie_flips=<t> inputs_differ=<d>`; `inputs_differ` > 0
= the run's inputs are not the reference's: the check is meaningless until they are pinned. Then step 4 with
`step4-explode.md`: twins on (`NAN_CAPTURE_TWIN_EVERY` beside the check-mode variables, with the `nan_doctor`
tool) and, when the switch names a component, an op filter for it (`NAN_CAPTURE_TWIN_OPS`,
`NAN_CAPTURE_OP_CLASSES`). Training: use the magnitude, not check mode.

## Locate: the first module that disagrees (no twin covers the op)

Two more runs, A and REF, each with `ND_DIFF_MODULES=<request or step index from diff.json a_vs_ref.diverge>`
and an eager configuration in `docker_env_vars` (`ND_VLLM_COMPILE=0`, `ND_VLLM_CUDAGRAPH_MODE=NONE`; keep A's
culprit switch on its buggy side), then:

```bash
madengine nan diff --modules <DIR2>/A/modules.jsonl --modules-ref <DIR2>/REF/modules.jsonl
# ND_DIFF_LOCATE verdict=FOUND module=<Class#n> | NONE_ABOVE | NO_OVERLAP
```

Every `nn.Module` call's output is summarised in both runs (l2, absmax, three signed sums) and compared in
completion order: children before their parents, so the innermost module that disagrees is named
(`ND_DIFF_MODULE` lines list the next ones). Module forwards must run in Python -- eager configurations
and an in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); SGLang runs the model in a scheduler
subprocess: not covered. The module's source is then read with `madengine nan capture-points` (step 4).

## Limits

* **Random weights hide silent bugs.** A random-init model's output distribution is nearly flat: a planted
  head swap moved the logprobs by one bf16 step and no threshold can see it. Use the user's checkpoint; a
  model with random weights is a valid test only of a bug that wrecks values.
* `ND_DIFF_FUNC` records tokens only: a differing token against the reference cannot be told from a near-tie
  (`unmargined=<n>`, never a divergence); against A2 it is nondeterminism.
* Only the catalogue's switches are bisected; a fix that is a model argument is an arm the user runs: a
  second comparison with that context as A.
* Cost (one 8-GPU node, image cached): a 0.5B model, 40 requests x 32 tokens -- 9
  vLLM arms 3-5 min side by side; madengine runs the arms one after another, so expect a few minutes per
  arm plus each run's start-up.

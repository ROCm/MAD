# Step 2, no signal: differential mode -- make the model's wrong outputs visible

Read when there is no `<NAN_SIGNAL>`: the model prints nothing when it is wrong, and the user suspects
silently wrong outputs ("is my model silently wrong", "the outputs look off", "validate this upgrade /
config change"), or asks to compare configurations. Not for a NaN, a spike or a parity line the model already
prints: those are `<NAN_SIGNAL>`s (`step1-ask.md`). Never added to a NaN investigation by default: it costs
at least three model runs.

**What it does.** The same model command runs under the user's configuration (A), A again (A2: the noise
floor), the conservative **reference** (REF: every catalogue switch on its safe side -- eager, no graphs,
native ops, no aiter, no TF32, math attention) and A with **one switch flipped** per switch (`flip_<s>`: the
bisection), side by side on one node. `scripts/diff_tap` records what each run returned -- no model edit;
`scripts/output_diff.py` compares. The switches: `python3 scripts/configs.py list` (vLLM, SGLang, plain
PyTorch / HF / training; environment-only).

## Run

```bash
CMD='<the model, once, as nd run takes it -- inputs pinned: fixed seed, fixed data order>' \
  ND_ENV=<ENV> scripts/nd submit --name diff --gpus 8 --exclusive --step step2 -- diff_wave.sh
nd wait diff --max 240     # last line: DIFF_WAVE_RESULT=<verdict> switch=<s|-> dir=<ART>/diff/<job>
```

Knobs: `FRAMEWORK` (auto from the script's imports), `ONLY` / `DROP` / `EXTRA=fp32` (switches),
`SETS='ND_VLLM_ATTENTION_BACKEND=TRITON_MLA'` (an MLA model's reference attention), `GPUS_PER_RUN` (TP),
`TAP` / `FUNC` (below), `ARMS` (a subset). Ledger the result line and `diff.txt`'s `ND_DIFF` line.

**What the tap sees** (`ND_DIFF_OUT` per arm, set by the job): vLLM `LLM.generate`, SGLang `Engine.generate`,
transformers `generate` -- per request the tokens, the chosen token's logprob and the top-1 / top-2 margin
(logprobs raised to >= 2 on a copy of the sampling params: the engine computes two more numbers, the math is
unchanged); training -- the loss per `backward()` and `clip_grad_norm_`'s result. A hand-written loop:
`TAP=fwd` (the outermost `nn.Module`'s output) or `FUNC=<function>` (the return value of a function of the
script, e.g. `FUNC=generate`: tokens only, no logprobs). A server (`vllm serve` / `sglang.launch_server`):
`CMD='bash $ND_REPO/scripts/serve_run.sh --server "<command with {port}>" --requests <file>'` -- the client
records the outputs (and runs check mode); the switches act in the server's processes.

## Read the verdict

| `DIFF_WAVE_RESULT` | meaning | next |
|---|---|---|
| `AGREE` | no request / step beyond the threshold under any switch (`tie_flips` = near-tie tokens resolved differently: benign) | report it: no silent divergence among these switches on these inputs. More requests, `EXTRA=fp32`, or the user's real prompts widen it |
| `DIVERGE switch=<s>` | A's outputs differ from the reference beyond max(dtype floor, 5 x A-vs-A2), and flipping `<s>` alone removes it | the signal: *Check mode* below, then step 4 with twins on the component `<s>` names |
| `DIVERGE switch=NONE_SINGLE` | no single switch removes it: two switches together, or the reference itself | `ARMS` with two switches set together; or *Locate* |
| `NONDETERMINISTIC` | A disagrees with A2: the configuration does not reproduce itself (inference: tokens differ between identical runs; training: one run's loss or grad norm is the odd one out at a step -- `ND_DIFF_SELF step=<t>`) | inference: *Check mode* against A2 (`ND_DIFF_REF=.../A2/outputs.jsonl ND_DIFF_SAME_CONFIG=1`) -- the requests that come out differently are hits. Training: the signal is the magnitude the `ND_DIFF_SELF` line names (`step4-explode.md`, `--signal-above loss=...` / grad norm) |
| `INPUTS_DIFFER` | the runs did not get the same prompts / batches | pin the inputs (seed, data order) and rerun |
| `NO_OUTPUTS` | the tap saw nothing | read the arm's log: `TAP=fwd` or `FUNC=<function>` |

The thresholds are calibrated on clean runs (2026-11-24): a trained bf16 model's chosen-token logprob differs
by <= 0.19 between correct implementations (Qwen2.5-0.5B, vLLM and SGLang, every switch; floor 0.25), and a
differing token counts as a near-tie while both runs' top-1 / top-2 margins are <= 2 x the threshold; training is judged three ways, because two runs of
one configuration drift apart by themselves after ~10 steps (backward atomics, amplified): a run counts as
the odd one out only where its gaps to both others exceed 5 x the closest pair's, and the reference only on a
**jump** (> 5 x its gap over the previous 5 steps) that persists 3 steps -- two correct implementations drift
smoothly (~3x every few steps), a bug jumps (x70 at the torchcompile example's spike).

## Check mode: the signal for steps 2-5

With a finished wave, any later run of the model prints a line per request that disagrees with the reference
-- `<NAN_SIGNAL>` = `ND_DIFF_HIT`, and steps 2-5 run unchanged (`wave_summary.py --signal ND_DIFF_HIT`, the
capture job's control arm). Add to the model's environment (spec lines, `CAPTURE_ENV`):

```
PYTHONPATH=$ND_REPO/scripts/diff_tap:$ND_REPO/scripts ND_DIFF_REF=$ND_ART/diff/<job>/REF/outputs.jsonl
ND_DIFF_THR=$ND_ART/diff/<job>/diff.json
```

Each run ends with `ND_DIFF_CHECK records=<n> hits=<h> tie_flips=<t> inputs_differ=<d>`; `inputs_differ` > 0
= the run's inputs are not the reference's: the check is meaningless until they are pinned. Then step 4 with
`step4-explode.md`: twins on (`NAN_CAPTURE_TWIN_EVERY`) and, when the switch names a component, an op filter
for it (`NAN_CAPTURE_TWIN_OPS`, `NAN_CAPTURE_OP_CLASSES`). Training: use the magnitude, not check mode.

## Locate: the first module that disagrees (no twin covers the op)

```bash
MODULES=<request or step index from diff.json a_vs_ref.diverge> ARMS='A REF' \
LOCATE_ENV='ND_VLLM_COMPILE=0 ND_VLLM_CUDAGRAPH_MODE=NONE' CMD='...' nd submit --name locate ... -- diff_wave.sh
# DIFF_LOCATE_RESULT=FOUND module=<Class#n> | NONE_ABOVE | NO_OVERLAP
```

Every `nn.Module` call's output is summarised in both runs (l2, absmax, three signed sums) and compared in
completion order: children before their parents, so the innermost module that disagrees is named
(`ND_DIFF_MODULE` lines list the next ones). Module forwards must run in Python -- eager configurations
(`LOCATE_ENV`; keep A's culprit switch on its buggy side) and an in-process engine
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`); SGLang runs the model in a scheduler subprocess: not covered. The
module's source is then read with `find_capture_points.py` (step 4).

## Limits

* **Random weights hide silent bugs.** A random-init model's output distribution is nearly flat: a planted
  head swap moved the examples' logprobs by one bf16 step and no threshold can see it (2026-11-24). Use the
  user's checkpoint; a model with random weights is a valid test only of a bug that wrecks values.
* `FUNC` records tokens only: a differing token against the reference cannot be told from a near-tie
  (`unmargined=<n>`, never a divergence); against A2 it is nondeterminism.
* Only the catalogue's switches are bisected; a fix that is a script argument (`--mode ws48`) is an arm the
  user runs: `ARMS`, or two diff waves.
* Cost (MI355X, one 8-GPU node, image cached): a 0.5B model, 40 requests x 32 tokens -- 9 vLLM arms 3-5 min,
  4 SGLang arms ~2 min; a 30-step training x 7 arms ~2.5 min; a check-mode run or a locate run ~1.5 min.

## Validated (2026-11-24, tests/diff)

A head swap planted in layer 3 of the default attention backend of a real Qwen2.5-0.5B: vLLM and SGLang
`DIVERGE` 14/40, `switch=attention`; check mode hits exactly those 14 requests; locate names `Attention#3`.
The blind MLA example (aiter#7631, no hand-written check): `NONDETERMINISTIC` 63/1536 -> check mode as the
signal -> the capture-to-reproducer job with twins on: `REPRODUCED`, `KERNEL_BUG`,
`mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co`. The torchcompile training example: `NONDETERMINISTIC` at
step 20 (a loss / grad-norm spike in one run), 26-58 steps before its NaN. Clean runs (vLLM, SGLang, HF,
eager training): `AGREE`.

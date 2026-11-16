# Step 1: ask how to run the model (required), declare contention

Never skipped; runs nothing but the `madengine nan exec` check (`<GPU_EXEC>` below = `madengine nan exec --image <IMAGE> --`).

| ask for | notes |
|---|---|
| `<MODEL_TAG>` | the madengine model/tag that runs the model (`madengine run --tags <tag>`), plus its `build_manifest.json` |
| `<NAN_SIGNAL>` | how its output says a NaN happened (a grep for a log line the model prints). A **magnitude** counts too -- `grad_norm: +([1-9][0-9]{2,}\|nan\|inf)` for exploding gradients: `step4-explode.md` |
| `<WORLD_SIZE>` | **derive, don't ask**: ranks of ONE job (`torchrun --nproc_per_node` x nodes, `NGPU`, the parallelism degrees). > 1 -> `step4-distributed.md` (wiring through the framework's step, manifest, origin rank). Independent trials sharing GPUs are co-tenants, world size 1 each |
| `<IMAGE>` | **don't ask**: the `docker_image` of the model in `build_manifest.json` |
| `<GPU_EXEC>` | `madengine nan exec --image <IMAGE> [--gpus ...] [--mount ...] [--env K=V] --`; runs any `python3 -m nan_doctor_rt.<module>` in the model's image. Steps 4-7 need it |

**Verify `<GPU_EXEC>` once:** `<GPU_EXEC> python3 -m nan_doctor gpu-check --torch`
prints torch's view of device 0 (name, arch, CUs,
memory) and a last line `GPU=... ARCH=... VERDICT=...` (`TORCH_GPU=NO` = the image or
GPU access is broken: fix it first). **Ledger the device line** (gfx942 = MI300X,
gfx950 = MI350X/MI355X), CUs and memory: rates, memory caps and reproducers are bound
to it (SKILL.md *GPUs*). `VERDICT=UNTESTED` here, not yet warned in step 0: print the
SKILL.md *GPU gate* warning now, before any wave, and continue.

**Injected vs stock: don't ask.** It matters only for reading step 7 (on an
injected image the root cause is the planted NaN, confirming the tooling, not a
bug), and the 5b `kernel.*.txt` or a `kernel_mutations --list-mutations` enumeration
tells: `nans_to_nops` matches a site = injected, `PatchError` = stock. Neither: report
it as unknown.

**Ask everything in step 1, one question at a time, each with a default** -- a
wall of questions overwhelms; a question later in the ladder stalls for as long as
the user is away. Each question states its default (from the model's run
configuration where possible) so "default" or "ok" is a complete answer. **After
question 1, offer the shortcut once**: one compact block listing questions 2-6 with
their computed defaults (one line each), "reply `default` to accept all, or name the
ones to change" -- then ask only the ones the user names, one at a time. Before each
question, drop what earlier answers already settled; never ask what you can read from
the model's run configuration -- state the derived value and let the user correct it.
In order:

1. **`<MODEL_TAG>`** -- no default. **If the user names a directory, not a tag:**
   pick the plain model entry -- the one that runs the model as-is, not a capture,
   timing or reproducer variant -- state the pick in one line and go on; ask only if
   two plain ones remain. **Then the PyTorch gate** (SKILL.md step 0,
   `madengine nan framework-check`) before anything else: `SKIP` = print its warning
   and stop -- no artifact dir, no ledger; `ASK` = its one question first. `RUN`: go
   on, and ledger its last line once the ledger exists. **Now create `<ARTIFACT_DIR>`**
   (SKILL.md rule 3; `<RUN_TAG>` = today's date + the model name) -- **`ls` it
   first**: the tag is only date + name, so an earlier session's investigation of the
   same model today already owns it. Its ledger has jobs or capsules: suffix the tag
   (`-2`). Only step-1 lines: tell the user and resume it (ledger a `resumed` note).
   Then start the ledger: `$LG start step1` (`$LG` = `madengine nan ledger <LEDGER>`),
   `$LG note step1` with `<MAD_ROOT>` and `<ARTIFACT_DIR>`. **Look for earlier
   investigations of the same model, any date**, in the same call: `grep -l '<MODEL_TAG>'
   <ARTIFACT_DIR>/../*/ledger.md`. A hit with a passing capture run or a tarball is a
   validated setup and maybe a finished reproducer: ledger it, and before step 4 offer
   the user a fresh capture (it confirms the old result still holds) or reuse of the
   old reproducer. Never carry its rates over (rule 5). Then read the model's run
   script and derive `<NAN_SIGNAL>`, the image, the trial geometry and **the data
   source** (public or test assets, or the user's own data: it sets the capsule data
   mode, question 3); report them in one line, don't ask. Ask for `<NAN_SIGNAL>` only
   if the script doesn't show it (preflight `SIGNAL ?  the run script greps no NaN
   line` = its greps only format output; take one of the `the source raises/logs:`
   candidates the preflight prints). Then, in ONE call, `madengine nan preflight
   <run script> --source <model.py>`: `IMAGE ... MISSING` = the image is not built or
   pulled -- get it now (a long pull is the user's call), before any run; `SIGNAL ...
   NOT RAISED ANYWHERE` means the script cannot see the NaN it greps for (raise it now,
   with the user -- not after a wave). The default fix, when the model hides NaNs
   (`nan_to_num` on the loss): one line after `backward()` in a working copy,
   `after_backward(model, loss, parts)` from `nan_doctor_rt.grad_trajectory`. It raises
   `NonFiniteDetected("non-finite GRADIENT at step N after backward: <modules>")` (add
   `NonFiniteDetected` to `<NAN_SIGNAL>`), and with `NAN_DOCTOR_GRADLOG` set it also
   logs the per-step gradient trajectory (step 8). Back the source up first, and
   restore it in step 9; `MASK` on a loss, `HOOK`, `ENV`, `LAZY` and `LOOP` settle
   step 4's wiring questions without further greps (`LOOP` is a static guess: the
   step-4 smoke test's locator confirms it).
   **Read the trigger notes too, in the same call**: `grep -niE
   'pressure|HBM|co-?tenan|share|race|stress|trials per GPU|control'` over the model
   directory's `README*`/`*.md` and the model source's docstring. A load trigger
   ("needs HBM pressure", "1 trial per GPU stays clean") means replay copies alone
   will likely not reproduce: ledger `TRIGGER: <quote, file:line>` and give the
   **first** replay its co-tenant (`step4-run.md` *With a TRIGGER*) instead of waiting
   for a solo 5b to come back `NOT_SET_UP`.
2. **Step 2** -- word it as: "Run the model once to make sure that it runs in this
   environment. You CAN skip this step and go directly to the NaN capturing step
   (step 4)." -- default skip. Running it means a wave of >= 8 runs, which also
   measures the rate (`step2-rate.md`). Always optional; skipped means *unmeasured*
   (and step 3 skipped), never a carried-over rate (`step2-rate.md`). Don't quote
   earlier runs' rates in the question.
3. **Capture-wave geometry** (number of runs, which are controls) -- default: the
   model's own geometry, controls per `step4-run.md`. A wave is N sequential
   `mad-run.sh` runs of `madengine run`; it stops at the first settled capsule -- say
   so in the question. **Same question, the capsule data mode**: real values
   (`NAN_CAPTURE_DATA=1`) by default when the data is public/test assets, shapes only
   otherwise (`step4-capture.md` 4.1 item 1) -- state which and why, so the user can
   override it now rather than after a clean synthetic replay.
4. **Root-causing (step 7) and fixing (step 8) wanted?** (no / ask me after 6 /
   yes) -- default: ask after 6. Both are loops over the step-6 reproducer with a
   wall-clock budget the user gives **when step 7 starts** (don't ask for it now) --
   both budgets in ONE question there when both steps are wanted (`step7-rootcause.md`
   *Entry*); fixing needs a root cause. Say that step 6 ends with a reproducer, not a
   fix, and that steps 7-8 are experimental in this version.
5. **`<GROUPS_TO_REPLAY>`: if the capture yields several capsule groups, replay only
   the largest, or all?** (largest / all) -- default: largest, i.e. the replayable
   group with the most capsules (`G1` in `groups_priority.txt`). `all` adds one 5b+6
   pass per further group. **The first time "largest" reaches the user** (this
   question, or its line in the shortcut block), define it: the group holding the most
   capsules -- size is the capsule count, not tensor size or memory.
6. **Time budget for the whole run** -- default computed from the answers above: only
   the runs they imply (waves only if step 2 runs; the capture run sized from the
   geometry and the model's runtime; with `all`, ~2 h per extra group; plus a reserve
   for the step-5N fallback -- default ~6 candidate runs of ~30 min and one locating
   wave -- spent only if no capsule reproduces; say so in the question). Within it,
   start waves, the capture run and per-group 5b replays without re-confirming; ask
   again only beyond it, or for the step-7/8 budgets (and each recipe sweep's
   mutators).

**Verify `<GPU_EXEC>` without a separate run when the next run exercises it anyway.**
Run the standalone `gpu-check` only if the first GPU job is something else -- **or if
step 0's GPU gate said `NO_GPU`**: then the GPU type is still unknown, so run it as its
own short command right after the step-1 questions, before any wave, so an
untested-GPU warning comes before work is spent.

**Ledger:** each answer (and whether it was the default), the derived inputs and image,
the `<GPU_EXEC>` check, containers per GPU and how derived.

## Contention: the one rule (every later replay)

A capsule stores *values*; a race is carried by *timing* (how many processes issue
at the device at once), which no `torch.save` keeps. A contention-gated capsule
replayed alone never fails -- a clean `0/N` that reads like evidence. **Every replay
rate carries the contention it was measured under**; a solo `0/N` on a
contention-gated fault is an experiment not set up, not a negative.

## Declare-contention (step 1; again whenever the geometry changes, e.g. step 3)

N = concurrent containers on the busiest **device**, injected into the capture run's
environment (never the model's own files) as `NAN_CAPTURE_CONTAINERS_PER_GPU` -- the
one fact the capture cannot see. `madengine run` runs one job per invocation, so
sequential runs give N = 1; ledger N and how derived. Anything else (several
containers on one GPU, several containers per run, uneven): `step1-contention.md`.

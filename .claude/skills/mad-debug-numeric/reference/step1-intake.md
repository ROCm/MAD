# Step 1 from an intake (no questions)

Read instead of `step1-ask.md` when the invocation names an **intake** file (`intake.json`; the `mad-nan-hunt` workflow writes one). The intake answers every step-1 question; the user is not asked any. `step1-ask.md` stays the reference for what each value means.

1. **Validate**, in one call: `madengine nan intake <intake.json>`. Exit 2 (`INTAKE_ERROR` lines) = stop and report them verbatim -- never guess a missing value. Exit 0 prints `KEY=VALUE` lines: `RUN_SCRIPT`, `NAN_SIGNAL`, `WORLD_SIZE`, `NNODES`, `ENV`, `ARTIFACT_DIR`, `IMAGE`, and for `ENV=madengine` `MODEL_TAG`, `DEPLOY_TARGET`, `ADDITIONAL_CONTEXT`; then `PREF_*`.
2. **Gates** (SKILL.md step 0) still run. `madengine nan framework-check` on the model source when the producer could not tell (it reads `madengine discover`, not the source): `SKIP` = stop with its warning, as interactively.
3. **`<ARTIFACT_DIR>`** = the intake's, created by the producer. `ls` it first (a resumed investigation keeps its ledger). Start the ledger: `madengine nan ledger <LEDGER> start step1`, then `madengine nan ledger <LEDGER> note step1 "intake <path> source=<source>"`, then one note with the `KEY=VALUE` block's values (<= 200 chars per line).
4. **Environment file:** `env-madengine.md` (`ENV=madengine`; `docker` is the same flow with `--image`).
5. **Preferences** replace questions 2-6 (ledger each with `(intake)`, not `(default)`; question 7, `<PRIORS>`, takes its default):

   | `PREF_*` | answers |
   |---|---|
   | `MEASURE_RATE` | Q2: `true` = step 2 runs; `false` = skipped, so step 3 is too |
   | `CAPTURE_DATA` | Q3 data mode: `values` = `NAN_CAPTURE_DATA=1` |
   | `ROOTCAUSE`, `BUDGET_MINUTES` | Q4: step 7 runs only if `true`, within the budget; `BUDGET_MINUTES=0` with `ROOTCAUSE=true` = stop after step 6 and ask |
   | `FIX` | Q4: step 8. An automated producer (the workflow) never sets it |
   | `GROUPS` | Q5 `<GROUPS_TO_REPLAY>` |

   Q3's geometry: `env-madengine.md` (one job per run). Q6's allocation budget: the producer's -- an automated run never asks for more; when the work it implies does not fit, stop at the last finished rung and report.
6. **Never ask the user anything an intake run did not leave open.** The only questions left are the ones the ladder raises with no intake answer (a step-7 budget with `BUDGET_MINUTES=0`, an image pull). In a non-interactive run (no user to answer), stop there and report what was needed.

Contention (`step1-ask.md` *Declare-contention*) still applies; `env-madengine.md` fixes it at 1.

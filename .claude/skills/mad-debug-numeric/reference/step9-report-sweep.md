# Step 9, if step 7's recipe track (b) ran: reporting a sweep

* Every mutation's outcome (`NAN`/`CLEAN`/`TIMEOUT`/`ERROR`); `identity` and
  `nans_to_nops` were not run (`--mutate-no-identity`; `<REPORT_JSON>`
  `identity_skipped: true`) -- say the sweep had no positive control and that each
  CLEAN row rests on its asm read-back.
* The tiers: which mutators ran in tier 1 and tier 2, and every stronger mutator
  reported as *predetermined CLEAN (not run)* because its weaker sibling was a
  powered `CLEAN`; any `PatchError` drop and why; any pair the user deferred for
  budget; whether the sweep was cut short by `<RC_BUDGET>` or by track (a) finding
  the root cause.
* Per pair, the reading (`step7-recipe-sweep.md` *Reading a pair*): which member is
  the minimal change that clears the NaN.
* **A CLEAN row whose `.diff` is empty** ("disassembles IDENTICALLY") = the patch
  never landed: not a finding. **Every row CLEAN** = suspect the setup first -- a
  contention-gated fault swept without `--contend`, a wrong `env:` banner,
  `--mutate-force` (`step7-recipe-sweep.md`).
* From `<REPORT_JSON>`: `payload_kind`, `trial_iters`, `contended`/`contend_n` --
  which record only what `--contend` started; a self-started co-tenant reads
  `false`/`0`: describe the real geometry from `<LEDGER>`.
* Chain sweep: `chain_culprit` (op, `kind`, `any_diverged`), `chain_order`,
  `mutate_stop_on_finding`, `mutate_chain_ops`; `kind=NON_FINITE` +
  `any_diverged=false` = unsound attribution; list ops left `SKIPPED`.
* Every CLEAN row with `expected_hits`; `UNDERPOWERED` is not a finding.
* `SKIPPED` rows, so the table doesn't read as full coverage.
* aiter / CK: op-to-kernel attribution was a heuristic.
* All `TIMEOUT` = failed sweep; name the remedy not applied.
* **A CLEAN row is the finding**: the mutation and instruction addresses from
  `<ASM_DIR>`'s `<MUTATOR>@<OP>.<KERNEL>.diff` (identical before/after = patch never
  landed). Match rows to kernels -- one op can dispatch several (`.co`).

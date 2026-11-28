# Step 4, conditional: shorten the time to a capsule

The capture catches the NaN within the first watched steps; **setup
dominates**. Measure before cutting:

1. Time the phases from the model's log (container start, imports, build, optimizer
   init, data prep, first steps, arming, capsule) in a full wave and with one trial
   alone; much slower in the wave = contended.
2. Right-size host threads: ~cores / concurrent processes, from the launcher's trial
   count.
3. Shrink one-time synthetic work (e.g. a cycled pre-generated batch pool) via an env
   knob (default = original), on capture and control arms alike.
4. Never below the control floor (`step4-capture.md` 4.3); keep the model's own check
   reachable.
5. **Timing changes can change the race** (threads raised the 1st NaN's rate):
   compare same-wave controls with `<FULL_MODEL_RATE>`; if moved, ledger the new rate
   and setting (SKILL.md rule 5).

Bigger levers, each A/B'd against the full model: a smaller capture model (fewer/
smaller embedding tables if the failing ops don't use them); several capsules per
trial; smaller allocations (one GPU, minimum co-tenants) that schedule sooner.

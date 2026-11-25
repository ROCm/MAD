# Contention in a single-process replay (outside 5b)

Only for a one-off arm or a real-model co-tenant (5b's copies are their own
contention; step 7's recipe sweep has its own section). Every rate carries its
contention. Start each long command through `mad-run.sh start`.

```bash
madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.replay_nan_op <CAPSULE_FILE> --iters 200000 --contend-auto
madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.replay_nan_op <CAPSULE_FILE> --iters 200000 \
    --contend-with "<FULL_MODEL_RUN_SCRIPT>"     # a second, independent arm
```

* `--contend N` vs a real model: no reliable ranking; both beat solo. Run both on
  separate GPUs if possible; compare magnitudes only within one host.
* `--iters` does not substitute for contention (a serial loop never has the trigger).
* **Size for duration**: ~20k single-op iters/s (`1000000` ~50 s; `2000` never
  overlaps). Chains are slower: time one run on this capsule, don't extrapolate from a
  short pilot (~half under).
* The built-in gate waits for device memory to drop (~256 MB) and prints
  `[verdict] CONTENTION_INVALID` if not -- it proves a process, not work.
  **Your own co-tenant: gate on its step line and prove overlap at both ends**
  (`until grep -q 'Step 100' co.log; do sleep 2; done`; after, its steps advanced);
  ledger both timestamps; no gate = invalid arm. Compare its step rate with step 3's
  loaded rate.
* Low contended rate, check in order: power (`p x N` >= ~3); co-tenant kind; overlap
  (log, not memory); only then process count.
* Two contended arms on one host are not independent (the second launched loses CPU):
  fine for presence/absence; for a comparison balance launch order or give each the
  node.

# Step 6: build, verify and pack a reproducer (`madengine nan package`)

**Package before 5b's verdict is final, verify from an extracted copy.** One command
builds `repro-<G>/` from the reproducing attempt; the verification is a reproducing run
of the extracted tarball -- no separate replay job.

```bash
bash .claude/skills/mad-common/mad-run.sh start package-<G> -- \
  madengine nan package --group <G> --artifact-dir <ARTIFACT_DIR> --image <IMAGE> \
    --capsule <CAPSULE> --attempt <reproducing attempt dir> \
    --groups <ARTIFACT_DIR>/groups.json --pack --ledger <LEDGER> --step 'step6[<G>]'
```

Then verify (below) and finish with `step6-finish.md`. A group that does not reproduce:
delete its `repro-<G>/` and `verify-<G>/` dirs. Move an earlier `repro-<G>` aside first:
the command refuses to overwrite.

## What

* One tarball per group with `n/N > 0`, holding the capsule 5b replayed. Groups on
  the same kernel fold into one (say which, on what basis).
* **Ship exactly what 5b ran**: whole chain, the reproducing `--concurrency` and
  `--iters`, through the shipped runner. No chain (`--chain-fallback`):
  the README says the operands were never validated. Tensile `KERNEL_BUG`: add the
  `--bench` one-liner (`step9-report.md`) as an extra.

## Layout (under `<ARTIFACT_DIR>`)

```
repro-<G>/
  run.sh  README.md  capsule/
  host/nan_doctor/            # host-side runner
  runtime/nan_doctor_rt/      # container-side code, mounted into the image
```

* It needs docker, python3 and the image -- not madengine. Never replace `run.sh` with
  a hand-written container command (run flags `--ipc=host`, `--shm-size`, memlock,
  seccomp shape timing).
* The capsule goes inside; no paths into `<CAPSULE_DIR>`.
* `run.sh` names `IMAGE` explicitly (the README quotes `captured build:` beside it),
  runs the shipped capsule at the reproducing concurrency and iteration count, keeps its
  output inside the extracted directory, and stays overridable for iterations,
  concurrency and GPU count; the result is `LADDER_RESULT=` + `summary.txt`.
* A loader path the replay holds back (`env:` banner `held back:`, e.g. `import torch`
  dying on a missing shared library): `run.sh` must export it -- add it by hand and
  re-pack (`step6-finish.md`).

## Verify from an extracted copy

Extract the tarball into a new directory (**not under `/tmp`**: a sibling of the
checkout; note the symptom in the README if it matters) and run *that*, not the build
directory, so a missing file cannot resolve through the checkout:

```bash
mkdir -p <ARTIFACT_DIR>/verify-<G> && tar -xzf <tarball> -C <ARTIFACT_DIR>/verify-<G>
bash .claude/skills/mad-common/mad-run.sh start verify-<G> -- bash <ARTIFACT_DIR>/verify-<G>/repro-<G>/run.sh
```

**Pass = a reproducing run (`n/N > 0` near the ledger rate)**, not exit 0; a `0/N` is a
failed build -- check `env:` banners and copies per GPU first.

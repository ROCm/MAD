# Step 5b, conditional: does the replay match the capture?

A wrong environment does not error -- it returns a clean `0/N`.

```
env:            3 difference(s) from capture (1 changed, 1 unset here, 1 set here only)  -- pass --restore-env
  HSA_DISABLE_CACHE                capture='0'  here='1'
```

* `matches capture` -- go on.
* `N difference(s)` -- read it before any rate (`madengine nan replay` always passes
  `--restore-env`; a hand-run `replay_nan_op` needs it explicitly).
  *changed* = overridden here; *unset here* = capture had it; *set here only* = your
  container added it (e.g. the image's `HSA_DISABLE_CACHE=1`) -- `--restore-env`
  unsets too.
* `STILL DIFFERENT AFTER --restore-env` -- something re-sets it after the re-exec;
  check the `--env K=V` flags passed to `madengine nan replay` / `madengine nan exec`.
* `held back:` (permanent) -- device selection (`HIP_VISIBLE_DEVICES`, would move off
  the pinned device) and loader paths (`LD_LIBRARY_PATH`, `LD_PRELOAD`, act before
  they could be applied). **A differing loader path is a different image** -- settle
  via `captured build:` first.
* `not carried:` -- set at capture but outside the capsule's namespace
  (`capsule['env_info']['not_captured']`); look there if the fault proves
  env-conditional.

`0/100` plain vs `100/100` with `--restore-env` is a finding: the named variable goes
in the bug report. Settle the environment before raising `--iters`.

**Build:** quote `captured build:` (ROCm, OS, the `libhipblaslt`/`librocblas`
mapped), not the image tag. Run flags (`--ipc=host`, `--shm-size`, memlock, seccomp)
come from `madengine nan exec`, not the capsule.

# Step 4, conditional: budget the capture's own memory

The capture can OOM at step 0 and blame the model:

* **Pre-call snapshot** clones each in-place op's written args; a huge out-param asks
  for a clone that size (`pre-call snapshot of <op> failed (OutOfMemoryError
  ...)` -- caught, op degrades to post-call; but a *successful* one can starve the
  co-tenant). **Fix with `snapshot_max_bytes`, never `snapshot_mutated=False`**
  (that loses pre-call fidelity for every in-place op, buys ~no speed).
* **Cap: below what a co-tenanted device can spare, above the writes you care
  about.** 1st NaN: 4 GiB -- under ~6.65 GiB worst free at 2 trials/GPU (both
  may snapshot at once), over Shampoo factor matrices (`max_preconditioner_dim`^2
  fp32, 64 MiB at 4096). The excluded op writes 4.81 and 67.52 GiB, so **do
  not round the cap up** (5 GiB copies 4.81 GiB every step). Those numbers are
  MI355X (288 GiB); on MI300X (192 GB) the same model leaves ~96 GB less per GPU,
  so measure worst free memory on this device in the smoke test and size the cap
  from that -- and expect fewer trials (and replay copies) per GPU to fit.
* **Read the skip lines**: one per distinct (op, size) with per-operand breakdown
  (older builds: once per op name, hid call B). Anything beyond the known
  giant buffers skipped = cap too low; raise only with headroom; ledger value and
  reason.
* **The post-call check can allocate too**: an `isfinite(t).all()` mask is
  full-size; the current max-abs reduction is not, and tensors above
  `NAN_CAPTURE_SCAN_CHUNK_ELEMS` (2^27) are reduced in slices (`_scan_chunks` in
  `op_nan_capture.py`). A `13 GiB`-class failure (shows as `Error at step N: CUDA
  out of memory`, not `[op-nan]`): check the build has `_scan_chunks` before
  adding a filter.

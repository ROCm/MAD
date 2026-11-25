# Step 4, conditional: budget the capture's own memory

The capture can OOM at step 0 and blame the model:

* **Pre-call snapshot** clones each in-place op's written args; a huge out-param asks
  for a clone that size (`pre-call snapshot of <op> failed (OutOfMemoryError ...)` --
  caught, op degrades to post-call; but a *successful* one can starve the co-tenant).
  **Fix with `snapshot_max_bytes`, never `snapshot_mutated=False`** (that loses
  pre-call fidelity for every in-place op, buys ~no speed).
* **Cap: below what a co-tenanted device can spare, above the writes you care
  about.** Set it under the worst free memory with the declared co-tenants (all may
  snapshot at once) and over the legitimate large writes (e.g. optimizer factor
  matrices, `max_preconditioner_dim`^2 fp32). **Do not round the cap up**: a cap just
  above a multi-GiB buffer copies it every step. Measure worst free memory on this
  device in the smoke test (MI355X has 288 GiB, MI300X 192 GB) and size the cap from
  that; expect fewer runs (and replay copies) per GPU to fit on the smaller device.
* **Read the skip lines**: one per distinct (op, size) with per-operand breakdown
  (older builds: once per op name, hiding a second call). Anything beyond the known
  giant buffers skipped = cap too low; raise only with headroom; ledger value and
  reason.
* **The post-call check can allocate too**: an `isfinite(t).all()` mask is full-size;
  the current max-abs reduction is not, and tensors above
  `NAN_CAPTURE_SCAN_CHUNK_ELEMS` (2^27) are reduced in slices (`_scan_chunks` in
  `op_nan_capture.py`). A multi-GiB failure (shows as `Error at step N: CUDA out of
  memory`, not `[op-nan]`): check the build has `_scan_chunks` before adding a filter.

# Wave hygiene -- before the first model wave (steps 2-4)

A wave is N sequential `mad-run.sh` runs of `madengine run --tags <tag> ...`
(`env-madengine.md`); controls are runs without the `nan_doctor` tool. There are no
trials, no controls inside a run, no early-stop wave.

* **Check who else is on the node, before and during**: another run of this skill
  there will clear load for its sweep (killing your runs), and your runs time its
  sweep out. `mad-run.sh` holds the host GPU lock; do not bypass it.
* **External kill signature:** a trapped signal cuts `wait` short, cleanup runs, and
  every run scores `ERROR` with exit 0; logs stop mid-training, no traceback/OOM, all
  within seconds (`madengine nan waves` flags it). Not the model.
* **Start every run detached** (`mad-run.sh start <run-id> -- madengine run ...`, then
  `mad-run.sh wait <run-id>` until `MAD_RUN_STATUS=succeeded|failed`);
  `PYTHONUNBUFFERED=1` (a frozen log = dead run; stderr is line-buffered already);
  `$HOST` in log paths.
* **Budget a stressed wave from stressed step lines**, never from an unstressed
  smoke test.
* Read the wave with `madengine nan waves` (SKILL.md rule 9).

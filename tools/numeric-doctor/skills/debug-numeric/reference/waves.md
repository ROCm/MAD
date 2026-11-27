# Wave hygiene -- before the first model wave (steps 2-4)

* **Check who else is on the node, before and during** (`env.md`): another run of
  this skill there will clear load for its sweep (killing your trials), and your
  trials time its sweep out.
* **External kill signature:** a trapped signal cuts `wait` short, cleanup runs, and
  every trial scores `ERROR` with exit 0 (Slurm `COMPLETED 0:0`); logs stop
  mid-training, no traceback/OOM, all within seconds (`wave_summary.py` flags it).
  Not the model.
* **Launch long waves to survive you**: detach (`setsid`, or a batch job);
  `PYTHONUNBUFFERED=1` (a frozen log = dead trial; stderr is line-buffered already);
  `$HOST` in log paths.
* **Budget a stressed wave from stressed step lines**, never from an unstressed
  smoke test.
* Read the wave with `wave_summary.py` (SKILL.md rule 9).

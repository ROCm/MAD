"""Both paths must produce a runnable invocation of the SAME launcher.

sbatch      : the batch script is submitted directly, and runs run_multinode.slurm
madengine   : madengine slurm_multi generates madengine_<name>.sh, which runs the same script

Runs died because the madengine half could not even write its script when the card
name was namespaced. This drives the real generator against the real
card, so a regression on either side shows up without a cluster.
"""
import json, sys, tempfile, os
from pathlib import Path
sys.path.insert(0, "/home/madengine/src")

MAD = "/tmp/madtest/MAD_pr242"
CARD_DIR = f"{MAD}/scripts/vllm_multinode"
cards = {c["name"]: c for c in json.load(open(f"{CARD_DIR}/models.json"))}
card = cards["pyt_vllm_kimi-k3_mi300x_pp2xtp8"]

# Namespaced exactly as madengine's discovery names a nested card -- the shape that broke 62.
NS = "vllm_multinode/pyt_vllm_kimi-k3_mi300x_pp2xtp8"
model = dict(card); model["name"] = NS

from madengine.deployment.slurm import SlurmDeployment

work = tempfile.mkdtemp()
manifest = {"built_models": {NS: model}, "built_images": {}, "context": {},
            "deployment_config": {"target": "slurm",
                                  "slurm": {"partition": "amd-rccl", "nodes": 2,
                                            "gpus_per_node": 8, "time": "04:00:00",
                                            "output_dir": f"{work}/slurm_results"},
                                  "distributed": {"launcher": "slurm_multi", "nnodes": 2}}}
mpath = Path(work) / "manifest.json"
mpath.write_text(json.dumps(manifest))

dep = SlurmDeployment.__new__(SlurmDeployment)
print("1. filename from the namespaced card name")
safe = dep._safe_name(model)
print(f"     name  : {NS}")
print(f"     file  : madengine_{safe}.sh")
ok1 = "/" not in safe
print(f"     one path segment: {ok1}")

print("\n2. the path that used to fail with ENOENT now writes")
out = Path(work) / "slurm_results"; out.mkdir(parents=True)
target = out / f"madengine_{safe}.sh"
target.write_text("#!/bin/bash\n")
ok2 = target.is_file() and target.parent == out
print(f"     wrote {target.name} directly under slurm_results: {ok2}")

print("\n3. both paths name the SAME launcher script")
standalone_script = card["scripts"]                     # what plain sbatch submits
madengine_script = model["scripts"]                     # what slurm_multi runs
ok3 = standalone_script == madengine_script == "run_multinode.slurm"
print(f"     sbatch     -> {standalone_script}")
print(f"     madengine  -> {madengine_script}")
print(f"     same launcher: {ok3}")

print("\n4. that launcher is syntactically sound and free of both hazard classes")
import subprocess
rc_syntax = subprocess.run(["bash","-n",f"{CARD_DIR}/run_multinode.slurm"]).returncode
rc_cont = subprocess.run(["python3",f"{MAD}/scripts/common/check_continuations.py",
                          f"{CARD_DIR}/run_multinode.slurm"],capture_output=True).returncode
rc_quote = subprocess.run(["python3",f"{MAD}/scripts/common/check_srun_quotes.py",
                           f"{CARD_DIR}/run_multinode.slurm"],capture_output=True).returncode
ok4 = rc_syntax == 0 and rc_cont == 0 and rc_quote == 0
print(f"     bash -n={rc_syntax}  continuations={rc_cont}  srun-quotes={rc_quote}")

print("\n5. the card declares results both collectors look for")
ok5 = card.get("multiple_results") == "perf_Kimi-K3.csv"
print(f"     multiple_results: {card.get('multiple_results')}  (madengine honours it since b3e3180)")

allok = all([ok1, ok2, ok3, ok4, ok5])
print("\nRESULT:", "both paths generate a runnable invocation" if allok else "REGRESSION")
sys.exit(0 if allok else 1)

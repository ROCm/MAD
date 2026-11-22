"""Both paths must produce a runnable invocation of the SAME launcher.

sbatch      : the batch script is submitted directly, and runs run_multinode.slurm
madengine   : madengine slurm_multi generates madengine_<name>.sh, which runs the same script

Runs died because the madengine half could not even write its script when the card
name was namespaced. This runs madengine's own generator, SlurmDeployment.prepare(),
on the real card and checks the script it writes, so a regression on either side
shows up without a cluster.

Needs madengine importable: installed, or a checkout's src/ in MADENGINE_SRC.
Usage: python3 scripts/common/verify_both_paths_generate.py   (from any directory)
"""
import json, sys, tempfile, os
from pathlib import Path
if os.environ.get("MADENGINE_SRC"):
    sys.path.insert(0, os.environ["MADENGINE_SRC"])

MAD = str(Path(__file__).resolve().parents[2])   # this repo's root
CARD_DIR = f"{MAD}/scripts/vllm_multinode"
cards = {c["name"]: c for c in json.load(open(f"{CARD_DIR}/models.json"))}
card = cards["pyt_vllm_kimi-k3_mi300x_pp2xtp8"]

# Named and pathed exactly as madengine's discovery gives a nested card
# (utils/discover_models.py): the name is namespaced -- the shape that broke runs --
# and the script path is relative to the MAD root.
NS = "vllm_multinode/pyt_vllm_kimi-k3_mi300x_pp2xtp8"
model = dict(card); model["name"] = NS
model["scripts"] = os.path.normpath(os.path.join("scripts", "vllm_multinode", card["scripts"]))

from madengine.deployment.base import DeploymentConfig
from madengine.deployment.slurm import SlurmDeployment

# madengine resolves the card's script against the manifest's directory, which is the
# MAD root when madengine runs. A temporary root whose scripts/ is this checkout's
# stands in for it, so nothing is written into the repository.
work = tempfile.mkdtemp()
os.symlink(f"{MAD}/scripts", f"{work}/scripts")
out = Path(work) / "slurm_results"
deployment_config = {"target": "slurm",
                     "slurm": {"partition": "<partition>", "nodes": 2,
                               "gpus_per_node": 8, "time": "04:00:00",
                               "output_dir": str(out)},
                     "distributed": {"launcher": "slurm_multi", "nnodes": 2}}
manifest = {"built_models": {NS: model}, "built_images": {NS: {"docker_image": "<image>"}},
            "context": {}, "deployment_config": deployment_config}
mpath = Path(work) / "build_manifest.json"
mpath.write_text(json.dumps(manifest))

print("1. madengine generates the job script for the namespaced card")
dep = SlurmDeployment(DeploymentConfig(target="slurm", manifest_file=str(mpath),
                                       additional_context=deployment_config))
ok_prep = dep.prepare() is True
target = Path(dep.script_path) if dep.script_path else None
print(f"     prepare(): {ok_prep}")
print(f"     name  : {NS}")
print(f"     file  : {target.name if target else None}")
ok1 = ok_prep and target is not None and target.is_file() and target.parent == out
print(f"     written directly under slurm_results: {ok1}")

print("\n2. the generated script runs the card's launcher")
generated = target.read_text() if ok1 else ""
ok2 = "run_multinode.slurm" in generated
print(f"     references run_multinode.slurm: {ok2}")

print("\n3. both paths name the SAME launcher script")
standalone_script = os.path.normpath(f"{CARD_DIR}/{card['scripts']}")   # what plain sbatch submits
madengine_script = os.path.normpath(f"{MAD}/{model['scripts']}")        # what slurm_multi runs
ok3 = standalone_script == madengine_script == os.path.normpath(f"{CARD_DIR}/run_multinode.slurm")
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

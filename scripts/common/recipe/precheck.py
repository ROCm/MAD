#!/usr/bin/env python3
"""The launchers' recipe gate, run before submitting: so a recipe that cannot run on the target
GPU is refused before the job waits in a queue. The launcher still checks on the allocated nodes
(cluster_check_recipe); this only refuses earlier, when the target GPU is known at submit time.

    precheck.py --script scripts/<dir>/<launcher> --arch gfx950      (the run's env: MODEL_NAME,
                                                                       MAD_PIN_SET, MAD_PLATFORM, ...)
Exit 0: resolves, or nothing to check; 1: refused (the reason on stderr).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recipe as R  # noqa: E402

# Each launcher directory: its engine, recipe file (relative to scripts/), the roles it runs and
# where the engine flags of roles without recipe flags come from.
LAUNCHERS = {
    "vllm_dissag": ("vllm", "vllm_dissag/models.yaml", None, None),
    "sglang_disagg": ("sglang", "sglang_disagg/models.yaml", None, None),
    "vllm_multinode": ("vllm", "vllm_dissag/models.yaml", "server", "COLOCATED_EXTRA_ARGS"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True)
    ap.add_argument("--arch", required=True)
    a = ap.parse_args(argv)
    env = os.environ
    launcher = os.path.basename(os.path.dirname(os.path.abspath(a.script)))
    if launcher not in LAUNCHERS or not env.get("MODEL_NAME") or env.get("RECIPE_CHECK") == "0":
        return 0
    engine, recipes, roles, flags_var = LAUNCHERS[launcher]
    scripts = os.path.dirname(os.path.dirname(HERE))
    argv = ["check", "--recipes", os.path.join(scripts, recipes), "--model", env["MODEL_NAME"],
            "--arch", a.arch, "--engine", engine]
    if env.get("MAD_PIN_SET"):
        argv += ["--pin-set", env["MAD_PIN_SET"]]
    if roles:
        argv += ["--roles", roles]
    if flags_var:
        argv += ["--flags", env.get(flags_var, "")]
    rc = R.main(argv)
    return 0 if rc in (0, 2) else 1


if __name__ == "__main__":
    sys.exit(main())

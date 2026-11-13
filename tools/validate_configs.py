#!/usr/bin/env python3
"""Check generated configs against the runner they are routed to, before a node.

Everything left in a config block after the runner pops its own keys is splatted
into the CSV row. A runner whose csv.DictWriter is strict raises on a key its
fieldnames do not contain -- after the server has loaded the checkpoint, tens of
minutes into a run. A runner opened with extrasaction="ignore" instead drops the
key silently. Which one applies is read from the runner source, not assumed: the
#290 rewrite switched all three runners to "ignore".

Checks per block:
  * no key outside the runner's CSV_HEADER plus the keys it pops -- an error
    for a strict writer, a count of silently-dropped keys for a lenient one;
  * the keys the runner reads unconditionally are present;
  * env / extra_args / agentic_env have the type the runner expects;
  * the manifest's config path matches the file actually emitted.

Reads CSV_HEADER straight out of each run_<engine>.py, so it cannot drift from
the runner it is checking.
"""

import ast
import glob
import json
import os
import sys

import yaml

if len(sys.argv) not in (2, 3):
    sys.exit(f"usage: {sys.argv[0]} [MAD_CHECKOUT] OUT_DIR\n"
             "  MAD_CHECKOUT defaults to the checkout this file is in;\n"
             "  OUT_DIR is the --out-dir infx2mad.py wrote manifests to")
MAD = (sys.argv[1] if len(sys.argv) == 3
       else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = sys.argv[-1]

# Keys each runner pops before building the CSV row (verified in the sources).
POPPED = {
    "sglang": {"env", "extra_args", "random_range_ratio", "agentic_env", "startup_timeout"},
    "vllm": {"env", "extra_args", "random_range_ratio", "agentic_env", "startup_timeout",
             "bench_args"},
    "atom": {"env", "extra_args", "random_range_ratio", "agentic_env", "bench_serving", "startup_timeout"},
}
# Keys the runner adds itself, so a block need not carry them.
ADDED = {"cmd", "num_prompts", "performance", "metric", "unit"}
REQUIRED = {"benchmark", "model", "tp", "max_concurrency"}
# Fields the SERVING command indexes unconditionally, beyond REQUIRED -- not
# every engine needs every one (ATOM's CSV has no `variant`, see RUNNERS in
# infx2mad.py), so these are checked per engine rather than folded into
# REQUIRED, which would false-positive on ATOM.
SERVING_REQUIRED = {
    "sglang": {"out", "variant", "dtype"},
    "vllm": {"out", "dtype"},
    "atom": {"out"},
}
# Fields each engine's run_agentic_point indexes unconditionally, beyond
# REQUIRED. An agentic block skips the serving checks, so without this a
# block missing dtype validates and then raises KeyError after staging.
AGENTIC_REQUIRED = {
    "sglang": {"dtype"},
    "vllm": {"dtype"},
    "atom": set(),
}


def lenient_writer(engine):
    """True when the runner's DictWriter ignores keys outside its fieldnames."""
    src = open(os.path.join(MAD, "scripts", engine, f"run_{engine}.py")).read()
    return 'extrasaction="ignore"' in src or "extrasaction='ignore'" in src


def csv_header(engine):
    src = open(os.path.join(MAD, "scripts", engine, f"run_{engine}.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CSV_HEADER":
            return {ast.literal_eval(e) for e in node.value.elts}
    raise SystemExit(f"CSV_HEADER not found in run_{engine}.py")


def main():
    headers = {e: csv_header(e) for e in POPPED}
    lenient = {e: lenient_writer(e) for e in POPPED}
    # run_sglang.py adds a `variant` column itself whenever a block carries one.
    headers["sglang"] = headers["sglang"] | {"variant"}
    dropped = {}
    problems, blocks, files = [], 0, 0

    # --all writes one directory per scenario, --scenario writes flat into
    # out_dir. Globbing only the nested layout made a single-scenario run report
    # zero files and OK, which is a false green.
    manifests = sorted(set(glob.glob(os.path.join(ROOT, "*", "*.manifest.json")) +
                           glob.glob(os.path.join(ROOT, "*.manifest.json"))))
    if not manifests:
        print(f"no manifests under {ROOT}")
        return 1
    for man_path in manifests:
        man = json.load(open(man_path))
        key = next(iter(man["built_models"]))
        model = man["built_models"][key]
        engine = model["scripts"].split("/")[1]
        if engine == "sglang_disagg":
            # Env-driven launcher: no config file to check, so check that the
            # topology the manifest states is internally consistent instead.
            env = man["context"]["docker_env_vars"]
            dist = man["deployment_config"]["distributed"]["sglang_disagg"]
            nodes = man["deployment_config"]["slurm"]["nodes"]
            if int(env["xP"]) + int(env["yD"]) != nodes:
                problems.append(f"{man_path}: xP+yD != slurm nodes")
            if (int(env["xP"]), int(env["yD"])) != (dist["prefill_nodes"], dist["decode_nodes"]):
                problems.append(f"{man_path}: env xP/yD disagree with distributed block")
            if env.get("RUN_MORI") == "1" and env.get("KV_TRANSFER_BACKEND") != "mori":
                problems.append(f"{man_path}: RUN_MORI=1 but KV_TRANSFER_BACKEND is not mori")
            if not env.get("NCCL_IB_HCA"):
                problems.append(f"{man_path}: multi-node manifest carries no fabric env")
            files += 1
            continue
        if engine not in headers:
            problems.append(f"{man_path}: unknown runner {engine!r}")
            continue

        cfg_name = model["args"].split("--config ")[1].split()[0].split("/")[-1]
        # The translator writes the config into the runner's configs/ when it
        # knows the MAD checkout, and beside the manifest otherwise; the runner
        # resolves --config relative to its own directory, so check both.
        runtime = os.path.join(MAD, "scripts", engine, "configs", cfg_name)
        beside = os.path.join(os.path.dirname(man_path), cfg_name)
        cfg_path = runtime if os.path.exists(runtime) else (
            beside if os.path.exists(beside) else None)
        if cfg_path is None:
            problems.append(
                f"{man_path}: manifest points at {cfg_name}, which is in neither "
                f"scripts/{engine}/configs/ nor the output dir")
            continue
        if cfg_path != runtime:
            # madengine stages only scripts/<engine>/ and the runner resolves
            # --config there; a sidecar copy is checked but would not launch.
            problems.append(
                f"{man_path}: {cfg_name} is only beside the manifest; copy it to "
                f"scripts/{engine}/configs/ (or translate with --mad-dir) before "
                "running")
        files += 1

        allowed = headers[engine] | POPPED[engine] | ADDED
        cfg_blocks = yaml.safe_load(open(cfg_path)) or []
        if not isinstance(cfg_blocks, list) or not cfg_blocks:
            # An empty config launches nothing; reporting it OK would let a
            # manifest with no runnable point through the pre-allocation check.
            problems.append(f"{man_path}: {cfg_name} has no config blocks")
            continue
        for i, b in enumerate(cfg_blocks):
            blocks += 1
            where = f"{os.path.basename(cfg_path)}[{i}] ({engine})"
            extra = set(b) - allowed
            if extra and lenient[engine]:
                for k in extra:
                    dropped.setdefault(engine, set()).add(k)
            elif extra:
                problems.append(f"{where}: keys the runner's CSV cannot hold: {sorted(extra)}")
            missing = REQUIRED - set(b)
            if missing:
                problems.append(f"{where}: missing {sorted(missing)}")
            for k in ("env", "extra_args", "agentic_env"):
                if k in b and not isinstance(b[k], dict):
                    problems.append(f"{where}: {k} must be a mapping, got {type(b[k]).__name__}")
            if b.get("benchmark") == "agentic":
                if "agentic_env" not in b:
                    problems.append(f"{where}: agentic block carries no agentic_env")
                missing_agentic = AGENTIC_REQUIRED.get(engine, set()) - set(b)
                if missing_agentic:
                    problems.append(
                        f"{where}: agentic block missing {sorted(missing_agentic)}, "
                        f"which {engine}'s agentic server command indexes unconditionally")
            if b.get("benchmark") == "serving":
                if not b.get("inp"):
                    problems.append(f"{where}: serving block has no input length")
                missing_serving = SERVING_REQUIRED.get(engine, set()) - set(b)
                if missing_serving:
                    problems.append(
                        f"{where}: serving block missing {sorted(missing_serving)}, "
                        f"which {engine}'s serving command indexes unconditionally")

    print(f"checked {files} config file(s), {blocks} block(s)")
    for engine, keys in sorted(dropped.items()):
        print(f"  note: run_{engine}.py drops these keys from the CSV row (lenient "
              f"writer): {sorted(keys)}")
    for p in problems:
        print("  FAIL", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

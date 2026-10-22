#!/usr/bin/env python3
"""Check generated configs against the runner they are routed to, before a node.

Everything left in a config block after the runner pops its own keys is splatted
into the CSV row, and csv.DictWriter raises on a key its fieldnames do not
contain. That failure happens after the server has loaded the checkpoint —
tens of minutes into a run — so it is worth catching from a laptop.

Checks per block:
  * no key outside the runner's CSV_HEADER plus the keys it pops;
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

MAD = sys.argv[1] if len(sys.argv) > 1 else "/home/ilkosare/pr-work/mad-rccl"
ROOT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/all"

# Keys each runner pops before building the CSV row (verified in the sources).
POPPED = {
    "sglang": {"env", "extra_args", "random_range_ratio", "agentic_env", "startup_timeout"},
    "vllm": {"env", "extra_args", "random_range_ratio", "agentic_env", "startup_timeout"},
    "atom": {"env", "extra_args", "random_range_ratio", "agentic_env", "bench_serving", "startup_timeout"},
}
# Keys the runner adds itself, so a block need not carry them.
ADDED = {"cmd", "num_prompts", "performance", "metric", "unit"}
REQUIRED = {"benchmark", "model", "tp", "max_concurrency"}


def csv_header(engine):
    src = open(os.path.join(MAD, "scripts", engine, f"run_{engine}.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CSV_HEADER":
            return {ast.literal_eval(e) for e in node.value.elts}
    raise SystemExit(f"CSV_HEADER not found in run_{engine}.py")


def main():
    headers = {e: csv_header(e) for e in POPPED}
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
        candidates = [os.path.join(MAD, "scripts", engine, "configs", cfg_name),
                      os.path.join(os.path.dirname(man_path), cfg_name)]
        cfg_path = next((c for c in candidates if os.path.exists(c)), None)
        if cfg_path is None:
            problems.append(
                f"{man_path}: manifest points at {cfg_name}, which is in neither "
                f"scripts/{engine}/configs/ nor the output dir")
            continue
        files += 1

        allowed = headers[engine] | POPPED[engine] | ADDED
        for i, b in enumerate(yaml.safe_load(open(cfg_path)) or []):
            blocks += 1
            where = f"{os.path.basename(cfg_path)}[{i}] ({engine})"
            extra = set(b) - allowed
            if extra:
                problems.append(f"{where}: keys the runner's CSV cannot hold: {sorted(extra)}")
            missing = REQUIRED - set(b)
            if missing:
                problems.append(f"{where}: missing {sorted(missing)}")
            for k in ("env", "extra_args", "agentic_env"):
                if k in b and not isinstance(b[k], dict):
                    problems.append(f"{where}: {k} must be a mapping, got {type(b[k]).__name__}")
            if b.get("benchmark") == "agentic" and "agentic_env" not in b:
                problems.append(f"{where}: agentic block carries no agentic_env")
            if b.get("benchmark") == "serving" and not b.get("inp"):
                problems.append(f"{where}: serving block has no input length")

    print(f"checked {files} config file(s), {blocks} block(s)")
    for p in problems:
        print("  FAIL", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

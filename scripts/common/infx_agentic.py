#################################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
"""Agentic trace-replay benchmark for the AGGREGATED serving runners.

MAD already owns the replay: scripts/common/benchmark_agentic.sh plus
agentic_lib.sh download and verify the SemiAnalysis trace corpus, drive aiperf
with the `inferencex-agentx-mvp` scenario, and aggregate through InferenceX's
own utils.agentic.aggregation. Until now that hook was reachable only from the
disaggregated launchers, which stand up their own router and export the env it
reads.

An aggregated run (scripts/sglang, scripts/vllm, scripts/atom) has no router, so
this module starts the engine's own server and points the same hook at it. The
replay, the corpus gate and the aggregation stay MAD's; this only bridges the
two. Shared by all three runners so they cannot drift apart in how an agentic
point is measured.
"""

import json
import os
import signal
import subprocess

import psutil

# Where each metric lives in the aggregate JSON, and the unit the CSV should
# carry. Paths rather than a flat search: the document nests per-metric
# distributions, so a bare key like "p50" occurs under ttft, e2el, itl and tpot
# alike and a flat lookup would silently report whichever came first.
# Latencies are seconds in the JSON and milliseconds in MAD's serving CSVs, so
# they are scaled to keep the two comparable.
AGENTIC_METRICS = (
    ("throughput_tot", ("request_metrics", "throughput", "total", "tokens_per_second"), "tok/sec", 1),
    ("throughput_gen", ("request_metrics", "throughput", "output", "tokens_per_second"), "tok/sec", 1),
    ("throughput_inp", ("request_metrics", "throughput", "input", "tokens_per_second"), "tok/sec", 1),
    ("median_ttft", ("request_metrics", "latency", "ttft", "p50"), "ms", 1000),
    ("median_tpot", ("request_metrics", "latency", "tpot", "p50"), "ms", 1000),
    ("median_itl", ("request_metrics", "latency", "itl", "p50"), "ms", 1000),
    ("median_e2el", ("request_metrics", "latency", "e2el", "p50"), "ms", 1000),
    # Accounting, not performance: a replay cut short by DURATION profiles only
    # part of its sessions, and a throughput number from a truncated run must
    # not look like a complete one.
    ("requests_profiled", ("request_accounting", "records_profiled"), "requests", 1),
    ("requests_total", ("request_accounting", "records_total"), "requests", 1),
    ("measured_duration", ("request_metrics", "throughput", "duration_seconds"), "s", 1),
)

RESULT_FILENAME = "agentic_result"


def _dig(doc, path):
    """Value at `path`, or None if any step is missing."""
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def run_agentic(model, config, backend, server_cmd, host, port, start_timeout):
    """Start `server_cmd`, replay the agentic corpus against it, return CSV rows.

    `config` is mutated the way the serving paths mutate it: runner-only keys are
    popped so that what remains can be splatted into the CSV row, whose columns
    are fixed by each runner's CSV_HEADER.
    """
    agentic_env = config.pop("agentic_env", {}) or {}
    if not config.get("num_prompts"):
        # Not a fixed-prompt-count workload, but the CSV column still wants one.
        config["num_prompts"] = "0"
    config["cmd"] = server_cmd

    print(server_cmd, flush=True)
    server = subprocess.Popen(server_cmd, shell=True)
    results = []
    try:
        status = subprocess.run(
            f"timeout {start_timeout} bash -c "
            f"'until curl -sf http://{host}:{port}/health; do sleep 30; done' || exit 1",
            shell=True,
        )
        if status.returncode != 0:
            raise Exception("Server failed to start")
        print(f"Server at {server.pid} contacted successfully", flush=True)

        # The hook writes fixed filenames (agentic_result.json,
        # benchmark_command.txt), so two replays sharing a directory lose the
        # first one's evidence. Give each its own, and do not reuse one that
        # already exists -- a repeated run of the same point is still a
        # separate measurement.
        tag = config.get("variant") or f"{config['model']}_tp{config['tp']}"
        base_dir = os.path.abspath(
            f"agentic_{tag}_c{config['max_concurrency']}".replace("/", "_")
        )
        result_dir, attempt = base_dir, 1
        while os.path.exists(result_dir):
            attempt += 1
            result_dir = f"{base_dir}_run{attempt}"
        os.makedirs(result_dir)

        hook_env = dict(os.environ)
        hook_env.update({str(k): str(v) for k, v in agentic_env.items()})
        hook_env.update({
            "MODEL_PATH": model,
            # MODEL is deliberately NOT set: the runners reduce config["model"]
            # to a basename, but vLLM serves under the full repository id or
            # checkpoint path and registers no alias, so a basename would miss
            # /v1/models and the replay would address a model the server does
            # not know. The hook resolves the advertised id itself.
            # agentic_lib.sh defaults AGENTIC_PORT to the backend's ROUTER port;
            # an aggregated server listens where the runner just launched it.
            "AGENTIC_PORT": str(port),
            "RESULT_DIR": result_dir,
            "AGENTIC_OUTPUT_DIR": result_dir,
            "AGENTIC_RESULT_FILENAME": RESULT_FILENAME,
            # benchmark_agentic.sh resolves the served context window from the
            # first endpoint here. Its disagg callers set it to the prefill
            # WORKERS because the router does not advertise max_model_len; an
            # aggregated server is its own worker, so it is the endpoint.
            # It must also be SET: the script runs under `set -u` and expands
            # ${AGENTIC_SERVER_METRICS%% *} unguarded, so leaving it unset is an
            # unbound-variable abort rather than the non-disagg fall-through its
            # own comment describes.
            "AGENTIC_SERVER_METRICS": f"{host}:{port}",
        })

        hook = os.path.join("..", "scripts", "common", "benchmark_agentic.sh")
        bench_cmd = f"bash {hook} --backend {backend}"
        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(bench_cmd, flush=True)
        subprocess.run(bench_cmd, shell=True, check=True, env=hook_env)

        agg = os.path.join(result_dir, f"{RESULT_FILENAME}.json")
        with open(agg) as f:
            doc = json.load(f)
        for name, path, unit, scale in AGENTIC_METRICS:
            value = _dig(doc, path)
            if value is None:
                continue
            results.append({
                "performance": str(value * scale),
                "metric": name,
                "unit": unit,
                **config,
            })
        if not results:
            raise Exception(
                f"No known agentic metrics in {agg}; top-level keys: {sorted(doc)}"
            )
    finally:
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        _ = server.communicate()
        del server

    return results

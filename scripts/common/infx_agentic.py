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
import shlex
import signal
import subprocess
import time

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


def _stop_server_group(server, grace=(60, 30, 10)):
    """Stop the server's whole process group: SIGINT, then SIGTERM, then SIGKILL.

    Waiting on the shell alone is not enough: an engine worker that ignores
    SIGINT or stalls in shutdown either hangs cleanup until the job timeout or
    outlives the shell and keeps its GPU memory into the next point. Each step
    waits a bounded time for every process in the group to go; a server that
    already exited is not an error, so the original failure is what surfaces.
    """
    # start_new_session made the shell the leader of a new group whose id is
    # its pid. Use that id directly: once the shell has exited and been
    # reaped, os.getpgid(server.pid) fails while engine workers it started may
    # still be in the group, holding GPU memory.
    pgid = server.pid

    def group_alive():
        # A killed member stays a zombie until its parent reaps it -- the
        # shell is our own child and is only reaped by server.wait() below --
        # so zombies are gone for this purpose, not alive.
        server.poll()
        return any(p.info["status"] != psutil.STATUS_ZOMBIE
                   for p in psutil.process_iter(["pid", "status"])
                   if p.info["pid"] != os.getpid() and _pgid_of(p.info["pid"]) == pgid)

    for sig, wait_s in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), grace):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + wait_s
        while time.time() < deadline and group_alive():
            time.sleep(1)
        if not group_alive():
            break
    else:
        print(f"[agentic] server process group {pgid} survived SIGKILL", flush=True)
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _pgid_of(pid):
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def _port_in_use(host, port):
    """True if something accepts TCP connections on host:port right now."""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def _tokenizer_for_replay(hook_env, model, server_cmd=""):
    """The tokenizer the replay should load, or None to leave aiperf's default.

    An explicit AGENTIC_TOKENIZER -- from the point's agentic_env or the
    caller's environment -- is the caller's decision (a server can be run with
    a tokenizer different from the checkpoint's) and is kept. The mounted
    checkpoint is only the fallback, used when it carries tokenizer files.
    An empty value counts as unset, as it does in agentic_lib.sh.
    """
    explicit = hook_env.get("AGENTIC_TOKENIZER")
    if explicit:
        return explicit
    # Next, the tokenizer the server itself was told to use (SGLang
    # --tokenizer-path, vLLM --tokenizer): replaying with the checkpoint's
    # tokenizer while the server tokenizes with another one would build and
    # count prompts differently.
    try:
        argv = shlex.split(server_cmd or "")
    except ValueError:
        argv = []
    for flag in ("--tokenizer-path", "--tokenizer"):
        for i, a in enumerate(argv):
            val = (argv[i + 1] if a == flag and i + 1 < len(argv)
                   else a.split("=", 1)[1] if a.startswith(flag + "=") else None)
            # A tekken tokenizer.tok.json (added by run_sglang.py for models
            # without tokenizer.json) is not something aiperf can load.
            if val and not val.endswith(".tok.json"):
                return val
    return model if _has_local_tokenizer(model) else None


def _window_for_replay(hook_env, model, server_cmd=""):
    """The context window to gate the replay corpus on, or None to leave it to the hook.

    Only for servers that do not advertise it. An explicit MAX_MODEL_LEN or
    AGENTIC_MAX_CONTEXT_LENGTH is the caller's decision. Otherwise the window
    the server was told to use (--max-model-len), else the checkpoint's own
    max_position_embeddings (top level, or under text_config for a multimodal
    config), which is what such a server serves when it is given no limit.
    """
    if hook_env.get("MAX_MODEL_LEN") or hook_env.get("AGENTIC_MAX_CONTEXT_LENGTH"):
        return None
    try:
        argv = shlex.split(server_cmd or "")
    except ValueError:
        argv = []
    for i, a in enumerate(argv):
        val = (argv[i + 1] if a == "--max-model-len" and i + 1 < len(argv)
               else a.split("=", 1)[1] if a.startswith("--max-model-len=") else None)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    try:
        with open(os.path.join(model, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    for section in (cfg, cfg.get("text_config")):
        val = section.get("max_position_embeddings") if isinstance(section, dict) else None
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None


def _has_local_tokenizer(model):
    """True when `model` is a directory carrying its own tokenizer files."""
    if not os.path.isdir(model):
        return False
    return any(os.path.isfile(os.path.join(model, f))
               for f in ("tokenizer.json", "tokenizer_config.json",
                         "tokenizer.model"))


def run_agentic(model, config, backend, server_cmd, host, port, start_timeout,
                window_from_checkpoint=False, server_metrics=True):
    """One agentic point; a failure costs that point, not the rest of the ladder.

    The serving paths record what they have and move on; an exception here would
    instead abort every later point of the sweep. The failed point leaves an
    unmeasured row (blank metric and performance, the shape a serving point
    that never got a measurement leaves), which the run's result check rejects.
    An empty list would not do: a point that differs from another only in its
    replay settings has the same CSV identity, and the other's rows would then
    hide the missing one.
    """
    try:
        return _run_agentic(model, config, backend, server_cmd, host, port, start_timeout,
                            window_from_checkpoint, server_metrics)
    except Exception:
        import traceback
        print("[infx] agentic point failed; continuing with the next one:", flush=True)
        traceback.print_exc()
        row = {k: v for k, v in config.items() if k not in ("agentic_env", "startup_timeout")}
        row.setdefault("num_prompts", "0")
        row["cmd"] = server_cmd
        return [{"performance": "", "metric": "", "unit": "", **row}]


def _check_measurement(results, doc, agg, hook_env):
    """Raise unless the aggregate is a measurement a CSV row may carry."""
    # Accounting rows alone are not a measurement: an aggregate with no
    # profiled request or no total throughput must fail the point rather
    # than land in the CSV as a successful row. submission_valid=false is
    # not checked here -- a short smoke window is a valid, labelled run.
    got = {r["metric"]: float(r["performance"]) for r in results}
    # INFX_MIN_AGENTIC_REQUESTS raises the bar above one request: a few long
    # traces at low concurrency can profile only a handful, which passes the
    # error-rate check but is too thin a sample to report. Off by default.
    min_req = max(1, int(hook_env.get("INFX_MIN_AGENTIC_REQUESTS", "1")))
    if got.get("requests_profiled", 0) < min_req or got.get("throughput_tot", 0) <= 0:
        raise Exception(
            f"agentic replay produced no measurement in {agg}: "
            f"requests_profiled={got.get('requests_profiled')} "
            f"(need {min_req}), throughput_tot={got.get('throughput_tot')}")
    # The shared hook only warns when validate_agentic_result.sh flags the
    # error rate; mostly-failed requests must not export as a normal row.
    # Same definition and threshold as that validator: dropped errors over
    # profiled + dropped, AGENTIC_MAX_ERROR_RATE (default 0.10).
    acct = doc.get("request_accounting") or {}
    errs = float(acct.get("records_error_dropped") or 0)
    ok = float(acct.get("records_profiled") or 0)
    # From the env the hook ran with, so a per-point AGENTIC_MAX_ERROR_RATE
    # applies here exactly as it did to the hook's own validator.
    max_err = float(hook_env.get("AGENTIC_MAX_ERROR_RATE", "0.10"))
    if errs and errs / (ok + errs) > max_err:
        raise Exception(
            f"agentic replay error rate {errs / (ok + errs):.1%} "
            f"({int(errs)} of {int(ok + errs)} measured requests) exceeds "
            f"{max_err:.0%} in {agg}")


def _run_agentic(model, config, backend, server_cmd, host, port, start_timeout,
                 window_from_checkpoint=False, server_metrics=True):
    """Start `server_cmd`, replay the agentic corpus against it, return CSV rows.

    `config` is mutated the way the serving paths mutate it: runner-only keys are
    popped so that what remains can be splatted into the CSV row, whose columns
    are fixed by each runner's CSV_HEADER.
    """
    agentic_env = config.pop("agentic_env", {}) or {}
    # The recipe states how long a cold load may take (Kimi-K3's MI355X vLLM
    # recipe allows 7200 s through its health check and
    # VLLM_ENGINE_READY_TIMEOUT_S). That env var reaches the server but not
    # this wait, so without carrying the budget a valid load is interrupted
    # before the replay starts. Popped, so it never reaches the CSV row.
    start_timeout = int(config.pop("startup_timeout", 0) or start_timeout)
    if not config.get("num_prompts"):
        # Not a fixed-prompt-count workload, but the CSV column still wants one.
        config["num_prompts"] = "0"
    config["cmd"] = server_cmd

    print(server_cmd, flush=True)
    # A listener already on the port (a previous point's server that has not
    # gone, another runner) would answer /health while this child is still
    # loading, and the replay would then measure that server under this
    # point's name. Refuse before launching rather than trust any listener.
    if _port_in_use(host, port):
        raise Exception(
            f"{host}:{port} is already accepting connections before this point's "
            "server was started; refusing to measure whatever is listening there")
    # Own process group: the shell, the engine and every worker it forks can
    # then be signalled and reaped as one, instead of only the shell.
    server = subprocess.Popen(server_cmd, shell=True, start_new_session=True)
    results = []
    try:
        # Liveness-aware: a server that dies during init (bad option, OOM) can
        # never become ready, and polling /health alone would hold the GPU
        # allocation for the whole cold-load budget -- up to 7200 s for
        # Kimi-K3 -- before reporting anything.
        deadline = time.time() + start_timeout
        while True:
            if server.poll() is not None:
                raise Exception(
                    f"Server exited with code {server.returncode} before becoming "
                    f"ready at http://{host}:{port}/health")
            # Bounded: a server that accepts the connection and then stalls
            # would otherwise hang this probe, and with it both the deadline
            # and the liveness check above.
            if subprocess.run(f"curl -sf --connect-timeout 5 --max-time 20 "
                              f"http://{host}:{port}/health",
                              shell=True, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                break
            if time.time() >= deadline:
                raise Exception(
                    f"Server not ready at http://{host}:{port}/health after "
                    f"{start_timeout} s")
            time.sleep(30)
        print(f"Server at {server.pid} contacted successfully", flush=True)

        # The hook writes fixed filenames (agentic_result.json,
        # benchmark_command.txt), so two replays sharing a directory lose the
        # first one's evidence. Give each its own, and do not reuse one that
        # already exists -- a repeated run of the same point is still a
        # separate measurement.
        tag = config.get("variant") or f"{config['model']}_tp{config['tp']}"
        # madengine's run_directory is deleted after a successful run unless
        # --keep-model-dir/--keep-alive, and the shared wrapper lifts only the
        # performance CSV out of it. The aggregate JSON, replay logs and AIPerf
        # artifacts -- including submission_valid, which never reaches the CSV
        # -- would be destroyed by exactly the runs that produced them.
        #
        # AGENTIC_RESULTS_DIR, when the translator set it, is an explicit bind
        # mount and therefore durable regardless of where madengine staged
        # run_directory. "one directory above cwd" is kept as a fallback for
        # manifests generated before this existed, or hand-written ones, but it
        # is only durable when that parent happens to be shared storage --
        # true for a single-node run whose workspace is the NFS submission
        # dir, false for one on node-local scratch and false for every
        # multi-node SLURM deployment this project's job.sh.j2 defaults to.
        results_root = os.environ.get("AGENTIC_RESULTS_DIR") or ".."
        base_dir = os.path.abspath(
            os.path.join(results_root, f"agentic_{tag}_c{config['max_concurrency']}"
                                       .replace("/", "_"))
        )
        # mkdir, not exists-then-mkdir: AGENTIC_RESULTS_DIR is shared across
        # concurrent runs of the same point (same tag, same concurrency), and
        # exists()-then-makedirs() is a check-then-act race -- two runs can
        # both see attempt N as free and then both call makedirs, one of them
        # raising FileExistsError after it has already loaded the checkpoint.
        result_dir, attempt = base_dir, 1
        while True:
            try:
                os.makedirs(result_dir)
                break
            except FileExistsError:
                attempt += 1
                result_dir = f"{base_dir}_run{attempt}"

        hook_env = dict(os.environ)
        # One runtime for the whole runner invocation: without SLURM_JOB_ID the
        # hook keys it by its own PID, which is new for every point, so each
        # point re-cloned InferenceX and rebuilt the aiperf environment after
        # its checkpoint had loaded. The runner's PID is the same for all its
        # points. A caller that sets the directory itself keeps it.
        hook_env.setdefault("AGENTIC_RUNTIME_DIR",
                            os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                         f"mad-agentic-runner-{os.getpid()}"))
        hook_env.update({str(k): str(v) for k, v in agentic_env.items()})
        # MODEL must be present but EMPTY. A non-empty value makes
        # benchmark_agentic.sh skip /v1/models discovery, and vLLM serves the
        # mounted checkpoint path under no alias, so the replay would address an
        # id the server never registered. Dropping it entirely is wrong too:
        # agentic_lib.sh assigns the discovered name with a plain `MODEL=...`,
        # which only reaches the aggregation subprocess if the variable already
        # carries the export attribute. Passing it empty gives discovery its
        # trigger and the aggregator its value.
        hook_env["MODEL"] = ""
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
            # Empty (still set) when the server's metrics are not to be
            # scraped: the replay is told not to collect them at all (aiperf
            # would otherwise scrape the endpoint's /metrics by itself) and the
            # aggregate carries request metrics only.
            "AGENTIC_SERVER_METRICS": f"{host}:{port}" if server_metrics else "",
            **({} if server_metrics else {"AGENTIC_NO_SERVER_METRICS": "1"}),
        })
        # The hook treats either of these as a request for the AgentX suite
        # driver. One inherited from the caller's environment must not send this
        # point's replay there instead.
        for selector in ("AGENTIC_CONFIG", "AGENTIC_WORKLOAD"):
            hook_env.pop(selector, None)
        # A translated point must not be accepted on a best-effort result: a
        # failed aggregation or an unusable measurement fails the point here
        # rather than after the sweep. (Same limits as the checks below.)
        hook_env.setdefault("AGENTIC_STRICT_RESULT", "1")
        # The server reads the mounted checkpoint; the replay's tokenizer must
        # come from the same files, not from the Hub under whatever id the
        # server advertises (gated repos and offline nodes fail there).
        tokenizer = _tokenizer_for_replay(hook_env, model, server_cmd)
        if tokenizer:
            hook_env["AGENTIC_TOKENIZER"] = tokenizer
        # A server that does not advertise its window (ATOM's /v1/models omits
        # max_model_len) leaves the hook with nothing to gate the corpus on.
        if window_from_checkpoint:
            window = _window_for_replay(hook_env, model, server_cmd)
            if window:
                hook_env["MAX_MODEL_LEN"] = str(window)

        # The hook is this module's sibling. Resolving it from the working
        # directory only worked inside madengine's run_directory; a runner
        # started from the checkout imported this file and then failed to find
        # the hook after the checkpoint had loaded.
        hook = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_agentic.sh")
        bench_cmd = f"bash {shlex.quote(hook)} --backend {shlex.quote(backend)}"
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
        _check_measurement(results, doc, agg, hook_env)
    finally:
        _stop_server_group(server)

    return results

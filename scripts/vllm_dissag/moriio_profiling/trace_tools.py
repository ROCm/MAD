#!/usr/bin/env python3
'''Combine traces, bucket kernels, extract request mappings, build trimmed summaries, and run the single-trace analysis pipeline.'''
import argparse, csv, fnmatch, glob, io, json, math, os, re, shutil, signal, socket, statistics, subprocess, sys, time

csv.field_size_limit(sys.maxsize)


def log(msg):
    print(f"[combine_traces] {msg}", flush=True)


_NODE_DIR_RE = re.compile(r"^rocprof_(prefill|decode)_NODE(\d+)$")
_RANK_RE = re.compile(r"^(?P<host>.+)_(?P<pid>\d+)_kernel_trace\.csv$")
_REQID_LOG_RE = re.compile(r"^(?P<role>prefill|decode)_NODE(?P<node_rank>\d+)\.log$")
_REQID_PID_RE = re.compile(r"\bpid=(?P<pid>\d+)\b")
_REQID_LINE_RE = re.compile(
    r"moriio_reqid_map\s+"
    r"dir=(?P<direction>\S+)\s+"
    r"request_id=(?P<request_id>\S+)\s+"
    r"transfer_id=(?P<transfer_id>\S+)\s+"
    r"layer=(?P<layer>\S+)\s+"
    r"write_uid=(?P<write_uid>\S+)"
)
_WORKER_RANK_RE = re.compile(
    r"\(Worker(?:_[^)]*)? pid=(?P<pid>\d+)\).*?"
    r"world_size=(?P<world_size>\d+)\s+rank=(?P<global_rank>\d+)\s+"
    r"local_rank=(?P<local_rank>\d+)\b"
)
_WORKER_ASSIGN_RE = re.compile(
    r"\(Worker(?:_[^)]*)? pid=(?P<pid>\d+)\).*?"
    r"rank (?P<global_rank>\d+) in world size (?P<world_size>\d+) is assigned as "
    r"DP rank (?P<dp_rank>\d+), PP rank (?P<pp_rank>\d+), "
    r"PCP rank (?P<pcp_rank>\d+), TP rank (?P<tp_rank>\d+), "
    r"EP rank (?P<ep_rank>\d+)"
)


def parse_reqid_maps(paths):
    rows = []
    for path in paths:
        path_match = _REQID_LOG_RE.fullmatch(os.path.basename(path))
        path_fields = (path_match.groupdict() if path_match
                       else {"role": "", "node_rank": ""})
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = _REQID_LINE_RE.search(line)
                if match:
                    row = match.groupdict()
                    row.update(path_fields)
                    pid_match = _REQID_PID_RE.search(line, 0, match.start())
                    row["pid"] = pid_match.group("pid") if pid_match else ""
                    rows.append(row)
    return rows


def discover_node_dirs(jobdir):
    out = []
    for name in sorted(os.listdir(jobdir)):
        m = _NODE_DIR_RE.match(name)
        if not m:
            continue
        full = os.path.join(jobdir, name)
        if os.path.isdir(full):
            out.append((m.group(1), int(m.group(2)), full))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


class RankManifestError(ValueError):
    pass


def _artifact_rows(dirpath):
    rows = {}
    for kcsv in sorted(glob.glob(os.path.join(dirpath, "*_kernel_trace.csv"))):
        match = _RANK_RE.fullmatch(os.path.basename(kcsv))
        if not match:
            continue
        pid = int(match.group("pid"))
        if pid in rows:
            raise RankManifestError(f"duplicate kernel trace for PID {pid} in {dirpath}")
        host = match.group("host")
        prefix = os.path.join(dirpath, f"{host}_{pid}")
        rows[pid] = {
            "hostname": host,
            "kernel_csv": kcsv,
            "pftrace": prefix + "_results.pftrace" if os.path.isfile(prefix + "_results.pftrace") else None,
            "rocprof_json": prefix + "_results.json" if os.path.isfile(prefix + "_results.json") else None,
        }
    return rows


def _parse_worker_log(path):
    workers = {}
    assignments = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            match = _WORKER_RANK_RE.search(line)
            if match:
                row = {key: int(value) for key, value in match.groupdict().items()}
                row["source_line"] = lineno
                pid = row["pid"]
                old = workers.get(pid)
                comparable = {k: row[k] for k in ("pid", "world_size", "global_rank", "local_rank")}
                if old and any(old[k] != comparable[k] for k in comparable):
                    raise RankManifestError(f"conflicting worker-rank lines for PID {pid} in {path}")
                workers[pid] = comparable | {"source_line": old["source_line"] if old else lineno}

            match = _WORKER_ASSIGN_RE.search(line)
            if match:
                row = {key: int(value) for key, value in match.groupdict().items()}
                row["source_line"] = lineno
                pid = row["pid"]
                old = assignments.get(pid)
                if old and any(old[k] != row[k] for k in row if k != "source_line"):
                    raise RankManifestError(f"conflicting rank-assignment lines for PID {pid} in {path}")
                assignments[pid] = old or row
    return workers, assignments


def build_rank_manifest(jobdir):
    jobdir = os.path.abspath(jobdir)
    node_dirs = discover_node_dirs(jobdir)
    if not node_dirs:
        raise RankManifestError(f"no rocprof_{{prefill,decode}}_NODE* dirs found under {jobdir}")

    manifest_rows = []
    selections = []
    for role, node_rank, dirpath in node_dirs:
        log_path = os.path.join(jobdir, f"{role}_NODE{node_rank}.log")
        if not os.path.isfile(log_path):
            raise RankManifestError(f"missing worker log for {role} NODE{node_rank}: {log_path}")
        workers, assignments = _parse_worker_log(log_path)
        artifacts = _artifact_rows(dirpath)
        if not workers:
            raise RankManifestError(f"no worker rank mappings found in {log_path}")
        if set(workers) != set(artifacts):
            missing_logs = sorted(set(artifacts) - set(workers))
            missing_traces = sorted(set(workers) - set(artifacts))
            raise RankManifestError(
                f"PID mapping/trace mismatch for {role} NODE{node_rank}: "
                f"unmapped_trace_pids={missing_logs}, mapped_pids_without_trace={missing_traces}"
            )

        local_ranks = [row["local_rank"] for row in workers.values()]
        if len(local_ranks) != len(set(local_ranks)):
            raise RankManifestError(f"duplicate local_rank in {log_path}: {sorted(local_ranks)}")
        expected = list(range(len(local_ranks)))
        if sorted(local_ranks) != expected:
            raise RankManifestError(
                f"missing/ambiguous local ranks in {log_path}: got {sorted(local_ranks)}, expected {expected}"
            )

        rank0 = [row for row in workers.values() if row["local_rank"] == 0]
        if len(rank0) != 1:
            raise RankManifestError(
                f"expected exactly one local_rank=0 for {role} NODE{node_rank}, found {len(rank0)}"
            )

        for pid, worker in sorted(workers.items(), key=lambda item: item[1]["local_rank"]):
            assignment = assignments.get(pid)
            if assignment and (
                assignment["global_rank"] != worker["global_rank"]
                or assignment["world_size"] != worker["world_size"]
            ):
                raise RankManifestError(f"rank assignment conflicts with init mapping for PID {pid} in {log_path}")
            artifact = artifacts[pid]
            row = {
                "role": role,
                "node_rank": node_rank,
                "hostname": artifact["hostname"],
                "pid": pid,
                "world_size": worker["world_size"],
                "global_rank": worker["global_rank"],
                "local_rank": worker["local_rank"],
                "dp_rank": assignment["dp_rank"] if assignment else None,
                "ep_rank": assignment["ep_rank"] if assignment else None,
                "pp_rank": assignment["pp_rank"] if assignment else None,
                "tp_rank": assignment["tp_rank"] if assignment else None,
                "source_log": os.path.relpath(log_path, jobdir),
                "source_line": worker["source_line"],
                "assignment_source_line": assignment["source_line"] if assignment else None,
                "artifacts": {
                    key: os.path.relpath(value, jobdir) if value else None
                    for key, value in artifact.items() if key != "hostname"
                },
            }
            manifest_rows.append(row)
            if row["local_rank"] == 0:
                selections.append({key: row[key] for key in (
                    "role", "node_rank", "hostname", "pid", "world_size",
                    "global_rank", "local_rank", "dp_rank", "ep_rank", "pp_rank", "tp_rank"
                )})

    manifest = {
        "schema_version": 1,
        "job_dir": jobdir,
        "selection_rule": "exactly local_rank == 0 per role/node; no PID or filename ordering fallback",
        "workers": manifest_rows,
        "selections": selections,
    }
    for key in sorted({(row["role"], int(row["node_rank"])) for row in manifest_rows}):
        selected_workers(manifest, *key)
    return manifest


def write_rank_manifest(jobdir, out_path):
    manifest = build_rank_manifest(jobdir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return manifest


def load_rank_manifest(path, jobdir=None):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != 1:
        raise RankManifestError(f"unsupported rank manifest schema in {path}")
    if jobdir and os.path.abspath(manifest.get("job_dir", "")) != os.path.abspath(jobdir):
        raise RankManifestError(f"rank manifest {path} belongs to {manifest.get('job_dir')}, not {jobdir}")
    try:
        keys = [(row["role"], int(row["node_rank"])) for row in manifest.get("workers", [])]
    except (KeyError, TypeError, ValueError) as exc:
        raise RankManifestError(f"malformed worker row in rank manifest {path}: {exc}") from exc
    for key in sorted(set(keys)):
        selected_workers(manifest, *key)
    return manifest


def selected_workers(manifest, role, node_rank):
    """Return every validated worker for one role/node in local-rank order."""
    if role not in ("prefill", "decode"):
        raise RankManifestError(f"invalid worker role {role!r}")
    try:
        workers = manifest["workers"]
        role_rows = [row for row in workers if row["role"] == role]
        world_sizes = {int(row["world_size"]) for row in role_rows}
        role_nodes = sorted({int(row["node_rank"]) for row in role_rows})
        global_ranks = [int(row["global_rank"]) for row in role_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise RankManifestError(f"malformed worker manifest for role {role}: {exc}") from exc
    if not role_rows or not role_nodes:
        raise RankManifestError(f"manifest has no workers for role {role}")
    if len(world_sizes) != 1:
        raise RankManifestError(f"ambiguous world_size values for role {role}: {sorted(world_sizes)}")
    world_size = next(iter(world_sizes))
    if world_size <= 0 or sorted(global_ranks) != list(range(world_size)):
        raise RankManifestError(
            f"missing/duplicate global ranks for role {role}: "
            f"got {sorted(global_ranks)}, expected {list(range(world_size))}"
        )
    if world_size % len(role_nodes):
        raise RankManifestError(
            f"world_size {world_size} is not divisible by {len(role_nodes)} nodes for role {role}"
        )
    expected_local_world_size = world_size // len(role_nodes)
    requested = []
    seen_artifacts = set()
    for current_node in role_nodes:
        rows = [row for row in role_rows if int(row["node_rank"]) == current_node]
        try:
            local_ranks = [int(row["local_rank"]) for row in rows]
            pids = [int(row["pid"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise RankManifestError(
                f"malformed worker identity for {role} NODE{current_node}: {exc}"
            ) from exc
        expected_ranks = list(range(expected_local_world_size))
        if sorted(local_ranks) != expected_ranks:
            raise RankManifestError(
                f"missing/duplicate local ranks for {role} NODE{current_node}: "
                f"got {sorted(local_ranks)}, expected {expected_ranks}"
            )
        if len(pids) != len(set(pids)):
            raise RankManifestError(f"duplicate worker PID for {role} NODE{current_node}: {pids}")
        hosts = {str(row.get("hostname", "")) for row in rows}
        if len(hosts) != 1 or not next(iter(hosts)):
            raise RankManifestError(
                f"ambiguous worker host for {role} NODE{current_node}: {sorted(hosts)}"
            )
        for row in rows:
            if row.get("source_log") != f"{role}_NODE{current_node}.log":
                raise RankManifestError(
                    f"wrong-role/node source log for PID {row['pid']}: {row.get('source_log')}"
                )
            artifact = row.get("artifacts", {}).get("kernel_csv")
            if not isinstance(artifact, str) or not artifact:
                raise RankManifestError(f"missing kernel CSV artifact for PID {row['pid']}")
            normalized = os.path.normpath(artifact).replace("\\", "/")
            parts = normalized.split("/")
            expected_dir = f"rocprof_{role}_NODE{current_node}"
            if (os.path.isabs(artifact) or normalized == ".." or normalized.startswith("../")
                    or len(parts) != 2 or parts[0] != expected_dir):
                raise RankManifestError(
                    f"wrong-role/node kernel CSV for PID {row['pid']}: {artifact}"
                )
            match = _RANK_RE.fullmatch(parts[1])
            if (not match or int(match.group("pid")) != int(row["pid"])
                    or match.group("host") != row["hostname"]):
                raise RankManifestError(
                    f"ambiguous kernel CSV identity for PID {row['pid']}: {artifact}"
                )
            if normalized in seen_artifacts:
                raise RankManifestError(f"duplicate kernel CSV in manifest: {artifact}")
            seen_artifacts.add(normalized)
        if current_node == int(node_rank):
            requested = sorted(rows, key=lambda row: int(row["local_rank"]))
    if not requested:
        raise RankManifestError(f"manifest has no workers for {role} NODE{node_rank}")
    return requested


def selected_worker(manifest, role, node_rank):
    rows = [row for row in manifest["workers"]
            if row["role"] == role and int(row["node_rank"]) == int(node_rank)
            and int(row["local_rank"]) == 0]
    if len(rows) != 1:
        raise RankManifestError(
            f"expected one manifest selection for {role} NODE{node_rank}, found {len(rows)}"
        )
    return rows[0]


class WindowMappingError(ValueError):
    pass


class ClockAlignmentError(ValueError):
    pass


def _append_jsonl(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _clock_pair():
    realtime_before_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    boottime_ns = (
        time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        if hasattr(time, "CLOCK_BOOTTIME") else None
    )
    realtime_after_ns = time.time_ns()
    pair = {
        "realtime_before_ns": realtime_before_ns,
        "realtime_after_ns": realtime_after_ns,
        "realtime_mid_ns": (realtime_before_ns + realtime_after_ns) // 2,
        "monotonic_ns": monotonic_ns,
        "pair_uncertainty_ns": (realtime_after_ns - realtime_before_ns + 1) // 2,
    }
    if boottime_ns is not None:
        pair["boottime_ns"] = boottime_ns
    return pair


def _clock_sync_snapshot():
    commands = (["chronyc", "tracking", "-n"], ["timedatectl", "timesync-status"])
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            continue
        values = {}
        for label in ("Last offset", "Root delay", "Root dispersion"):
            match = re.search(rf"^{re.escape(label)}\s*:\s*([+-]?[0-9.eE+-]+)\s+seconds", text, re.MULTILINE)
            if match:
                values[label] = abs(float(match.group(1)))
        leap_ok = bool(re.search(r"^Leap status\s*:\s*Normal\s*$", text, re.MULTILINE))
        if leap_ok and "Root dispersion" in values:
            uncertainty_s = values.get("Last offset", 0.0) + values["Root dispersion"] + values.get("Root delay", 0.0) / 2.0
            return {
                "available": True,
                "command": cmd,
                "uncertainty_ns": int(math.ceil(uncertainty_s * 1e9)),
                "raw": text,
            }
        return {"available": False, "command": cmd, "uncertainty_ns": None, "raw": text}
    return {"available": False, "command": None, "uncertainty_ns": None, "raw": "no clock-sync status command available"}


def run_clock_sampler(out_path, node_rank, interval_s):
    if interval_s <= 0:
        raise ValueError(f"clock sample interval must be positive, got {interval_s}")
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        boot_id = open("/proc/sys/kernel/random/boot_id", encoding="utf-8").read().strip()
    except OSError:
        boot_id = None
    _append_jsonl(out_path, {
        "kind": "clock_sampler_metadata",
        "schema_version": 1,
        "timestamp_unit": "ns",
        "source_trace_clock": "CLOCK_BOOTTIME",
        "node_rank": int(node_rank),
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "clock": time.get_clock_info("monotonic").__dict__,
        "clock_sync": _clock_sync_snapshot(),
    })
    while not stop:
        _append_jsonl(out_path, {
            "kind": "clock_sample",
            "node_rank": int(node_rank),
            "hostname": socket.gethostname(),
            **_clock_pair(),
        })
        deadline = time.monotonic() + interval_s
        while not stop and time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    _append_jsonl(out_path, {
        "kind": "clock_sampler_stop",
        "node_rank": int(node_rank),
        "hostname": socket.gethostname(),
        **_clock_pair(),
    })


def record_benchmark_marker(out_path, event, step_id, fields):
    row = {
        "kind": "benchmark_marker",
        "event": event,
        "step_id": step_id,
        "hostname": socket.gethostname(),
        **_clock_pair(),
    }
    row.update({key: value for key, value in fields.items() if value is not None})
    _append_jsonl(out_path, row)
    return row


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise WindowMappingError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def _map_wall_to_monotonic(samples, wall_ns, sync_uncertainty_ns, max_gap_ns):
    samples = sorted(samples, key=lambda row: int(row["realtime_mid_ns"]))
    before = None
    after = None
    for sample in samples:
        sample_wall = int(sample["realtime_mid_ns"])
        if sample_wall <= wall_ns:
            before = sample
        if sample_wall >= wall_ns:
            after = sample
            break
    if before is None or after is None:
        raise WindowMappingError(f"wall timestamp {wall_ns} is not bracketed by node clock samples")
    r1 = int(before["realtime_mid_ns"])
    r2 = int(after["realtime_mid_ns"])
    if r2 - r1 > max_gap_ns:
        raise WindowMappingError(f"clock-sample gap {r2-r1} ns exceeds {max_gap_ns} ns")
    o1 = int(before["monotonic_ns"]) - r1
    o2 = int(after["monotonic_ns"]) - r2
    offset = o1 if r2 == r1 else o1 + ((o2 - o1) * (wall_ns - r1)) // (r2 - r1)
    uncertainty = (
        max(int(before.get("pair_uncertainty_ns", 0)), int(after.get("pair_uncertainty_ns", 0)))
        + abs(o2 - o1) + int(sync_uncertainty_ns)
    )
    return wall_ns + offset, uncertainty, {
        "formula": "trace_monotonic_ns = wall_realtime_ns + linearly_interpolated(monotonic_ns - realtime_mid_ns)",
        "before_realtime_ns": r1,
        "after_realtime_ns": r2,
        "before_offset_ns": o1,
        "after_offset_ns": o2,
    }


def build_window_manifest(jobdir, rank_manifest):
    jobdir = os.path.abspath(jobdir)
    timing_path = os.path.join(jobdir, "benchmark_timing.jsonl")
    clock_paths = sorted(glob.glob(os.path.join(jobdir, "clock_NODE*.jsonl")))
    historical_sources = []
    for log_path in sorted(glob.glob(os.path.join(jobdir, "benchmark_*_CONCURRENCY.log"))):
        initial_utc = None
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if line.startswith("UTC Time:"):
                    initial_utc = line.strip()
                    break
        historical_sources.append({
            "benchmark_log": os.path.relpath(log_path, jobdir),
            "initial_log_timestamp": initial_utc,
            "exact_measured_start_end_present": False,
        })
    unavailable = {
        "schema_version": 1,
        "job_dir": jobdir,
        "available": False,
        "analysis_scope": "whole_capture",
        "reason": (
            "historical job has no benchmark_timing.jsonl/per-node clock samples; "
            "wall/log timestamps cannot be mapped reliably to trace monotonic time"
        ),
        "source_timestamps": historical_sources,
        "node_windows": [],
    }
    if not os.path.isfile(timing_path) and not clock_paths:
        return unavailable
    if not os.path.isfile(timing_path):
        raise WindowMappingError("clock samples exist but benchmark_timing.jsonl is missing")

    markers = [row for row in _read_jsonl(timing_path) if row.get("kind") == "benchmark_marker"]
    by_step = {}
    for marker in markers:
        key = (str(marker.get("step_id")), str(marker.get("event")))
        if key in by_step:
            raise WindowMappingError(f"duplicate benchmark marker {key} in {timing_path}")
        by_step[key] = marker
    step_ids = sorted({key[0] for key in by_step})
    if not step_ids:
        raise WindowMappingError(f"no benchmark markers in {timing_path}")
    starts = []
    ends = []
    for step_id in step_ids:
        start = by_step.get((step_id, "start"))
        end = by_step.get((step_id, "end"))
        if not start or not end:
            raise WindowMappingError(f"unpaired benchmark markers for step {step_id}")
        if int(end.get("return_code", 0)) != 0:
            raise WindowMappingError(f"benchmark step {step_id} ended with status {end.get('return_code')}")
        starts.append(start)
        ends.append(end)
    wall_start_ns = min(int(row["realtime_mid_ns"]) for row in starts)
    wall_end_ns = max(int(row["realtime_mid_ns"]) for row in ends)
    if wall_end_ns <= wall_start_ns:
        raise WindowMappingError("benchmark end is not after benchmark start")

    max_gap_ns = int(os.environ.get("CLOCK_SAMPLE_MAX_GAP_NS", "5000000000"))
    max_uncertainty_ns = int(os.environ.get("WINDOW_MAX_UNCERTAINTY_NS", "20000000"))
    node_windows = []
    for selection in rank_manifest["selections"]:
        role = selection["role"]
        node_rank = int(selection["node_rank"])
        clock_path = os.path.join(jobdir, f"clock_NODE{node_rank}.jsonl")
        if not os.path.isfile(clock_path):
            raise WindowMappingError(f"missing clock samples for NODE{node_rank}: {clock_path}")
        rows = _read_jsonl(clock_path)
        metadata = [row for row in rows if row.get("kind") == "clock_sampler_metadata"]
        samples = [row for row in rows if row.get("kind") == "clock_sample"]
        if len(metadata) != 1 or len(samples) < 2:
            raise WindowMappingError(f"insufficient clock metadata/samples in {clock_path}")
        sync = metadata[0].get("clock_sync") or {}
        if not sync.get("available") or sync.get("uncertainty_ns") is None:
            raise WindowMappingError(f"NODE{node_rank} lacks usable clock-sync uncertainty evidence")
        start_ns, start_unc, start_formula = _map_wall_to_monotonic(
            samples, wall_start_ns, int(sync["uncertainty_ns"]), max_gap_ns)
        end_ns, end_unc, end_formula = _map_wall_to_monotonic(
            samples, wall_end_ns, int(sync["uncertainty_ns"]), max_gap_ns)
        uncertainty_ns = max(start_unc, end_unc)
        if uncertainty_ns > max_uncertainty_ns:
            raise WindowMappingError(
                f"NODE{node_rank} clock uncertainty {uncertainty_ns} ns exceeds {max_uncertainty_ns} ns"
            )
        node_windows.append({
            "role": role,
            "node_rank": node_rank,
            "hostname": selection["hostname"],
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "uncertainty_ns": uncertainty_ns,
            "clock_sync": sync,
            "start_conversion": start_formula,
            "end_conversion": end_formula,
        })
    return {
        "schema_version": 1,
        "job_dir": jobdir,
        "available": True,
        "analysis_scope": "benchmark_window",
        "wall_start_realtime_ns": wall_start_ns,
        "wall_end_realtime_ns": wall_end_ns,
        "source_timestamps": markers,
        "node_windows": node_windows,
        "max_clock_uncertainty_ns": max_uncertainty_ns,
    }


def write_window_manifest(jobdir, rank_manifest, out_path):
    manifest = build_window_manifest(jobdir, rank_manifest)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return manifest


def load_window_manifest(path, role, node_rank):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("available"):
        return manifest, None
    rows = [row for row in manifest.get("node_windows", [])
            if row["role"] == role and int(row["node_rank"]) == int(node_rank)]
    if len(rows) != 1:
        raise WindowMappingError(f"expected one window for {role} NODE{node_rank}, found {len(rows)}")
    return manifest, (int(rows[0]["start_monotonic_ns"]), int(rows[0]["end_monotonic_ns"]))


def discover_ranks(dirpath):
    'Discover trace artifacts; callers must join them to the rank manifest.'
    ranks = []
    for kcsv in glob.glob(os.path.join(dirpath, "*_kernel_trace.csv")):
        m = _RANK_RE.match(os.path.basename(kcsv))
        if not m:
            continue
        host, pid = m.group("host"), m.group("pid")
        marker = os.path.join(dirpath, f"{host}_{pid}_marker_api_trace.csv")
        pftrace = os.path.join(dirpath, f"{host}_{pid}_results.pftrace")
        ranks.append((
            int(pid), host, kcsv,
            marker if os.path.exists(marker) else None,
            pftrace if os.path.exists(pftrace) else None,
        ))
    ranks.sort(key=lambda x: x[0])
    return ranks


_NATIVE_CLOCK_CACHE = {}


def _checked_ns(value, label):
    if isinstance(value, bool):
        raise ClockAlignmentError(f"{label} is not an integer nanosecond timestamp")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ClockAlignmentError(f"{label} is not an integer nanosecond timestamp: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ClockAlignmentError(f"{label} is not an integer nanosecond timestamp: {value!r}")
    if result < 0 or result > INT64_MAX:
        raise ClockAlignmentError(f"{label} is outside signed-int64 nanoseconds: {result}")
    return result


def _wall_ns(source_ns, offset_ns, label):
    wall_ns = source_ns + offset_ns
    if wall_ns < 0 or wall_ns > INT64_MAX:
        raise ClockAlignmentError(f"{label} wall-clock alignment overflows signed-int64: {wall_ns}")
    return wall_ns


def _load_explicit_clock_manifest(path, jobdir):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != 1 or manifest.get("timestamp_unit") != "ns":
        raise ClockAlignmentError(
            f"clock manifest {path} must have schema_version=1 and timestamp_unit='ns'"
        )
    manifest_jobdir = manifest.get("job_dir")
    if manifest_jobdir and os.path.abspath(manifest_jobdir) != os.path.abspath(jobdir):
        raise ClockAlignmentError(f"clock manifest {path} belongs to {manifest_jobdir}, not {jobdir}")
    alignments = {}
    for index, row in enumerate(manifest.get("nodes", [])):
        try:
            key = (str(row["role"]), int(row["node_rank"]))
            hostname = str(row["hostname"])
            source_clock = str(row["source_clock"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClockAlignmentError(f"invalid node identity at {path}:nodes[{index}]") from exc
        if key in alignments:
            raise ClockAlignmentError(f"duplicate clock mapping for {key} in {path}")
        if key[0] not in ("prefill", "decode") or key[1] < 0 or not hostname:
            raise ClockAlignmentError(f"invalid role/node/hostname at {path}:nodes[{index}]")
        if row.get("timestamp_unit", "ns") != "ns":
            raise ClockAlignmentError(f"non-nanosecond clock mapping for {key} in {path}")
        if source_clock not in ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC", "CLOCK_MONOTONIC_RAW"):
            raise ClockAlignmentError(f"unsupported source clock {source_clock} for {key} in {path}")
        source_ns = _checked_ns(row.get("source_sample_ns"), f"{path}:{key} source_sample_ns")
        monotonic_ns = _checked_ns(row.get("monotonic_sample_ns"), f"{path}:{key} monotonic_sample_ns")
        realtime_ns = _checked_ns(row.get("realtime_sample_ns"), f"{path}:{key} realtime_sample_ns")
        uncertainty_ns = _checked_ns(row.get("uncertainty_ns"), f"{path}:{key} uncertainty_ns")
        if source_clock == "CLOCK_MONOTONIC" and source_ns != monotonic_ns:
            raise ClockAlignmentError(f"{path}:{key} CLOCK_MONOTONIC sample does not match monotonic sample")
        offset_ns = (monotonic_ns - source_ns) + (realtime_ns - monotonic_ns)
        if row.get("offset_ns") is not None and int(row["offset_ns"]) != offset_ns:
            raise ClockAlignmentError(f"{path}:{key} offset does not match its clock samples")
        if (row.get("sample_start_source_ns") is None) != (row.get("sample_end_source_ns") is None):
            raise ClockAlignmentError(f"{path}:{key} must provide both sample range endpoints or neither")
        alignments[key] = {
            "role": key[0],
            "node_rank": key[1],
            "hostname": hostname,
            "source_clock": source_clock,
            "offset_ns": offset_ns,
            "uncertainty_ns": uncertainty_ns,
            "sample_start_source_ns": (
                _checked_ns(row["sample_start_source_ns"], f"{path}:{key} sample start")
                if row.get("sample_start_source_ns") is not None else None
            ),
            "sample_end_source_ns": (
                _checked_ns(row["sample_end_source_ns"], f"{path}:{key} sample end")
                if row.get("sample_end_source_ns") is not None else None
            ),
            "correlation": {
                "source_sample_ns": source_ns,
                "monotonic_sample_ns": monotonic_ns,
                "realtime_sample_ns": realtime_ns,
            },
            "provenance": {
                "kind": "explicit_clock_manifest",
                "path": os.path.abspath(path),
                "detail": row.get("provenance"),
            },
        }
    if not alignments:
        raise ClockAlignmentError(f"clock manifest {path} has no node mappings")
    return alignments


def _parse_native_clock_snapshot(block, pftrace):
    primary_match = re.search(r"primary_trace_clock:\s*(BUILTIN_CLOCK_[A-Z_]+)", block)
    if not primary_match:
        raise ClockAlignmentError(f"native clock snapshot in {pftrace} has no primary trace clock")
    primary = primary_match.group(1)
    primary_ids = {
        "BUILTIN_CLOCK_REALTIME": (1, "CLOCK_REALTIME"),
        "BUILTIN_CLOCK_MONOTONIC": (3, "CLOCK_MONOTONIC"),
        "BUILTIN_CLOCK_MONOTONIC_RAW": (5, "CLOCK_MONOTONIC_RAW"),
        "BUILTIN_CLOCK_BOOTTIME": (6, "CLOCK_BOOTTIME"),
    }
    if primary not in primary_ids:
        raise ClockAlignmentError(f"unsupported primary trace clock {primary} in {pftrace}")
    clocks = {}
    for clock_block in re.findall(r"clocks\s*\{(.*?)\}", block, re.DOTALL):
        id_match = re.search(r"clock_id:\s*(\d+)", clock_block)
        ts_match = re.search(r"timestamp:\s*(\d+)", clock_block)
        if not id_match or not ts_match:
            continue
        clock_id = int(id_match.group(1))
        multiplier_match = re.search(r"unit_multiplier_ns:\s*(\d+)", clock_block)
        multiplier = int(multiplier_match.group(1)) if multiplier_match else 1
        if multiplier != 1:
            raise ClockAlignmentError(
                f"native clock {clock_id} in {pftrace} uses unsupported {multiplier} ns units"
            )
        timestamp = _checked_ns(ts_match.group(1), f"{pftrace} native clock {clock_id}")
        if clock_id in clocks and clocks[clock_id] != timestamp:
            raise ClockAlignmentError(f"duplicate conflicting native clock {clock_id} in {pftrace}")
        clocks[clock_id] = timestamp
    source_id, source_clock = primary_ids[primary]
    for required_id, name in ((1, "REALTIME"), (3, "MONOTONIC"), (source_id, source_clock)):
        if required_id not in clocks:
            raise ClockAlignmentError(f"native {name} clock is missing from {pftrace}")
    return {
        "source_clock": source_clock,
        "source_sample_ns": clocks[source_id],
        "monotonic_sample_ns": clocks[3],
        "realtime_sample_ns": clocks[1],
    }


def _extract_native_clock_alignment(pftrace, traceconv):
    cache_key = (os.path.abspath(pftrace), os.path.abspath(traceconv))
    if cache_key in _NATIVE_CLOCK_CACHE:
        return _NATIVE_CLOCK_CACHE[cache_key]
    try:
        with open(traceconv, "rb") as f:
            is_elf = f.read(4) == b"\x7fELF"
    except OSError as exc:
        raise ClockAlignmentError(f"cannot inspect native clock extractor {traceconv}: {exc}") from exc
    cmd = ([traceconv] if is_elf else ["python3", traceconv]) + [
        "text", "--skip-unknown", pftrace, "/dev/stdout",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise ClockAlignmentError(f"cannot start native clock extraction for {pftrace}: {exc}") from exc
    snapshots = []
    block = []
    depth = 0
    try:
        for line in proc.stdout:
            if not block:
                if "clock_snapshot {" not in line:
                    continue
                block = [line]
                depth = line.count("{") - line.count("}")
            else:
                block.append(line)
                depth += line.count("{") - line.count("}")
            if block and depth == 0:
                snapshots.append(_parse_native_clock_snapshot("".join(block), pftrace))
                block = []
                if len(snapshots) == 2:
                    break
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if len(snapshots) != 2:
        raise ClockAlignmentError(
            f"{pftrace} did not expose two native MONOTONIC/REALTIME clock snapshots"
        )
    if snapshots[0]["source_clock"] != snapshots[1]["source_clock"]:
        raise ClockAlignmentError(f"native primary clock changes between snapshots in {pftrace}")
    source_clock = snapshots[1]["source_clock"]
    offsets = [
        row["realtime_sample_ns"] - row["source_sample_ns"]
        for row in snapshots
    ]
    sample_span_ns = max(
        abs(snapshots[1][field] - snapshots[0][field])
        for field in ("source_sample_ns", "monotonic_sample_ns", "realtime_sample_ns")
    )
    uncertainty_ns = max(sample_span_ns, abs(offsets[1] - offsets[0]))
    selected = snapshots[1]
    result = {
        "source_clock": source_clock,
        "offset_ns": offsets[1],
        "uncertainty_ns": uncertainty_ns,
        "sample_start_source_ns": None,
        "sample_end_source_ns": None,
        "correlation": selected,
        "provenance": {
            "kind": "native_perfetto_clock_snapshot",
            "pftrace": os.path.abspath(pftrace),
            "traceconv": os.path.abspath(traceconv),
            "snapshot_count": 2,
            "uncertainty_scope": "back-to-back native clock snapshot span",
        },
    }
    _NATIVE_CLOCK_CACHE[cache_key] = result
    return result


def _load_host_clock_alignment(path, role, node_rank, hostname, native):
    try:
        rows = _read_jsonl(path)
    except WindowMappingError as exc:
        raise ClockAlignmentError(str(exc)) from exc
    metadata = [row for row in rows if row.get("kind") == "clock_sampler_metadata"]
    samples = [row for row in rows if row.get("kind") in ("clock_sample", "clock_sampler_stop")]
    if len(metadata) != 1 or len(samples) < 2:
        raise ClockAlignmentError(f"{path} must contain one metadata row and at least two clock samples")
    meta = metadata[0]
    if meta.get("timestamp_unit", "ns") != "ns":
        raise ClockAlignmentError(f"{path} does not use nanosecond clock units")
    if int(meta.get("node_rank", -1)) != node_rank or meta.get("hostname") != hostname:
        raise ClockAlignmentError(
            f"{path} identity does not match {role} NODE{node_rank} host {hostname}"
        )
    clock_info = meta.get("clock") or {}
    if clock_info and (not clock_info.get("monotonic") or clock_info.get("adjustable")):
        raise ClockAlignmentError(f"{path} does not describe a stable monotonic clock")
    sync = meta.get("clock_sync") or {}
    if not sync.get("available") or sync.get("uncertainty_ns") is None:
        raise ClockAlignmentError(f"{path} lacks usable clock-sync uncertainty evidence")
    sync_uncertainty_ns = _checked_ns(sync["uncertainty_ns"], f"{path} sync uncertainty")
    source_clock = native["source_clock"] if native else meta.get("source_trace_clock")
    if not source_clock:
        raise ClockAlignmentError(
            f"{path} does not identify the source trace clock and no native PFTrace snapshot is available"
        )
    if meta.get("source_trace_clock") and meta["source_trace_clock"] != source_clock:
        raise ClockAlignmentError(f"{path} source clock conflicts with the native PFTrace clock")
    source_to_monotonic = (
        native["correlation"]["monotonic_sample_ns"] - native["correlation"]["source_sample_ns"]
        if native else 0
    )
    normalized = []
    previous_source = None
    max_gap_ns = int(os.environ.get("CLOCK_SAMPLE_MAX_GAP_NS", "5000000000"))
    for index, sample in enumerate(samples):
        if int(sample.get("node_rank", -1)) != node_rank or sample.get("hostname") != hostname:
            raise ClockAlignmentError(f"{path} sample {index} has the wrong node identity")
        before_ns = _checked_ns(sample.get("realtime_before_ns"), f"{path} sample {index} realtime before")
        realtime_ns = _checked_ns(sample.get("realtime_mid_ns"), f"{path} sample {index} realtime")
        after_ns = _checked_ns(sample.get("realtime_after_ns"), f"{path} sample {index} realtime after")
        monotonic_ns = _checked_ns(sample.get("monotonic_ns"), f"{path} sample {index} monotonic")
        pair_uncertainty_ns = _checked_ns(
            sample.get("pair_uncertainty_ns"), f"{path} sample {index} pair uncertainty")
        if not before_ns <= realtime_ns <= after_ns:
            raise ClockAlignmentError(f"{path} sample {index} has an invalid realtime bracket")
        if pair_uncertainty_ns < (after_ns - before_ns + 1) // 2:
            raise ClockAlignmentError(f"{path} sample {index} understates pair uncertainty")
        if source_clock == "CLOCK_BOOTTIME" and sample.get("boottime_ns") is not None:
            source_ns = _checked_ns(sample["boottime_ns"], f"{path} sample {index} boottime")
        elif source_clock == "CLOCK_MONOTONIC":
            source_ns = monotonic_ns
        elif native:
            source_ns = monotonic_ns - source_to_monotonic
            if source_ns < 0:
                raise ClockAlignmentError(f"{path} sample {index} source-clock conversion is negative")
        else:
            raise ClockAlignmentError(
                f"{path} sample {index} cannot map {source_clock} to MONOTONIC"
            )
        if previous_source is not None:
            gap_ns = source_ns - previous_source
            if gap_ns <= 0 or gap_ns > max_gap_ns:
                raise ClockAlignmentError(
                    f"{path} source-clock sample gap {gap_ns} ns is invalid or exceeds {max_gap_ns} ns"
                )
        previous_source = source_ns
        normalized.append({
            "source_ns": source_ns,
            "monotonic_ns": monotonic_ns,
            "realtime_ns": realtime_ns,
            "pair_uncertainty_ns": pair_uncertainty_ns,
            "offset_ns": realtime_ns - source_ns,
        })
    selected = normalized[len(normalized) // 2]
    uncertainty_ns = (
        sync_uncertainty_ns
        + max(row["pair_uncertainty_ns"] for row in normalized)
        + max(abs(row["offset_ns"] - selected["offset_ns"]) for row in normalized)
        + (native["uncertainty_ns"] if native and source_clock != "CLOCK_BOOTTIME" else 0)
    )
    native_delta_ns = None
    if native:
        native_delta_ns = selected["offset_ns"] - native["offset_ns"]
        uncertainty_ns = max(uncertainty_ns, abs(native_delta_ns) + native["uncertainty_ns"])
    return {
        "role": role,
        "node_rank": node_rank,
        "hostname": hostname,
        "source_clock": source_clock,
        "offset_ns": selected["offset_ns"],
        "uncertainty_ns": uncertainty_ns,
        "sample_start_source_ns": normalized[0]["source_ns"],
        "sample_end_source_ns": normalized[-1]["source_ns"],
        "correlation": {
            "source_sample_ns": selected["source_ns"],
            "monotonic_sample_ns": selected["monotonic_ns"],
            "realtime_sample_ns": selected["realtime_ns"],
        },
        "provenance": {
            "kind": "host_clock_samples",
            "path": os.path.abspath(path),
            "sample_count": len(normalized),
            "clock_sync_command": sync.get("command"),
            "clock_sync_uncertainty_ns": sync_uncertainty_ns,
            "native_crosscheck_delta_ns": native_delta_ns,
            "native_pftrace": native["provenance"]["pftrace"] if native else None,
        },
    }


def _resolve_clock_alignment(jobdir, role, node_rank, hostname, pftrace,
                             explicit_alignments, traceconv):
    key = (role, node_rank)
    max_uncertainty_ns = int(os.environ.get(
        "COMBINE_MAX_CLOCK_UNCERTAINTY_NS",
        os.environ.get("WINDOW_MAX_UNCERTAINTY_NS", "20000000"),
    ))
    if max_uncertainty_ns < 0:
        raise ClockAlignmentError("COMBINE_MAX_CLOCK_UNCERTAINTY_NS must be non-negative")
    if explicit_alignments is not None:
        if key not in explicit_alignments:
            raise ClockAlignmentError(f"clock manifest has no mapping for {role} NODE{node_rank}")
        alignment = dict(explicit_alignments[key])
        if alignment["hostname"] != hostname:
            raise ClockAlignmentError(
                f"clock manifest host {alignment['hostname']} does not match {hostname} for {key}"
            )
    else:
        native = None
        native_available = bool(
            pftrace and traceconv and os.path.isfile(pftrace)
            and os.path.isfile(traceconv) and os.access(traceconv, os.X_OK)
            and os.path.exists("/dev/stdout")
        )
        if native_available:
            native = _extract_native_clock_alignment(pftrace, traceconv)
        clock_path = os.path.join(jobdir, f"clock_NODE{node_rank}.jsonl")
        if os.path.isfile(clock_path):
            alignment = _load_host_clock_alignment(
                clock_path, role, node_rank, hostname, native)
        elif native:
            alignment = {
                **native,
                "role": role,
                "node_rank": node_rank,
                "hostname": hostname,
            }
        else:
            raise ClockAlignmentError(
                f"no valid cross-node clock correlation for {role} NODE{node_rank}: "
                f"missing {clock_path} and no extractable native PFTrace clock snapshot; "
                "future captures must retain clock_NODE*.jsonl, or pass --clock-manifest"
            )
    if alignment["uncertainty_ns"] > max_uncertainty_ns:
        raise ClockAlignmentError(
            f"{role} NODE{node_rank} clock uncertainty {alignment['uncertainty_ns']} ns "
            f"exceeds {max_uncertainty_ns} ns"
        )
    return alignment


def _timestamp_range(path):
    first_start_ns = last_end_ns = None
    count = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None, 0
        if "Start_Timestamp" not in header or "End_Timestamp" not in header:
            raise ClockAlignmentError(f"{path} has no Start_Timestamp/End_Timestamp columns")
        start_index = header.index("Start_Timestamp")
        end_index = header.index("End_Timestamp")
        for lineno, row in enumerate(reader, 2):
            if not row:
                continue
            try:
                start_ns = _checked_ns(row[start_index], f"{path}:{lineno} Start_Timestamp")
                end_ns = _checked_ns(row[end_index], f"{path}:{lineno} End_Timestamp")
            except IndexError as exc:
                raise ClockAlignmentError(f"{path}:{lineno} has an incomplete timestamp row") from exc
            if end_ns < start_ns:
                raise ClockAlignmentError(f"{path}:{lineno} has end before start")
            first_start_ns = start_ns if first_start_ns is None else min(first_start_ns, start_ns)
            last_end_ns = end_ns if last_end_ns is None else max(last_end_ns, end_ns)
            count += 1
    return first_start_ns, last_end_ns, count


def _stream_csv_events(path, name_column, default_name, alignment, global_wall_min_ns,
                       pid, tid, out_fh, first_written):
    count = 0
    first_source_ns = last_end_ns = None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        required = ("Start_Timestamp", "End_Timestamp", name_column)
        if not header or any(column not in header for column in required):
            raise ClockAlignmentError(f"{path} is missing required columns {required}")
        start_index = header.index("Start_Timestamp")
        end_index = header.index("End_Timestamp")
        name_index = header.index(name_column)
        for lineno, row in enumerate(reader, 2):
            if not row:
                continue
            try:
                start_ns = _checked_ns(row[start_index], f"{path}:{lineno} Start_Timestamp")
                end_ns = _checked_ns(row[end_index], f"{path}:{lineno} End_Timestamp")
            except IndexError as exc:
                raise ClockAlignmentError(f"{path}:{lineno} has an incomplete event row") from exc
            if end_ns < start_ns:
                raise ClockAlignmentError(f"{path}:{lineno} has end before start")
            if end_ns - start_ns > INT64_MAX:
                raise ClockAlignmentError(f"{path}:{lineno} duration overflows signed-int64")
            first_source_ns = start_ns if first_source_ns is None else min(first_source_ns, start_ns)
            last_end_ns = max(last_end_ns or end_ns, end_ns)
            wall_start_ns = _wall_ns(start_ns, alignment["offset_ns"], f"{path}:{lineno} start")
            wall_end_ns = _wall_ns(end_ns, alignment["offset_ns"], f"{path}:{lineno} end")
            display_start_ns = wall_start_ns - global_wall_min_ns
            display_end_ns = wall_end_ns - global_wall_min_ns
            if display_start_ns < 0 or display_end_ns < display_start_ns:
                raise ClockAlignmentError(f"{path}:{lineno} produced invalid aligned coordinates")
            name = row[name_index] if name_index < len(row) else default_name
            event = {
                "ph": "X", "name": name or default_name,
                "ts": display_start_ns / 1000.0,
                "dur": (end_ns - start_ns) / 1000.0,
                "pid": pid, "tid": tid,
            }
            if first_written:
                out_fh.write(",")
            out_fh.write(json.dumps(event))
            first_written = True
            count += 1
    return count, first_written, first_source_ns, last_end_ns


def stream_kernel_events(kernel_csv, alignment, global_wall_min_ns,
                         pid, tid, out_fh, first_written):
    return _stream_csv_events(
        kernel_csv, "Kernel_Name", "kernel", alignment, global_wall_min_ns,
        pid, tid, out_fh, first_written,
    )


def stream_marker_events(marker_csv, alignment, global_wall_min_ns,
                         pid, tid, out_fh, first_written):
    return _stream_csv_events(
        marker_csv, "Function", "marker", alignment, global_wall_min_ns,
        pid, tid, out_fh, first_written,
    )


def build_combined(jobdir, out_path, rank0_only, rank_manifest, role_filter=None,
                   clock_manifest_path=None, traceconv=None):
    node_dirs = discover_node_dirs(jobdir)
    if not node_dirs:
        raise SystemExit(f"no rocprof_{{prefill,decode}}_NODE* dirs found under {jobdir}")
    if role_filter:
        node_dirs = [node_dir for node_dir in node_dirs if node_dir[0] == role_filter]
        if not node_dirs:
            raise RankManifestError(f"no {role_filter} rocprof dirs found under {jobdir}")

    t_start = time.time()
    pid_counter = 0
    total_kernel_events = 0
    total_marker_events = 0
    summary = []
    explicit_alignments = _load_explicit_clock_manifest(clock_manifest_path, jobdir)
    nodes = []
    for role, node_idx, dirpath in node_dirs:
        ranks = discover_ranks(dirpath)
        if not ranks:
            raise RankManifestError(f"no ranks discovered in {dirpath}")
        rows = [row for row in rank_manifest["workers"]
                if row["role"] == role and int(row["node_rank"]) == node_idx]
        by_pid = {int(row["pid"]): row for row in rows}
        if len(by_pid) != len(rows):
            raise RankManifestError(f"duplicate manifest PID for {role} NODE{node_idx}")
        artifact_pids = {rank[0] for rank in ranks}
        if set(by_pid) != artifact_pids:
            raise RankManifestError(
                f"manifest/artifact PID mismatch for {role} NODE{node_idx}: "
                f"manifest={sorted(by_pid)}, artifacts={sorted(artifact_pids)}"
            )
        for pid_val, host, *_ in ranks:
            if by_pid[pid_val]["hostname"] != host:
                raise RankManifestError(
                    f"manifest host {by_pid[pid_val]['hostname']} does not match artifact host "
                    f"{host} for PID {pid_val}"
                )
        ranks.sort(key=lambda rank: int(by_pid[rank[0]]["local_rank"]))
        selected = selected_worker(rank_manifest, role, node_idx)
        selected_ranks = [rank for rank in ranks if rank[0] == int(selected["pid"])]
        if len(selected_ranks) != 1:
            raise RankManifestError(
                f"selected PID {selected['pid']} missing/duplicate in {dirpath}"
            )
        if rank0_only:
            ranks = selected_ranks
        hostname = selected["hostname"]
        alignment = _resolve_clock_alignment(
            jobdir, role, node_idx, hostname, selected_ranks[0][4],
            explicit_alignments, traceconv,
        )
        nodes.append({
            "role": role, "node_rank": node_idx, "ranks": ranks,
            "by_pid": by_pid, "alignment": alignment,
        })

    first_wall_timestamps = []
    for node in nodes:
        for _, _, kernel_csv, marker_csv, _ in node["ranks"]:
            for source_path in (kernel_csv, marker_csv):
                if not source_path:
                    continue
                first_source_ns, _, _ = _timestamp_range(source_path)
                if first_source_ns is not None:
                    first_wall_timestamps.append(_wall_ns(
                        first_source_ns, node["alignment"]["offset_ns"],
                        f"{source_path} first event",
                    ))
    if not first_wall_timestamps:
        raise ClockAlignmentError("no timestamped events found in selected combined-trace inputs")
    global_wall_min_ns = min(first_wall_timestamps)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    node_ranges = {}
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as out_fh:
            out_fh.write('{"traceEvents":[')
            first_written = False
            for node in nodes:
                role = node["role"]
                node_idx = node["node_rank"]
                alignment = node["alignment"]
                range_key = (role, node_idx)
                node_ranges[range_key] = [None, None]
                for pid_val, host, kcsv, marker_csv, pftrace in node["ranks"]:
                    mapping = node["by_pid"][pid_val]
                    local_rank = int(mapping["local_rank"])
                    global_rank = int(mapping["global_rank"])
                    out_pid = 1000 + pid_counter
                    pid_counter += 1
                    proc_name = (
                        f"{role.upper()} NODE{node_idx} LOCAL_RANK{local_rank} "
                        f"GLOBAL_RANK{global_rank} ({host}:{pid_val})"
                    )
                    process_args = {
                        "name": proc_name, "role": role, "node_rank": node_idx,
                        "hostname": host, "source_pid": pid_val,
                        **{key: mapping.get(key) for key in (
                            "world_size", "global_rank", "local_rank", "dp_rank",
                            "ep_rank", "pp_rank", "tp_rank",
                        )},
                    }
                    meta = [
                        {"ph": "M", "name": "process_name", "pid": out_pid, "tid": 0,
                         "args": process_args},
                        {"ph": "M", "name": "thread_name", "pid": out_pid, "tid": 1,
                         "args": {"name": f"GPU kernels ({role} NODE{node_idx} local_rank{local_rank})"}},
                    ]
                    for meta_event in meta:
                        if first_written:
                            out_fh.write(",")
                        out_fh.write(json.dumps(meta_event))
                        first_written = True
                    nk, first_written, kernel_first, kernel_last = stream_kernel_events(
                        kcsv, alignment, global_wall_min_ns, out_pid, 1, out_fh, first_written)
                    nmk = 0
                    marker_first = marker_last = None
                    if marker_csv:
                        if first_written:
                            out_fh.write(",")
                        out_fh.write(json.dumps({
                            "ph": "M", "name": "thread_name", "pid": out_pid, "tid": 2,
                            "args": {"name": "Marker API (reqstats + MORI-IO)"},
                        }))
                        first_written = True
                        nmk, first_written, marker_first, marker_last = stream_marker_events(
                            marker_csv, alignment, global_wall_min_ns,
                            out_pid, 2, out_fh, first_written,
                        )
                    starts = [value for value in (kernel_first, marker_first) if value is not None]
                    ends = [value for value in (kernel_last, marker_last) if value is not None]
                    if starts:
                        old_start = node_ranges[range_key][0]
                        node_ranges[range_key][0] = min(starts + ([old_start] if old_start is not None else []))
                    if ends:
                        old_end = node_ranges[range_key][1]
                        node_ranges[range_key][1] = max(ends + ([old_end] if old_end is not None else []))
                    total_kernel_events += nk
                    total_marker_events += nmk
                    has_native_pftrace = pftrace is not None
                    summary.append((proc_name, nk, nmk, has_native_pftrace))
                    log(f"{proc_name}: kernel_events={nk} marker_events={nmk} "
                        f"native_pftrace={'yes' if has_native_pftrace else 'no'}")
            for node in nodes:
                key = (node["role"], node["node_rank"])
                source_start, source_end = node_ranges[key]
                sample_start = node["alignment"]["sample_start_source_ns"]
                sample_end = node["alignment"]["sample_end_source_ns"]
                if sample_start is not None and (
                    source_start is None or source_start < sample_start or source_end > sample_end
                ):
                    raise ClockAlignmentError(
                        f"{key} trace range {source_start}..{source_end} is not bracketed by "
                        f"clock samples {sample_start}..{sample_end}"
                    )
            alignment_metadata = {
                "schema_version": 1,
                "coordinate_domain": "wall_realtime_ns_minus_one_global_minimum",
                "formula": (
                    "wall_ns = source_ns + (monotonic_sample_ns - source_sample_ns) "
                    "+ (realtime_sample_ns - monotonic_sample_ns)"
                ),
                "source_timestamp_unit": "ns",
                "event_coordinate_unit": "us",
                "global_wall_origin_ns": global_wall_min_ns,
                "nodes": [node["alignment"] for node in nodes],
            }
            out_fh.write('],"displayTimeUnit":"ns","clockAlignment":')
            out_fh.write(json.dumps(alignment_metadata, sort_keys=True))
            out_fh.write("}")
            out_fh.flush()
            os.fsync(out_fh.fileno())
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    elapsed = time.time() - t_start
    size = os.path.getsize(out_path)
    log(f"wrote {out_path} ({size/1e6:.1f} MB, {pid_counter} lanes, "
        f"{total_kernel_events} kernel events, {total_marker_events} marker events) "
        f"in {elapsed:.1f}s")
    return summary, size



def categorize_kernel(name):
    n = name.lower()

    # Preserve classifier precedence for fused kernels.
    if ('rmsnorm' in n or 'fused_rms' in n or 'rms_norm' in n or
        ('rsqrt' in n and 'mean' in n and 'mul' in n)):
        return 'RMSNorm'
    # ROPE must precede other fused-kernel matches.
    if 'rope' in n: return 'ROPE'

    if 'reshape' in n and 'cache' in n:
        return 'KVCacheReshape'

    if 'kernel_unified_attention' in n: return 'Attention'
    if '_fwd_kernel' in name: return 'TritonAttention'
    if 'fmha' in n: return 'FMHA'
    if 'mla' in n: return 'MLA'
    if 'aiter::pa' in name: return 'PA'
    if 'paged_attention' in n: return 'PagedAttn'

    if 'routing' in n or 'route' in n: return 'MoE_Router'
    if 'moesorting' in n or 'opus_moe_sorting_entry' in n: return 'MoE_Sort'
    if 'epcombine' in n and 'syncbarrier' in n: return 'EP_Barrier'
    # Only intra/inter-node payload kernels are MORI EP; support kernels are communication.
    if any(x in n for x in ('epdispatchinternode', 'epcombineinternode', 'epdispatchintranode', 'epcombineintranode')): return 'MORI EP'
    if 'epdispatch' in n or 'epcombine' in n: return 'Communication'
    if 'fmoe' in n: return 'MoE_Fused'
    if 'kernel_moe' in n: return 'MoE_Unfused'
    if 'topk' in n: return 'MoE_TopK'

    if 'gemm' in n or 'cijk' in n or 'wvsplit' in n or 'matmul' in n: return 'GEMM'
    # Activation must precede quantization for fused kernels.
    if 'act_and_mul' in n or 'silu' in n: return 'Activation'
    if 'quant' in n: return 'Quant'

    if 'allreduce' in n or 'cross_device' in n or 'nccl' in n: return 'Communication'

    if 'poi' in n or 'elementwise' in n: return 'Elementwise'
    return 'Other'


def _find_column(headers, predicate, what):
    for h in headers:
        if predicate((h or '').lower()):
            return h
    sys.exit(
        f"ERROR: could not find the {what} column.\n"
        f"  Headers found: {headers}"
    )


def find_kernel_name_column(headers):
    return _find_column(
        headers,
        lambda h: 'kernel' in h and 'name' in h,
        "kernel name (a column containing both 'kernel' and 'name')",
    )


def find_duration_sum_column(headers):
    return _find_column(
        headers,
        lambda h: 'duration' in h and '_sum' in h,
        "duration sum (a column containing both 'duration' and '_sum')",
    )


def find_duration_count_column(headers):
    for h in headers:
        hl = (h or '').lower()
        if 'duration' in hl and '_count' in hl:
            return h
    return None  # Count is optional; default to one launch per row.


def _to_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _to_int(value):
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0


def process(in_path, out_per_kernel, out_by_category, label):
    # Accept producer BOMs and micro-sign headers.
    with open(in_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        sys.exit(f"ERROR: input CSV '{in_path}' is empty.")

    headers = rows[0]
    data_rows = rows[1:]

    name_col = find_kernel_name_column(headers)
    dur_sum_col = find_duration_sum_column(headers)
    count_col = find_duration_count_column(headers)

    name_idx = headers.index(name_col)
    dur_idx = headers.index(dur_sum_col)
    count_idx = headers.index(count_col) if count_col else None


    out_headers = ['Category'] + headers
    agg = {}
    grand_count = 0
    grand_us = 0.0

    with open(out_per_kernel, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        for row in data_rows:
            if not row or all((c or '').strip() == '' for c in row):
                continue
            kernel_name = row[name_idx] if name_idx < len(row) else ''
            category = categorize_kernel(kernel_name)
            writer.writerow([category] + row)

            total_us = _to_float(row[dur_idx]) if dur_idx < len(row) else 0.0
            n_kernels = (
                _to_int(row[count_idx])
                if count_idx is not None and count_idx < len(row)
                else 1
            )
            bucket = agg.setdefault(category, [0, 0.0])
            bucket[0] += n_kernels
            bucket[1] += total_us
            grand_count += n_kernels
            grand_us += total_us


    with open(out_by_category, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Num_Kernels', 'Total_us', 'Total_ms',
                         'Pct_of_Kernel_Time'])
        for category, (n_kernels, total_us) in sorted(
                agg.items(), key=lambda kv: kv[1][1], reverse=True):
            pct = (total_us / grand_us * 100) if grand_us > 0 else 0.0
            writer.writerow([
                category,
                n_kernels,
                f"{total_us:.2f}",
                f"{total_us / 1000.0:.4f}",
                f"{pct:.1f}%",
            ])
        writer.writerow([
            'TOTAL',
            grand_count,
            f"{grand_us:.2f}",
            f"{grand_us / 1000.0:.4f}",
            "100.0%",
        ])

    _print_summary(label, in_path, headers, name_col, dur_sum_col, count_col,
                   agg, grand_count, grand_us, out_per_kernel, out_by_category)


def _print_summary(label, in_path, headers, name_col, dur_sum_col, count_col,
                   agg, grand_count, grand_us, out_per_kernel, out_by_category):
    print('==== %s ====' % label)
    print('input: %s' % in_path)
    print("kernel-name column:   %r" % name_col)
    print("duration-sum column:  %r" % dur_sum_col)
    print("duration-count column:%s" % (
        (' %r' % count_col) if count_col else ' (not found; counted 1 row per kernel)'))
    print('total kernel time: %.1f us  (%.3f ms)   total kernels: %d'
          % (grand_us, grand_us / 1000.0, grand_count))
    print()
    print('%-15s %10s %15s %9s' % ('Category', '#kernels', 'total_us', '%time'))
    for category, (n_kernels, total_us) in sorted(
            agg.items(), key=lambda kv: kv[1][1], reverse=True):
        pct = (total_us / grand_us * 100) if grand_us > 0 else 0.0
        print('%-15s %10d %15.1f %8.2f%%' % (category, n_kernels, total_us, pct))
    print('%-15s %10d %15.1f %8.2f%%' % ('TOTAL', grand_count, grand_us, 100.0))
    print()
    print('wrote:', out_per_kernel)
    print('wrote:', out_by_category)



OUTPUT_COLUMNS = [
    "Time", "Total Time", "Instances", "Avg", "Med", "Min", "Max", "StdDev",
    "GridXYZ", "BlockXYZ", "VGPR", "AccumVGPR", "SGPR", "LDS", "Scratch", "Name",
    "Time %", "Total Time (ns)", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)",
    "StdDev (ns)", "GridX", "GridY", "GridZ", "BlockX", "BlockY", "BlockZ",
    "n_trimmed", "instances_before_trim",
]


def pretty_ns(ns):
    if ns >= 1e6:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.3f} \u00b5s"
    return f"{ns:.3f} ns"


def _n_to_trim(count, trim_pct):
    'Return the ceiling-based trim count for eligible kernels.'
    if trim_pct <= 0:
        return 0
    n = math.ceil(count * trim_pct / 100.0)
    return max(n, 1)


_KERNEL_RESOURCE_FIELDS = {
    "GridX": "Grid_Size_X", "GridY": "Grid_Size_Y", "GridZ": "Grid_Size_Z",
    "BlockX": "Workgroup_Size_X", "BlockY": "Workgroup_Size_Y", "BlockZ": "Workgroup_Size_Z",
    "VGPR": "VGPR_Count", "AccumVGPR": "Accum_VGPR_Count", "SGPR": "SGPR_Count",
    "LDS": "LDS_Block_Size", "Scratch": "Scratch_Size",
}
_NORMALIZED_HELP_COLUMNS = [
    "Kernel name", "kernel_duration_us_sum", "kernel_duration_us_count",
]


def _scan_kernel_traces(kernel_trace_csvs, window=None):
    """Stream raw worker CSVs and retain only durations plus one resource tuple per name."""
    if isinstance(kernel_trace_csvs, (str, os.PathLike)):
        paths = [os.path.abspath(os.fspath(kernel_trace_csvs))]
    else:
        paths = [os.path.abspath(os.fspath(path)) for path in kernel_trace_csvs]
    if not paths:
        raise SystemExit("ERROR: no kernel trace CSV inputs")
    if len(paths) != len(set(paths)):
        raise SystemExit(f"ERROR: duplicate kernel trace CSV inputs: {paths}")

    groups = {}
    raw_rows = invalid_count = total_dispatches = excluded_count = clipped_count = 0
    invalid_correlation_ids = []
    input_audits = []
    for kernel_trace_csv in paths:
        input_audit = {
            "path": kernel_trace_csv,
            "raw_row_count": 0,
            "dispatch_row_count": 0,
            "invalid_event_count": 0,
            "window_excluded_events": 0,
            "window_clipped_events": 0,
            "included_event_count": 0,
        }
        with open(kernel_trace_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            name_col = "Kernel_Name" if "Kernel_Name" in fieldnames else None
            start_col = "Start_Timestamp" if "Start_Timestamp" in fieldnames else None
            end_col = "End_Timestamp" if "End_Timestamp" in fieldnames else None
            if not (name_col and start_col and end_col):
                raise SystemExit(
                    f"ERROR: {kernel_trace_csv} does not look like a rocprofv3 "
                    f"kernel_trace.csv (need Kernel_Name/Start_Timestamp/End_Timestamp, "
                    f"got columns: {fieldnames})"
                )
            for row in reader:
                raw_rows += 1
                input_audit["raw_row_count"] += 1
                if row.get("Kind") and row["Kind"] != "KERNEL_DISPATCH":
                    continue
                total_dispatches += 1
                input_audit["dispatch_row_count"] += 1
                correlation_id = (
                    row.get("Correlation_Id") or row.get("Correlation_ID")
                    or row.get("Dispatch_Id") or row.get("Dispatch_ID") or ""
                )
                try:
                    start_ns = int(row[start_col])
                    end_ns = int(row[end_col])
                except (TypeError, ValueError):
                    invalid_count += 1
                    input_audit["invalid_event_count"] += 1
                    invalid_correlation_ids.append(correlation_id)
                    continue
                dur_ns = end_ns - start_ns
                if end_ns < start_ns or dur_ns > INT64_MAX:
                    invalid_count += 1
                    input_audit["invalid_event_count"] += 1
                    invalid_correlation_ids.append(correlation_id)
                    continue
                if window is not None:
                    clipped_start = max(start_ns, int(window[0]))
                    clipped_end = min(end_ns, int(window[1]))
                    if clipped_end <= clipped_start:
                        excluded_count += 1
                        input_audit["window_excluded_events"] += 1
                        continue
                    if clipped_start != start_ns or clipped_end != end_ns:
                        clipped_count += 1
                        input_audit["window_clipped_events"] += 1
                    dur_ns = clipped_end - clipped_start
                name = row[name_col] or ""
                if name not in groups:
                    groups[name] = {
                        "durations": [],
                        "resources": {
                            output: row.get(source, "")
                            for output, source in _KERNEL_RESOURCE_FIELDS.items()
                        },
                    }
                groups[name]["durations"].append(dur_ns)
                input_audit["included_event_count"] += 1
        input_audits.append(input_audit)

    max_count, max_fraction = _invalid_event_limits()
    invalid_fraction = invalid_count / total_dispatches if total_dispatches else 0.0
    if invalid_count > max_count or invalid_fraction > max_fraction:
        raise TraceSanitizationError(
            f"pooled-summary invalid-event guard rejected {invalid_count}/{total_dispatches} events"
        )
    return groups, {
        "inputs": input_audits,
        "pooled_raw_row_count": raw_rows,
        "pooled_dispatch_row_count": total_dispatches,
        "pooled_included_row_count": sum(
            row["included_event_count"] for row in input_audits),
        "dispatches_before": total_dispatches,
        "invalid_event_count": invalid_count,
        "invalid_event_fraction": invalid_fraction,
        "invalid_correlation_ids": invalid_correlation_ids,
        "window_excluded_events": excluded_count,
        "window_clipped_events": clipped_count,
    }


def _build_summary_rows(groups, trim_pct):
    summary_rows = []
    grand_total_ns = 0
    for name, g in groups.items():
        durations = g["durations"]
        count_before = len(durations)
        if count_before >= 20 and trim_pct > 0:
            n_trim = _n_to_trim(count_before, trim_pct)
            n_trim = min(n_trim, count_before - 1)  # Never drop every call.
            kept = sorted(durations)[: count_before - n_trim]
        else:
            n_trim = 0
            kept = durations

        total = sum(kept)
        count = len(kept)
        avg = total / count
        med = statistics.median(kept)
        mn = min(kept)
        mx = max(kept)
        stddev = statistics.pstdev(kept) if count > 1 else 0.0

        grand_total_ns += total
        summary_rows.append(g["resources"] | {
            "Name": name,
            "Instances": count,
            "instances_before_trim": count_before,
            "n_trimmed": n_trim,
            "Total Time (ns)": total,
            "Avg (ns)": avg,
            "Med (ns)": med,
            "Min (ns)": mn,
            "Max (ns)": mx,
            "StdDev (ns)": stddev,
        })

    summary_rows.sort(key=lambda x: x["Total Time (ns)"], reverse=True)
    for r in summary_rows:
        r["Time %"] = (100.0 * r["Total Time (ns)"] / grand_total_ns) if grand_total_ns else 0.0
    return summary_rows, grand_total_ns


def build_pooled_kernel_summaries(kernel_trace_csvs, trim_pct, window=None):
    """Build canonical untrimmed and post-pooling-trimmed node summaries in one scan."""
    groups, audit = _scan_kernel_traces(kernel_trace_csvs, window)
    normalized_rows, normalized_total_ns = _build_summary_rows(groups, 0)
    trimmed_rows, trimmed_total_ns = _build_summary_rows(groups, trim_pct)
    return (
        normalized_rows, normalized_total_ns,
        trimmed_rows, trimmed_total_ns, audit,
    )


def build_trimmed_summary(kernel_trace_csv, trim_pct, window=None):
    'Build trimmed rows and per-kernel trim metadata from one or more worker CSVs.'
    groups, audit = _scan_kernel_traces(kernel_trace_csv, window)
    rows, grand_total_ns = _build_summary_rows(groups, trim_pct)
    return rows, grand_total_ns, audit


def write_csv(rows, out_path, add_category, categorize_kernel, normalized=False):
    columns = list(OUTPUT_COLUMNS[:-2] if normalized else OUTPUT_COLUMNS)
    if normalized:
        columns += _NORMALIZED_HELP_COLUMNS
    if add_category and categorize_kernel is not None:
        columns = ["Category"] + columns
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            grid_xyz = f"{r['GridX']} {r['GridY']} {r['GridZ']}"
            block_xyz = f"{r['BlockX']} {r['BlockY']} {r['BlockZ']}"
            out = [
                f"{r['Time %']:.1f}%",
                pretty_ns(r["Total Time (ns)"]),
                r["Instances"],
                pretty_ns(r["Avg (ns)"]),
                pretty_ns(r["Med (ns)"]),
                pretty_ns(r["Min (ns)"]),
                pretty_ns(r["Max (ns)"]),
                pretty_ns(r["StdDev (ns)"]),
                grid_xyz,
                block_xyz,
                r["VGPR"], r["AccumVGPR"], r["SGPR"], r["LDS"], r["Scratch"],
                r["Name"],
                r["Time %"],
                r["Total Time (ns)"], r["Avg (ns)"], r["Med (ns)"], r["Min (ns)"],
                r["Max (ns)"], r["StdDev (ns)"],
                r["GridX"], r["GridY"], r["GridZ"],
                r["BlockX"], r["BlockY"], r["BlockZ"],
                r["n_trimmed"], r["instances_before_trim"],
            ]
            if normalized:
                out = out[:-2] + [
                    r["Name"], f"{r['Total Time (ns)'] / 1000.0:.4f}", r["Instances"],
                ]
            if add_category and categorize_kernel is not None:
                out = [categorize_kernel(r["Name"])] + out
            w.writerow(out)


def _add_combine_parser(subparsers):
    p = subparsers.add_parser(
        "combine",
        help="stream rocprofv3 CSV shards into combined Chrome JSON traces",
        description=(
            "Stream rocprofv3 CSV shards into combined Chrome JSON traces; every node is "
            "mapped to shared wall time before one global display origin is subtracted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("jobdir", help="job output dir, e.g. /shared_inference/aarai/my_prof_run/<jobid>")
    p.add_argument("--out-dir", default=None,
                   help="where to write combined_all.pftrace / combined_rank0.pftrace / combined_prefill.chrome.json (default: <jobdir> itself)")
    p.add_argument(
        "--clock-manifest", default=None,
        help=(
            "validated schema-v1 cross-node clock manifest (default: auto-discover "
            "clock_NODE*.jsonl or native PFTrace snapshots)"
        ),
    )
    p.add_argument("--only", choices=["all", "rank0", "prefill", "both"], default="both",
                   help="which trace(s) to build; both includes prefill-only (default: both)")
    p.set_defaults(func=_run_combine)


def _run_combine(a):
    try:
        return _run_combine_impl(a)
    except (ClockAlignmentError, RankManifestError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def _run_combine_impl(a):
    jobdir = os.path.abspath(a.jobdir)
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else jobdir
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "rank_manifest.json")
    manifest = write_rank_manifest(jobdir, manifest_path)
    log(f"validated PID/rank manifest -> {manifest_path}")
    clock_manifest = os.path.abspath(a.clock_manifest) if a.clock_manifest else None
    here = os.path.dirname(os.path.abspath(__file__))
    traceconv = os.environ.get(
        "TRACECONV", os.path.join(here, "external_copies", "traceconv_bin", "traceconv"))

    if a.only in ("all", "both"):
        log(f"building combined_all.pftrace (ALL ranks, ALL nodes) from {jobdir}")
        build_combined(
            jobdir, os.path.join(out_dir, "combined_all.pftrace"), False, manifest,
            clock_manifest_path=clock_manifest, traceconv=traceconv,
        )
    if a.only in ("rank0", "both"):
        log(f"building combined_rank0.pftrace (local_rank=0 per node) from {jobdir}")
        build_combined(
            jobdir, os.path.join(out_dir, "combined_rank0.pftrace"), True, manifest,
            clock_manifest_path=clock_manifest, traceconv=traceconv,
        )
    if a.only in ("prefill", "both"):
        log(f"building combined_prefill.chrome.json (ALL prefill ranks, ALL prefill nodes) from {jobdir}")
        build_combined(
            jobdir, os.path.join(out_dir, "combined_prefill.chrome.json"),
            False, manifest, role_filter="prefill",
            clock_manifest_path=clock_manifest, traceconv=traceconv,
        )


def _add_extract_reqid_parser(subparsers):
    p = subparsers.add_parser(
        "extract-reqid",
        help="extract request-to-MoRI write UID mappings from worker logs",
        description="Extract request-to-MoRI write UID mappings from worker logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("logs", nargs="+", help="vLLM worker log file(s) to parse")
    p.add_argument("-o", "--out", default=None,
                   help="output CSV path (default: stdout)")
    p.set_defaults(func=_run_extract_reqid)


def _run_extract_reqid(args):
    rows = parse_reqid_maps(args.logs)
    fields = ["write_uid", "direction", "request_id", "transfer_id", "layer",
              "role", "node_rank", "pid"]
    out = (open(args.out, "w", newline="", encoding="utf-8")
           if args.out else sys.stdout)
    try:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    finally:
        if args.out:
            out.close()

    print(
        f"[trace_tools extract-reqid] parsed {len(rows)} mapping line(s) "
        f"from {len(args.logs)} file(s)"
        + (f" -> {args.out}" if args.out else ""),
        file=sys.stderr,
    )


def _add_buckets_parser(subparsers):
    p = subparsers.add_parser(
        "buckets",
        help="apply first-party kernel categories to TraceLens summaries",
        description="Add first-party kernel bucket categories to a TraceLens kernel-summary CSV.",
    )
    p.add_argument('--in', dest='in_path', required=True,
                   help='Input TraceLens kernel_summary CSV.')
    p.add_argument('--out-per-kernel', required=True,
                   help='Output per-kernel CSV (Category + all original columns).')
    p.add_argument('--out-by-category', required=True,
                   help='Output by-category rollup CSV.')
    p.add_argument('--label', default=None,
                   help='Label for the stdout summary banner (default: input path).')
    p.set_defaults(func=_run_buckets)


def _run_buckets(args):
    label = args.label if args.label is not None else args.in_path
    process(args.in_path, args.out_per_kernel, args.out_by_category, label)


def _add_trimmed_summary_parser(subparsers):
    p = subparsers.add_parser(
        "trimmed-summary",
        help="build per-kernel summaries after dropping the slowest eligible calls",
        description="Build per-kernel summaries after dropping the slowest eligible calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kernel-trace", required=True, help="Raw rocprofv3 *_kernel_trace.csv (per-dispatch rows)")
    p.add_argument("--out", default=None, help="Output CSV path (default: <stem>_summary_trimmed.csv next to input)")
    p.add_argument("--trim-pct", type=float, default=5.0, help="Percent of each eligible kernel's slowest calls to drop (default: 5)")
    p.add_argument("--add-category", action="store_true", help="Prepend a Category column via the shared first-party categorizer")
    p.set_defaults(func=_run_trimmed_summary)
    return p


def run_trimmed_summary(kernel_trace, out_path, trim_pct, add_category, window=None):
    'Shared trimmed-summary implementation used by both the CLI subcommand and `analyze`.'
    if not os.path.isfile(kernel_trace):
        raise SystemExit(f"ERROR: not found: {kernel_trace}")
    if trim_pct < 0 or trim_pct >= 100:
        raise SystemExit(f"ERROR: --trim-pct must be in [0, 100), got {trim_pct}")

    classifier = categorize_kernel if add_category else None

    rows, grand_total_ns, audit = build_trimmed_summary(kernel_trace, trim_pct, window)
    write_csv(rows, out_path, add_category, classifier)

    n_eligible = sum(1 for r in rows if r["instances_before_trim"] >= 20)
    n_trimmed_kernels = sum(1 for r in rows if r["n_trimmed"] > 0)
    total_calls_dropped = sum(r["n_trimmed"] for r in rows)
    print(f"wrote {out_path}")
    print(f"  {len(rows)} distinct kernels; {n_eligible} eligible (>=20 calls); "
          f"{n_trimmed_kernels} trimmed at {trim_pct}%; {total_calls_dropped} total calls dropped")
    print(f"  trimmed grand total: {pretty_ns(grand_total_ns)}")
    print(
        f"  malformed events discarded: {audit['invalid_event_count']} "
        f"of {audit['dispatches_before']}; window excluded/clipped: "
        f"{audit['window_excluded_events']}/{audit['window_clipped_events']}"
    )


def run_pooled_kernel_summaries(kernel_workers, normalized_path, trimmed_path,
                                trim_pct, window=None):
    """Write canonical all-local-worker summaries and return JSON-safe provenance."""
    if trim_pct < 0 or trim_pct >= 100:
        raise SystemExit(f"ERROR: --trim-pct must be in [0, 100), got {trim_pct}")
    paths = [worker["kernel_csv"] for worker in kernel_workers]
    normalized, normalized_total, trimmed, trimmed_total, audit = (
        build_pooled_kernel_summaries(paths, trim_pct, window)
    )
    write_csv(normalized, normalized_path, False, None, normalized=True)
    write_csv(trimmed, trimmed_path, True, categorize_kernel)

    provenance = []
    for worker, input_audit in zip(kernel_workers, audit["inputs"]):
        if os.path.abspath(worker["kernel_csv"]) != input_audit["path"]:
            raise RankManifestError("pooled kernel input order diverged from manifest local-rank order")
        provenance.append({
            key: worker[key] for key in (
                "role", "node_rank", "hostname", "pid", "world_size",
                "global_rank", "local_rank", "filename", "kernel_csv",
            )
        } | {
            key: input_audit[key] for key in (
                "raw_row_count", "dispatch_row_count", "invalid_event_count",
                "window_excluded_events", "window_clipped_events", "included_event_count",
            )
        })
    audit = dict(audit)
    audit.update({
        "analysis_scope": "all_local_workers_per_node",
        "expected_local_world_size": len(kernel_workers),
        "selected_local_ranks": [worker["local_rank"] for worker in kernel_workers],
        "selected_pids": [worker["pid"] for worker in kernel_workers],
        "workers": provenance,
        "normalized_distinct_kernel_count": len(normalized),
        "normalized_total_ns": normalized_total,
        "trimmed_distinct_kernel_count": len(trimmed),
        "trimmed_total_ns": trimmed_total,
        "trimmed_calls_dropped": sum(row["n_trimmed"] for row in trimmed),
    })

    print("kernel_analysis_scope=all_local_workers_per_node")
    print(f"expected_local_world_size={len(kernel_workers)}")
    print("selected_local_ranks=" + ",".join(str(rank) for rank in audit["selected_local_ranks"]))
    print("selected_pids=" + ",".join(str(pid) for pid in audit["selected_pids"]))
    for worker in provenance:
        print(
            f"pooled_input local_rank={worker['local_rank']} pid={worker['pid']} "
            f"file={worker['filename']} raw_rows={worker['raw_row_count']} "
            f"dispatch_rows={worker['dispatch_row_count']} "
            f"included_rows={worker['included_event_count']}"
        )
    print(
        f"pooled_total raw_rows={audit['pooled_raw_row_count']} "
        f"dispatch_rows={audit['pooled_dispatch_row_count']} "
        f"included_rows={audit['pooled_included_row_count']}"
    )
    print(
        f"normalized distinct_kernels={len(normalized)} total_ns={normalized_total} "
        f"-> {normalized_path}"
    )
    print(
        f"trimmed distinct_kernels={len(trimmed)} total_ns={trimmed_total} "
        f"calls_dropped={audit['trimmed_calls_dropped']} -> {trimmed_path}"
    )
    print(
        f"malformed events discarded: {audit['invalid_event_count']} "
        f"of {audit['dispatches_before']}; window excluded/clipped: "
        f"{audit['window_excluded_events']}/{audit['window_clipped_events']}"
    )
    return audit


def _run_trimmed_summary(args):
    out_path = args.out
    if out_path is None:
        base = os.path.basename(args.kernel_trace)
        stem = base[: -len("_kernel_trace.csv")] if base.endswith("_kernel_trace.csv") else os.path.splitext(base)[0]
        out_path = os.path.join(os.path.dirname(args.kernel_trace) or ".", f"{stem}_kernel_summary_trimmed.csv")
    run_trimmed_summary(args.kernel_trace, out_path, args.trim_pct, args.add_category)


def _date_u():
    'Match the exact banner formatting of the shell `date -u` command.'
    try:
        return subprocess.run(["date", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return time.strftime("%a %b %e %H:%M:%S UTC %Y", time.gmtime())


def _tail_lines(text, n):
    'Return the last n lines of text, mirroring the shell `tail -N` helper the old script relied on.'
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _print_tail(text, n):
    tail = _tail_lines(text, n)
    if tail:
        print(tail)


def _call_captured(func, *args, **kwargs):
    '''Call func with stdout/stderr captured (like a subprocess with 2>&1), returning
    (exit_code, captured_text). SystemExit / exceptions are converted to a nonzero
    exit code instead of propagating, matching how the old script inspected $? after
    shelling back out to trace_tools.py for buckets/trimmed-summary.'''
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    exit_code = 0
    try:
        sys.stdout = buf
        sys.stderr = buf
        func(*args, **kwargs)
    except SystemExit as e:
        code = e.code
        if code is None:
            exit_code = 0
        elif isinstance(code, int):
            exit_code = code
        else:
            print(str(code))
            exit_code = 1
    except Exception as e:
        print(f"ERROR: {e}")
        exit_code = 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return exit_code, buf.getvalue()


def _find_first(dirpath, pattern):
    'find "$dirpath" -maxdepth 1 -name pattern | head -1, deterministically sorted.'
    matches = sorted(glob.glob(os.path.join(dirpath, pattern)))
    return matches[0] if matches else None


def _find_iname_first(dirpath, pattern):
    'find "$dirpath" -maxdepth 1 -iname pattern | head -1, deterministically sorted.'
    if not os.path.isdir(dirpath):
        return None
    rx = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    matches = sorted(name for name in os.listdir(dirpath) if rx.match(name))
    return os.path.join(dirpath, matches[0]) if matches else None


def _ls_lh(path):
    'Print `ls -lh path` output; return False (after printing path-appropriate stderr) if it fails.'
    proc = subprocess.run(["ls", "-lh", path], capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return False
    return True


def _print_output_listing(dirs):
    'Equivalent of `ls -la "$TL_DIR" "$TL_DIR/out_csvs" "$BK_DIR" 2>/dev/null`.'
    proc = subprocess.run(["ls", "-la"] + list(dirs), capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)


INT64_MAX = (1 << 63) - 1


class TraceSanitizationError(ValueError):
    pass


def _invalid_event_limits():
    # The known incident had 24 malformed dispatches in a multi-hundred-thousand
    # event trace. Both guards must pass; overrides are explicit environment knobs.
    max_count = int(os.environ.get("INVALID_EVENT_MAX_COUNT", "100"))
    max_fraction = float(os.environ.get("INVALID_EVENT_MAX_FRACTION", "0.0001"))
    if max_count < 0 or not 0 <= max_fraction <= 1:
        raise TraceSanitizationError(
            f"invalid sanitizer thresholds: count={max_count}, fraction={max_fraction}"
        )
    return max_count, max_fraction


def _write_json_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _pair_key(event):
    return (
        event.get("cat"), event.get("name"), event.get("pid"), event.get("tid"),
        json.dumps(event.get("id2"), sort_keys=True),
    )


def _parse_trace_event_line(line, path, lineno):
    stripped = line.strip()
    if stripped.endswith(","):
        stripped = stripped[:-1]
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TraceSanitizationError(f"invalid event JSON at {path}:{lineno}: {exc}") from exc
    if not isinstance(event, dict):
        raise TraceSanitizationError(f"non-object trace event at {path}:{lineno}")
    return event, stripped


def _stream_traceconv_events(src, path):
    """Yield (event, original_object_text), then (None, top-level trailer)."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    max_record_bytes = 16 * 1024 * 1024
    while True:
        buffer = buffer.lstrip()
        while buffer.startswith(","):
            buffer = buffer[1:].lstrip()
        if buffer.startswith("]"):
            yield None, buffer + src.read()
            return
        if not buffer and eof:
            raise TraceSanitizationError(f"unexpected EOF before traceEvents trailer in {path}")
        try:
            event, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError as exc:
            if eof:
                raise TraceSanitizationError(f"invalid traceconv JSON in {path}: {exc}") from exc
            chunk = src.readline()
            if chunk:
                buffer += chunk
                if len(buffer.encode("utf-8")) > max_record_bytes:
                    raise TraceSanitizationError(
                        f"trace record exceeds {max_record_bytes} bytes or is malformed in {path}"
                    )
            else:
                eof = True
            continue
        if not isinstance(event, dict):
            raise TraceSanitizationError(f"non-object trace event in {path}")
        raw = buffer[:end]
        buffer = buffer[end:]
        yield event, raw


def _invalid_record(domain, begin_ns, end_ns, correlation_id, event=None):
    delta = end_ns - begin_ns
    reasons = []
    if end_ns < begin_ns:
        reasons.append("end_before_begin")
    if delta > INT64_MAX or delta < 0 and delta % (1 << 64) > INT64_MAX:
        reasons.append("unsigned_duration_exceeds_int64")
    return {
        "domain": domain,
        "correlation_id": correlation_id,
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "signed_duration_ns": delta,
        "unsigned_duration_ns": delta % (1 << 64),
        "reasons": reasons,
        "name": event.get("name") if event else None,
        "pid": event.get("pid") if event else None,
        "tid": event.get("tid") if event else None,
    }


def _guard_invalid(report, report_path):
    count = report["invalid_event_count"]
    denominator = report["duration_events_before"]
    fraction = count / denominator if denominator else 0.0
    max_count, max_fraction = _invalid_event_limits()
    report["invalid_event_fraction"] = fraction
    report["guard"] = {
        "max_count": max_count,
        "max_fraction": max_fraction,
        "passed": count <= max_count and fraction <= max_fraction,
        "override_environment": ["INVALID_EVENT_MAX_COUNT", "INVALID_EVENT_MAX_FRACTION"],
    }
    if not report["guard"]["passed"]:
        report["status"] = "rejected_by_guard"
        _write_json_atomic(report_path, report)
        raise TraceSanitizationError(
            f"invalid-event guard rejected {count}/{denominator} events "
            f"({fraction:.6%}); limits are {max_count} and {max_fraction:.6%}"
        )


def sanitize_perfetto_json(in_path, out_path, report_path, window=None):
    """Stream traceconv JSON, removing only malformed duration-bearing begin records.

    traceconv globally sorts phase records and omits correlation IDs from ``ph=e``
    records, so an end record cannot be uniquely paired after conversion. TraceLens
    consumes only the duration-bearing ``ph=b`` record (the one with ``agent`` and
    begin/end/delta fields); retaining phase-end records preserves every record that
    cannot be proven malformed while removing exactly the bad TraceLens input.
    """
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise TraceSanitizationError("sanitizer input and output must differ")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    invalid = []
    stats = {
        "trace_records_before": 0,
        "trace_records_after": 0,
        "duration_events_before": 0,
        "duration_events_after": 0,
        "duration_ns_before": 0,
        "duration_ns_after": 0,
        "window_excluded_events": 0,
        "window_clipped_events": 0,
    }
    try:
        with open(in_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
            header = src.readline()
            if header.strip() != '{"traceEvents":[':
                raise TraceSanitizationError(
                    f"{in_path} is not line-oriented traceconv JSON (unexpected first line)"
                )
            dst.write(header)
            previous = None

            def emit(raw):
                nonlocal previous
                if previous is not None:
                    dst.write(previous + ",\n")
                previous = raw

            trailer = None
            for event, raw in _stream_traceconv_events(src, in_path):
                if event is None:
                    trailer = raw
                    break
                stats["trace_records_before"] += 1
                args = event.get("args") or {}
                is_duration = event.get("ph") == "b" and "begin_ns" in args and "end_ns" in args
                if not is_duration:
                    emit(raw)
                    stats["trace_records_after"] += 1
                    continue

                begin_ns = int(args["begin_ns"])
                end_ns = int(args["end_ns"])
                delta_ns = end_ns - begin_ns
                stats["duration_events_before"] += 1
                if end_ns < begin_ns or delta_ns > INT64_MAX:
                    invalid.append(_invalid_record(
                        event.get("cat", "unknown"), begin_ns, end_ns,
                        args.get("corr_id", (event.get("id2") or {}).get("local")), event,
                    ))
                    continue
                stats["duration_ns_before"] += delta_ns

                clipped_begin = begin_ns
                clipped_end = end_ns
                if window is not None:
                    clipped_begin = max(begin_ns, int(window[0]))
                    clipped_end = min(end_ns, int(window[1]))
                    if clipped_end <= clipped_begin:
                        stats["window_excluded_events"] += 1
                        continue
                    if clipped_begin != begin_ns or clipped_end != end_ns:
                        stats["window_clipped_events"] += 1
                        event_args = dict(args)
                        event_args["begin_ns"] = clipped_begin
                        event_args["end_ns"] = clipped_end
                        event_args["delta_ns"] = clipped_end - clipped_begin
                        event = dict(event)
                        event["args"] = event_args
                        event["ts"] = clipped_begin // 1000
                        raw = json.dumps(event, separators=(",", ":"), sort_keys=True)
                emit(raw)
                stats["trace_records_after"] += 1
                stats["duration_events_after"] += 1
                stats["duration_ns_after"] += clipped_end - clipped_begin

            if trailer is None:
                raise TraceSanitizationError(f"traceEvents trailer missing in {in_path}")
            if previous is not None:
                dst.write(previous + "\n")
            dst.write(trailer)
            dst.flush()
            os.fsync(dst.fileno())

        report = {
            "schema_version": 1,
            "status": "sanitized",
            "input": os.path.abspath(in_path),
            "output": os.path.abspath(out_path),
            "input_bytes": os.path.getsize(in_path),
            "output_bytes": os.path.getsize(tmp),
            "window_monotonic_ns": list(window) if window is not None else None,
            "invalid_event_count": len(invalid),
            "discarded_trace_records": len(invalid),
            "phase_end_records_retained": True,
            "phase_end_rationale": (
                "traceconv end records have no correlation ID and are globally timestamp-sorted; "
                "TraceLens consumes the duration-bearing begin record"
            ),
            "invalid_events": invalid,
            **stats,
        }
        _guard_invalid(report, report_path)
        os.replace(tmp, out_path)
        report["status"] = "ok"
        report["output_bytes"] = os.path.getsize(out_path)
        _write_json_atomic(report_path, report)
        return report
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)



def sanitize_native_rocprof_data(data, source_path, report_path, window=None):
    try:
        tool_data = data["rocprofiler-sdk-tool"][0]
        buffer_records = tool_data["buffer_records"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TraceSanitizationError(f"invalid rocprofv3 JSON structure in {source_path}") from exc
    invalid = []
    duration_before = duration_after = duration_ns_before = duration_ns_after = 0
    excluded = clipped = 0
    for domain, records in list(buffer_records.items()):
        if not isinstance(records, list):
            continue
        kept = []
        for record in records:
            if not isinstance(record, dict) or "start_timestamp" not in record or "end_timestamp" not in record:
                kept.append(record)
                continue
            begin_ns = int(record["start_timestamp"])
            end_ns = int(record["end_timestamp"])
            delta_ns = end_ns - begin_ns
            duration_before += 1
            correlation = record.get("correlation_id") or record.get("dispatch_info", {}).get("dispatch_id")
            if end_ns < begin_ns or delta_ns > INT64_MAX:
                invalid.append(_invalid_record(domain, begin_ns, end_ns, correlation))
                continue
            duration_ns_before += delta_ns
            clipped_begin = begin_ns
            clipped_end = end_ns
            if window is not None:
                clipped_begin = max(begin_ns, int(window[0]))
                clipped_end = min(end_ns, int(window[1]))
                if clipped_end <= clipped_begin:
                    excluded += 1
                    continue
                if clipped_begin != begin_ns or clipped_end != end_ns:
                    clipped += 1
                    record = dict(record)
                    record["start_timestamp"] = clipped_begin
                    record["end_timestamp"] = clipped_end
            kept.append(record)
            duration_after += 1
            duration_ns_after += clipped_end - clipped_begin
        buffer_records[domain] = kept
    report = {
        "schema_version": 1,
        "status": "sanitized_in_memory",
        "input": os.path.abspath(source_path),
        "output": "TraceLens in-memory rocprof data",
        "input_bytes": os.path.getsize(source_path),
        "window_monotonic_ns": list(window) if window is not None else None,
        "invalid_event_count": len(invalid),
        "discarded_trace_records": len(invalid),
        "invalid_events": invalid,
        "duration_events_before": duration_before,
        "duration_events_after": duration_after,
        "duration_ns_before": duration_ns_before,
        "duration_ns_after": duration_ns_after,
        "window_excluded_events": excluded,
        "window_clipped_events": clipped,
    }
    _guard_invalid(report, report_path)
    report["status"] = "ok"
    _write_json_atomic(report_path, report)
    return report


def _json_trace_kind(path):
    with open(path, "rb") as f:
        prefix = f.read(4096).lstrip()
    if prefix.startswith(b'{"traceEvents"'):
        return "perfetto_json"
    if prefix.startswith(b'{"rocprofiler-sdk-tool"'):
        return "rocprof_json"
    return None


def _detect_trace(trace, rank_manifest_path=None):
    """Resolve one exact analysis input without PID/filename-order fallback."""
    trace = os.path.abspath(trace)
    if os.path.isdir(trace):
        match = _NODE_DIR_RE.fullmatch(os.path.basename(trace))
        if not match:
            print(f"ERROR: trace directory name must match rocprof_(prefill|decode)_NODE<N>: {trace}")
            return None
        if not rank_manifest_path:
            print("ERROR: a rank manifest is required when analyzing a rocprof node directory")
            return None
        role, node_rank = match.group(1), int(match.group(2))
        jobdir = os.path.dirname(trace)
        manifest = load_rank_manifest(rank_manifest_path, jobdir)
        workers = selected_workers(manifest, role, node_rank)
        selected = workers[0]
        kernel_workers = []
        for worker in workers:
            kernel_csv = os.path.join(jobdir, worker["artifacts"]["kernel_csv"])
            if not os.path.isfile(kernel_csv):
                print(
                    f"ERROR: worker local_rank={worker['local_rank']} PID {worker['pid']} "
                    f"has no kernel CSV: {kernel_csv}"
                )
                return None
            kernel_workers.append({
                key: worker[key] for key in (
                    "role", "node_rank", "hostname", "pid", "world_size",
                    "global_rank", "local_rank",
                )
            } | {
                "kernel_csv": kernel_csv,
                "filename": os.path.basename(kernel_csv),
            })
        artifacts = {
            key: (os.path.join(jobdir, value) if value else None)
            for key, value in selected["artifacts"].items()
        }
        kernel_csv = artifacts.get("kernel_csv")
        if artifacts.get("pftrace") and os.path.isfile(artifacts["pftrace"]):
            primary_kind = "pftrace"
            primary = artifacts["pftrace"]
        elif artifacts.get("rocprof_json") and os.path.isfile(artifacts["rocprof_json"]):
            primary_kind = "rocprof_json"
            primary = artifacts["rocprof_json"]
        else:
            print(
                f"ERROR: selected PID {selected['pid']} has neither native PFTrace nor rocprof JSON; "
                "raw CSV is not a TraceLens fallback"
            )
            return None
        return {
            "kind": primary_kind,
            "path": primary,
            "kernel_csv": kernel_csv,
            "jobdir": jobdir,
            "role": role,
            "node_rank": node_rank,
            "selection": selected,
            "rank_manifest": os.path.abspath(rank_manifest_path),
            "kernel_workers": kernel_workers,
            "kernel_analysis_scope": "all_local_workers_per_node",
        }

    if not os.path.isfile(trace):
        print(f"ERROR: trace not found: {trace}")
        return None
    if trace.endswith(".pftrace"):
        kind = "pftrace"
    elif trace.endswith(".json") or trace.endswith(".json.gz"):
        if trace.endswith(".json.gz"):
            print("ERROR: compressed JSON is not supported by the strict sanitizer")
            return None
        kind = _json_trace_kind(trace)
        if not kind:
            print(f"ERROR: unrecognized JSON trace schema: {trace}")
            return None
    elif trace.endswith("_kernel_trace.csv"):
        print("ERROR: raw kernel CSV is not accepted as a TraceLens analysis input")
        return None
    else:
        print(f"ERROR: '{trace}' is not a recognized trace file/dir")
        return None
    stem = trace[:-len("_results.pftrace")] if trace.endswith("_results.pftrace") else None
    kernel_csv = stem + "_kernel_trace.csv" if stem and os.path.isfile(stem + "_kernel_trace.csv") else None
    return {
        "kind": kind,
        "path": trace,
        "kernel_csv": kernel_csv,
        "jobdir": os.path.dirname(trace),
        "role": None,
        "node_rank": None,
        "selection": None,
        "rank_manifest": None,
        "kernel_workers": None,
        "kernel_analysis_scope": "single_trace",
    }



def _normalize_kernel_summary(src, out):
    '''Reproduce the analyze_kernels.sh inline Python heredoc normalizer exactly. Prints its own
    success/error message (matching the old unredirected heredoc) and returns an exit-code-like
    int (0 success, 1 failure) instead of calling sys.exit, since a failure here must not abort
    the rest of `analyze`.'''
    if not src or not os.path.isfile(src):
        print(f"ERROR: normalizer input CSV not found: {src}")
        return 1
    with open(src, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        print("ERROR: normalizer input CSV is empty: %s" % src)
        return 1
    hdr = rows[0]
    low = [(h or '').strip().lower() for h in hdr]

    def col(*cands):
        for c in cands:
            if c in low:
                return low.index(c)
        return None

    name_i  = col('name', 'kernel name', 'kernel_name')
    ns_i    = col('total time (ns)')
    inst_i  = col('instances', 'total count')
    start_i = col('start_timestamp')
    end_i   = col('end_timestamp')
    native_total_i = col('total kernel time (ms)')
    native_count_i = col('count')

    HELP = ['Kernel name', 'kernel_duration_us_sum', 'kernel_duration_us_count']

    if start_i is not None and end_i is not None and name_i is not None:
        agg = {}
        for r in rows[1:]:
            if not r or name_i >= len(r):
                continue
            nm = r[name_i]
            try:
                dur_us = (float(r[end_i]) - float(r[start_i])) / 1000.0
            except (ValueError, IndexError):
                continue
            a = agg.setdefault(nm, [0.0, 0])
            a[0] += dur_us
            a[1] += 1
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['Name', 'Total Time (ns)', 'Instances'] + HELP)
            for nm, (us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
                w.writerow([nm, us * 1000.0, cnt, nm, '%.4f' % us, cnt])
        print("normalized rocprofv3 kernel_trace.csv: %d distinct kernels" % len(agg))
        return 0
    elif name_i is not None and native_total_i is not None and native_count_i is not None:
        # TraceLens's rocprof analyzer currently labels this column as milliseconds,
        # but its implementation stores total_ns / 1000 (microseconds). Normalize
        # from that known producer behavior into the same ns/us columns as PFTrace.
        n = 0
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['Name', 'Total Time (ns)', 'Instances'] + HELP)
            for r in rows[1:]:
                if not r or name_i >= len(r):
                    continue
                try:
                    total_us = float(r[native_total_i])
                    count = int(float(r[native_count_i]))
                except (ValueError, IndexError):
                    continue
                name = r[name_i]
                w.writerow([name, total_us * 1000.0, count, name, total_us, count])
                n += 1
        print("normalized TraceLens rocprof JSON kernel_summary.csv: %d rows" % n)
        return 0
    elif name_i is not None:
        n = 0
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(hdr + HELP)
            for r in rows[1:]:
                if not r or all((c or '').strip() == '' for c in r):
                    continue
                nm = r[name_i] if name_i < len(r) else ''
                us = ''
                if ns_i is not None and ns_i < len(r):
                    try:
                        us = '%.4f' % (float(r[ns_i]) / 1000.0)
                    except ValueError:
                        us = ''
                cnt = ''
                if inst_i is not None and inst_i < len(r):
                    try:
                        cnt = int(float(r[inst_i]))
                    except ValueError:
                        cnt = ''
                w.writerow(r + [nm, us, cnt])
                n += 1
        print("normalized TraceLens kernel_summary.csv: %d rows" % n)
        return 0
    else:
        print("ERROR: unrecognized kernel-summary schema; headers=%r" % hdr)
        return 1


def _run_native_tracelens(profile_json, out_csvs, sanitizer_report, window):
    """Use TraceLens's official rocprof parser/analyzer on a narrowly filtered in-memory trace."""
    from TraceLens.util import RocprofParser
    from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer

    data = RocprofParser.load_rocprof_data(profile_json)
    report = sanitize_native_rocprof_data(data, profile_json, sanitizer_report, window)
    kernel_events = RocprofParser.extract_kernel_events(data)
    memory_events = RocprofParser.extract_memory_events(data)
    api_events = RocprofParser.extract_api_events(data)
    metadata = RocprofParser.get_metadata(data)
    if window is not None:
        metadata["init_time"], metadata["fini_time"] = int(window[0]), int(window[1])
    analyzer = RocprofAnalyzer(kernel_events, memory_events, api_events, metadata)
    frames = {
        "gpu_timeline": analyzer.get_df_gpu_timeline(),
        "kernel_summary": analyzer.get_df_kernel_summary(),
        "kernel_summary_by_category": analyzer.get_df_kernel_summary_by_category(),
    }
    os.makedirs(out_csvs, exist_ok=True)
    for name, frame in frames.items():
        path = os.path.join(out_csvs, f"{name}.csv")
        frame.to_csv(path, index=False)
        print(f"TraceLens native rocprof: {path} ({len(frame)} rows)")
    print(
        f"TraceLens native rocprof sanitized {report['invalid_event_count']} malformed "
        f"event(s); retained {len(kernel_events)} kernel event(s)"
    )


def run_analyze(trace, outdir, label, venv, traceconv, trim_pct,
                rank_manifest_path=None, window_manifest_path=None):
    """Strict single-trace analysis with manifest selection and no CSV fallback."""
    print(f"############ [{label}] START {_date_u()} ############")
    print(f"TRACE={trace}")

    detected = _detect_trace(trace, rank_manifest_path)
    if detected is None:
        return 2
    if os.path.isdir(outdir) and os.listdir(outdir):
        print(f"ERROR: output directory is not empty (refusing stale-output success): {outdir}")
        return 2

    tl_dir = os.path.join(outdir, "tracelens")
    out_csvs = os.path.join(tl_dir, "out_csvs")
    bk_dir = os.path.join(outdir, "buckets")
    os.makedirs(out_csvs, exist_ok=True)
    os.makedirs(bk_dir, exist_ok=True)
    metadata_path = os.path.join(outdir, "analysis_metadata.json")
    kernel_workers = detected.get("kernel_workers")
    node_aggregate = bool(kernel_workers)
    tracelens_scope = "supplementary_local_rank_0" if node_aggregate else "primary_single_trace"
    metadata = {
        "schema_version": 1,
        "status": "running",
        "label": label,
        "primary_input_kind": detected["kind"],
        "primary_input": detected["path"],
        "raw_csv_fallback": False,
        "selection": detected["selection"],
        "rank_manifest": detected["rank_manifest"],
        "window_manifest": os.path.abspath(window_manifest_path) if window_manifest_path else None,
        "tracelens": {
            "scope": tracelens_scope,
            "selection": detected["selection"],
            "primary_input": detected["path"],
        },
        "kernel_analysis": {
            "analysis_scope": detected["kernel_analysis_scope"],
            "canonical_source": (
                "pooled_raw_worker_kernel_csvs" if node_aggregate
                else "tracelens_single_trace_summary"
            ),
            "expected_local_world_size": len(kernel_workers) if node_aggregate else None,
            "selected_local_ranks": (
                [worker["local_rank"] for worker in kernel_workers] if node_aggregate else None
            ),
            "workers": kernel_workers,
        },
    }
    print(f"tracelens_scope={tracelens_scope}")
    print(f"kernel_analysis_scope={detected['kernel_analysis_scope']}")

    def fail(code, message):
        print(f"ERROR: {message}")
        metadata["status"] = "failed"
        metadata["error"] = message
        _write_json_atomic(metadata_path, metadata)
        return code

    if not os.path.isfile(os.path.join(venv, "bin", "activate")):
        return fail(3, f"venv not found at '{venv}'")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = venv
    env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")

    window_manifest = {
        "available": False,
        "analysis_scope": "whole_capture",
        "reason": "no window manifest supplied",
    }
    window = None
    if window_manifest_path:
        if detected["role"] is None:
            return fail(2, "windowed analysis requires a manifest-selected role/node directory")
        try:
            window_manifest, window = load_window_manifest(
                window_manifest_path, detected["role"], detected["node_rank"])
        except (OSError, ValueError) as exc:
            return fail(2, f"invalid window manifest: {exc}")
    metadata["measurement_window"] = {
        "available": bool(window_manifest.get("available")),
        "analysis_scope": window_manifest.get("analysis_scope", "whole_capture"),
        "reason": window_manifest.get("reason"),
        "monotonic_ns": list(window) if window else None,
    }
    print(f"analysis_scope={metadata['measurement_window']['analysis_scope']}")
    if not window:
        print(f"measurement_window_unavailable={metadata['measurement_window']['reason']}")
    _write_json_atomic(metadata_path, metadata)

    sanitizer_report = os.path.join(tl_dir, "sanitizer_report.json")
    kind = detected["kind"]
    primary = detected["path"]
    ks = None
    catsum = None

    if kind in ("pftrace", "perfetto_json"):
        if kind == "pftrace":
            print(
                f"=== [{label}] (a0) {tracelens_scope} traceconv: "
                "pftrace -> Perfetto JSON ==="
            )
            if not _ls_lh(primary):
                return fail(2, "selected PFTrace is missing")
            if not os.path.isfile(traceconv) or not os.access(traceconv, os.X_OK):
                return fail(3, f"traceconv is not executable: {traceconv}")
            base = os.path.basename(primary)
            if base.endswith(".pftrace"):
                base = base[:-len(".pftrace")]
            converted = os.path.join(tl_dir, base + ".traceconv.tmp.json")
            try:
                with open(traceconv, "rb") as f:
                    is_elf = f.read(4) == b"\x7fELF"
            except OSError as exc:
                return fail(3, f"cannot inspect traceconv: {exc}")
            cmd = [traceconv, "json", primary, converted] if is_elf else ["python3", traceconv, "json", primary, converted]
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            except OSError as exc:
                return fail(3, f"traceconv invocation failed: {exc}")
            traceconv_text = (proc.stdout or "") + (proc.stderr or "")
            with open(os.path.join(tl_dir, "traceconv_stdout.txt"), "w", encoding="utf-8") as f:
                f.write(traceconv_text)
            print(f"traceconv_exit={proc.returncode} -> {converted}")
            _print_tail(traceconv_text, 8)
            if proc.returncode != 0 or not (os.path.isfile(converted) and os.path.getsize(converted) > 0):
                return fail(3, f"traceconv failed with status {proc.returncode}")
        else:
            converted = primary

        sanitized = os.path.join(tl_dir, os.path.basename(converted).replace(".traceconv.tmp", "") + ".sanitized.json")
        try:
            report = sanitize_perfetto_json(converted, sanitized, sanitizer_report, window)
        except (OSError, ValueError) as exc:
            return fail(4, f"trace sanitizer failed: {exc}")
        print(
            f"sanitizer_discarded={report['invalid_event_count']} "
            f"window_before={report['duration_events_before']} window_after={report['duration_events_after']}"
        )
        if kind == "pftrace":
            os.unlink(converted)

        print(
            f"=== [{label}] (a) {tracelens_scope} TraceLens Perfetto parsing "
            "and kernel aggregation ==="
        )
        tl_cmd = [
            "TraceLens_generate_perf_report_pftrace_hip_activity",
            "--trace_path", sanitized,
            "--output_xlsx_path", os.path.join(tl_dir, "report_TP0.xlsx"),
            "--output_csvs_dir", out_csvs,
            "--output_md_path", os.path.join(tl_dir, "report.md"),
            "--traceconv", traceconv,
            "--min_event_ns", "0",
            "--write_md",
        ]
        try:
            proc = subprocess.run(tl_cmd, env=env, capture_output=True, text=True)
        except OSError as exc:
            return fail(4, f"TraceLens invocation failed: {exc}")
        tl_text = (proc.stdout or "") + (proc.stderr or "")
        with open(os.path.join(tl_dir, "tracelens_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(tl_text)
        print(f"tracelens_exit={proc.returncode}")
        _print_tail(tl_text, 12)
        if proc.returncode != 0 or "Traceback (most recent call last)" in tl_text or re.search(r"\bERROR\s+-", tl_text):
            return fail(4, f"TraceLens Perfetto analysis failed with status {proc.returncode}")
        ks = _find_iname_first(out_csvs, "kernel_summary*.csv")
        catsum = _find_iname_first(out_csvs, "category_summary*.csv")
        required = [ks, catsum, os.path.join(tl_dir, "report.md")]
        if any(not path or not os.path.isfile(path) or os.path.getsize(path) == 0 for path in required):
            return fail(4, "TraceLens exited without all required Perfetto outputs")
    elif kind == "rocprof_json":
        print(
            f"=== [{label}] (a) {tracelens_scope} TraceLens native rocprof JSON "
            "analysis (PFTrace unavailable) ==="
        )
        exit_code, tl_text = _call_captured(
            _run_native_tracelens, primary, out_csvs, sanitizer_report, window)
        with open(os.path.join(tl_dir, "tracelens_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(tl_text)
        print(f"tracelens_exit={exit_code}")
        _print_tail(tl_text, 12)
        if exit_code != 0:
            return fail(4, f"TraceLens native rocprof analysis failed with status {exit_code}")
        ks = os.path.join(out_csvs, "kernel_summary.csv")
        catsum = os.path.join(out_csvs, "kernel_summary_by_category.csv")
        required = [ks, catsum, os.path.join(out_csvs, "gpu_timeline.csv")]
        if any(not os.path.isfile(path) or os.path.getsize(path) == 0 for path in required):
            return fail(4, "TraceLens native analysis did not produce all required outputs")
    else:
        return fail(2, f"unsupported strict analysis input kind: {kind}")

    print(f"{tracelens_scope}_tracelens_kernel_summary_csv={ks}")
    native_category_name = (
        "supplementary_rank0_tracelens_native_category_summary.csv"
        if node_aggregate else "tracelens_native_category_summary.csv"
    )
    native_category_copy = os.path.join(bk_dir, native_category_name)
    shutil.copyfile(catsum, native_category_copy)
    metadata["tracelens"].update({
        "kernel_summary": ks,
        "native_category_summary": native_category_copy,
        "sanitizer_report": sanitizer_report,
    })

    norm = os.path.join(bk_dir, "kernel_summary_normalized.csv")
    trimmed_path = os.path.join(bk_dir, "kernel_summary_trimmed.csv")
    if node_aggregate:
        print(
            f"=== [{label}] (b) canonical pooled all-local-worker summaries "
            f"(TRIM_PCT={trim_pct:g}%) ==="
        )
        pooled_result = {}

        def build_pooled():
            pooled_result["audit"] = run_pooled_kernel_summaries(
                kernel_workers, norm, trimmed_path, trim_pct, window)

        exit_code, pooled_text = _call_captured(build_pooled)
        with open(os.path.join(bk_dir, "trimmed_summary_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(pooled_text)
        print(f"pooled_summary_exit={exit_code}")
        if pooled_text:
            sys.stdout.write(pooled_text)
        if (exit_code != 0 or "audit" not in pooled_result
                or any(not os.path.isfile(path) or os.path.getsize(path) == 0
                       for path in (norm, trimmed_path))):
            return fail(5, "pooled all-local-worker summaries failed or produced missing outputs")
        metadata["kernel_analysis"].update(pooled_result["audit"])
        metadata["kernel_analysis"].update({
            "normalized_output": norm,
            "trimmed_output": trimmed_path,
        })
    else:
        print(f"=== [{label}] (b) TraceLens kernel-summary normalization ===")
        norm_exit = _normalize_kernel_summary(ks, norm)
        print(f"normalize_exit={norm_exit} -> {norm}")
        if norm_exit != 0 or not os.path.isfile(norm) or os.path.getsize(norm) == 0:
            return fail(5, "kernel summary normalization failed")

    print(f"=== [{label}] (c) first-party category buckets (name-based) ===")
    exit_code, bucket_text = _call_captured(
        process, norm,
        os.path.join(bk_dir, "perkernel_buckets.csv"),
        os.path.join(bk_dir, "bycat_buckets.csv"),
        f"{label} ({'all local workers/node' if node_aggregate else 'TraceLens'})",
    )
    with open(os.path.join(bk_dir, "buckets_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(bucket_text)
    print(f"buckets_exit={exit_code}")
    _print_tail(bucket_text, 12)
    bucket_outputs = [os.path.join(bk_dir, "perkernel_buckets.csv"), os.path.join(bk_dir, "bycat_buckets.csv")]
    if exit_code != 0 or any(not os.path.isfile(path) or os.path.getsize(path) == 0 for path in bucket_outputs):
        return fail(5, "first-party bucketing failed or produced missing outputs")

    if not node_aggregate:
        print(f"=== [{label}] (d) trimmed single-worker kernel summary (TRIM_PCT={trim_pct:g}%) ===")
        trim_src = detected.get("kernel_csv")
        if not trim_src or not os.path.isfile(trim_src):
            return fail(5, "selected worker has no raw kernel CSV for the auxiliary trimmed summary")
        print(f"trim_src={trim_src}")
        exit_code, trim_text = _call_captured(
            run_trimmed_summary, trim_src, trimmed_path, trim_pct, True, window)
        with open(os.path.join(bk_dir, "trimmed_summary_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(trim_text)
        print(f"trimmed_summary_exit={exit_code}")
        _print_tail(trim_text, 8)
        if exit_code != 0 or not os.path.isfile(trimmed_path) or os.path.getsize(trimmed_path) == 0:
            return fail(5, "trimmed summary failed or produced no output")

    expected_outputs = [metadata_path, sanitizer_report, ks, native_category_copy, norm, trimmed_path] + bucket_outputs
    if any(not os.path.isfile(path) or os.path.getsize(path) == 0 for path in expected_outputs[1:]):
        return fail(6, "final completeness check found a missing/empty output")
    metadata["status"] = "success"
    metadata["sanitizer_report"] = sanitizer_report
    metadata["sanitizer_scope"] = tracelens_scope
    metadata["outputs"] = expected_outputs[1:]
    _write_json_atomic(metadata_path, metadata)
    print(f"=== [{label}] OUTPUT LISTING ===")
    _print_output_listing([tl_dir, out_csvs, bk_dir])
    print(f"############ [{label}] DONE {_date_u()} ############")
    return 0



def _add_rank_manifest_parser(subparsers):
    p = subparsers.add_parser("rank-manifest", help="reconstruct and validate PID/rank mappings from worker logs")
    p.add_argument("jobdir")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_run_rank_manifest)


def _run_rank_manifest(args):
    manifest = write_rank_manifest(args.jobdir, args.out)
    for row in manifest["selections"]:
        print(
            f"selected role={row['role']} node_rank={row['node_rank']} pid={row['pid']} "
            f"local_rank={row['local_rank']} global_rank={row['global_rank']} "
            f"dp_rank={row['dp_rank']} ep_rank={row['ep_rank']} host={row['hostname']}"
        )
    print(f"wrote {args.out}")


def _add_window_manifest_parser(subparsers):
    p = subparsers.add_parser("window-manifest", help="map benchmark wall timestamps to each node trace clock")
    p.add_argument("jobdir")
    p.add_argument("--rank-manifest", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_run_window_manifest)


def _run_window_manifest(args):
    ranks = load_rank_manifest(args.rank_manifest, args.jobdir)
    manifest = write_window_manifest(args.jobdir, ranks, args.out)
    print(f"measurement_window_available={str(manifest['available']).lower()}")
    print(f"analysis_scope={manifest['analysis_scope']}")
    if not manifest["available"]:
        print(f"reason={manifest['reason']}")
    print(f"wrote {args.out}")


def _add_sanitize_parser(subparsers):
    p = subparsers.add_parser("sanitize", help="discard only malformed traceconv duration pairs")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--window-start-ns", type=int)
    p.add_argument("--window-end-ns", type=int)
    p.set_defaults(func=_run_sanitize)


def _run_sanitize(args):
    if (args.window_start_ns is None) != (args.window_end_ns is None):
        raise SystemExit("ERROR: both --window-start-ns and --window-end-ns are required")
    window = None if args.window_start_ns is None else (args.window_start_ns, args.window_end_ns)
    report = sanitize_perfetto_json(args.in_path, args.out, args.report, window)
    print(f"discarded_invalid_events={report['invalid_event_count']}")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


def _add_clock_sampler_parser(subparsers):
    p = subparsers.add_parser("clock-sampler", help="record per-node realtime/monotonic clock correlations")
    p.add_argument("--out", required=True)
    p.add_argument("--node-rank", required=True, type=int)
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=lambda args: run_clock_sampler(args.out, args.node_rank, args.interval))


def _add_benchmark_marker_parser(subparsers):
    p = subparsers.add_parser("benchmark-marker", help="record an exact measured benchmark boundary")
    p.add_argument("--out", required=True)
    p.add_argument("--event", required=True, choices=("start", "end"))
    p.add_argument("--step-id", required=True)
    p.add_argument("--iteration", type=int)
    p.add_argument("--isl", type=int)
    p.add_argument("--osl", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--num-prompts", type=int)
    p.add_argument("--return-code", type=int)
    p.set_defaults(func=_run_benchmark_marker)


def _run_benchmark_marker(args):
    fields = {key: getattr(args, key) for key in (
        "iteration", "isl", "osl", "concurrency", "num_prompts", "return_code")}
    row = record_benchmark_marker(args.out, args.event, args.step_id, fields)
    print(json.dumps(row, sort_keys=True))


def _add_analyze_parser(subparsers):
    p = subparsers.add_parser(
        "analyze",
        help="run the single-trace analysis pipeline (traceconv + TraceLens + first-party buckets + trimmed summary)",
        description=(
            "Strict manifest-selected analysis: native PFTrace is converted and narrowly sanitized "
            "before TraceLens; when a capture has no PFTrace, TraceLens's native rocprof JSON path "
            "is used. Raw kernel CSV is never a TraceLens fallback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("trace", help="*.pftrace / *.json file, or a rocprof_*_NODE* dir")
    p.add_argument("outdir", help="new output directory (must be empty)")
    p.add_argument("label", help="short label used in banners and stdout summaries")
    p.add_argument("--rank-manifest", default=None)
    p.add_argument("--window-manifest", default=None)
    p.set_defaults(func=_run_analyze)
    return p


def _run_analyze(args):
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.environ.get("VENV", os.path.join(here, "external_copies", "venv"))
    traceconv = os.environ.get("TRACECONV", os.path.join(here, "external_copies", "traceconv_bin", "traceconv"))
    try:
        trim_pct = float(os.environ.get("TRIM_PCT", "5"))
    except ValueError:
        trim_pct = 5.0
    return run_analyze(
        args.trace, args.outdir, args.label, venv, traceconv, trim_pct,
        args.rank_manifest, args.window_manifest,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subparsers = p.add_subparsers(dest='command', required=True)
    _add_combine_parser(subparsers)
    _add_buckets_parser(subparsers)
    _add_extract_reqid_parser(subparsers)
    _add_rank_manifest_parser(subparsers)
    _add_window_manifest_parser(subparsers)
    _add_sanitize_parser(subparsers)
    _add_clock_sampler_parser(subparsers)
    _add_benchmark_marker_parser(subparsers)
    _add_analyze_parser(subparsers)
    trimmed_parser = _add_trimmed_summary_parser(subparsers)
    effective_argv = sys.argv[1:] if argv is None else argv
    if effective_argv and effective_argv[0] == 'trimmed-summary':
        args = trimmed_parser.parse_args(effective_argv[1:])
    else:
        args = p.parse_args(effective_argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())

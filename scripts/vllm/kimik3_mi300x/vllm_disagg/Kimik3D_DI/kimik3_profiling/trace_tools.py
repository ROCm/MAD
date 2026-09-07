#!/usr/bin/env python3
'''Combine traces, bucket kernels, extract request mappings, build trimmed summaries, and run the single-trace analysis pipeline.'''
import argparse, csv, fnmatch, glob, io, json, math, os, re, shutil, statistics, subprocess, sys, time

csv.field_size_limit(sys.maxsize)


def log(msg):
    print(f"[combine_traces] {msg}", flush=True)


_NODE_DIR_RE = re.compile(r"^rocprof_(prefill|decode)_NODE(\d+)$")
_RANK_RE = re.compile(r"^(?P<host>.+)_(?P<pid>\d+)_kernel_trace\.csv$")
_REQID_LINE_RE = re.compile(
    r"moriio_reqid_map\s+"
    r"dir=(?P<direction>\S+)\s+"
    r"request_id=(?P<request_id>\S+)\s+"
    r"transfer_id=(?P<transfer_id>\S+)\s+"
    r"layer=(?P<layer>\S+)\s+"
    r"write_uid=(?P<write_uid>\S+)"
)


def parse_reqid_maps(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = _REQID_LINE_RE.search(line)
                if match:
                    rows.append(match.groupdict())
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


def discover_ranks(dirpath):
    'Return ranks ordered by PID as a rank proxy.'
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


# First rows approximate each node capture start without rescanning large files.


def _peek_first_start_ts(kernel_csv):
    with open(kernel_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return None
        try:
            si = header.index("Start_Timestamp")
        except ValueError:
            return None
        for row in r:
            try:
                return int(row[si])
            except (IndexError, ValueError):
                continue
    return None


def node_t0(ranks):
    ts = [t for t in (_peek_first_start_ts(k) for _, _, k, _, _ in ranks) if t is not None]
    return min(ts) if ts else 0


def stream_kernel_events(kernel_csv, t0, pid, tid, out_fh, first_written):
    n = 0
    with open(kernel_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return 0, first_written
        try:
            si = header.index("Start_Timestamp")
            ei = header.index("End_Timestamp")
            ni = header.index("Kernel_Name")
        except ValueError:
            return 0, first_written
        for row in r:
            try:
                s = int(row[si]); e = int(row[ei])
            except (IndexError, ValueError):
                continue
            name = row[ni] if ni < len(row) else "kernel"
            ev = {"ph": "X", "name": name, "ts": (s - t0) / 1000.0,
                  "dur": max((e - s) / 1000.0, 0.001), "pid": pid, "tid": tid}
            if first_written:
                out_fh.write(",")
            out_fh.write(json.dumps(ev))
            first_written = True
            n += 1
    return n, first_written


def stream_marker_events(marker_csv, t0, pid, tid, out_fh, first_written):
    n = 0
    with open(marker_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return 0, first_written
        try:
            si = header.index("Start_Timestamp")
            ei = header.index("End_Timestamp")
            fi = header.index("Function")
        except ValueError:
            return 0, first_written
        for row in r:
            try:
                s = int(row[si]); e = int(row[ei])
            except (IndexError, ValueError):
                continue
            name = row[fi] if fi < len(row) else "marker"
            ev = {"ph": "X", "name": name, "ts": (s - t0) / 1000.0,
                  "dur": max((e - s) / 1000.0, 0.001), "pid": pid, "tid": tid}
            if first_written:
                out_fh.write(",")
            out_fh.write(json.dumps(ev))
            first_written = True
            n += 1
    return n, first_written


def build_combined(jobdir, out_path, rank0_only, role_filter=None):
    node_dirs = discover_node_dirs(jobdir)
    if not node_dirs:
        raise SystemExit(f"no rocprof_{{prefill,decode}}_NODE* dirs found under {jobdir}")
    if role_filter:
        node_dirs = [node_dir for node_dir in node_dirs if node_dir[0] == role_filter]
        if not node_dirs:
            log(f"WARNING: no {role_filter} rocprof dirs found under {jobdir}; writing empty trace")

    t_start = time.time()
    pid_counter = 0
    total_kernel_events = 0
    total_marker_events = 0
    summary = []

    with open(out_path, "w") as out_fh:
        out_fh.write('{"traceEvents":[')
        first_written = False
        for role, node_idx, dirpath in node_dirs:
            ranks = discover_ranks(dirpath)
            if not ranks:
                log(f"WARNING: no ranks discovered in {dirpath}, skipping")
                continue
            if rank0_only:
                ranks = ranks[:1]
            t0 = node_t0(ranks)
            for local_idx, (pid_val, host, kcsv, marker_csv, pftrace) in enumerate(ranks):
                out_pid = 1000 + pid_counter
                pid_counter += 1
                proc_name = f"{role.upper()} NODE{node_idx} RANK{local_idx} ({host}:{pid_val})"
                meta = [
                    {"ph": "M", "name": "process_name", "pid": out_pid, "tid": 0,
                     "args": {"name": proc_name}},
                    {"ph": "M", "name": "thread_name", "pid": out_pid, "tid": 1,
                     "args": {"name": f"GPU kernels ({role} NODE{node_idx} rank{local_idx})"}},
                ]
                for m in meta:
                    if first_written:
                        out_fh.write(",")
                    out_fh.write(json.dumps(m))
                    first_written = True
                nk, first_written = stream_kernel_events(kcsv, t0, out_pid, 1, out_fh, first_written)
                nmk = 0
                if marker_csv:
                    out_fh.write(",")
                    out_fh.write(json.dumps({"ph": "M", "name": "thread_name", "pid": out_pid,
                                              "tid": 2, "args": {"name": "Marker API (reqstats + MORI-IO)"}}))
                    nmk, first_written = stream_marker_events(marker_csv, t0, out_pid, 2, out_fh, first_written)
                total_kernel_events += nk
                total_marker_events += nmk
                has_native_pftrace = pftrace is not None
                summary.append((proc_name, nk, nmk, has_native_pftrace))
                log(f"{proc_name}: kernel_events={nk} marker_events={nmk} "
                    f"native_pftrace={'yes' if has_native_pftrace else 'no (used CSV fallback)'}")
        out_fh.write('],"displayTimeUnit":"ns"}')

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
    if 'aiter::fmoe' in name: return 'MoE_Fused'
    if 'kernel_moe' in n: return 'MoE_Unfused'
    if 'moesorting' in n: return 'MoE_Sort'
    if 'topk' in n: return 'MoE_TopK'
    # Only inter-node payload kernels are MORI EP; support kernels are communication.
    if 'epdispatchinternode' in n or 'epcombineinternode' in n: return 'MORI EP'
    if 'epdispatch' in n or 'epcombine' in n: return 'Communication'

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


def build_trimmed_summary(kernel_trace_csv, trim_pct):
    'Build trimmed rows and per-kernel trim metadata.'
    groups = {}
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
            if row.get("Kind") and row["Kind"] != "KERNEL_DISPATCH":
                continue
            name = row[name_col]
            try:
                dur_ns = float(row[end_col]) - float(row[start_col])
            except (TypeError, ValueError):
                continue
            g = groups.setdefault(name, {"durations": [], "row": row})
            g["durations"].append(dur_ns)

    summary_rows = []
    grand_total_ns = 0.0
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

        r = g["row"]
        grand_total_ns += total
        summary_rows.append({
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
            "GridX": r.get("Grid_Size_X", ""),
            "GridY": r.get("Grid_Size_Y", ""),
            "GridZ": r.get("Grid_Size_Z", ""),
            "BlockX": r.get("Workgroup_Size_X", ""),
            "BlockY": r.get("Workgroup_Size_Y", ""),
            "BlockZ": r.get("Workgroup_Size_Z", ""),
            "VGPR": r.get("VGPR_Count", ""),
            "AccumVGPR": r.get("Accum_VGPR_Count", ""),
            "SGPR": r.get("SGPR_Count", ""),
            "LDS": r.get("LDS_Block_Size", ""),
            "Scratch": r.get("Scratch_Size", ""),
        })

    summary_rows.sort(key=lambda x: x["Total Time (ns)"], reverse=True)
    for r in summary_rows:
        r["Time %"] = (100.0 * r["Total Time (ns)"] / grand_total_ns) if grand_total_ns else 0.0

    return summary_rows, grand_total_ns


def write_csv(rows, out_path, add_category, categorize_kernel):
    columns = list(OUTPUT_COLUMNS)
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
            if add_category and categorize_kernel is not None:
                out = [categorize_kernel(r["Name"])] + out
            w.writerow(out)


def _add_combine_parser(subparsers):
    p = subparsers.add_parser(
        "combine",
        help="stream rocprofv3 CSV shards into combined Chrome JSON traces",
        description="Stream rocprofv3 CSV shards into combined Chrome JSON traces; timestamps are normalized per node.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("jobdir", help="job output dir, e.g. /shared_inference/aarai/my_prof_run/<jobid>")
    p.add_argument("--out-dir", default=None,
                   help="where to write combined_all.pftrace / combined_rank0.pftrace / combined_prefill.chrome.json (default: <jobdir> itself)")
    p.add_argument("--only", choices=["all", "rank0", "prefill", "both"], default="both",
                   help="which trace(s) to build; both includes prefill-only (default: both)")
    p.set_defaults(func=_run_combine)


def _run_combine(a):
    jobdir = os.path.abspath(a.jobdir)
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else jobdir
    os.makedirs(out_dir, exist_ok=True)

    if a.only in ("all", "both"):
        log(f"building combined_all.pftrace (ALL ranks, ALL nodes) from {jobdir}")
        build_combined(jobdir, os.path.join(out_dir, "combined_all.pftrace"), rank0_only=False)
    if a.only in ("rank0", "both"):
        log(f"building combined_rank0.pftrace (rank0 per node) from {jobdir}")
        build_combined(jobdir, os.path.join(out_dir, "combined_rank0.pftrace"), rank0_only=True)
    if a.only in ("prefill", "both"):
        log(f"building combined_prefill.chrome.json (ALL prefill ranks, ALL prefill nodes) from {jobdir}")
        build_combined(
            jobdir, os.path.join(out_dir, "combined_prefill.chrome.json"),
            rank0_only=False, role_filter="prefill",
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
    fields = ["write_uid", "direction", "request_id", "transfer_id", "layer"]
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


def run_trimmed_summary(kernel_trace, out_path, trim_pct, add_category):
    'Shared trimmed-summary implementation used by both the CLI subcommand and `analyze`.'
    if not os.path.isfile(kernel_trace):
        raise SystemExit(f"ERROR: not found: {kernel_trace}")
    if trim_pct < 0 or trim_pct >= 100:
        raise SystemExit(f"ERROR: --trim-pct must be in [0, 100), got {trim_pct}")

    classifier = categorize_kernel if add_category else None

    rows, grand_total_ns = build_trimmed_summary(kernel_trace, trim_pct)
    write_csv(rows, out_path, add_category, classifier)

    n_eligible = sum(1 for r in rows if r["instances_before_trim"] >= 20)
    n_trimmed_kernels = sum(1 for r in rows if r["n_trimmed"] > 0)
    total_calls_dropped = sum(r["n_trimmed"] for r in rows)
    print(f"wrote {out_path}")
    print(f"  {len(rows)} distinct kernels; {n_eligible} eligible (>=20 calls); "
          f"{n_trimmed_kernels} trimmed at {trim_pct}%; {total_calls_dropped} total calls dropped")
    print(f"  trimmed grand total: {pretty_ns(grand_total_ns)}")


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


def _detect_trace(trace):
    '''Reproduce analyze_kernels.sh's trace-type detection. Returns (pft, kcsv, pjson) with
    exactly one of the three set, or None if an error was already printed (caller should exit 2).'''
    if trace.endswith(".pftrace"):
        return trace, None, None
    if trace.endswith("_kernel_trace.csv"):
        return None, trace, None
    if trace.endswith(".json") or trace.endswith(".json.gz"):
        return None, None, trace
    if os.path.isdir(trace):
        cand = _find_first(trace, "*_results.pftrace") or _find_first(trace, "*.pftrace")
        if cand:
            print(f"detected rocprof dir; using pftrace: {cand}")
            return cand, None, None
        cand = _find_first(trace, "*_kernel_trace.csv")
        if cand:
            print(f"detected rocprof dir; using kernel_trace csv: {cand}")
            return None, cand, None
        print(f"ERROR: no *.pftrace or *_kernel_trace.csv found under '{trace}'.")
        return None
    print(f"ERROR: '{trace}' is not a recognized trace file/dir.")
    print("       Pass a *.pftrace / *.json / *_kernel_trace.csv, or a rocprof_*_NODE* dir.")
    return None


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


def run_analyze(trace, outdir, label, venv, traceconv, trim_pct):
    '''Python reimplementation of analyze_kernels.sh: single-trace orchestration (trace-type
    detection, traceconv, TraceLens, first-party buckets, trimmed summary). Returns a process-style
    exit code (0 success, 2 unrecognized/missing trace, 3 missing venv or traceconv failure).'''
    print(f"############ [{label}] START {_date_u()} ############")
    print(f"TRACE={trace}")

    detected = _detect_trace(trace)
    if detected is None:
        return 2
    pft, kcsv, pjson = detected

    tl_dir = os.path.join(outdir, "tracelens")
    bk_dir = os.path.join(outdir, "buckets")
    os.makedirs(os.path.join(tl_dir, "out_csvs"), exist_ok=True)
    os.makedirs(bk_dir, exist_ok=True)

    if not os.path.isfile(os.path.join(venv, "bin", "activate")):
        print(f"ERROR: venv not found at '{venv}'. Run posthoc.sh --kernels for setup and analysis.")
        return 3

    # Replicate `source $VENV/bin/activate`'s effect on command resolution without sourcing it.
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = venv
    env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")

    ks = None

    if kcsv:
        print(f"=== [{label}] (a) rocprofv3 kernel_trace.csv -> aggregated kernel_summary ===")
        if not _ls_lh(kcsv):
            print("KERNEL_TRACE CSV MISSING")
            return 2
        ks = os.path.join(tl_dir, "out_csvs", "kernel_summary.csv")
    else:
        if pft:
            print(f"=== [{label}] (a0) traceconv: pftrace -> Perfetto JSON ===")
            if not _ls_lh(pft):
                print("PFTRACE MISSING")
                return 2
            base = os.path.basename(pft)
            if base.endswith(".pftrace"):
                base = base[: -len(".pftrace")]
            pjson = os.path.join(tl_dir, base + ".json")

            if not (os.path.isfile(traceconv) or os.access(traceconv, os.X_OK)):
                print(f"ERROR: traceconv not found at '{traceconv}'. Set TRACECONV=/path/to/traceconv.")
                return 3
            try:
                st = os.stat(traceconv)
                os.chmod(traceconv, st.st_mode | 0o111)
            except OSError:
                pass

            # Accept native traceconv binaries and wrapper scripts.
            try:
                with open(traceconv, "rb") as f:
                    is_elf = f.read(4) == b"\x7fELF"
            except OSError:
                is_elf = False
            cmd = [traceconv, "json", pft, pjson] if is_elf else ["python3", traceconv, "json", pft, pjson]

            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            tc_text = (proc.stdout or "") + (proc.stderr or "")
            with open(os.path.join(tl_dir, "traceconv_stdout.txt"), "w", encoding="utf-8") as f:
                f.write(tc_text)
            tc_exit = proc.returncode
            print(f"traceconv_exit={tc_exit} -> {pjson}")
            _print_tail(tc_text, 5)
            if tc_exit != 0 or not (os.path.isfile(pjson) and os.path.getsize(pjson) > 0):
                print("ERROR: traceconv failed (no internet on this node and no cached binary?).")
                return 3

        print(f"=== [{label}] (a) TraceLens trace parsing and kernel aggregation ===")
        if not _ls_lh(pjson):
            print("PERFETTO JSON MISSING")
            return 2
        tl_cmd = [
            "TraceLens_generate_perf_report_pftrace_hip_activity",
            "--trace_path", pjson,
            "--output_xlsx_path", os.path.join(tl_dir, "report_TP0.xlsx"),
            "--output_csvs_dir", os.path.join(tl_dir, "out_csvs"),
            "--output_md_path", os.path.join(tl_dir, "report.md"),
            "--traceconv", traceconv,
            "--write_md",
        ]
        proc = subprocess.run(tl_cmd, env=env, capture_output=True, text=True)
        tl_text = (proc.stdout or "") + (proc.stderr or "")
        with open(os.path.join(tl_dir, "tracelens_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(tl_text)
        print(f"tracelens_exit={proc.returncode}")
        _print_tail(tl_text, 8)

        ks = _find_iname_first(os.path.join(tl_dir, "out_csvs"), "kernel_summary*.csv")
        print(f"kernel_summary_csv={ks or ''}")
        catsum = _find_iname_first(os.path.join(tl_dir, "out_csvs"), "category_summary*.csv")
        if catsum and os.path.isfile(catsum):
            dest = os.path.join(bk_dir, "tracelens_native_category_summary.csv")
            shutil.copyfile(catsum, dest)
            print(f"copied TraceLens native category_summary -> {dest}")

    print(f"=== [{label}] (b) first-party category buckets (name-based) ===")
    norm = os.path.join(bk_dir, "kernel_summary_normalized.csv")
    norm_src = kcsv if kcsv else ks
    norm_exit = _normalize_kernel_summary(norm_src, norm)
    print(f"normalize_exit={norm_exit} -> {norm}")

    if norm_exit == 0 and os.path.isfile(norm) and os.path.getsize(norm) > 0:
        exit_code, text = _call_captured(
            process, norm,
            os.path.join(bk_dir, "perkernel_buckets.csv"),
            os.path.join(bk_dir, "bycat_buckets.csv"),
            f"{label} (rocprof)",
        )
        with open(os.path.join(bk_dir, "buckets_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"buckets_exit={exit_code}")
        _print_tail(text, 12)
    else:
        print("buckets_SKIPPED: normalizer produced no usable kernel summary")

    print(f"=== [{label}] (c) trimmed kernel summary (outlier-robust, TRIM_PCT={trim_pct:g}%) ===")
    trim_src = kcsv
    if not trim_src and pft:
        suffix = "_results.pftrace"
        stem = pft[: -len(suffix)] if pft.endswith(suffix) else pft
        cand = stem + "_kernel_trace.csv"
        if os.path.isfile(cand):
            trim_src = cand
        else:
            trim_src = _find_first(os.path.dirname(pft), "*_kernel_trace.csv")

    if trim_src and os.path.isfile(trim_src):
        print(f"trim_src={trim_src}")
        exit_code, text = _call_captured(
            run_trimmed_summary, trim_src,
            os.path.join(bk_dir, "kernel_summary_trimmed.csv"),
            trim_pct, True,
        )
        with open(os.path.join(bk_dir, "trimmed_summary_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"trimmed_summary_exit={exit_code}")
        _print_tail(text, 8)
    else:
        print(f"trimmed_summary_SKIPPED: no raw *_kernel_trace.csv found (TRIM_SRC='{trim_src or ''}')")

    print(f"=== [{label}] OUTPUT LISTING ===")
    _print_output_listing([tl_dir, os.path.join(tl_dir, "out_csvs"), bk_dir])
    print(f"############ [{label}] DONE {_date_u()} ############")
    return 0


def _add_analyze_parser(subparsers):
    p = subparsers.add_parser(
        "analyze",
        help="run the single-trace analysis pipeline (traceconv + TraceLens + first-party buckets + trimmed summary)",
        description=(
            "Single-trace orchestration: detect trace type (*.pftrace / *_kernel_trace.csv / "
            "*.json(.gz) / a rocprof_*_NODE* dir), convert pftrace->Perfetto JSON via traceconv "
            "when needed, run the TraceLens kernel-aggregation console script, normalize the "
            "resulting kernel-summary CSV, apply first-party category buckets, and build a "
            "trimmed (outlier-robust) kernel summary. Requires a Python venv with TraceLens "
            "installed (see posthoc.sh's embedded setup) and a traceconv binary/script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("trace", help="*.pftrace / *.json(.gz) / *_kernel_trace.csv file, or a rocprof_*_NODE* dir")
    p.add_argument("outdir", help="output directory (tracelens/ and buckets/ subdirs are created under it)")
    p.add_argument("label", help="short label used in banners and stdout summaries")
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
    return run_analyze(args.trace, args.outdir, args.label, venv, traceconv, trim_pct)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subparsers = p.add_subparsers(dest='command', required=True)
    _add_combine_parser(subparsers)
    _add_buckets_parser(subparsers)
    _add_extract_reqid_parser(subparsers)
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

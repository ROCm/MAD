#!/usr/bin/env python3
"""Offline ROCTX trace construction, request attribution, and kernel analysis."""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_csv_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_limit)
        break
    except OverflowError:
        _csv_limit //= 10

# ---- clean-trace construction ----
_MAP_RE = re.compile('\\broom=(\\d+)\\s+uid=(\\d+)')
_IDV_RE = re.compile('\\bid=(\\d+)')
_BYTES_RE = re.compile('\\bbytes=(\\d+)')
_WORKER_GPU0_RE = re.compile(r'\bProcess (?P<pid>\d+) gpu_id 0 is running on CPUs:')
_WORKER_GPU_RE = re.compile(
    r'(?:\bDP(?P<dp_rank>\d+)\s+TP(?P<tp_rank>\d+)\s+'
    r'EP(?P<ep_rank>\d+)\]\s+)?Process (?P<pid>\d+) '
    r'gpu_id (?P<local_rank>\d+) is running on CPUs:'
)
_ANALYSIS_PID_RE = re.compile(r'^.+_(?P<pid>\d+)_(?:results\.pftrace|kernel_trace\.csv)$')
_NODE_CAPTURE_RE = re.compile(r'^rocprof_(?P<role>prefill|decode)_NODE(?P<node_rank>\d+)$')

def iter_kernels(path):
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                s = int(r['Start_Timestamp'])
                e = int(r['End_Timestamp'])
            except Exception:
                continue
            if _duration_or_none(e - s) is None:
                continue
            yield (s, e, r.get('Kernel_Name', 'kernel'))

def read_kernels(path):
    return list(iter_kernels(path))

def iter_marks(path):
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            fn = r.get('Function', '')
            if not fn.startswith('reqstats'):
                continue
            try:
                s = int(r['Start_Timestamp'])
            except Exception:
                continue
            m = re.search('id=(\\S+)', fn)
            room = m.group(1) if m else ''
            event = fn.split(' id=')[0]
            yield (s, event, room)

def read_marks(path):
    return list(iter_marks(path))

def iter_mori(path):
    """MORI-IO host-send roctx RANGES (mori.*): duration slices (Start..End).

    Excludes the `mori.map` correlation marks (instant, zero-dur): those are consumed by
    read_map for attribution, not rendered as transfer ranges."""
    out = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            fn = r.get('Function', '')
            if not fn.startswith('mori'):
                continue
            if fn.startswith('mori.map'):
                continue
            try:
                s = int(r['Start_Timestamp'])
                e = int(r['End_Timestamp'])
            except Exception:
                continue
            yield (s, e, fn)

def read_mori(path):
    return list(iter_mori(path))

def read_map(path, lo=None, hi=None):
    """Parse `mori.map room=R uid=U` marks -> {uid(str): room(str)} for one pid.
    Empty on pre-patch traces (callers then fall back to timestamp containment)."""
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            fn = r.get('Function', '')
            if not fn.startswith('mori.map'):
                continue
            if lo is not None or hi is not None:
                try:
                    ts = int(r['Start_Timestamp'])
                except Exception:
                    continue
                if lo is not None and ts < lo:
                    continue
                if hi is not None and ts > hi:
                    continue
            m = _MAP_RE.search(fn)
            if m:
                out[m.group(2)] = m.group(1)
    return out

def mori_rid_assign(probe, mori_ranges, uid_to_room=None):
    """Return a list (parallel to mori_ranges) of the owning rid (or None).

    Prefer EXACT-by-id attribution via the `mori.map` uid->room map (uid_to_room) when a
    range carries an `id=N` that maps to a real room; otherwise fall back to the original
    per-pid timestamp-containment rule (inner [prefill_kv_transfer_start,
    prefill_kv_transfer_finish] window, tightest enclosing, outer [forward_entry,
    completion] fallback). The fallback keeps pre-patch traces (no mori.map) working and
    doubles as a cross-check. probe: (ts,event,room) for THIS pid; mori_ranges: (s,e,name)."""
    uid_to_room = uid_to_room or {}
    rooms = {}
    for s, event, room in probe:
        if room in ('', '0'):
            continue
        leaf = event.split('.')[-1]
        rooms.setdefault(room, {})[leaf] = s
    inner, outer = ({}, {})
    for room, st in rooms.items():
        if 'prefill_kv_transfer_start' in st and 'prefill_kv_transfer_finish' in st:
            a, b = (st['prefill_kv_transfer_start'], st['prefill_kv_transfer_finish'])
            if b >= a:
                inner[room] = (a, b)
        if 'forward_entry' in st and 'completion' in st:
            a, b = (st['forward_entry'], st['completion'])
            if b >= a:
                outer[room] = (a, b)
    rids = []
    for s, e, name in mori_ranges:
        m = _IDV_RE.search(name)
        room = uid_to_room.get(m.group(1)) if m else None
        if room is not None and room not in ('', '0'):
            rids.append(room)
            continue
        mid = (s + e) // 2
        hits = [(room, iv[1] - iv[0]) for room, iv in inner.items() if iv[0] <= mid <= iv[1]]
        if not hits:
            hits = [(room, iv[1] - iv[0]) for room, iv in outer.items() if iv[0] <= mid <= iv[1]]
        if hits:
            hits.sort(key=lambda x: x[1])
            rids.append(hits[0][0])
        else:
            rids.append(None)
    return rids

def emit_kv_stack(ev, kv_intervals, base_pid, kvt, gtp, us, max_rows=50):
    """Emit overlapping KV ranges on greedy-stacked complete-event rows.
    ``max_rows`` bounds reserved track IDs; overflow uses the last row.
    Returns ``(emitted_count, maximum_depth)``.
    """
    kv_intervals.sort(key=lambda x: x[0])
    last_end = []
    nkvt = 0
    kv_depth = 0
    warned = False
    for s, e, label, rid, nbytes in kv_intervals:
        k = next((idx for idx in range(len(last_end)) if last_end[idx] <= s), None)
        if k is None:
            k = len(last_end)
            last_end.append(e)
        else:
            last_end[k] = e
        if k >= max_rows:
            if not warned:
                print(f'[per-GPU][WARN] TP{gtp} kv_transfer stack depth {k + 1} exceeds {max_rows} sub-rows -- clamping to row {max_rows - 1} (tid {kvt + max_rows - 1})')
                warned = True
            k_emit = max_rows - 1
        else:
            k_emit = k
        _a = {}
        if rid:
            _a['rid'] = rid
        if nbytes:
            _a['bytes'] = nbytes
        ev.append({'ph': 'X', 'name': label, 'ts': us(s), 'dur': max((e - s) / 1000.0, 0.0), 'pid': base_pid, 'tid': kvt + k_emit, 'args': _a})
        nkvt += 1
        kv_depth = max(kv_depth, k + 1)
    for k in range(1, min(kv_depth, max_rows)):
        ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': kvt + k, 'args': {'name': f'TP{gtp} MORI-IO KV transfer #{k + 1}'}})
    return (nkvt, kv_depth)

def emit_stacked(ev, intervals, base_pid, base_tid, us, row_label, max_rows=12):
    """Greedily assign intervals to the lowest non-overlapping row.
    Returns ``(emitted_count, maximum_depth)``.
    """
    intervals.sort(key=lambda x: x[0])
    last_end = []
    n = 0
    depth = 0
    for s, e, name, args in intervals:
        k = next((idx for idx in range(len(last_end)) if last_end[idx] <= s), None)
        if k is None:
            k = len(last_end)
            last_end.append(e)
        else:
            last_end[k] = e
        k_emit = k if k < max_rows else max_rows - 1
        ev.append({'ph': 'X', 'name': name, 'ts': us(s), 'dur': max((e - s) / 1000.0, 0.0), 'pid': base_pid, 'tid': base_tid + k_emit, 'args': args})
        n += 1
        depth = max(depth, k + 1)
    for k in range(1, min(depth, max_rows)):
        ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': base_tid + k, 'args': {'name': row_label(k)}})
    return (n, depth)

def build_stage_span_intervals(byroom, gap_mult=3.0):
    """Build adjacent stage spans, decode-step indices, and long-gap flags.
    A gap is flagged above ``gap_mult`` times its pair-type median; pair types with
    fewer than two samples are never flagged. Existing pairing/order is preserved.
    """
    durations_by_type = {}
    for room, seq in byroom.items():
        for i in range(len(seq) - 1):
            (s0, _e0), (s1, _e1) = (seq[i], seq[i + 1])
            if s1 <= s0:
                continue
            short0 = _e0.split('.')[-1]
            short1 = _e1.split('.')[-1]
            durations_by_type.setdefault(f'{short0}->{short1}', []).append(s1 - s0)
    thresholds = {pt: gap_mult * statistics.median(d) for pt, d in durations_by_type.items() if len(d) >= 2}
    out = []
    for room, seq in byroom.items():
        dc = 0
        for i in range(len(seq) - 1):
            (s0, e0), (s1, e1) = (seq[i], seq[i + 1])
            if s1 <= s0:
                continue
            short0 = e0.split('.')[-1]
            short1 = e1.split('.')[-1]
            pair_type = f'{short0}->{short1}'
            args = {'room': room, 'pair_type': pair_type}
            if short1 == 'decode_finish' and short0 in ('decode_finish', 'decode_prebuilt_finish'):
                name = f'{pair_type}[step={dc}]'
                args['step'] = dc
                dc += 1
            else:
                name = pair_type
            thr = thresholds.get(pair_type)
            if thr is not None and s1 - s0 > thr:
                args['long_gap'] = 1
                args['median_ns'] = int(statistics.median(durations_by_type[pair_type]))
                name += ' [LONG GAP]'
            out.append((s0, s1, name, args))
    return out

def build_decode_ct_counters(probe, label, base_pid, us):
    """Emit decode progress and admission-backlog counters for one worker.
    Returns no counters when the worker has no decode_finish markers.
    """
    total = 0
    room_dc = {}
    backlog = set()
    out = []
    for s, event, room in probe:
        if room in ('', '0'):
            continue
        leaf = event.split('.')[-1]
        if leaf == 'decode_finish':
            total += 1
            c = room_dc.get(room, 0) + 1
            room_dc[room] = c
            if c == 1:
                backlog.add(room)
            elif room in backlog:
                backlog.discard(room)
            out.append({'ph': 'C', 'name': f'{label} decode_ct', 'ts': us(s), 'pid': base_pid, 'args': {'decode_ct': total}})
            out.append({'ph': 'C', 'name': f'{label} decode_admission_backlog', 'ts': us(s), 'pid': base_pid, 'args': {'decode_admission_backlog': len(backlog)}})
        elif leaf == 'completion':
            backlog.discard(room)
            room_dc.pop(room, None)
    return out

def _pid_from_marker(path):
    m = re.search('_(\\d+)_marker_api_trace\\.csv$', os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

def discover_pairs_in_dir(d):
    """Topology-aware pair discovery, keyed on the MARKER csv (always present) with the
    kernel csv treated as OPTIONAL (None if absent). This is what makes the builder work
    for BOTH kernel+marker captures AND marker-trace-only captures -- the EP / full-V3
    2P2D runs use `ROCPROF_FLAGS=--marker-trace` only (a kernel trace would be millions of
    rows/worker at conc-32), so they have no *_kernel_trace.csv. Returns sorted
    [(pid, kernel_csv_or_None, marker_csv)] for every TP worker in the dir."""
    pairs = []
    for mk in glob.glob(os.path.join(d, '*_marker_api_trace.csv')):
        kc = mk[:-len('_marker_api_trace.csv')] + '_kernel_trace.csv'
        pairs.append((_pid_from_marker(mk), kc if os.path.exists(kc) else None, mk))
    pairs.sort(key=lambda x: x[0])
    return pairs

def engine_events(kernel_csv, marker_csv, pid, proc_name, pad_ns=2000000, probe_only=False, gap_ns=30000000000):
    marks = read_marks(marker_csv)
    probe = [m for m in marks if m[2] not in ('', '0')]
    probe.sort(key=lambda x: x[0])
    if probe_only and probe:
        clusters, cur = ([], [probe[0]])
        for prev, r in zip(probe, probe[1:]):
            if r[0] - prev[0] > gap_ns:
                clusters.append(cur)
                cur = [r]
            else:
                cur.append(r)
        clusters.append(cur)
        probe = clusters[-1]
    ev = []
    meta = [{'ph': 'M', 'name': 'process_name', 'pid': pid, 'tid': 0, 'args': {'name': proc_name}}, {'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': 1, 'args': {'name': 'GPU kernels'}}, {'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': 2, 'args': {'name': 'reqstats markers'}}, {'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': 3, 'args': {'name': 'per-request stage spans'}}, {'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': 4, 'args': {'name': 'MORI-IO host send (ibv_post_send)'}}]
    if not probe:
        probe = marks
    t_lo = min((s for s, _, _ in probe)) - pad_ns
    t_hi = max((s for s, _, _ in probe)) + pad_ns
    t0 = t_lo
    us = lambda ns: (ns - t0) / 1000.0
    nk = 0
    for s, e, name in read_kernels(kernel_csv):
        if s < t_lo or s > t_hi:
            continue
        ev.append({'ph': 'X', 'name': name, 'ts': us(s), 'dur': (e - s) / 1000.0, 'pid': pid, 'tid': 1})
        nk += 1
    for s, event, room in probe:
        ev.append({'ph': 'i', 'name': event + (' room=' + room if room else ''), 'ts': us(s), 'pid': pid, 'tid': 2, 's': 't'})
    byroom = {}
    for s, event, room in sorted(probe, key=lambda x: x[0]):
        byroom.setdefault(room, []).append((s, event))
    nspan = 0
    for s0, s1, name, args in build_stage_span_intervals(byroom):
        ev.append({'ph': 'X', 'name': name, 'ts': us(s0), 'dur': (s1 - s0) / 1000.0, 'pid': pid, 'tid': 3, 'args': args})
        nspan += 1
    ev += build_decode_ct_counters(probe, proc_name, pid, us)
    nmori = 0
    for s, e, name in read_mori(marker_csv):
        if s < t_lo or s > t_hi:
            continue
        ev.append({'ph': 'X', 'name': name, 'ts': us(s), 'dur': max((e - s) / 1000.0, 0.05), 'pid': pid, 'tid': 4})
        nmori += 1
    return (meta + ev, nk, len(probe), nspan, nmori)

def _select_probe(marker_csv, probe_only, rid_rooms=None, gap_ns=30000000000):
    if rid_rooms is not None:
        return sorted(
            (mark for mark in read_marks(marker_csv) if mark[2] in rid_rooms),
            key=lambda mark: mark[0],
        )
    if not probe_only:
        marks = read_marks(marker_csv)
        probe = [m for m in marks if m[2] not in ('', '0')]
        return probe or marks
    probe, prev_s = ([], None)
    for mark in iter_marks(marker_csv):
        if mark[2] in ('', '0'):
            continue
        if prev_s is not None and mark[0] - prev_s > gap_ns:
            probe = []
        probe.append(mark)
        prev_s = mark[0]
    probe.sort(key=lambda x: x[0])
    return probe

def _pid_from(path):
    m = re.search('_(\\d+)_kernel_trace\\.csv$', os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

def discover_pairs(repr_kernel_csv):
    d = os.path.dirname(os.path.abspath(repr_kernel_csv))
    pairs = []
    for k in glob.glob(os.path.join(d, '*_kernel_trace.csv')):
        mk = k[:-len('_kernel_trace.csv')] + '_marker_api_trace.csv'
        if os.path.exists(mk):
            pairs.append((_pid_from(k), k, mk))
    pairs.sort(key=lambda x: x[0])
    return pairs

def lane_window(pairs, probe_only, pad_ns=2000000, rid_rooms=None):
    per, los, his = ([], [], [])
    for pid, kcsv, mcsv in pairs:
        probe = _select_probe(mcsv, probe_only, rid_rooms=rid_rooms)
        per.append((pid, kcsv, mcsv, probe))
        if probe:
            los.append(min((s for s, _, _ in probe)))
            his.append(max((s for s, _, _ in probe)))
    lo = min(los) - pad_ns if los else 0
    hi = max(his) + pad_ns if his else 0
    return (per, lo, hi)

def pergpu_engine(
    per,
    lo,
    hi,
    base_pid,
    proc_name,
    t0,
    with_mori=True,
    tp_offset=0,
    no_reqstats=False,
    lane_index_base=0,
):
    """Emit one lane set per TP worker using global ranks from ``tp_offset``.
    Track-ID spacing reserves rows for stacked stage, host-send, and KV intervals.
    """
    us = lambda ns: (ns - t0) / 1000.0
    ev = [{'ph': 'M', 'name': 'process_name', 'pid': base_pid, 'tid': 0, 'args': {'name': proc_name}}]
    stats = []
    for j, (pid, kcsv, mcsv, probe) in enumerate(per):
        lane_index = lane_index_base + j
        gtp = tp_offset + lane_index
        b = 1000 + lane_index * 100
        kt, mt, st, ot, kvt = (b + 0, b + 1, b + 10, b + 30, b + 50)
        ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': kt, 'args': {'name': f'TP{gtp} GPU kernels (GPU{lane_index})'}})
        if not no_reqstats:
            ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': mt, 'args': {'name': f'TP{gtp} reqstats markers'}})
            ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': st, 'args': {'name': f'TP{gtp} per-request stage spans'}})
        if with_mori:
            ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': ot, 'args': {'name': f'TP{gtp} MORI-IO host send (post us)'}})
            ev.append({'ph': 'M', 'name': 'thread_name', 'pid': base_pid, 'tid': kvt, 'args': {'name': f'TP{gtp} MORI-IO KV transfer (post->cq ms)'}})
        nk = 0
        if kcsv:
            for s, e, name in iter_kernels(kcsv):
                if s < lo or s > hi:
                    continue
                ev.append({'ph': 'X', 'name': name, 'ts': us(s), 'dur': (e - s) / 1000.0, 'pid': base_pid, 'tid': kt})
                nk += 1
        nspan, span_depth = (0, 0)
        if not no_reqstats:
            for s, event, room in probe:
                ev.append({'ph': 'i', 'name': event + (' room=' + room if room else ''), 'ts': us(s), 'pid': base_pid, 'tid': mt, 's': 't'})
            byroom = {}
            for s, event, room in sorted(probe, key=lambda x: x[0]):
                byroom.setdefault(room, []).append((s, event))
            span_intervals = build_stage_span_intervals(byroom)
            nspan, span_depth = emit_stacked(ev, span_intervals, base_pid, st, us, lambda k: f'TP{gtp} per-request stage spans #{k + 1}', max_rows=20)
        ev += build_decode_ct_counters(probe, f'TP{gtp}', base_pid, us)
        nmori = 0
        nkvt = 0
        kv_depth = 0
        if with_mori:
            mori = [m for m in iter_mori(mcsv) if lo <= m[0] <= hi]
            uid_to_room = read_map(mcsv, lo, hi)
            rids = mori_rid_assign(probe, mori, uid_to_room)
            kv_intervals = []
            host_intervals = []
            for (s, e, name), rid in zip(mori, rids):
                label = name + (f' rid={rid}' if rid else '')
                _mb = _BYTES_RE.search(name)
                nbytes = int(_mb.group(1)) if _mb else 0
                if 'kv_transfer' in name:
                    kv_intervals.append((s, e, label, rid, nbytes))
                else:
                    _ha = {}
                    if rid:
                        _ha['rid'] = rid
                    if nbytes:
                        _ha['bytes'] = nbytes
                    host_intervals.append((s, e, label, _ha))
                    nmori += 1
            emit_stacked(ev, host_intervals, base_pid, ot, us, lambda k: f'TP{gtp} MORI-IO host send #{k + 1}', max_rows=20)
            nkvt, kv_depth = emit_kv_stack(ev, kv_intervals, base_pid, kvt, gtp, us, max_rows=50)
        stats.append((pid, nk, len(probe), nspan, nmori, nkvt, kv_depth))
    return (ev, stats)

def write_pergpu(a, out):
    pp = discover_pairs(a.prefill_kernel)
    dp = discover_pairs(a.decode_kernel)
    p_per, p_lo, p_hi = lane_window(pp, a.probe_only)
    d_per, d_lo, d_hi = lane_window(dp, a.probe_only)
    no_reqstats = getattr(a, 'no_reqstats_lanes', False)
    pev, pst = pergpu_engine(p_per, p_lo, p_hi, 10, 'PREFILL (NODE0)', p_lo, with_mori=True, no_reqstats=no_reqstats)
    dev, dst = pergpu_engine(d_per, d_lo, d_hi, 20, 'DECODE (NODE1)', d_lo, with_mori=False, no_reqstats=no_reqstats)
    with open(out, 'w') as f:
        json.dump({'traceEvents': pev + dev, 'displayTimeUnit': 'ns'}, f)
    print(f'[per-GPU] PREFILL lanes={len(pst)} (TP0..TP{len(pst) - 1}) kernels={[s[1] for s in pst]} mori_hostsend={[s[4] for s in pst]} kv_transfer={[s[5] for s in pst]} kv_depth={[s[6] for s in pst]}')
    print(f'[per-GPU] DECODE  lanes={len(dst)} (TP0..TP{len(dst) - 1}) kernels={[s[1] for s in dst]} mori_hostsend={[s[4] for s in dst]} kv_transfer={[s[5] for s in dst]} kv_depth={[s[6] for s in dst]}')
    print(f'[per-GPU] wrote {out} ({len(pev) + len(dev)} events) -- 8+8 = {len(pst) + len(dst)} GPU lanes')


def _new_trace_validation(path):
    return {
        "path": os.path.abspath(path),
        "events": 0,
        "processes": [],
        "worker_lanes": 0,
        "metadata": set(),
        "slices": 0,
    }


def _record_trace_event(validation, event):
    validation["events"] += 1
    name = event.get("name", "")
    if event.get("ph") == "M":
        args = event.get("args", {})
        signature = (
            name,
            event.get("pid"),
            event.get("tid"),
            json.dumps(args, sort_keys=True),
        )
        if signature in validation["metadata"]:
            raise SystemExit(
                "ERROR: duplicate metadata signature in "
                f"{validation['path']}: {signature}"
            )
        validation["metadata"].add(signature)
        if name == "process_name":
            validation["processes"].append(args.get("name", ""))
        elif name == "thread_name" and "reqstats markers" in args.get("name", ""):
            validation["worker_lanes"] += 1
    elif event.get("ph") in ("X", "i", "C"):
        validation["slices"] += 1


def write_multinode(
    prefill_dirs,
    decode_dirs,
    out,
    probe_only,
    no_reqstats=False,
    rid_rooms=None,
):
    """TOPOLOGY-AWARE path: one process per source node, GLOBAL TP-rank labels, and
    depth-stacked stage-span / host-send / KV lanes. Handles 1P1D (1 prefill + 1 decode
    dir -> 16 lanes) up through xPyD (xP prefill dirs + yD decode dirs -> (xP+yD)*8 lanes,
    e.g. 2P2D -> 32). Each node is normalized to its OWN probe-window t0 (the nodes'
    clocks are unaligned); processes overlay and are delineated by name."""
    base = 10
    summary = []
    validation = _new_trace_validation(out)
    with open(out, 'w') as f:
        f.write('{"traceEvents": [')
        first_event = True
        total_events = 0
        for role, dirs, with_mori in (
            ('PREFILL', prefill_dirs, True),
            ('DECODE', decode_dirs, False),
        ):
            for local_idx, d in enumerate(dirs):
                m = re.search('NODE(\\d+)', os.path.basename(os.path.normpath(d)))
                nr = m.group(1) if m else str(local_idx)
                tp_off = local_idx * 8
                pairs = discover_pairs_in_dir(d)
                per, lo, hi = lane_window(
                    pairs, probe_only, rid_rooms=rid_rooms
                )
                g0 = tp_off
                g1 = tp_off + (len(per) - 1 if per else 0)
                pname = f'{role} NODE{nr} [TP{g0}-{g1}]'
                st = []
                wrote_process = False
                lane_inputs = list(enumerate(per))
                if not lane_inputs:
                    lane_inputs = [(0, None)]
                for lane_index, lane in lane_inputs:
                    if lane is None:
                        events = [{
                            'ph': 'M',
                            'name': 'process_name',
                            'pid': base,
                            'tid': 0,
                            'args': {'name': pname},
                        }]
                        lane_stats = []
                    else:
                        events, lane_stats = pergpu_engine(
                            [lane],
                            lo,
                            hi,
                            base,
                            pname,
                            lo,
                            with_mori=with_mori,
                            tp_offset=tp_off,
                            no_reqstats=no_reqstats,
                            lane_index_base=lane_index,
                        )
                    if wrote_process:
                        events = events[1:]
                    else:
                        wrote_process = True
                    for event in events:
                        _record_trace_event(validation, event)
                        if not first_event:
                            f.write(', ')
                        json.dump(event, f)
                        first_event = False
                        total_events += 1
                    st.extend(lane_stats)
                base += 10
                print(f'[{pname}] lanes={len(st)} kernels={[s[1] for s in st]} stage_spans={[s[3] for s in st]} mori_hostsend={[s[4] for s in st]} kv_transfer={[s[5] for s in st]} kv_depth={[s[6] for s in st]}')
                summary.append((pname, st))
        f.write('], "displayTimeUnit": "ns"}')
    total = sum((len(st) for _, st in summary))
    global _LAST_TRACE_VALIDATION
    _LAST_TRACE_VALIDATION = validation
    print(f'[multinode] wrote {out} ({total_events} events) -- {total} GPU lanes across {len(summary)} processes (xP={len(prefill_dirs)} yD={len(decode_dirs)})')

def write_aggregated(a, out):
    pe, pk, pm, ps, pmo = engine_events(a.prefill_kernel, a.prefill_marker, 10, 'PREFILL (NODE0)', probe_only=a.probe_only)
    de, dk, dm, ds, dmo = engine_events(a.decode_kernel, a.decode_marker, 20, 'DECODE (NODE1)', probe_only=a.probe_only)
    with open(out, 'w') as f:
        json.dump({'traceEvents': pe + de, 'displayTimeUnit': 'ns'}, f)
    print(f'[aggregated] PREFILL: kernels={pk} reqstats={pm} stage_spans={ps} mori={pmo}')
    print(f'[aggregated] DECODE : kernels={dk} reqstats={dm} stage_spans={ds} mori={dmo}')
    print(f'[aggregated] wrote {out} ({len(pe) + len(de)} events)')

def _legacy_build_trace():
    ap = argparse.ArgumentParser(description='Clean per-GPU Perfetto/Chrome-JSON trace builder. Two input modes: (1) single-file 1P1D (--prefill-kernel/-marker + --decode-kernel/-marker); (2) TOPOLOGY-AWARE multi-node (--prefill-dir ... --decode-dir ...) which auto-scales to 1P1D/2P2D/xPyD with global TP-rank labels.')
    ap.add_argument('--prefill-dir', nargs='+', help='rocprof_prefill_NODE* dir(s) -- topology-aware multi-node mode')
    ap.add_argument('--decode-dir', nargs='+', help='rocprof_decode_NODE* dir(s) -- topology-aware multi-node mode')
    ap.add_argument('--prefill-kernel')
    ap.add_argument('--prefill-marker')
    ap.add_argument('--decode-kernel')
    ap.add_argument('--decode-marker')
    ap.add_argument('--out', required=True)
    ap.add_argument('--probe-only', action='store_true', help='legacy: trim to the final request cluster after a >30s gap')
    ap.add_argument('--rid-prefix', help='select requests whose ReqTimeStats RID starts with this prefix')
    ap.add_argument('--request-logs', nargs='*', default=[], help='ReqTimeStats logs used with --rid-prefix')
    ap.add_argument('--expect-workers', type=int, help='validate the number of request-marker worker lanes')
    ap.add_argument('--no-aggregated', action='store_true', help='skip the secondary <out>_aggregated.json merged view (mode 1 only)')
    ap.add_argument('--no-perlane', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--no-reqstats-lanes', action='store_true', dest='no_reqstats_lanes', help="exclude reqstats marker and per-request stage-span lanes from trace JSON; direct correlate/reqstats analysis remains unaffected")
    a = ap.parse_args()
    rid_rooms = None
    if a.rid_prefix:
        room_to_request = load_request_ids(a.request_logs)
        rid_rooms = {
            room
            for room, rid in room_to_request.items()
            if rid.startswith(a.rid_prefix)
        }
        if not rid_rooms:
            ap.error(f'no ReqTimeStats requests matched --rid-prefix={a.rid_prefix!r}')
    if a.prefill_dir or a.decode_dir:
        if not (a.prefill_dir and a.decode_dir):
            ap.error('--prefill-dir and --decode-dir must be given together')
        write_multinode(
            sorted(a.prefill_dir),
            sorted(a.decode_dir),
            a.out,
            a.probe_only,
            no_reqstats=a.no_reqstats_lanes,
            rid_rooms=rid_rooms,
        )
        return
    missing = [n for n in ('prefill_kernel', 'prefill_marker', 'decode_kernel', 'decode_marker') if getattr(a, n) is None]
    if missing:
        ap.error('single-file mode needs --%s (or use --prefill-dir/--decode-dir)' % ', --'.join((m.replace('_', '-') for m in missing)))
    write_pergpu(a, a.out)
    if not a.no_aggregated:
        agg_out = a.out[:-5] + '_aggregated.json' if a.out.endswith('.json') else a.out + '_aggregated'
        write_aggregated(a, agg_out)

# ---- MORI/request correlation ----
STAGES_INNER = ('prefill_kv_transfer_start', 'prefill_kv_transfer_finish')
STAGES_OUTER = ('forward_entry', 'completion')
_MAP_RE = re.compile('\\broom=(\\d+)\\s+uid=(\\d+)')
_BYTES_RE = re.compile('\\bbytes=(\\d+)')

def _bytes(fn):
    m = _BYTES_RE.search(fn)
    return int(m.group(1)) if m else 0

def _basename(fn):
    return re.sub('\\s+id=\\S+$', '', fn)

def _idtag(fn):
    m = re.search('\\bid=(\\S+)$', fn)
    return m.group(1) if m else None

def _is_real_rid(idv):
    return idv is not None and idv != '0' and idv.isdigit() and (len(idv) >= 12)
_REQ_RE = re.compile('ReqTimeStats\\(rid=(?P<rid>[^,]+), bootstrap_room=(?P<room>\\d+),')

def load_request_ids(logpaths):
    """Return bootstrap_room -> stable SGLang request ID from ReqTimeStats logs."""
    out = {}
    for path in logpaths or []:
        if not path or not os.path.exists(path):
            continue
        with open(path, errors='ignore') as fh:
            for line in fh:
                m = _REQ_RE.search(line)
                if m and _is_real_rid(m.group('room')):
                    out[m.group('room')] = m.group('rid')
    return out

def _latest_probe_bounds(path, gap_ns=30000000000, pad_ns=2000000):
    """Return the final RID-tagged request cluster bounds without retaining the sweep."""
    lo = hi = prev = None
    with open(path, newline='') as fh:
        rows = csv.reader(fh)
        next(rows, None)
        for row in rows:
            if len(row) < 7 or not row[1].startswith('reqstats.sched.prefill.'):
                continue
            if not _is_real_rid(_idtag(row[1])):
                continue
            try:
                ts = int(row[5])
            except ValueError:
                continue
            if prev is not None and ts - prev > gap_ns:
                lo = ts
            elif lo is None:
                lo = ts
            hi = prev = ts
    return (lo - pad_ns, hi + pad_ns) if lo is not None else None

def _rid_bounds(path, rid_rooms, pad_ns=2000000):
    """Return marker-clock bounds for explicitly selected bootstrap rooms."""
    stamps = []
    with open(path, newline='') as fh:
        rows = csv.reader(fh)
        next(rows, None)
        for row in rows:
            if len(row) < 7 or not row[1].startswith('reqstats.sched.'):
                continue
            if _idtag(row[1]) not in rid_rooms:
                continue
            try:
                stamps.append(int(row[5]))
            except ValueError:
                continue
    if not stamps:
        return None
    return (min(stamps) - pad_ns, max(stamps) + pad_ns)


def parse_pid_file(path, probe_only=False, rid_rooms=None):
    """Return (rid_stamps, mori_ranges, uid_to_room) for one TP-worker marker CSV.
    rid_stamps: rid -> {stage_basename_suffix: ts}
    mori_ranges: list of (kind, start, end, idv) with kind in {io, rdma, kvt};
                 idv is the transfer_uid string from `id=N` (None for io, which has none)
    uid_to_room: transfer_uid(str) -> bootstrap_room(str) harvested from `mori.map` marks
    """
    rid_stamps = defaultdict(dict)
    mori = []
    uid_to_room = {}
    bounds = (
        _rid_bounds(path, rid_rooms)
        if rid_rooms is not None
        else (_latest_probe_bounds(path) if probe_only else None)
    )
    if rid_rooms is not None and bounds is None:
        return (rid_stamps, mori, uid_to_room)
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) < 7:
                continue
            fn = row[1]
            try:
                start = int(row[5])
                end = int(row[6])
            except ValueError:
                continue
            if bounds is not None and (not bounds[0] <= start <= bounds[1]):
                continue
            if fn.startswith('reqstats.sched.prefill.'):
                idv = _idtag(fn)
                if _is_real_rid(idv) and (
                    rid_rooms is None or idv in rid_rooms
                ):
                    stage = _basename(fn).split('reqstats.sched.prefill.')[-1]
                    if stage not in rid_stamps[idv]:
                        rid_stamps[idv][stage] = start
            elif fn.startswith('mori.map'):
                m = _MAP_RE.search(fn)
                if m:
                    uid_to_room[m.group(2)] = m.group(1)
            elif fn.startswith('mori.io.engine_batch_write'):
                mori.append(('io', start, end, None, 0))
            elif fn.startswith('mori.rdma.batch_post.write'):
                mori.append(('rdma', start, end, _idtag(fn), _bytes(fn)))
            elif fn.startswith('mori.rdma.kv_transfer'):
                mori.append(('kvt', start, end, _idtag(fn), _bytes(fn)))
    return (rid_stamps, mori, uid_to_room)

def windows_for(rid_stamps):
    """rid -> dict(inner=(s,e)|None, outer=(s,e)|None)."""
    out = {}
    for rid, st in rid_stamps.items():
        inner = None
        outer = None
        if STAGES_INNER[0] in st and STAGES_INNER[1] in st:
            a, b = (st[STAGES_INNER[0]], st[STAGES_INNER[1]])
            if b >= a:
                inner = (a, b)
        if STAGES_OUTER[0] in st and STAGES_OUTER[1] in st:
            a, b = (st[STAGES_OUTER[0]], st[STAGES_OUTER[1]])
            if b >= a:
                outer = (a, b)
        out[rid] = {'inner': inner, 'outer': outer}
    return out

def assign(mid, wins, key):
    """Return list of (rid, width) whose `key` window contains mid."""
    hits = []
    for rid, w in wins.items():
        iv = w[key]
        if iv and iv[0] <= mid <= iv[1]:
            hits.append((rid, iv[1] - iv[0]))
    return hits

def aggregate_by_room(
    prefill_dir,
    tp_glob='*marker_api_trace.csv',
    probe_only=False,
    rid_rooms=None,
):
    """Reusable per-room (== bootstrap_room) MORI aggregation, using the SAME
    parse/attribution logic as main() (exact-by-id via mori.map first, then per-pid
    timestamp containment fallback). Returns a dict:

        { room(str): {
            "mori_io_sends", "mori_rdma_posts", "mori_kv_transfers",   # counts
            "io_dur_us", "rdma_dur_us", "kv_transfer_dur_us",          # summed durations
            "rdma_bytes", "kv_transfer_bytes", "kv_eff_bw_GBps",       # bytes + eff BW
            "via_exact", "via_inner", "via_outer",                    # attribution tally
        } }

    Only rooms that had >=1 MORI range assigned are present. This is consumed by
    the reqstats subcommand so the per-request CSV's MORI columns (incl. the
    post->CQ "mori io time") are IDENTICAL to this tool's `--out-csv`. main() is
    left untouched (this is purely additive)."""
    agg = defaultdict(lambda: {'io': 0, 'rdma': 0, 'kvt': 0, 'io_dur': 0, 'rdma_dur': 0, 'kvt_dur': 0, 'rdma_bytes': 0, 'kvt_bytes': 0, 'via_exact': 0, 'via_inner': 0, 'via_outer': 0})
    for f in sorted(glob.glob(os.path.join(prefill_dir, tp_glob))):
        rid_stamps, mori, uid_to_room = parse_pid_file(
            f, probe_only=probe_only, rid_rooms=rid_rooms
        )
        wins = windows_for(rid_stamps)
        for kind, s, e, idv, nbytes in mori:
            mid = (s + e) // 2
            rid = None
            via = None
            room = uid_to_room.get(idv) if idv is not None else None
            if rid_rooms is not None and _is_real_rid(room) and room not in rid_rooms:
                continue
            if room is not None and _is_real_rid(room):
                rid = room
                via = 'exact'
            else:
                hits = assign(mid, wins, 'inner')
                via = 'inner'
                if not hits:
                    hits = assign(mid, wins, 'outer')
                    via = 'outer'
                if not hits:
                    continue
                hits.sort(key=lambda x: x[1])
                best_w = hits[0][1]
                tied = [h for h in hits if h[1] == best_w]
                if len(tied) > 1:
                    continue
                rid = tied[0][0]
            a = agg[rid]
            a[kind] += 1
            a[kind + '_dur'] += e - s
            if kind in ('rdma', 'kvt'):
                a[kind + '_bytes'] += nbytes
            a['via_' + via] += 1
    out = {}
    for room, a in agg.items():
        bw = a['kvt_bytes'] / a['kvt_dur'] if a['kvt_dur'] else 0.0
        out[room] = {'mori_io_sends': a['io'], 'mori_rdma_posts': a['rdma'], 'mori_kv_transfers': a['kvt'], 'io_dur_us': round(a['io_dur'] / 1000.0, 1), 'rdma_dur_us': round(a['rdma_dur'] / 1000.0, 1), 'kv_transfer_dur_us': round(a['kvt_dur'] / 1000.0, 1), 'rdma_bytes': a['rdma_bytes'], 'kv_transfer_bytes': a['kvt_bytes'], 'kv_eff_bw_GBps': round(bw, 3), 'via_exact': a['via_exact'], 'via_inner': a['via_inner'], 'via_outer': a['via_outer']}
    return out

def _legacy_correlate():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefill-dir', required=True, nargs='+', help='one or more rocprof_prefill_NODE* directories')
    ap.add_argument('--prefill-logs', nargs='*', default=[], help='ReqTimeStats logs used for request_id <-> bootstrap_room')
    ap.add_argument('--tp-glob', default='*marker_api_trace.csv')
    ap.add_argument('--probe-only', action='store_true', help='legacy: correlate only the final cluster after a >30s gap')
    ap.add_argument('--rid-prefix', help='correlate only ReqTimeStats RIDs with this prefix')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--require-complete', action='store_true', help='fail unless every KV-transfer has exact mori.map and request-ID attribution')
    args = ap.parse_args()
    files = []
    for d in args.prefill_dir:
        files.extend(glob.glob(os.path.join(d, args.tp_glob)))
    files = sorted(files)
    if not files:
        print(f'NO marker CSVs under {args.prefill_dir} / {args.tp_glob}', file=sys.stderr)
        sys.exit(2)
    room_to_request = load_request_ids(args.prefill_logs)
    rid_rooms = None
    if args.rid_prefix:
        rid_rooms = {
            room
            for room, rid in room_to_request.items()
            if rid.startswith(args.rid_prefix)
        }
        if not rid_rooms:
            ap.error(f'no ReqTimeStats requests matched --rid-prefix={args.rid_prefix!r}')
    agg = defaultdict(lambda: {'io': 0, 'rdma': 0, 'kvt': 0, 'io_dur': 0, 'rdma_dur': 0, 'kvt_dur': 0, 'rdma_bytes': 0, 'kvt_bytes': 0, 'pids': set(), 'via_exact': 0, 'via_inner': 0, 'via_outer': 0, 'inner_w': [], 'outer_w': []})
    n_mori = 0
    n_assigned = 0
    n_unassigned = 0
    n_ambiguous = 0
    n_io = 0
    n_rdma = 0
    n_kvt = 0
    n_exact = 0
    n_time = 0
    n_kvt_assigned = 0
    n_kvt_exact = 0
    n_kvt_unassigned = 0
    n_kvt_ambiguous = 0
    geom = {'mori_inside_kv': 0, 'kv_inside_mori': 0, 'overlap_partial': 0, 'kv_missing': 0}
    pid_count = 0
    rid_global_pids = defaultdict(set)
    for f in files:
        pid = os.path.basename(f).split('_')[1] if '_' in os.path.basename(f) else os.path.basename(f)
        rid_stamps, mori, uid_to_room = parse_pid_file(
            f, probe_only=args.probe_only, rid_rooms=rid_rooms
        )
        wins = windows_for(rid_stamps)
        if mori:
            pid_count += 1
        for rid in wins:
            rid_global_pids[rid].add(pid)
        for kind, s, e, idv, nbytes in mori:
            room = uid_to_room.get(idv) if idv is not None else None
            if rid_rooms is not None and _is_real_rid(room) and room not in rid_rooms:
                continue
            n_mori += 1
            if kind == 'io':
                n_io += 1
            elif kind == 'rdma':
                n_rdma += 1
            else:
                n_kvt += 1
            mid = (s + e) // 2
            rid = None
            via = None
            if room is not None and _is_real_rid(room):
                rid = room
                via = 'exact'
            else:
                hits = assign(mid, wins, 'inner')
                via = 'inner'
                if not hits:
                    hits = assign(mid, wins, 'outer')
                    via = 'outer'
                if not hits:
                    n_unassigned += 1
                    if kind == 'kvt':
                        n_kvt_unassigned += 1
                    continue
                hits.sort(key=lambda x: x[1])
                best_w = hits[0][1]
                tied = [h for h in hits if h[1] == best_w]
                if len(tied) > 1:
                    n_ambiguous += 1
                    if kind == 'kvt':
                        n_kvt_ambiguous += 1
                    continue
                rid = tied[0][0]
            n_assigned += 1
            if kind == 'kvt':
                n_kvt_assigned += 1
                if via == 'exact':
                    n_kvt_exact += 1
            if via == 'exact':
                n_exact += 1
            else:
                n_time += 1
            a = agg[rid]
            a[kind] += 1
            a[kind + '_dur'] += e - s
            if kind in ('rdma', 'kvt'):
                a[kind + '_bytes'] += nbytes
            a['pids'].add(pid)
            a['via_' + via] += 1
            kv = wins.get(rid, {}).get('inner')
            if kv is None:
                geom['kv_missing'] += 1
            else:
                if kv[0] <= s and e <= kv[1]:
                    geom['mori_inside_kv'] += 1
                elif s <= kv[0] and kv[1] <= e:
                    geom['kv_inside_mori'] += 1
                else:
                    geom['overlap_partial'] += 1
                a['inner_w'].append(kv[1] - kv[0])
    all_rids = set(rid_rooms) if rid_rooms is not None else set(rid_global_pids.keys()) | set(agg.keys())
    rids_with_mori = set(agg.keys())
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['request_id', 'bootstrap_room', 'n_pids_seen', 'n_pids_with_mori', 'mori_io_sends', 'mori_rdma_posts', 'mori_kv_transfers', 'io_dur_us', 'rdma_dur_us', 'kv_transfer_dur_us', 'assigned_via_exact', 'assigned_via_inner', 'assigned_via_outer', 'kv_window_us_avg', 'rdma_bytes', 'kv_transfer_bytes', 'kv_eff_bw_GBps'])
        for rid in sorted(all_rids):
            a = agg.get(rid)
            if a:
                kvw = sum(a['inner_w']) / len(a['inner_w']) / 1000.0 if a['inner_w'] else 0.0
                bw = a['kvt_bytes'] / a['kvt_dur'] if a['kvt_dur'] else 0.0
                w.writerow([room_to_request.get(rid, ''), rid, len(rid_global_pids[rid]), len(a['pids']), a['io'], a['rdma'], a['kvt'], round(a['io_dur'] / 1000.0, 1), round(a['rdma_dur'] / 1000.0, 1), round(a['kvt_dur'] / 1000.0, 1), a['via_exact'], a['via_inner'], a['via_outer'], round(kvw, 1), a['rdma_bytes'], a['kvt_bytes'], round(bw, 3)])
            else:
                w.writerow([room_to_request.get(rid, ''), rid, len(rid_global_pids[rid]), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0.0])
    lines = []
    lines.append('# MORI-IO -> request correlation summary')
    lines.append('')
    lines.append(f'- TP-worker marker CSVs parsed: {len(files)} (pids with mori marks: {pid_count})')
    lines.append(f'- real rids (non-id=0) seen on reqstats.sched.prefill: {len(all_rids)}')
    lines.append(f'- real rids that got >=1 MORI send assigned: {len(rids_with_mori)}')
    lines.append(f'- MORI ranges total: {n_mori}  (io={n_io}, rdma={n_rdma}, kv_transfer={n_kvt})')
    _tot_rdma_b = sum((a['rdma_bytes'] for a in agg.values()))
    _tot_kvt_b = sum((a['kvt_bytes'] for a in agg.values()))
    _tot_kvt_dur = sum((a['kvt_dur'] for a in agg.values()))
    _agg_bw = _tot_kvt_b / _tot_kvt_dur if _tot_kvt_dur else 0.0
    lines.append(f'- bytes (optional bytes= token): rdma_batch_post={_tot_rdma_b} B, kv_transfer={_tot_kvt_b} B; aggregate kv effective BW = {_agg_bw:.3f} GB/s (0 if pre-bytes trace)')
    lines.append(f'- assigned: {n_assigned}   unassigned: {n_unassigned}   ambiguous(>1 tightest): {n_ambiguous}')
    lines.append(f'- attribution method: exact-by-id (mori.map) = {n_exact}   by-time (containment fallback) = {n_time}')
    lines.append(f'- KV-transfer mapping: total={n_kvt} assigned={n_kvt_assigned} exact={n_kvt_exact} unassigned={n_kvt_unassigned} ambiguous={n_kvt_ambiguous}')
    lines.append(f'- ReqTimeStats request-ID bridges loaded: {len(room_to_request)}')
    lines.append(f'- assigned_check: assigned+unassigned+ambiguous = {n_assigned + n_unassigned + n_ambiguous} (== total {n_mori}: {n_assigned + n_unassigned + n_ambiguous == n_mori})')
    lines.append(f'- method_check: exact+by-time = {n_exact + n_time} (== assigned {n_assigned}: {n_exact + n_time == n_assigned})')
    lines.append('')
    lines.append("## geometry: mori range vs the rid's inner KV-transfer window")
    lines.append(f"- mori range fully INSIDE kv window : {geom['mori_inside_kv']}")
    lines.append(f"- kv window fully INSIDE mori range : {geom['kv_inside_mori']}")
    lines.append(f"- partial overlap only             : {geom['overlap_partial']}")
    lines.append(f"- kv window missing for that rid   : {geom['kv_missing']}")
    lines.append('')
    lines.append('## per-rid (aggregated over pids)')
    lines.append('| rid | pids_with_mori | io_sends | rdma_posts | kv_transfers | io_dur_us | rdma_dur_us | kv_dur_us | via_exact | via_inner | via_outer | kv_bytes | kv_GBps |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for rid in sorted(all_rids):
        a = agg.get(rid)
        if a:
            _bw = a['kvt_bytes'] / a['kvt_dur'] if a['kvt_dur'] else 0.0
            lines.append(f"| {rid} | {len(a['pids'])} | {a['io']} | {a['rdma']} | {a['kvt']} | {a['io_dur'] / 1000.0:.1f} | {a['rdma_dur'] / 1000.0:.1f} | {a['kvt_dur'] / 1000.0:.1f} | {a['via_exact']} | {a['via_inner']} | {a['via_outer']} | {a['kvt_bytes']} | {_bw:.3f} |")
        else:
            lines.append(f'| {rid} | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.000 |')
    summary = '\n'.join(lines) + '\n'
    with open(args.out_summary, 'w') as fh:
        fh.write(summary)
    print(summary)
    if args.require_complete:
        mapped_rooms = {room for room, a in agg.items() if a['kvt']}
        missing_request_ids = sorted((room for room in mapped_rooms if not room_to_request.get(room)))
        errors = []
        if rid_rooms is not None:
            output_rows = [(room_to_request.get(room, ''), room) for room in sorted(all_rids)]
            expected_rows = {(room_to_request[room], room) for room in rid_rooms}
            if len(output_rows) != len(expected_rows) or set(output_rows) != expected_rows:
                errors.append('filtered output rows do not exactly match selected requests')
            if any(not request_id or not room for request_id, room in output_rows):
                errors.append('filtered output contains blank request or room IDs')
            if len({request_id for request_id, _ in output_rows}) != len(output_rows) or len({room for _, room in output_rows}) != len(output_rows):
                errors.append('filtered output contains duplicate request or room IDs')
        if n_kvt == 0:
            errors.append('no mori.rdma.kv_transfer markers found')
        if n_kvt_assigned != n_kvt or n_kvt_unassigned or n_kvt_ambiguous:
            errors.append(f'KV transfers not uniquely assigned ({n_kvt_assigned}/{n_kvt})')
        if n_kvt_exact != n_kvt:
            errors.append(f'mori.map exact attribution incomplete ({n_kvt_exact}/{n_kvt})')
        if missing_request_ids:
            errors.append(f'missing ReqTimeStats request IDs for {len(missing_request_ids)} bootstrap rooms')
        if errors:
            for err in errors:
                print(f'[correlate_mori] ERROR: {err}', file=sys.stderr)
            sys.exit(3)

# ---- per-request statistics ----
RUN_OUT_BASE = os.environ.get('RUN_OUT_BASE', '/shared_inference/%s/model_blog_logs' % (os.environ.get('USER') or 'aarai'))
NS = 1000000.0

def _splitlist(vals):
    """Accept nargs='+' AND comma-separated; flatten + drop empties."""
    out = []
    if not vals:
        return out
    for v in vals:
        out.extend((x for x in str(v).split(',') if x))
    return out

def _is_real_room(r):
    return r is not None and r != '0' and r.isdigit() and (len(r) >= 12)

def collect_stage_ts(dirs, side):
    """side in {prefill,decode}. dirs is a LIST of rocprof_*_NODE* dirs (one per node
    on that side). Returns room -> {stage: ts_ns} (earliest across ALL TP workers on
    ALL nodes of that side), and room -> last decode_finish ts."""
    pref = f'reqstats.sched.{side}.'
    stamps = defaultdict(dict)
    dfin_last = defaultdict(int)
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*marker_api_trace.csv'))))
    for f in files:
        with open(f, newline='') as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) < 7:
                    continue
                fn = row[1]
                if not fn.startswith(pref):
                    continue
                m = re.search('\\bid=(\\S+)$', fn)
                room = m.group(1) if m else None
                if not _is_real_room(room):
                    continue
                try:
                    ts = int(row[5])
                except ValueError:
                    continue
                stage = fn.split(pref)[-1].split(' id=')[0]
                if stage == 'decode_finish':
                    if ts > dfin_last[room]:
                        dfin_last[room] = ts
                    continue
                if stage not in stamps[room] or ts < stamps[room][stage]:
                    stamps[room][stage] = ts
    return (stamps, dfin_last)

def _dur_ms(st, a, b):
    if a in st and b in st and (st[b] >= st[a]):
        return round((st[b] - st[a]) / NS, 3)
    return ''

def prefill_derived(st):
    return {'pm_bootstrap_ms': _dur_ms(st, 'prefill_bootstrap_queue_entry', 'bootstrap_done'), 'pm_queue_ms': _dur_ms(st, 'wait_queue_entry', 'forward_entry'), 'pm_forward_ms': _dur_ms(st, 'forward_entry', 'prefill_finished'), 'pm_kv_transfer_ms': _dur_ms(st, 'prefill_kv_transfer_start', 'prefill_kv_transfer_finish'), 'pm_total_ms': _dur_ms(st, 'recv', 'completion'), 'pm_recv_ns': st.get('recv', ''), 'pm_forward_entry_ns': st.get('forward_entry', ''), 'pm_prefill_finished_ns': st.get('prefill_finished', ''), 'pm_kv_start_ns': st.get('prefill_kv_transfer_start', ''), 'pm_kv_finish_ns': st.get('prefill_kv_transfer_finish', ''), 'pm_completion_ns': st.get('completion', '')}

def decode_derived(st, dfin_last):
    fwd_end = dfin_last if dfin_last else st.get('completion', None)
    dm_forward = ''
    if 'forward_entry' in st and fwd_end and (fwd_end >= st['forward_entry']):
        dm_forward = round((fwd_end - st['forward_entry']) / NS, 3)
    return {'dm_queue_ms': _dur_ms(st, 'wait_queue_entry', 'forward_entry'), 'dm_forward_ms': dm_forward, 'dm_total_ms': _dur_ms(st, 'recv', 'completion'), 'dm_recv_ns': st.get('recv', ''), 'dm_forward_entry_ns': st.get('forward_entry', ''), 'dm_completion_ns': st.get('completion', ''), 'dm_last_decode_finish_ns': dfin_last or ''}
RTS = re.compile('ReqTimeStats\\(rid=(?P<rid>[^,]+), bootstrap_room=(?P<room>\\d+), input_len=(?P<il>\\d+), cached_input_len=(?P<cil>\\d+), output_len=(?P<ol>\\d+), (?:attempts=\\d+, )?type=(?P<type>\\w+)\\):(?P<rest>.*)')

def parse_reqtimestats(logpaths):
    """logpaths is a LIST of engine logs (one per node on that side). Returns
    room -> dict of log fields (last occurrence wins across all logs)."""
    out = {}
    for logpath in logpaths:
        if not logpath or not os.path.exists(logpath):
            continue
        with open(logpath, errors='ignore') as fh:
            for line in fh:
                m = RTS.search(line)
                if not m:
                    continue
                room = m.group('room')
                if not _is_real_room(room):
                    continue
                d = {'rid': m.group('rid'), 'bootstrap_room': room, 'input_len': m.group('il'), 'cached_input_len': m.group('cil'), 'output_len': m.group('ol')}
                for k, v in re.findall('([#\\w]+)=([\\d.]+)', m.group('rest')):
                    d[k] = v
                out[room] = d
    return out
_MORI_NUM_KEYS = ('mori_io_sends', 'mori_rdma_posts', 'mori_kv_transfers', 'io_dur_us', 'rdma_dur_us', 'kv_transfer_dur_us', 'rdma_bytes', 'kv_transfer_bytes', 'via_exact', 'via_inner', 'via_outer')

def _merge_mori(per_dir):
    """Merge correlate_mori.aggregate_by_room() outputs across prefill dirs. A given
    bootstrap_room is served by ONE prefill node, but we SUM defensively; effective
    BW is recomputed from the summed bytes/duration."""
    out = {}
    for dct in per_dir:
        for room, m in dct.items():
            o = out.setdefault(room, {k: 0 for k in _MORI_NUM_KEYS})
            for k in _MORI_NUM_KEYS:
                o[k] = o.get(k, 0) + (m.get(k, 0) or 0)
    for room, o in out.items():
        dur_ns = o['kv_transfer_dur_us'] * 1000.0
        o['kv_eff_bw_GBps'] = round(o['kv_transfer_bytes'] / dur_ns, 3) if dur_ns else 0.0
    return out

def mori_per_room(prefill_dirs, rid_rooms=None):
    per_dir = [
        aggregate_by_room(
            d, probe_only=(rid_rooms is None), rid_rooms=rid_rooms
        )
        for d in prefill_dirs
    ]
    return _merge_mori(per_dir)

def _client_ms(row, start_key, end_key):
    try:
        return round((int(row[end_key]) - int(row[start_key])) / NS, 3)
    except (KeyError, TypeError, ValueError):
        return ''

def load_client(client_csv):
    out = {}
    if not client_csv or not os.path.exists(client_csv):
        return out
    with open(client_csv, newline='') as fh:
        for row in csv.DictReader(fh):
            out[row.get('rid', '')] = row
    return out

def load_manifest_rids(client_manifest):
    if not client_manifest or not os.path.exists(client_manifest):
        return set()
    with open(client_manifest) as fh:
        manifest = json.load(fh)
    return {
        request.get('rid', '')
        for request in manifest.get('requests', [])
        if request.get('rid')
    }

def _legacy_reqstats():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job')
    ap.add_argument('--xp', type=int, default=1, help='# prefill nodes (auto-derive dirs/logs from --job)')
    ap.add_argument('--yd', type=int, default=1, help='# decode nodes (auto-derive dirs/logs from --job)')
    ap.add_argument('--prefill-dir')
    ap.add_argument('--decode-dir')
    ap.add_argument('--prefill-log')
    ap.add_argument('--decode-log')
    ap.add_argument('--prefill-dirs', nargs='+')
    ap.add_argument('--decode-dirs', nargs='+')
    ap.add_argument('--prefill-logs', nargs='+')
    ap.add_argument('--decode-logs', nargs='+')
    ap.add_argument('--client-csv')
    ap.add_argument('--client-manifest')
    ap.add_argument('--rid-prefix', help='keep only requests whose ReqTimeStats RID starts with this prefix')
    ap.add_argument('--out-dir')
    ap.add_argument('--splits', action='store_true', help='also write _prefill/_decode CSVs')
    ap.add_argument('--require-data', action='store_true', help='fail if marker/log request data cannot be reconstructed')
    ap.add_argument('--require-client', action='store_true', help='fail unless every tagged client RID joins to ReqTimeStats')
    ap.add_argument('--no-mori', action='store_true', help='skip MoRI attribution and leave MoRI columns zero')
    a = ap.parse_args()
    J = a.job or 'run'
    base = os.path.join(RUN_OUT_BASE, J) if a.job else None
    if _splitlist(a.prefill_dirs):
        pdirs = _splitlist(a.prefill_dirs)
    elif a.prefill_dir:
        pdirs = [a.prefill_dir]
    elif base:
        pdirs = [os.path.join(base, f'rocprof_prefill_NODE{i}') for i in range(a.xp)]
    else:
        pdirs = []
    if _splitlist(a.decode_dirs):
        ddirs = _splitlist(a.decode_dirs)
    elif a.decode_dir:
        ddirs = [a.decode_dir]
    elif base:
        ddirs = [os.path.join(base, f'rocprof_decode_NODE{a.xp + j}') for j in range(a.yd)]
    else:
        ddirs = []
    if _splitlist(a.prefill_logs):
        plogs = _splitlist(a.prefill_logs)
    elif a.prefill_log:
        plogs = [a.prefill_log]
    elif base:
        plogs = [os.path.join(base, f'prefill_NODE{i}.log') for i in range(a.xp)]
    else:
        plogs = []
    if _splitlist(a.decode_logs):
        dlogs = _splitlist(a.decode_logs)
    elif a.decode_log:
        dlogs = [a.decode_log]
    elif base:
        dlogs = [os.path.join(base, f'decode_NODE{a.xp + j}.log') for j in range(a.yd)]
    else:
        dlogs = []
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[reqstats_per_request] xp={a.xp} yd={a.yd}')
    print(f'[reqstats_per_request] prefill dirs: {pdirs}')
    print(f'[reqstats_per_request] decode  dirs: {ddirs}')
    print(f'[reqstats_per_request] prefill logs: {plogs}')
    print(f'[reqstats_per_request] decode  logs: {dlogs}')
    p_stamps, _ = collect_stage_ts(pdirs, 'prefill') if pdirs else ({}, {})
    d_stamps, d_dfin = collect_stage_ts(ddirs, 'decode') if ddirs else ({}, {})
    p_log = parse_reqtimestats(plogs)
    d_log = parse_reqtimestats(dlogs)
    rid_rooms = None
    if a.rid_prefix:
        rid_rooms = {
            room
            for room, row in {**p_log, **d_log}.items()
            if row.get('rid', '').startswith(a.rid_prefix)
        }
        if not rid_rooms:
            ap.error(f'no ReqTimeStats requests matched --rid-prefix={a.rid_prefix!r}')
        p_stamps = {room: value for room, value in p_stamps.items() if room in rid_rooms}
        d_stamps = {room: value for room, value in d_stamps.items() if room in rid_rooms}
        d_dfin = {room: value for room, value in d_dfin.items() if room in rid_rooms}
        p_log = {room: value for room, value in p_log.items() if room in rid_rooms}
        d_log = {room: value for room, value in d_log.items() if room in rid_rooms}
    mori = (
        mori_per_room(pdirs, rid_rooms=rid_rooms)
        if pdirs and not a.no_mori
        else {}
    )
    client = load_client(a.client_csv or (os.path.join(base, 'rocprof_probe_client.csv') if base else None))
    manifest_rids = load_manifest_rids(a.client_manifest)
    if a.client_manifest and not manifest_rids:
        ap.error(f'client manifest is missing or has no requests: {a.client_manifest}')
    if a.rid_prefix and any(
        not rid.startswith(a.rid_prefix) for rid in manifest_rids
    ):
        ap.error('client manifest contains RIDs outside --rid-prefix')
    rooms = set(p_stamps) | set(d_stamps) | set(p_log) | set(d_log)
    cols = ['rid', 'bootstrap_room', 'sides', 'input_len', 'cached_input_len', 'output_len', 'p_bootstrap_ms', 'p_queue_ms', 'p_forward_ms', 'p_entry_time', 'p_transfer_speed_GBps', 'p_transfer_total_MB', 'p_retries', 'pm_bootstrap_ms', 'pm_queue_ms', 'pm_forward_ms', 'pm_kv_transfer_ms', 'pm_total_ms', 'pm_recv_ns', 'pm_forward_entry_ns', 'pm_prefill_finished_ns', 'pm_kv_start_ns', 'pm_kv_finish_ns', 'pm_completion_ns', 'd_bootstrap_ms', 'd_alloc_wait_ms', 'd_transfer_ms', 'd_queue_ms', 'd_forward_ms', 'd_entry_time', 'dm_queue_ms', 'dm_forward_ms', 'dm_total_ms', 'dm_recv_ns', 'dm_forward_entry_ns', 'dm_completion_ns', 'dm_last_decode_finish_ns', 'mori_io_sends', 'mori_rdma_posts', 'mori_io_dur_us', 'mori_kv_transfers', 'mori_io_time_ms', 'mori_kv_bytes', 'mori_kv_eff_bw_GBps', 'client_send_wall_ns', 'client_first_token_wall_ns', 'client_done_wall_ns', 'client_ttft_ms', 'client_e2e_ms', 'cli_e2e_latency_s', 'cli_queue_time_s', 'cli_completion_tokens', 'cli_decode_throughput', 'cli_first_token_ts', 'cli_request_finished_ts']
    rows = []
    for room in rooms:
        pl = p_log.get(room, {})
        dl = d_log.get(room, {})
        rid = pl.get('rid') or dl.get('rid') or ''
        sides = ('P' if room in p_stamps or pl else '') + ('D' if room in d_stamps or dl else '')
        il = pl.get('input_len') or dl.get('input_len') or ''
        cil = pl.get('cached_input_len') or dl.get('cached_input_len') or ''
        ol = dl.get('output_len') or pl.get('output_len') or ''
        pm = prefill_derived(p_stamps.get(room, {}))
        dm = decode_derived(d_stamps.get(room, {}), d_dfin.get(room, 0))
        mo = mori.get(room, {})
        cli = client.get(rid, {})
        row = {'rid': rid, 'bootstrap_room': room, 'sides': sides, 'input_len': il, 'cached_input_len': cil, 'output_len': ol, 'p_bootstrap_ms': pl.get('bootstrap_duration', ''), 'p_queue_ms': pl.get('queue_duration', ''), 'p_forward_ms': pl.get('forward_duration', ''), 'p_entry_time': pl.get('entry_time', ''), 'p_transfer_speed_GBps': pl.get('transfer_speed', ''), 'p_transfer_total_MB': pl.get('transfer_total', ''), 'p_retries': pl.get('#retries', ''), 'd_bootstrap_ms': dl.get('bootstrap_duration', ''), 'd_alloc_wait_ms': dl.get('alloc_wait_duration', ''), 'd_transfer_ms': dl.get('transfer_duration', ''), 'd_queue_ms': dl.get('queue_duration', ''), 'd_forward_ms': dl.get('forward_duration', ''), 'd_entry_time': dl.get('entry_time', ''), 'mori_io_sends': mo.get('mori_io_sends', 0), 'mori_rdma_posts': mo.get('mori_rdma_posts', 0), 'mori_io_dur_us': mo.get('io_dur_us', 0.0), 'mori_kv_transfers': mo.get('mori_kv_transfers', 0), 'mori_io_time_ms': round(mo.get('kv_transfer_dur_us', 0.0) / 1000.0, 4), 'mori_kv_bytes': mo.get('kv_transfer_bytes', 0), 'mori_kv_eff_bw_GBps': mo.get('kv_eff_bw_GBps', 0.0), 'client_send_wall_ns': cli.get('client_send_wall_ns', ''), 'client_first_token_wall_ns': cli.get('client_first_token_wall_ns', ''), 'client_done_wall_ns': cli.get('client_done_wall_ns', ''), 'client_ttft_ms': _client_ms(cli, 'client_send_wall_ns', 'client_first_token_wall_ns'), 'client_e2e_ms': _client_ms(cli, 'client_send_wall_ns', 'client_done_wall_ns'), 'cli_e2e_latency_s': cli.get('mi_e2e_latency', ''), 'cli_queue_time_s': cli.get('mi_queue_time', ''), 'cli_completion_tokens': cli.get('mi_completion_tokens', ''), 'cli_decode_throughput': cli.get('mi_decode_throughput', ''), 'cli_first_token_ts': cli.get('mi_first_token_ts', ''), 'cli_request_finished_ts': cli.get('mi_request_finished_ts', '')}
        row.update(pm)
        row.update(dm)
        rows.append(row)
    rows.sort(key=lambda r: (float(r['p_entry_time']) if r['p_entry_time'] else float('inf'), r['bootstrap_room']))
    merged = os.path.join(out_dir, f'reqstats_per_request_{J}.csv')
    with open(merged, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[reqstats_per_request] wrote {merged}  ({len(rows)} requests, {len(cols)} columns)')
    if a.splits:
        for side, keep in (('prefill', [c for c in cols if not c.startswith(('d_', 'dm_'))]), ('decode', [c for c in cols if not c.startswith(('p_', 'pm_', 'mori_'))])):
            sp = os.path.join(out_dir, f'reqstats_per_request_{J}_{side}.csv')
            with open(sp, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=keep, extrasaction='ignore')
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f'[reqstats_per_request] wrote {sp}')
    row_rids = {r['rid'] for r in rows if r['rid']}
    client_rids = {rid for rid in client if rid}
    joined_rids = row_rids & client_rids
    missing_client = sorted(client_rids - row_rids)
    mori_rids = {r['rid'] for r in rows if r['rid'] and r['mori_kv_transfers']}
    missing_mori = sorted(client_rids - mori_rids) if not a.no_mori else []
    n_join = len(joined_rids)
    n_mori = sum((1 for r in rows if r['mori_io_sends'] or r['mori_kv_transfers']))
    n_iotime = sum((1 for r in rows if r['mori_io_time_ms']))
    print(f'[reqstats_per_request] rids={len(rows)}  client-joined={n_join}/{len(client_rids)}  with-MORI={n_mori}  with-mori_io_time={n_iotime}')
    errors = []
    if a.require_data and (not rooms or not p_stamps or (not p_log)):
        errors.append(f'insufficient request data rooms={len(rooms)} prefill_markers={len(p_stamps)} prefill_logs={len(p_log)}')
    if a.require_client:
        if not client_rids:
            errors.append('tagged client CSV is missing or empty')
        if a.client_manifest and manifest_rids != client_rids:
            errors.append(
                'client CSV/manifest RID mismatch '
                f'(csv_only={sorted(client_rids - manifest_rids)[:3]}, '
                f'manifest_only={sorted(manifest_rids - client_rids)[:3]})'
            )
        if missing_client:
            errors.append(f'{len(missing_client)} tagged client RIDs missing from ReqTimeStats (sample={missing_client[:3]})')
        if not a.no_mori and missing_mori:
            errors.append(f'{len(missing_mori)} tagged client RIDs missing exact MORI KV mappings (sample={missing_mori[:3]})')
    if errors:
        for err in errors:
            print(f'[reqstats_per_request] ERROR: {err}', file=sys.stderr)
        sys.exit(3)

# ---- MIT-attributed kernel categorization ----
# SPDX-License-Identifier: MIT
# Rules derived from ROCm/llmscope layer_detection.py categorize_kernel:
# https://github.com/ROCm/llmscope/blob/di_analysis_branch/llmscope/layer_detection.py
def categorize_kernel(name):
    """Categorize a kernel by its function."""
    n = name.lower()
    if 'rmsnorm' in n or 'fused_rms' in n or 'rms_norm' in n or ('rsqrt' in n and 'mean' in n and ('mul' in n)):
        return 'RMSNorm'
    if 'rope' in n:
        return 'ROPE'
    if 'reshape' in n and 'cache' in n:
        return 'KVCacheReshape'
    if 'kernel_unified_attention' in n:
        return 'Attention'
    if '_fwd_kernel' in name:
        return 'TritonAttention'
    if 'fmha' in n:
        return 'FMHA'
    if 'mla' in n:
        return 'MLA'
    if 'aiter::pa' in name:
        return 'PA'
    if 'paged_attention' in n:
        return 'PagedAttn'
    if 'routing' in n or 'route' in n:
        return 'MoE_Router'
    if 'aiter::fmoe' in name:
        return 'MoE_Fused'
    if 'kernel_moe' in n:
        return 'MoE_Unfused'
    if 'moesorting' in n:
        return 'MoE_Sort'
    if 'topk' in n:
        return 'MoE_TopK'
    if any(x in n for x in ('epdispatchinternode', 'epcombineinternode', 'epdispatchintranode', 'epcombineintranode')):
        return 'MORI EP'
    if 'epdispatch' in n or 'epcombine' in n:
        return 'Communication'
    if 'gemm' in n or 'cijk' in n or 'wvsplit' in n or ('matmul' in n):
        return 'GEMM'
    if 'act_and_mul' in n or 'silu' in n:
        return 'Activation'
    if 'quant' in n:
        return 'Quant'
    if 'allreduce' in n or 'cross_device' in n or 'nccl' in n or ('allgather' in n):
        return 'Communication'
    if 'poi' in n or 'elementwise' in n:
        return 'Elementwise'
    return 'Other'

# ---- kernel bucket generation ----
def _find_column(headers, predicate, what):
    """Return the first header matching ``predicate`` (called on lowercased name)."""
    for h in headers:
        if predicate((h or '').lower()):
            return h
    sys.exit(f'ERROR: could not find the {what} column.\n  Headers found: {headers}')

def find_kernel_name_column(headers):
    """Locate the kernel-name column: contains both 'kernel' and 'name'."""
    return _find_column(headers, lambda h: 'kernel' in h and 'name' in h, "kernel name (a column containing both 'kernel' and 'name')")

def find_duration_sum_column(headers):
    """Locate the per-row total-duration column: contains 'duration' and '_sum'.

    Tolerant of the micro sign (matches whether the header uses 'µs' or 'us').
    """
    return _find_column(headers, lambda h: 'duration' in h and '_sum' in h, "duration sum (a column containing both 'duration' and '_sum')")

def find_duration_count_column(headers):
    """Locate the per-row kernel-count column: contains 'duration' and '_count'."""
    for h in headers:
        hl = (h or '').lower()
        if 'duration' in hl and '_count' in hl:
            return h
    return None

def _duration_or_none(value):
    try:
        value = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None

def _to_int(value):
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0

def process(in_path, out_per_kernel, out_by_category, categorize_kernel, label):
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
            if not row or all(((c or '').strip() == '' for c in row)):
                continue
            total_us = _duration_or_none(row[dur_idx] if dur_idx < len(row) else None)
            if total_us is None:
                continue
            kernel_name = row[name_idx] if name_idx < len(row) else ''
            category = categorize_kernel(kernel_name)
            writer.writerow([category] + row)
            n_kernels = _to_int(row[count_idx]) if count_idx is not None and count_idx < len(row) else 1
            bucket = agg.setdefault(category, [0, 0.0])
            bucket[0] += n_kernels
            bucket[1] += total_us
            grand_count += n_kernels
            grand_us += total_us
    with open(out_by_category, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Num_Kernels', 'Total_us', 'Total_ms', 'Pct_of_Kernel_Time'])
        for category, (n_kernels, total_us) in sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True):
            pct = total_us / grand_us * 100 if grand_us > 0 else 0.0
            writer.writerow([category, n_kernels, f'{total_us:.2f}', f'{total_us / 1000.0:.4f}', f'{pct:.1f}%'])
        writer.writerow(['TOTAL', grand_count, f'{grand_us:.2f}', f'{grand_us / 1000.0:.4f}', '100.0%'])
    _print_summary(label, in_path, headers, name_col, dur_sum_col, count_col, agg, grand_count, grand_us, out_per_kernel, out_by_category)

def _print_summary(label, in_path, headers, name_col, dur_sum_col, count_col, agg, grand_count, grand_us, out_per_kernel, out_by_category):
    print('==== %s ====' % label)
    print('input: %s' % in_path)
    print('kernel-name column:   %r' % name_col)
    print('duration-sum column:  %r' % dur_sum_col)
    print('duration-count column:%s' % (' %r' % count_col if count_col else ' (not found; counted 1 row per kernel)'))
    print('total kernel time: %.1f us  (%.3f ms)   total kernels: %d' % (grand_us, grand_us / 1000.0, grand_count))
    print()
    print('%-15s %10s %15s %9s' % ('Category', '#kernels', 'total_us', '%time'))
    for category, (n_kernels, total_us) in sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True):
        pct = total_us / grand_us * 100 if grand_us > 0 else 0.0
        print('%-15s %10d %15.1f %8.2f%%' % (category, n_kernels, total_us, pct))
    print('%-15s %10d %15.1f %8.2f%%' % ('TOTAL', grand_count, grand_us, 100.0))
    print()
    print('wrote:', out_per_kernel)
    print('wrote:', out_by_category)

def _legacy_buckets():
    p = argparse.ArgumentParser(description='Add local function-based kernel categories onto a TraceLens kernel-summary CSV.')
    p.add_argument('--in', dest='in_path', required=True, help='Input TraceLens kernel_summary CSV.')
    p.add_argument('--out-per-kernel', required=True, help='Output per-kernel CSV (Category + all original columns).')
    p.add_argument('--out-by-category', required=True, help='Output by-category rollup CSV.')
    p.add_argument('--label', default=None, help='Label for the stdout summary banner (default: input path).')
    args = p.parse_args()
    label = args.label if args.label is not None else args.in_path
    process(args.in_path, args.out_per_kernel, args.out_by_category, categorize_kernel, label)

# ---- trimmed kernel summaries ----
OUTPUT_COLUMNS = ['Time', 'Total Time', 'Instances', 'Avg', 'Med', 'Min', 'Max', 'StdDev', 'GridXYZ', 'BlockXYZ', 'VGPR', 'AccumVGPR', 'SGPR', 'LDS', 'Scratch', 'Name', 'Time %', 'Total Time (ns)', 'Avg (ns)', 'Med (ns)', 'Min (ns)', 'Max (ns)', 'StdDev (ns)', 'GridX', 'GridY', 'GridZ', 'BlockX', 'BlockY', 'BlockZ', 'n_trimmed', 'instances_before_trim']

def pretty_ns(ns):
    """Format a nanosecond value the way TraceLens' kernel_summary.csv does:
    ms if >= 1e6 ns, us if >= 1e3 ns, else ns; 3 decimal places."""
    if ns >= 1000000.0:
        return f'{ns / 1000000.0:.3f} ms'
    if ns >= 1000.0:
        return f'{ns / 1000.0:.3f} µs'
    return f'{ns:.3f} ns'

def _n_to_trim(count, trim_pct):
    """Return the number of slowest calls to drop, with at least one when eligible."""
    if trim_pct <= 0:
        return 0
    n = math.ceil(count * trim_pct / 100.0)
    return max(n, 1)

def load_categorize_kernel():
    return categorize_kernel

_KERNEL_RESOURCE_FIELDS = {
    'GridX': 'Grid_Size_X',
    'GridY': 'Grid_Size_Y',
    'GridZ': 'Grid_Size_Z',
    'BlockX': 'Workgroup_Size_X',
    'BlockY': 'Workgroup_Size_Y',
    'BlockZ': 'Workgroup_Size_Z',
    'VGPR': 'VGPR_Count',
    'AccumVGPR': 'Accum_VGPR_Count',
    'SGPR': 'SGPR_Count',
    'LDS': 'LDS_Block_Size',
    'Scratch': 'Scratch_Size',
}
_NORMALIZED_HELP_COLUMNS = [
    'Kernel name',
    'kernel_duration_us_sum',
    'kernel_duration_us_count',
]

def _scan_kernel_traces(kernel_trace_csvs):
    """Pool exact-name dispatch durations from one or more raw worker CSVs."""
    if isinstance(kernel_trace_csvs, (str, os.PathLike)):
        paths = [os.path.abspath(os.fspath(kernel_trace_csvs))]
    else:
        paths = [
            os.path.abspath(os.fspath(path))
            for path in kernel_trace_csvs
        ]
    if not paths:
        raise SystemExit('ERROR: no kernel trace CSV inputs')
    if len(paths) != len(set(paths)):
        raise SystemExit(f'ERROR: duplicate kernel trace CSV inputs: {paths}')

    groups = {}
    input_audits = []
    for kernel_trace_csv in paths:
        input_audit = {
            'path': kernel_trace_csv,
            'raw_row_count': 0,
            'dispatch_row_count': 0,
            'invalid_event_count': 0,
            'included_event_count': 0,
        }
        with open(kernel_trace_csv, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            name_col = 'Kernel_Name' if 'Kernel_Name' in fieldnames else None
            start_col = 'Start_Timestamp' if 'Start_Timestamp' in fieldnames else None
            end_col = 'End_Timestamp' if 'End_Timestamp' in fieldnames else None
            if not (name_col and start_col and end_col):
                raise SystemExit(
                    f'ERROR: {kernel_trace_csv} does not look like a rocprofv3 '
                    'kernel_trace.csv (need Kernel_Name/Start_Timestamp/'
                    f'End_Timestamp, got columns: {fieldnames})'
                )
            for row in reader:
                input_audit['raw_row_count'] += 1
                if row.get('Kind') and row['Kind'] != 'KERNEL_DISPATCH':
                    continue
                input_audit['dispatch_row_count'] += 1
                try:
                    dur_ns = _duration_or_none(
                        float(row[end_col]) - float(row[start_col])
                    )
                except (TypeError, ValueError):
                    dur_ns = None
                if dur_ns is None:
                    input_audit['invalid_event_count'] += 1
                    continue
                name = row[name_col]
                if name not in groups:
                    groups[name] = {
                        'durations': [],
                        'resources': {
                            output: row.get(source, '')
                            for output, source in _KERNEL_RESOURCE_FIELDS.items()
                        },
                    }
                groups[name]['durations'].append(dur_ns)
                input_audit['included_event_count'] += 1
        input_audits.append(input_audit)

    return groups, {
        'inputs': input_audits,
        'pooled_raw_row_count': sum(
            row['raw_row_count'] for row in input_audits
        ),
        'pooled_dispatch_row_count': sum(
            row['dispatch_row_count'] for row in input_audits
        ),
        'pooled_included_row_count': sum(
            row['included_event_count'] for row in input_audits
        ),
        'invalid_event_count': sum(
            row['invalid_event_count'] for row in input_audits
        ),
    }

def _build_summary_rows(groups, trim_pct):
    summary_rows = []
    grand_total_ns = 0.0
    for name, g in groups.items():
        durations = g['durations']
        count_before = len(durations)
        if count_before >= 20 and trim_pct > 0:
            n_trim = _n_to_trim(count_before, trim_pct)
            n_trim = min(n_trim, count_before - 1)
            kept = sorted(durations)[:count_before - n_trim]
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
        summary_rows.append(g['resources'] | {
            'Name': name,
            'Instances': count,
            'instances_before_trim': count_before,
            'n_trimmed': n_trim,
            'Total Time (ns)': total,
            'Avg (ns)': avg,
            'Med (ns)': med,
            'Min (ns)': mn,
            'Max (ns)': mx,
            'StdDev (ns)': stddev,
        })
    summary_rows.sort(key=lambda x: x['Total Time (ns)'], reverse=True)
    for r in summary_rows:
        r['Time %'] = 100.0 * r['Total Time (ns)'] / grand_total_ns if grand_total_ns else 0.0
    return (summary_rows, grand_total_ns)

def build_pooled_kernel_summaries(kernel_trace_csvs, trim_pct):
    """Build canonical untrimmed and post-pooling-trimmed summaries in one scan."""
    groups, audit = _scan_kernel_traces(kernel_trace_csvs)
    normalized_rows, normalized_total_ns = _build_summary_rows(groups, 0)
    trimmed_rows, trimmed_total_ns = _build_summary_rows(groups, trim_pct)
    return (
        normalized_rows,
        normalized_total_ns,
        trimmed_rows,
        trimmed_total_ns,
        audit,
    )

def build_trimmed_summary(kernel_trace_csv, trim_pct):
    """Build a trimmed summary for the explicitly supplied raw kernel CSV."""
    groups, _audit = _scan_kernel_traces(kernel_trace_csv)
    return _build_summary_rows(groups, trim_pct)

def write_csv(rows, out_path, add_category, categorize_kernel, normalized=False):
    columns = list(OUTPUT_COLUMNS[:-2] if normalized else OUTPUT_COLUMNS)
    if normalized:
        columns += _NORMALIZED_HELP_COLUMNS
    if add_category and categorize_kernel is not None:
        columns = ['Category'] + columns
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            grid_xyz = f"{r['GridX']} {r['GridY']} {r['GridZ']}"
            block_xyz = f"{r['BlockX']} {r['BlockY']} {r['BlockZ']}"
            out = [f"{r['Time %']:.1f}%", pretty_ns(r['Total Time (ns)']), r['Instances'], pretty_ns(r['Avg (ns)']), pretty_ns(r['Med (ns)']), pretty_ns(r['Min (ns)']), pretty_ns(r['Max (ns)']), pretty_ns(r['StdDev (ns)']), grid_xyz, block_xyz, r['VGPR'], r['AccumVGPR'], r['SGPR'], r['LDS'], r['Scratch'], r['Name'], r['Time %'], r['Total Time (ns)'], r['Avg (ns)'], r['Med (ns)'], r['Min (ns)'], r['Max (ns)'], r['StdDev (ns)'], r['GridX'], r['GridY'], r['GridZ'], r['BlockX'], r['BlockY'], r['BlockZ'], r['n_trimmed'], r['instances_before_trim']]
            if normalized:
                out = out[:-2] + [
                    r['Name'],
                    f"{r['Total Time (ns)'] / 1000.0:.4f}",
                    r['Instances'],
                ]
            if add_category and categorize_kernel is not None:
                out = [categorize_kernel(r['Name'])] + out
            w.writerow(out)


def run_pooled_kernel_summaries(
    coverage,
    normalized_path,
    trimmed_path,
    trim_pct,
    log_path=None,
):
    """Write canonical all-worker node summaries and provenance."""
    if trim_pct < 0 or trim_pct >= 100:
        raise SystemExit(
            f'ERROR: --trim-pct must be in [0, 100), got {trim_pct}'
        )
    workers = coverage['workers']
    paths = [worker['kernel_csv'] for worker in workers]
    (
        normalized,
        normalized_total_ns,
        trimmed,
        trimmed_total_ns,
        audit,
    ) = build_pooled_kernel_summaries(paths, trim_pct)
    write_csv(
        normalized,
        normalized_path,
        False,
        None,
        normalized=True,
    )
    write_csv(
        trimmed,
        trimmed_path,
        True,
        categorize_kernel,
    )

    provenance = []
    for worker, input_audit in zip(workers, audit['inputs']):
        if os.path.abspath(worker['kernel_csv']) != input_audit['path']:
            raise SystemExit(
                'ERROR: pooled kernel input order diverged from verified '
                'worker source order'
            )
        provenance.append({
            **worker,
            **input_audit,
        })
    audit = dict(audit)
    audit.update({
        'analysis_scope': 'pooled all workers',
        'aggregation_semantics': (
            'raw dispatch rows pooled before exact-name grouping and trimming; '
            'durations and call counts are summed, never averaged'
        ),
        'activity_interpretation': (
            'summed GPU kernel activity; not wall time or utilization'
        ),
        'role': coverage['role'],
        'node_rank': coverage['node_rank'],
        'expected_worker_count': coverage['expected_worker_count'],
        'included_worker_count': coverage['included_worker_count'],
        'rank_source': coverage['rank_source'],
        'ranks_derivable': coverage['ranks_derivable'],
        'selected_pids': [worker['pid'] for worker in workers],
        'selected_local_ranks': [
            worker['local_rank'] for worker in workers
        ],
        'source_files': [worker['kernel_csv'] for worker in workers],
        'workers': provenance,
        'normalized_distinct_kernel_count': len(normalized),
        'normalized_total_ns': normalized_total_ns,
        'trimmed_distinct_kernel_count': len(trimmed),
        'trimmed_total_ns': trimmed_total_ns,
        'trimmed_calls_dropped': sum(
            row['n_trimmed'] for row in trimmed
        ),
    })

    lines = [
        'kernel_analysis_scope=pooled all workers',
        (
            'aggregation_semantics=raw dispatch rows pooled before exact-name '
            'grouping and trimming; durations and call counts are summed, '
            'never averaged'
        ),
        (
            'activity_interpretation=summed GPU kernel activity; '
            'not wall time or utilization'
        ),
        f"role={coverage['role'] if coverage['role'] is not None else 'unknown'}",
        (
            'node_rank='
            + (
                str(coverage['node_rank'])
                if coverage['node_rank'] is not None else 'unknown'
            )
        ),
        f"expected_worker_count={coverage['expected_worker_count']}",
        f"included_worker_count={coverage['included_worker_count']}",
        f"ranks_derivable={str(coverage['ranks_derivable']).lower()}",
        (
            'rank_source='
            + (
                coverage['rank_source']
                if coverage['rank_source'] is not None else 'unavailable'
            )
        ),
    ]
    for worker in provenance:
        rank_fields = ' '.join(
            f"{key}={worker[key] if worker[key] is not None else 'unknown'}"
            for key in ('local_rank', 'dp_rank', 'tp_rank', 'ep_rank')
        )
        lines.append(
            f"pooled_input source_order={worker['source_order']} "
            f"pid={worker['pid']} {rank_fields} "
            f"source_file={worker['kernel_csv']} "
            f"raw_rows={worker['raw_row_count']} "
            f"dispatch_rows={worker['dispatch_row_count']} "
            f"included_rows={worker['included_event_count']} "
            f"invalid_rows={worker['invalid_event_count']}"
        )
    lines.extend([
        (
            f"pooled_total raw_rows={audit['pooled_raw_row_count']} "
            f"dispatch_rows={audit['pooled_dispatch_row_count']} "
            f"included_rows={audit['pooled_included_row_count']} "
            f"invalid_rows={audit['invalid_event_count']}"
        ),
        (
            f"normalized distinct_kernels={len(normalized)} "
            f"total_ns={normalized_total_ns} output={normalized_path}"
        ),
        (
            f"trimmed distinct_kernels={len(trimmed)} "
            f"total_ns={trimmed_total_ns} "
            f"calls_dropped={audit['trimmed_calls_dropped']} "
            f"output={trimmed_path}"
        ),
    ])
    if log_path is not None:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write('\n'.join(lines) + '\n')
    for line in lines:
        print(line)
    return audit


def _legacy_trimmed_summary():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kernel-trace', required=True, help='Raw rocprofv3 *_kernel_trace.csv (per-dispatch rows)')
    ap.add_argument('--out', default=None, help='Output CSV path (default: <stem>_summary_trimmed.csv next to input)')
    ap.add_argument('--trim-pct', type=float, default=5.0, help="Percent of each eligible kernel's slowest calls to drop (default: 5)")
    ap.add_argument('--add-category', action='store_true', help='Prepend a Category column via the vendored categorize_kernel()')
    args = ap.parse_args()
    if not os.path.isfile(args.kernel_trace):
        raise SystemExit(f'ERROR: not found: {args.kernel_trace}')
    if args.trim_pct < 0 or args.trim_pct >= 100:
        raise SystemExit(f'ERROR: --trim-pct must be in [0, 100), got {args.trim_pct}')
    out_path = args.out
    if out_path is None:
        base = os.path.basename(args.kernel_trace)
        stem = base[:-len('_kernel_trace.csv')] if base.endswith('_kernel_trace.csv') else os.path.splitext(base)[0]
        out_path = os.path.join(os.path.dirname(args.kernel_trace) or '.', f'{stem}_kernel_summary_trimmed.csv')
    categorize_kernel = load_categorize_kernel() if args.add_category else None
    if args.add_category and categorize_kernel is None:
        print('WARNING: kernel categorizer unavailable; proceeding without it.', file=sys.stderr)
    rows, grand_total_ns = build_trimmed_summary(args.kernel_trace, args.trim_pct)
    write_csv(rows, out_path, args.add_category, categorize_kernel)
    n_eligible = sum((1 for r in rows if r['instances_before_trim'] >= 20))
    n_trimmed_kernels = sum((1 for r in rows if r['n_trimmed'] > 0))
    total_calls_dropped = sum((r['n_trimmed'] for r in rows))
    print(f'wrote {out_path}')
    print(f'  {len(rows)} distinct kernels; {n_eligible} eligible (>=20 calls); {n_trimmed_kernels} trimmed at {args.trim_pct}%; {total_calls_dropped} total calls dropped')
    print(f'  trimmed grand total: {pretty_ns(grand_total_ns)}')

# ---- consolidated CLI and analysis orchestration ----

def _temporary_output(final_path):
    final = Path(final_path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".tmp", dir=str(final.parent)
    )
    os.close(fd)
    os.unlink(name)
    return final, Path(name)


def _validate_trace(path, expected_workers=None):
    recorded = globals().get("_LAST_TRACE_VALIDATION")
    if (
        recorded is not None
        and recorded.get("path") == os.path.abspath(path)
    ):
        event_count = recorded["events"]
        processes = recorded["processes"]
        worker_lanes = recorded["worker_lanes"]
        slices = recorded["slices"]
    else:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        events = payload.get("traceEvents")
        if not isinstance(events, list) or not events:
            raise SystemExit(f"ERROR: trace has no events: {path}")
        validation = _new_trace_validation(path)
        for event in events:
            _record_trace_event(validation, event)
        event_count = validation["events"]
        processes = validation["processes"]
        worker_lanes = validation["worker_lanes"]
        slices = validation["slices"]

    if len(processes) != len(set(processes)):
        raise SystemExit(f"ERROR: duplicate process names in {path}")
    if not event_count or not processes or not slices:
        raise SystemExit(
            f"ERROR: trace is structurally empty: processes={len(processes)} slices={slices}"
        )
    if expected_workers is not None and worker_lanes != expected_workers:
        raise SystemExit(
            f"ERROR: trace has {worker_lanes} request-marker lanes; "
            f"expected {expected_workers}"
        )
    print(
        f"[trace_tools] validated events={event_count} processes={len(processes)} "
        f"worker_lanes={worker_lanes} slices={slices}"
    )
    return event_count


def _run_absorbed(command, function, argv):
    """Run a preserved CLI implementation with atomic explicit outputs."""
    output_flags = {
        "build-trace": ("--out",),
        "correlate": ("--out-csv", "--out-summary"),
        "buckets": ("--out-per-kernel", "--out-by-category"),
        "trimmed-summary": ("--out",),
    }.get(command, ())
    cooked = list(argv)
    pending = []
    for flag in output_flags:
        if flag not in cooked:
            continue
        index = cooked.index(flag) + 1
        final, temporary = _temporary_output(cooked[index])
        cooked[index] = str(temporary)
        pending.append((final, temporary))

    expected_workers = None
    if command == "build-trace" and "--expect-workers" in cooked:
        index = cooked.index("--expect-workers")
        try:
            expected_workers = int(cooked[index + 1])
        except (IndexError, ValueError):
            raise SystemExit("ERROR: --expect-workers needs an integer")

    old_argv = sys.argv
    try:
        sys.argv = [f"{old_argv[0]} {command}", *cooked]
        result = function()
        if result not in (None, 0):
            return result
        if command == "build-trace":
            output = next((temporary for final, temporary in pending
                           if "--out" in output_flags), None)
            _validate_trace(output, expected_workers)
            if "--prefill-dir" not in cooked and "--no-aggregated" not in cooked:
                aggregate_temporary = Path(
                    str(output) + "_aggregated"
                    if not str(output).endswith(".json")
                    else str(output)[:-5] + "_aggregated.json"
                )
                aggregate_final = next(final for final, temporary in pending)
                aggregate_final = Path(
                    str(aggregate_final)[:-5] + "_aggregated.json"
                    if str(aggregate_final).endswith(".json")
                    else str(aggregate_final) + "_aggregated"
                )
                _validate_trace(aggregate_temporary)
                pending.append((aggregate_final, aggregate_temporary))
        for final, temporary in pending:
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise SystemExit(f"ERROR: {command} did not produce {temporary}")
        for final, temporary in pending:
            os.replace(temporary, final)
        return 0
    finally:
        sys.argv = old_argv
        for _final, temporary in pending:
            if temporary.exists():
                temporary.unlink()


def _column_indexes(header):
    lowered = [(value or "").strip().lower() for value in header]
    def column(*names):
        return next((lowered.index(name) for name in names if name in lowered), None)
    return {
        "name": column("name", "kernel name", "kernel_name"),
        "total_ns": column("total time (ns)"),
        "instances": column("instances", "total count"),
        "start": column("start_timestamp"),
        "end": column("end_timestamp"),
    }


def normalize_kernel_summary(source, output):
    """Normalize TraceLens or raw rocprof CSV without materializing dispatch rows."""
    with open(source, newline="", encoding="utf-8-sig") as src:
        reader = csv.reader(src)
        header = next(reader, None)
        if not header:
            raise SystemExit(f"ERROR: normalizer input CSV is empty: {source}")
        indexes = _column_indexes(header)
        if indexes["name"] is None:
            raise SystemExit(
                f"ERROR: unrecognized kernel-summary schema; headers={header!r}"
            )
        helper = [
            "Kernel name",
            "kernel_duration_us_sum",
            "kernel_duration_us_count",
        ]
        if indexes["start"] is not None and indexes["end"] is not None:
            aggregate = {}
            for row in reader:
                try:
                    name = row[indexes["name"]]
                    duration_us = _duration_or_none((
                        float(row[indexes["end"]]) - float(row[indexes["start"]])
                    ) / 1000.0)
                except (IndexError, ValueError):
                    continue
                if duration_us is None:
                    continue
                bucket = aggregate.setdefault(name, [0.0, 0])
                bucket[0] += duration_us
                bucket[1] += 1
            with open(output, "w", newline="", encoding="utf-8-sig") as dst:
                writer = csv.writer(dst)
                writer.writerow(["Name", "Total Time (ns)", "Instances", *helper])
                for name, (duration_us, count) in sorted(
                    aggregate.items(), key=lambda item: -item[1][0]
                ):
                    writer.writerow([
                        name,
                        duration_us * 1000.0,
                        count,
                        name,
                        f"{duration_us:.4f}",
                        count,
                    ])
            return len(aggregate)

        count = 0
        with open(output, "w", newline="", encoding="utf-8-sig") as dst:
            writer = csv.writer(dst)
            writer.writerow([*header, *helper])
            for row in reader:
                if not row or all(not (cell or "").strip() for cell in row):
                    continue
                name = row[indexes["name"]] if indexes["name"] < len(row) else ""
                try:
                    duration_ns = _duration_or_none(row[indexes["total_ns"]])
                except (IndexError, TypeError):
                    duration_ns = None
                if duration_ns is None:
                    continue
                duration_us = f"{duration_ns / 1000.0:.4f}"
                instances = ""
                try:
                    if indexes["instances"] is not None:
                        instances = int(float(row[indexes["instances"]]))
                except (IndexError, ValueError):
                    pass
                writer.writerow([*row, name, duration_us, instances])
                count += 1
        return count


def _artifact_pid_map(root, suffix):
    """Return a PID-keyed map for one top-level rocprof worker artifact type."""
    root = Path(root)
    pattern = re.compile(rf'^.+_(?P<pid>\d+)_{re.escape(suffix)}$')
    by_pid = {}
    for path in sorted(root.glob(f'*_{suffix}')):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise SystemExit(
                f'ERROR: cannot derive worker PID from {path.name!r}'
            )
        pid = int(match.group('pid'))
        if pid in by_pid:
            raise SystemExit(
                f'ERROR: duplicate {suffix} artifacts for PID {pid}: '
                f'{by_pid[pid]}, {path}'
            )
        by_pid[pid] = path.resolve()
    return by_pid


def _expected_local_worker_count():
    raw = os.environ.get('ROCPROF_EXPECT_PER_NODE', '8')
    try:
        expected = int(raw)
    except ValueError as error:
        raise SystemExit(
            f'ERROR: invalid ROCPROF_EXPECT_PER_NODE={raw!r}'
        ) from error
    if expected <= 0:
        raise SystemExit(
            f'ERROR: ROCPROF_EXPECT_PER_NODE must be positive, got {expected}'
        )
    return expected


def _worker_rank_map(log_path):
    """Derive only ranks explicitly tied to a worker PID in the node log."""
    if not log_path.is_file():
        return {}
    workers = {}
    with open(log_path, encoding='utf-8', errors='replace') as log_file:
        for lineno, line in enumerate(log_file, 1):
            match = _WORKER_GPU_RE.search(line)
            if match is None:
                continue
            row = {
                key: (
                    int(match.group(key))
                    if match.group(key) is not None else None
                )
                for key in (
                    'pid',
                    'local_rank',
                    'dp_rank',
                    'tp_rank',
                    'ep_rank',
                )
            }
            row['source_line'] = lineno
            pid = row['pid']
            previous = workers.get(pid)
            if previous is None:
                workers[pid] = row
                continue
            if previous['local_rank'] != row['local_rank']:
                raise SystemExit(
                    f'ERROR: conflicting gpu_id mappings for PID {pid} '
                    f'in {log_path}'
                )
            for key in ('dp_rank', 'tp_rank', 'ep_rank'):
                if (
                    previous[key] is not None
                    and row[key] is not None
                    and previous[key] != row[key]
                ):
                    raise SystemExit(
                        f'ERROR: conflicting {key} mappings for PID {pid} '
                        f'in {log_path}'
                    )
                if previous[key] is None and row[key] is not None:
                    previous[key] = row[key]
    return workers


def _discover_kernel_workers(root):
    """Verify and deterministically order every local GPU worker in one node dir."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise SystemExit(f'ERROR: capture directory missing: {root}')
    expected = _expected_local_worker_count()
    kernels = _artifact_pid_map(root, 'kernel_trace.csv')
    markers = _artifact_pid_map(root, 'marker_api_trace.csv')
    results = _artifact_pid_map(root, 'results.json')
    if len(kernels) != expected:
        raise SystemExit(
            f'ERROR: {root.name} has {len(kernels)}/{expected} '
            'kernel_trace.csv worker files'
        )
    kernel_pids = set(kernels)
    for suffix, artifacts in (
        ('marker_api_trace.csv', markers),
        ('results.json', results),
    ):
        artifact_pids = set(artifacts)
        if artifact_pids != kernel_pids:
            raise SystemExit(
                f'ERROR: PID mismatch for {suffix} in {root.name}: '
                f'missing={sorted(kernel_pids - artifact_pids)}, '
                f'extra={sorted(artifact_pids - kernel_pids)}'
            )
    for pid, kernel_csv in kernels.items():
        prefix = kernel_csv.name[:-len('_kernel_trace.csv')]
        expected_names = {
            'marker_api_trace.csv': f'{prefix}_marker_api_trace.csv',
            'results.json': f'{prefix}_results.json',
        }
        for suffix, artifacts in (
            ('marker_api_trace.csv', markers),
            ('results.json', results),
        ):
            if artifacts[pid].name != expected_names[suffix]:
                raise SystemExit(
                    f'ERROR: worker PID {pid} has mismatched {suffix}: '
                    f'{artifacts[pid].name!r}, expected '
                    f'{expected_names[suffix]!r}'
                )

    log_path = (
        root.parent / f"{root.name.removeprefix('rocprof_')}.log"
    )
    rank_map = _worker_rank_map(log_path)
    known = {
        pid: rank_map[pid]
        for pid in kernel_pids
        if pid in rank_map
    }
    local_ranks = [row['local_rank'] for row in known.values()]
    if len(local_ranks) != len(set(local_ranks)):
        raise SystemExit(
            f'ERROR: duplicate derived gpu_id ranks in {log_path}: '
            f'{sorted(local_ranks)}'
        )
    if any(rank < 0 or rank >= expected for rank in local_ranks):
        raise SystemExit(
            f'ERROR: out-of-range derived gpu_id ranks in {log_path}: '
            f'{sorted(local_ranks)}'
        )
    ranks_derivable = len(known) == expected
    if ranks_derivable and sorted(local_ranks) != list(range(expected)):
        raise SystemExit(
            f'ERROR: incomplete derived local ranks in {log_path}: '
            f'got {sorted(local_ranks)}, expected {list(range(expected))}'
        )
    if ranks_derivable:
        ordered_pids = sorted(
            kernel_pids,
            key=lambda pid: rank_map[pid]['local_rank'],
        )
    else:
        ordered_pids = sorted(
            kernel_pids,
            key=lambda pid: (pid, kernels[pid].name),
        )

    workers = []
    for source_order, pid in enumerate(ordered_pids):
        kernel_csv = kernels[pid]
        prefix = kernel_csv.name[:-len('_kernel_trace.csv')]
        hostname = prefix.rsplit('_', 1)[0]
        rank = rank_map.get(pid, {})
        workers.append({
            'source_order': source_order,
            'pid': pid,
            'hostname': hostname,
            'local_rank': rank.get('local_rank'),
            'dp_rank': rank.get('dp_rank'),
            'tp_rank': rank.get('tp_rank'),
            'ep_rank': rank.get('ep_rank'),
            'rank_source_line': rank.get('source_line'),
            'filename': kernel_csv.name,
            'kernel_csv': str(kernel_csv),
            'marker_csv': str(markers[pid]),
            'results_json': str(results[pid]),
        })

    node_match = _NODE_CAPTURE_RE.fullmatch(root.name)
    return {
        'root': str(root),
        'role': node_match.group('role') if node_match else None,
        'node_rank': int(node_match.group('node_rank')) if node_match else None,
        'expected_worker_count': expected,
        'included_worker_count': len(workers),
        'rank_source': str(log_path.resolve()) if log_path.is_file() else None,
        'ranks_derivable': ranks_derivable,
        'workers': workers,
    }


def _analysis_artifact_for_pid(root, pid):
    root = Path(root)
    for pattern in ('*_results.pftrace', '*_kernel_trace.csv'):
        artifacts = [
            path for path in sorted(root.rglob(pattern))
            if (match := _ANALYSIS_PID_RE.fullmatch(path.name))
            and int(match.group('pid')) == int(pid)
        ]
        if artifacts:
            break
    if len(artifacts) != 1:
        raise SystemExit(
            f'ERROR: expected one analysis artifact for PID {pid}: {artifacts}'
        )
    return artifacts[0]


def _local_rank_zero_trace(root):
    root = Path(root)
    log_path = root.parent / f"{root.name.removeprefix('rocprof_')}.log"
    if not log_path.is_file():
        raise SystemExit(f"ERROR: worker log missing for {root.name}: {log_path}")
    with open(log_path, encoding="utf-8", errors="replace") as log_file:
        pids = {match.group("pid") for line in log_file
                if (match := _WORKER_GPU0_RE.search(line))}
    if len(pids) != 1:
        raise SystemExit(f"ERROR: expected one local-rank-0 PID in {log_path}, found {sorted(pids)}")
    pid = pids.pop()
    return _analysis_artifact_for_pid(root, pid)


def _first_match(root, patterns):
    root = Path(root)
    if not root.is_dir():
        return None
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _run_logged(command, log_path, required=True, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [str(item) for item in command],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
            )
        except FileNotFoundError as error:
            if required:
                raise SystemExit(f"ERROR: command not found: {command[0]}") from error
            return 127
    if required and result.returncode:
        raise SystemExit(
            f"ERROR: command failed ({result.returncode}); see {log_path}"
        )
    return result.returncode


def _analyze(argv):
    parser = argparse.ArgumentParser(
        prog="trace_tools.py analyze",
        description=(
            "Analyze one Torch/Kineto or rocprof trace. The rocprof CSV-only "
            "path is fully offline; TraceLens/traceconv are optional paths."
        ),
    )
    parser.add_argument("trace")
    parser.add_argument("outdir")
    parser.add_argument("label")
    parser.add_argument(
        "--mode",
        choices=("torch", "rocprof"),
        default=os.environ.get("ANALYZE_MODE") or None,
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=float(os.environ.get("TRIM_PCT", "5")),
    )
    parser.add_argument(
        "--traceconv",
        default=os.environ.get(
            "TRACECONV",
            str(
                Path(os.environ.get("TRACELENS_DIR", Path.home()))
                / "tracelens_test"
                / "tryout"
                / "traceconv"
            ),
        ),
    )
    args = parser.parse_args(argv)
    source = Path(args.trace)
    output = Path(args.outdir)
    tracelens_dir = output / "tracelens"
    csv_dir = tracelens_dir / "out_csvs"
    bucket_dir = output / "buckets"
    csv_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    venv = Path(
        command_env.get(
            "VENV",
            str(
                Path(command_env.get("TRACELENS_DIR", Path.home()))
                / "tracelens_test"
                / "venv"
            ),
        )
    )
    if (venv / "bin").is_dir():
        command_env["VIRTUAL_ENV"] = str(venv)
        command_env["PATH"] = (
            str(venv / "bin") + os.pathsep + command_env.get("PATH", "")
        )

    mode = args.mode or (
        "torch"
        if str(source).endswith((".trace.json", ".trace.json.gz"))
        else "rocprof"
    )
    kernel_csv = None
    pftrace = None
    perfetto_json = None
    coverage = None
    supplementary_worker = None
    if source.is_dir():
        coverage = _discover_kernel_workers(source)
        rank_zero = [
            worker for worker in coverage['workers']
            if worker['local_rank'] == 0
        ]
        if len(rank_zero) > 1:
            raise SystemExit(
                f"ERROR: multiple derived local-rank-0 workers in {source}"
            )
        if rank_zero:
            supplementary_worker = rank_zero[0]
            selected = _analysis_artifact_for_pid(
                source,
                supplementary_worker['pid'],
            )
        else:
            selected = Path(coverage['workers'][0]['kernel_csv'])
        is_csv = str(selected).endswith("_kernel_trace.csv")
        pftrace, kernel_csv = (None, selected) if is_csv else (selected, None)
        print(
            '[trace_tools analyze] canonical_kernel_scope='
            'pooled all workers'
        )
        print(
            '[trace_tools analyze] aggregation_semantics='
            'durations and call counts are summed, never averaged; '
            'summed activity is not wall time or utilization'
        )
        if supplementary_worker is not None:
            print(
                '[trace_tools analyze] supplementary_trace_scope='
                f"local_rank=0 pid={supplementary_worker['pid']} "
                f"source={selected}"
            )
        else:
            print(
                '[trace_tools analyze] supplementary_trace_scope='
                'unavailable (local rank not derivable); deterministic '
                'first source is used only for legacy input plumbing'
            )
    elif str(source).endswith("_kernel_trace.csv"):
        kernel_csv = source
    elif source.suffix == ".pftrace":
        pftrace = source
    elif str(source).endswith((".json", ".json.gz")):
        perfetto_json = source

    summary_csv = None
    if mode == "torch":
        if not source.is_file():
            raise SystemExit(f"ERROR: trace missing: {source}")
        _run_logged(
            [
                "TraceLens_generate_perf_report_pytorch",
                "--profile_json_path", source,
                "--output_xlsx_path", tracelens_dir / "report_TP0.xlsx",
                "--output_csvs_dir", csv_dir,
                "--enable_kernel_summary",
            ],
            tracelens_dir / "tracelens_stdout.txt",
            env=command_env,
        )
        summary_csv = _first_match(csv_dir, ("kernel_summary*.csv",))
    else:
        if pftrace is not None:
            traceconv = Path(args.traceconv)
            if not traceconv.is_file():
                raise SystemExit(f"ERROR: traceconv not found: {traceconv}")
            perfetto_json = tracelens_dir / f"{pftrace.stem}.json"
            _run_logged(
                [sys.executable, traceconv, "json", pftrace, perfetto_json],
                tracelens_dir / "traceconv_stdout.txt",
                env=command_env,
            )
        if kernel_csv is not None:
            summary_csv = kernel_csv
        else:
            if perfetto_json is None or not perfetto_json.is_file():
                raise SystemExit(f"ERROR: no usable rocprof input: {source}")
            _run_logged(
                [
                    "TraceLens_generate_perf_report_pftrace_hip_activity",
                    "--trace_path", perfetto_json,
                    "--output_xlsx_path", tracelens_dir / "report_TP0.xlsx",
                    "--output_csvs_dir", csv_dir,
                    "--output_md_path", tracelens_dir / "report.md",
                    "--traceconv", args.traceconv,
                    "--kernel_summary_include_rccl",
                    "--write_md",
                ],
                tracelens_dir / "tracelens_stdout.txt",
                env=command_env,
            )
            for suffix, executable in (
                ("hip_api", "TraceLens_generate_perf_report_pftrace_hip_api"),
                ("memcpy", "TraceLens_generate_perf_report_pftrace_memory_copy"),
            ):
                _run_logged(
                    [
                        executable,
                        "--trace_path", perfetto_json,
                        "--output_csvs_dir", tracelens_dir / f"out_csvs_{suffix}",
                        "--traceconv", args.traceconv,
                    ],
                    tracelens_dir / f"tracelens_{suffix}_stdout.txt",
                    required=False,
                    env=command_env,
                )
            summary_csv = _first_match(csv_dir, ("kernel_summary*.csv",))
            category_summary = _first_match(
                csv_dir, ("category_summary*.csv",)
            )
            if category_summary is not None:
                shutil.copy2(
                    category_summary,
                    bucket_dir / "tracelens_native_category_summary.csv",
                )
            if coverage is not None:
                with open(
                    tracelens_dir / "tracelens_stdout.txt",
                    "a",
                    encoding="utf-8",
                ) as tracelens_log:
                    tracelens_log.write(
                        "\n[trace_tools] scope=supplementary local_rank=0; "
                        "canonical kernel CSVs are pooled all-worker outputs\n"
                    )

    if summary_csv is None:
        raise SystemExit("ERROR: no kernel summary was produced")
    normalized = bucket_dir / "kernel_summary_normalized.csv"
    if coverage is not None:
        pooled_audit = run_pooled_kernel_summaries(
            coverage,
            normalized,
            bucket_dir / "kernel_summary_trimmed.csv",
            args.trim_pct,
            bucket_dir / "trimmed_summary_stdout.txt",
        )
        normalized_count = pooled_audit[
            'normalized_distinct_kernel_count'
        ]
    else:
        normalized_count = normalize_kernel_summary(summary_csv, normalized)
    process(
        normalized,
        bucket_dir / "perkernel_buckets.csv",
        bucket_dir / "bycat_buckets.csv",
        categorize_kernel,
        (
            f"{args.label} (pooled all workers)"
            if coverage is not None else f"{args.label} ({mode})"
        ),
    )

    if coverage is None:
        trim_source = kernel_csv
        if trim_source is None and pftrace is not None:
            sibling = Path(str(pftrace).replace(
                "_results.pftrace", "_kernel_trace.csv"
            ))
            trim_source = sibling if sibling.is_file() else _first_match(
                pftrace.parent, ("*_kernel_trace.csv",)
            )
        if trim_source is not None:
            rows, total_ns = build_trimmed_summary(
                trim_source,
                args.trim_pct,
            )
            write_csv(
                rows,
                bucket_dir / "kernel_summary_trimmed.csv",
                True,
                categorize_kernel,
            )
            print(
                f"[trace_tools analyze] trimmed={len(rows)} kernels "
                f"total={pretty_ns(total_ns)}"
            )
    print(
        f"[trace_tools analyze] {args.label}: normalized={normalized_count} "
        f"outputs={bucket_dir}"
    )
    return 0


def _self_test_categories(argv):
    parser = argparse.ArgumentParser(
        prog="trace_tools.py self-test-categories",
        description="Run kernel classifier regression cases.",
    )
    parser.parse_args(argv)
    cases = [
        ("EpDispatchInterNodeV1Kernel[LowLatency]_fp8_fnuz.kd", "MORI EP"),
        ("EpCombineInterNodeV1Kernel[LowLatency]_bf16.kd", "MORI EP"),
        ("EpDispatchIntraNodeKernel_fp8_fnuz", "MORI EP"),
        ("EpCombineIntraNodeKernel_bf16_nop2p", "MORI EP"),
        ("EpCombineSyncBarrier_bf16.kd", "Communication"),
        ("EpDispatchCopyToStaging_fp8_fnuz.kd", "Communication"),
        ("rmsnorm_kernel", "RMSNorm"),
        ("some_rope_kernel", "ROPE"),
        ("_ZN5aiter13allgather_vecIfEEvPT_S2_ii", "Communication"),
        ("Cijk_Ailk_Bljk_HHS_BH_MT128x128", "GEMM"),
        ("moe_topk_softmax_kernel", "MoE_TopK"),
        ("some_unknown_kernel_xyz", "Other"),
    ]
    failures = [
        (name, expected, categorize_kernel(name))
        for name, expected in cases
        if categorize_kernel(name) != expected
    ]
    for name, expected, actual in failures:
        print(
            f"FAIL: {name!r} expected {expected!r}, got {actual!r}",
            file=sys.stderr,
        )
    if failures:
        return 1
    print(f"OK: {len(cases)}/{len(cases)} category regression cases passed.")
    return 0


def main(argv=None):
    commands = {
        "build-trace": _legacy_build_trace,
        "correlate": _legacy_correlate,
        "reqstats": _legacy_reqstats,
        "buckets": _legacy_buckets,
        "trimmed-summary": _legacy_trimmed_summary,
        "analyze": _analyze,
        "self-test-categories": _self_test_categories,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command in {
        "build-trace",
        "correlate",
        "reqstats",
        "buckets",
        "trimmed-summary",
    }:
        return _run_absorbed(
            args.command, commands[args.command], args.arguments
        )
    return commands[args.command](args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())

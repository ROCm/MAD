#!/usr/bin/env python3
"""Aggregate an rtl (rocpd SQLite) trace into the client-perf-hub upload payload (COMPACT v2).

Compact/columnar schema to avoid string+key duplication (a trace repeats ~dozens of kernel
names millions of times). Names live once in KernelNames[]; every kernel/gap is referenced by
integer index; per-pass items are stored as parallel columnar arrays (no per-item dict keys).

  SchemaVersion : 2
  Config        : optional {TP, Precision, ISL, OSL}
  KernelNames[] : index -> kernel name (deduped once per payload)
  Kernels[]     : one row per (GPU, Stream, kernel) -- {GPU, Stream, K:<idx>, Calls, TotalUs,
                  MinUs, MaxUs}
  Trace.Stages[]: one per stage; Stage is an int code (0 prefill, 1 mixed, 2 decode).
    Representative / Passes[] : each pass = columnar {K:[...], D:[...], L:[...]}
        K[i] = kernel index, or -1 for a gap        (Type is implied: K==-1 -> gap)
        D[i] = duration (us)
        L[i] = layer number, or -1 for none          (was the "Layer N" Group string)
      PassIndex is positional (1-based). Each individual pass also carries "Ann": the raw vLLM
      roctx label (execute_context_<nctx>(<ctxTok>)_generation_<ngen>(<genTok>)) so the frontend
      can reconstruct the true per-pass annotation (and derive prefill/mixed/decode + batch/token
      counts) from it. The stage int code stays as the grouping key; classification is also
      derivable from Ann. Representative is a synthetic modal pass, so it carries no Ann.

  Pass capping (curation). Decode is 10^4-10^5 near-identical passes (DeepSeek-V3: 81,872
  passes, only ~1% distinct kernel sequences), which no downstream storage/dashboard can
  browse per-pass. So --full does NOT emit every pass; for any stage with more than
  --min-stage-cap passes it emits a CURATED set via two INDEPENDENT filters (a pass can clear
  both): (1) unique = one pass per DISTINCT kernel sequence (first occurrence); (2) outlier =
  per sequence, any pass whose total wall (kernels + idle) µs exceeds median + k*MAD (robust to the
  skew/outliers that inflate a plain stddev; the first occurrence is tested too), capped at
  --cap total (uniques first, then most-deviant outliers). Each emitted pass carries: PassIndex
  (ORIGINAL 1-based position within the stage -- the "index of PassCount"); Kind ("unique" |
  "slow_outlier" | "unique_slow_outlier" | "sample"); and, when an outlier, OutlierRatio =
  total/median (e.g. 2.83x the typical duration for its exact sequence) + OutlierNmad =
  (total-median)/spread -- so the dashboard shows WHERE and WHAT is outlying. PassCount stays
  the REAL total; low-count stages (prefill/mixed) are emitted whole; --no-cap restores every
  pass. Kernels[] aggregates are over ALL ops and are unaffected by curation.
"""
import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict

LABEL_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXPLANATION FROM NIKHIL ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each entry is a substring matched against kernel names.
# typically uses attention as an anchor that we can observe once per pass, and then backtracks to find a norm marker which kicks off a layer. 
# This is the most effective way to calculate passes because attention fires once per layer and layers start at norm markers, but norm markers can also fire at other points during layers (and typically do). 
# We can figure out the pass identity (prefill, decode, or mixed) using ROCTx annotations originating from vllm (other frameworks also may do this differently). 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ATTENTION_ANCHORS = (
    "paged_attention_ll4mi_QKV",   # vLLM ROCm paged attention (Llama/Mistral/Mixtral/OPT/Qwen…)
    "kernel_unified_attention",    # vLLM v1 unified attention
    "concat_and_cache_mla_kernel", # MLA (DeepSeek-V2/V3) per-layer KV-cache write, added as we did not have coverage for deepseek models layer annotations.
    "reshape_and_cache_kernel",    # aiter per-layer KV-cache write (ATOM native: Qwen3/Llama/Mixtral/...); verified 28 layers in prefill+decode
    # Prefill often runs a DIFFERENT attention kernel from decode (chunked/context FMHA vs paged/
    # unified decode attention), so the anchor is discovered PER STAGE (see build_trace). These CK
    # ck_tile FMHA kernels each fire once per layer in prefill:
    "FmhaFwdKernel",               # CK ck_tile FMHA forward — GPT-OSS (vLLM) prefill; general CK prefill
    "FmhaBatchPrefill",            # CK ck_tile paged-KV batch prefill — SGLang prefill
    "reshape_and_cache_flash",     # vLLM flash KV-cache write (GPT-OSS): per-layer, fires in both stages
)
# Layer boundaries start at the pre-attention norm. Most engines emit a NATIVE fused RMSNorm/LayerNorm
# kernel (name contains "rmsnorm"/"layernorm"), but some models are normed via a torch.compile/inductor
# FUSED TRITON kernel whose name carries the RMSNorm *math* instead of "norm" — e.g. Gemma-2 emits
# `triton_red_fused__to_copy_add_mean_mul_pow_..._rsqrt_N` (mean(x^2) -> rsqrt). Without matching these,
# the backward search finds no norm and each layer wrongly starts at its attention kernel (pre-attention
# norm + QKV get mis-attributed to the previous layer). `rsqrt`/`mean_mul_pow` are RMSNorm-specific and
# were verified to match nothing but real norms across vLLM/SGLang/ATOM traces (zero false positives).
NORM_MARKERS = ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "mean_mul_pow", "rsqrt")
STAGE_CODE = {"prefill": 0, "mixed": 1, "decode": 2}
MIN_ANCHOR_REPEATS = 4
# Regex-only detector for "attention-like" kernels that tend to repeat once per layer.
# This is intentionally broad enough to cover vLLM/SGLang/ATOM naming drift while excluding
# unrelated high-frequency kernels.
ATTENTION_ANCHOR_RE = re.compile(
    r"(attn|attention|paged_attention|kernel_unified_attention|reshape_and_cache|"
    r"concat_and_cache|store_kvcache|kvcache|flash_c\d+_(decode|prefill)|qkv|mla)",
    re.IGNORECASE,
)


def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.execute(
        "SELECT o.gpuId, o.queueId, s.string, o.start, o.end FROM rocpd_op o "
        "JOIN rocpd_string s ON o.description_id = s.id WHERE o.start IS NOT NULL"
    )
    pass_markers, ops = [], []
    for g, q, n, st, en in cur:
        if g is not None and g < 0:
            if LABEL_RE.search(n or ""):
                pass_markers.append((n, st, en))
        elif en is not None:
            ops.append((g, q, n, st, en))
    con.close()
    pass_markers.sort(key=lambda r: r[1])
    ops.sort(key=lambda r: r[3])
    return pass_markers, ops


def stage_of(label):
    m = LABEL_RE.search(label or "")
    ctx_tokens, gen_tokens = int(m[2]), int(m[4])
    return "prefill" if gen_tokens == 0 else "decode" if ctx_tokens == 0 else "mixed"


def is_norm(name):
    low = (name or "").lower()
    return any(marker in low for marker in NORM_MARKERS)


def _periodicity_cv(positions):
    if len(positions) < 3:
        return float("inf")
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return float("inf")
    return statistics.pstdev(gaps) / mean_gap


def _count_substring_positions(ops, needle):
    positions = []
    for idx, (_g, _q, name, _s, _e) in enumerate(ops):
        if needle in (name or ""):
            positions.append(idx)
    return positions


def discover_anchor(ops, override=None, expected_layers=None):
    if override:
        return override
    # Candidate pool:
    #   1) historical allowlist substrings (stable across previous runs),
    #   2) dynamic regex-discovered attention-like kernels (better generalization).
    candidates = []
    for anchor in ATTENTION_ANCHORS:
        positions = _count_substring_positions(ops, anchor)
        if len(positions) >= MIN_ANCHOR_REPEATS:
            candidates.append((anchor, len(positions), positions, True))

    exact_counts = Counter()
    exact_positions = defaultdict(list)
    for idx, (_g, _q, name, _s, _e) in enumerate(ops):
        if ATTENTION_ANCHOR_RE.search(name or ""):
            exact_counts[name] += 1
            exact_positions[name].append(idx)
    for name, count in exact_counts.items():
        if count >= MIN_ANCHOR_REPEATS:
            candidates.append((name, count, exact_positions[name], False))

    if not candidates:
        return None

    def score(row):
        anchor, count, positions, is_allowlist = row
        cv = _periodicity_cv(positions)
        # A true once-per-layer anchor fires layers*passes times, i.e. an exact MULTIPLE of the layer
        # count. Sub-sampling variants leave a large remainder: e.g. Gemma-2 alternates sliding-window
        # and global attention, exposing kernel_unified_attention_2d/_3d that each fire on HALF the
        # layers. Those halves are deceptively MORE periodic (lower CV), so CV must NOT be the primary
        # key or we undercount layers 2x. Rank by: expected-layer multiple (when the caller knows the
        # layer count) > curated allowlist anchor > fuller per-layer coverage > periodicity.
        if expected_layers:
            rem = count % expected_layers
            multiple_penalty = min(rem, expected_layers - rem)
        else:
            multiple_penalty = 0
        return (multiple_penalty, 0 if is_allowlist else 1, -count, cv, positions[0], len(anchor))

    best = min(candidates, key=score)
    return best[0]


def _stream(queue):
    return f"0x{queue:x}" if isinstance(queue, int) else str(queue)


def build_names(ops):
    """Stable index for every distinct kernel name (first-seen order over time-sorted ops)."""
    idx, names = {}, []
    for _g, _q, name, _s, _e in ops:
        if name not in idx:
            idx[name] = len(names)
            names.append(name)
    return idx, names


def aggregate(ops, name_to_idx):
    calls, total, mn, mx = defaultdict(int), defaultdict(float), {}, defaultdict(float)
    for gpu, queue, name, start, end in ops:
        key, dur = (gpu, queue, name), max(0, end - start)
        calls[key] += 1
        total[key] += dur
        mx[key] = max(mx[key], dur)
        mn[key] = dur if key not in mn else min(mn[key], dur)
    rows = [{"GPU": g, "Stream": _stream(q), "K": name_to_idx[n], "Calls": c,
             "TotalUs": round(total[(g, q, n)] / 1000, 3),
             "MinUs": round(mn[(g, q, n)] / 1000, 3),
             "MaxUs": round(mx[(g, q, n)] / 1000, 3)}
            for (g, q, n), c in calls.items()]
    rows.sort(key=lambda r: r["TotalUs"], reverse=True)
    return rows


def layer_index(names, anchor):
    out = [-1] * len(names)
    anchor_pos = [i for i, nm in enumerate(names) if anchor and anchor in (nm or "")]
    if not anchor_pos:
        return out
    starts, lower = [], 0
    for a in anchor_pos:
        start = next((i for i in range(a - 1, lower - 1, -1) if is_norm(names[i])), a)
        starts.append(start)
        lower = start + 1
    trailing = [i for i in range(starts[-1] + 1, len(names)) if is_norm(names[i])]
    bounds = starts + [trailing[1] if len(trailing) >= 2 else len(names)]
    for li in range(len(anchor_pos)):
        for i in range(bounds[li], bounds[li + 1]):
            out[i] = li
    return out


def pass_columns(pass_ops, anchor, min_gap_us, name_to_idx):
    """Columnar {K,D,L} for one pass. Gap -> K=-1,L=-1. Layer -1 = none."""
    layers = layer_index([o[2] for o in pass_ops], anchor)
    K, D, L = [], [], []
    prev_end = None
    for i, (_g, _q, name, start, end) in enumerate(pass_ops):
        if prev_end is not None:
            gap = (start - prev_end) / 1000.0
            if gap > 0 and gap >= min_gap_us:
                K.append(-1); D.append(round(gap, 3)); L.append(-1)
        K.append(name_to_idx[name]); D.append(round(max(0, end - start) / 1000.0, 3)); L.append(layers[i])
        # Track the max end seen, not just this op's end: kernels on concurrent streams/queues can be
        # launch-ordered so a later-starting op finishes BEFORE an earlier one; using its (smaller)
        # end would fabricate a positive gap before the next op.
        prev_end = end if prev_end is None else max(prev_end, end)
    return {"K": K, "D": D, "L": L}


def _pass_signature(cols):
    """Order-sensitive STRUCTURAL key for a pass: the (kernel_index, layer) sequence, GAPS EXCLUDED.

    Sequence identity is about which kernels run in which order per layer — idle gaps are timing,
    not structure, so a K==-1 gap never changes the signature (two passes with the same kernels in
    the same order share a signature no matter how the idle between them differs). This is
    intentionally coarser than the downstream cph template key (which still carries gap items);
    that is safe because template dedup runs on this already-gap-collapsed curated set. Durations
    are excluded here too — timing is handled by _pass_total_us (the outlier metric).
    """
    return tuple((k, l) for k, l in zip(cols["K"], cols["L"]) if k != -1)


def _pass_total_us(cols):
    """Total pass duration for outlier ranking: sum of launch-ordered kernel + gap durations (µs).

    Not true GPU wall clock — overlapping kernels on different streams are summed (same as the
    prior kernel-only metric), but idle gaps are included so bubble-heavy passes surface as
    outliers. (Sequence identity ignores gaps — see _pass_signature.)"""
    return sum(cols["D"])


def _curation_fields(reason):
    """Payload fields that tell the dashboard WHY a pass was kept + WHAT is outlying.

    `Kind` is the summary marker; the two filters are evaluated INDEPENDENTLY so a pass can be
    both (a first-occurrence that is also slow -> "unique_slow_outlier"). For an outlier we also
    emit the quantitative "what": OutlierRatio = its total pass duration (kernels + gaps, µs) /
    the sequence's median (e.g. 2.83 => 2.83x the typical duration for its exact kernel sequence)
    and OutlierNmad = the robust deviation (total-median)/spread in MAD-scaled units.
    """
    uniq = bool(reason.get("uniq"))
    ratio = reason.get("ratio")
    if reason.get("sample"):
        kind = "sample"
    elif uniq and ratio is not None:
        kind = "unique_slow_outlier"
    elif uniq:
        kind = "unique"
    elif ratio is not None:
        kind = "slow_outlier"
    else:
        kind = "sample"
    out = {"Kind": kind}
    if ratio is not None:
        out["OutlierRatio"] = ratio
        out["OutlierNmad"] = reason.get("nmad")
    return out


def select_curated_passes(cols_list, k_mad, cap, min_stage, rel_floor=0.02):
    """Pick a small, information-preserving subset of a stage's passes, tagging WHY each is kept.

    Returns a sorted list of (orig_index, reason) where orig_index is the 0-based launch-order
    position and reason is a dict {uniq: bool, ratio: float|None, nmad: float|None}. The two
    filters are INDEPENDENT (a pass can clear both):
      * unique filter  -> one pass per DISTINCT kernel sequence (first occurrence): reason.uniq=True.
      * outlier filter -> per sequence, any pass whose total wall (kernels + idle) µs exceeds median + k*MAD
        (MAD scaled by 1.4826; a relative floor rel_floor*median avoids flagging f32 noise when a
        near-constant sequence collapses MAD to 0). Only sequences with >=3 samples are tested. The
        first occurrence is ALSO tested, so a unique pass that is slow is marked as both. reason.ratio
        = total/median, reason.nmad = (total-median)/spread.
    A pass is kept iff uniq or ratio is not None. Hard cap: all uniques first (by launch order), then
    the most-deviant NON-unique outliers, so `cap` bounds a pathological run.

    Stages with <= min_stage passes are NOT curated (caller emits them whole, reason.sample=True).
    """
    n = len(cols_list)
    if n <= min_stage:
        return [(i, {"sample": True}) for i in range(n)]

    groups = defaultdict(list)                       # signature -> [orig_index, ...] (ascending)
    for i, cols in enumerate(cols_list):
        groups[_pass_signature(cols)].append(i)

    reason = {}                                       # orig_index -> reason dict
    outlier_rank = []                                 # (nmad, orig_index) for non-unique outliers
    for idxs in groups.values():
        reason[idxs[0]] = {"uniq": True, "ratio": None, "nmad": None}
        if len(idxs) < 3:
            continue
        durs = [_pass_total_us(cols_list[i]) for i in idxs]
        med = statistics.median(durs)
        mad = statistics.median([abs(d - med) for d in durs])
        spread = max(1.4826 * mad, rel_floor * med)
        if spread <= 0:
            continue
        thr = med + k_mad * spread
        for i, d in zip(idxs, durs):
            if d <= thr:
                continue
            nmad = (d - med) / spread
            r = reason.get(i) or {"uniq": False, "ratio": None, "nmad": None}
            r["ratio"] = round(d / med, 3) if med > 0 else None
            r["nmad"] = round(nmad, 2)
            reason[i] = r
            if not r["uniq"]:
                outlier_rank.append((nmad, i))

    kept = {i: r for i, r in reason.items() if r["uniq"]}
    if len(kept) > cap:                               # pathological: more distinct sequences than cap
        kept = {i: kept[i] for i in sorted(kept)[:cap]}
    budget = cap - len(kept)
    if budget > 0 and outlier_rank:
        outlier_rank.sort(reverse=True)               # most anomalous first
        for _nm, i in outlier_rank:
            if i not in kept:
                kept[i] = reason[i]
                budget -= 1
                if budget <= 0:
                    break
    return sorted(kept.items())


def build_trace(pass_markers, ops, anchor, sample, min_gap_us, name_to_idx,
                full=False, trace_gpu=None, cap=1500, min_stage=512, k_mad=3.0,
                anchor_override=None, expected_layers=None):
    if trace_gpu is not None:
        ops = [o for o in ops if o[0] == trace_gpu]
    by_stage, ki, n = defaultdict(list), 0, len(ops)
    for label, mstart, mend in pass_markers:
        while ki < n and ops[ki][3] < mstart:
            ki += 1
        j = ki
        while j < n and ops[j][3] <= mend:
            j += 1
        by_stage[stage_of(label)].append((label, ops[ki:j]))

    stages = []
    for stage in ("prefill", "mixed", "decode"):
        plist = by_stage.get(stage)
        if not plist:
            continue
        count = len(plist)
        # Anchor is discovered PER STAGE: prefill and decode frequently run different attention
        # kernels (e.g. GPT-OSS uses ck_tile FmhaFwdKernel in prefill but kernel_unified_attention in
        # decode; SGLang uses FmhaBatchPrefill vs paged_attention). A single run-global anchor only
        # labels the stage whose kernel it matched, leaving the other stage with 0 layers — that was
        # why GPT-OSS/SGLang prefill showed no layerwise breakdown. Explicit --anchor still overrides.
        stage_ops = [o for _l, po in plist for o in po]
        stage_anchor = anchor_override or discover_anchor(stage_ops, expected_layers=expected_layers) or anchor
        # Columnar form for every pass (transient); needed for signature grouping + curation below.
        cols_list = [pass_columns(po, stage_anchor, min_gap_us, name_to_idx) for _l, po in plist]
        # Representative = a TYPICAL instance of the MOST COMMON kernel SEQUENCE. Two bugs are fixed
        # here: (1) the old code grouped passes by kernel COUNT (Counter(len(po))), not their actual
        # ordered sequence, and (2) it took the FIRST such pass -- which is the warmup / first-token
        # step, a RARE one-off (e.g. 1 of 32 distinct decode sequences across 5072 passes) whose
        # ~200 ms prefill->decode + graph-capture transition gap made decode read ~96% idle. Decode is
        # highly repetitive (one sequence spans most of the passes), so the modal (kernel, layer)
        # signature is unambiguous; taking the MEDIAN wall-span within that group keeps warmup/outlier
        # timings from defining the representative.
        sigs = [_pass_signature(c) for c in cols_list]
        modal_sig = Counter(sigs).most_common(1)[0][0]
        modal_idxs = [i for i, s in enumerate(sigs) if s == modal_sig]
        modal_idxs.sort(key=lambda i: max(o[4] for o in plist[i][1]) - min(o[3] for o in plist[i][1]))
        rep_ops = plist[modal_idxs[len(modal_idxs) // 2]][1]
        labels = [label for label, _po in plist]
        if not full:
            kept = [(i, {"sample": True}) for i in range(min(count, sample))]
        elif cap:
            kept = select_curated_passes(cols_list, k_mad, cap, min_stage)
        else:                                     # --no-cap: every pass (offline validation)
            kept = [(i, {"sample": True}) for i in range(count)]
        stages.append({
            "Stage": STAGE_CODE[stage],
            "PassCount": count,                   # REAL pass total (unaffected by curation)
            "Representative": pass_columns(rep_ops, stage_anchor, min_gap_us, name_to_idx),
            # Each emitted pass keeps its ORIGINAL 1-based positional index (PassIndex) within the
            # stage + curation markers (Kind, and OutlierRatio/OutlierNmad when a slow outlier) + the
            # raw roctx label (Ann) so the frontend shows WHERE (index of PassCount) and WHY it was
            # kept. The synthetic Representative carries none of these.
            "Passes": [dict(cols_list[i], Ann=labels[i], PassIndex=i + 1, **_curation_fields(reason))
                       for i, reason in kept],
        })
    return {"Stages": stages}


def pick_trace_gpu(ops):
    counts = Counter(g for g, _q, _n, _s, _e in ops if isinstance(g, int) and g >= 0)
    if not counts:
        return None
    return min(counts, key=lambda g: (-counts[g], g))


def build_config(args):
    cfg = {}
    for key, val in (("TP", args.tp), ("Precision", args.precision),
                     ("ISL", args.isl), ("OSL", args.osl)):
        if val is not None:
            cfg[key] = val
    return cfg


def main():
    ap = argparse.ArgumentParser(description="rtl trace -> client-perf-hub compact payload (v2).")
    ap.add_argument("db")
    ap.add_argument("-o", "--out", default="profile/kernel_summary_payload.json")
    ap.add_argument("--tp", type=int)
    ap.add_argument("--precision")
    ap.add_argument("--isl", type=int)
    ap.add_argument("--osl", type=int)
    ap.add_argument("--trace-passes", type=int, default=5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--trace-all-gpus", action="store_true")
    ap.add_argument("--min-gap-us", type=float, default=0.1,
                    help="emit inter-kernel idle gaps >= this many us (default 0.1us=100ns; below that is "
                         "timer-quantization noise -- measured negligible: <1pct of gaps, ~0pct of idle time)")
    ap.add_argument("--anchor")
    ap.add_argument("--expected-layers", type=int,
                    help="optional prior for hidden-layer count used during anchor selection")
    ap.add_argument("--indent", type=int, default=None)
    # Pass curation (with --full): high-count stages keep 1 pass per distinct kernel sequence
    # + per-sequence slow-duration outliers, capped. Keeps the payload/parquet/dashboard small
    # without losing distinct structures or timing anomalies. See select_curated_passes.
    ap.add_argument("--cap", type=int, default=1500,
                    help="max curated passes per high-count stage (with --full); 0 disables the cap")
    ap.add_argument("--min-stage-cap", type=int, default=512,
                    help="only stages with more passes than this are curated (with --full)")
    ap.add_argument("--k-mad", type=float, default=3.0,
                    help="slow-outlier threshold = median + k*MAD of per-pass wall microseconds (kernels + idle)")
    ap.add_argument("--no-cap", action="store_true",
                    help="emit EVERY pass with --full (disables curation; for offline validation)")
    args = ap.parse_args()

    pass_markers, ops = load(args.db)
    if not ops:
        raise SystemExit(f"no GPU ops (gpuId>=0) found in {args.db}")
    name_to_idx, names = build_names(ops)
    doc = {"SchemaVersion": 2, "Config": build_config(args),
           "KernelNames": names, "Kernels": aggregate(ops, name_to_idx)}
    if pass_markers:
        anchor = discover_anchor(ops, args.anchor)
        trace_gpu = None if args.trace_all_gpus else pick_trace_gpu(ops)
        cap = 0 if args.no_cap else args.cap
        doc["Trace"] = build_trace(pass_markers, ops, anchor, args.trace_passes,
                                   args.min_gap_us, name_to_idx, args.full, trace_gpu,
                                   cap=cap, min_stage=args.min_stage_cap, k_mad=args.k_mad,
                                   anchor_override=args.anchor, expected_layers=args.expected_layers)
    separators = None if args.indent is not None else (",", ":")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=args.indent, separators=separators)
    stages = doc.get("Trace", {}).get("Stages", [])
    stage_name = {v: k for k, v in STAGE_CODE.items()}
    kept = ", ".join(f"{stage_name.get(s['Stage'], s['Stage'])} {len(s.get('Passes', []))}/{s.get('PassCount')}"
                     for s in stages)
    print(f"wrote {len(doc['Kernels'])} kernel rows + {len(names)} names + "
          f"Trace[{len(stages)} stages: {kept} kept/total] -> {args.out}")


if __name__ == "__main__":
    main()

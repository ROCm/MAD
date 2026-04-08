#!/usr/bin/env python3
"""
Post-process benchmark results: merge results_rixl.json, results_mori.json,
and results_mooncake.json from a shared directory into:
  - results_merged.json  (combined, normalized list)
  - results_merged.csv   (pivot: one row per transfer size, one column per backend)
  - report.html          (self-contained HTML report with table + throughput chart)

Use --kv-cache-estimator-file to load KV cache mapping from a CSV file.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULT_FILES = [
    "results_rixl.json",
    "results_mori.json",
    "results_mooncake.json",
]

BACKENDS = ["rixl", "mori", "mooncake"]

BACKEND_COLORS = {
    "rixl": "#4472C4",
    "mori": "#ED7D31",
    "mooncake": "#A5A5A5",
}


def _bytes_to_label(size_bytes: int) -> str:
    """Convert byte count to human-readable label (e.g. 4096 -> '4KB')."""
    if size_bytes >= 1 << 30:
        val = size_bytes / (1 << 30)
        return f"{val:g}GB"
    if size_bytes >= 1 << 20:
        val = size_bytes / (1 << 20)
        return f"{val:g}MB"
    if size_bytes >= 1 << 10:
        val = size_bytes / (1 << 10)
        return f"{val:g}KB"
    return f"{size_bytes}B"


def _normalize_result(entry: dict) -> dict | None:
    """
    Normalize a single result entry from any backend into a common format:
    { backend, transfer_size, throughput_gbs, timestamp }

    Mori uses flat keys: transfer_size, throughput
    RIXL/Mooncake use nested: test_parameters.size_bytes, results.bandwidth_gbs_avg
    """
    backend = entry.get("backend", "").strip().lower()

    if "transfer_size" in entry:
        size = entry["transfer_size"]
        throughput = entry.get("throughput", 0)
        timestamp = entry.get("date-time", "")
    elif "test_parameters" in entry:
        size = entry["test_parameters"].get("size_bytes", 0)
        res = entry.get("results") or {}
        throughput = res.get("bandwidth_gbs_avg", res.get("bandwidth_gbs", res.get("throughput", 0)))
        timestamp = entry.get("timestamp_utc", "")
    else:
        return None

    if size is None or size == 0:
        return None

    return {
        "backend": backend,
        "transfer_size": int(size),
        "throughput_gbs": round(float(throughput), 4) if throughput else 0,
        "timestamp": timestamp,
    }


def merge_results(input_dir: Path, output_path: Path, verbose: bool = True, kv_cache_estimator_file: Path | None = None) -> dict:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    merged = {"metadata": {}, "results": []}

    for filename in RESULT_FILES:
        path = input_dir / filename
        if not path.exists():
            if verbose:
                print(f"Skipping (not found): {path}", file=sys.stderr)
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)
            continue

        backend_name = filename.replace("results_", "").replace(".json", "")
        meta = data.get("metadata", data.get("version_info", {}))
        merged["metadata"][backend_name] = meta

        raw_results = data.get("results", [])
        normalized = [_normalize_result(r) for r in raw_results]
        normalized = [r for r in normalized if r is not None]
        merged["results"].extend(normalized)
        if verbose:
            print(f"Added {len(normalized)} result(s) from {filename} (backend={backend_name})")

    merged["results"].sort(key=lambda r: (r.get("backend", ""), r.get("transfer_size", 0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    if verbose:
        print(f"Merged JSON written to {output_path} ({len(merged['results'])} total results)")

    # --- Pivot CSV ---
    by_size: dict[int, dict] = {}
    for r in merged["results"]:
        size = r["transfer_size"]
        backend = r["backend"]
        if size not in by_size:
            by_size[size] = {"transfer_size": size, "size_label": _bytes_to_label(size)}
            for b in BACKENDS:
                by_size[size][f"{b}_throughput"] = ""
        if backend in BACKENDS:
            by_size[size][f"{backend}_throughput"] = r["throughput_gbs"]

    rows = sorted(by_size.values(), key=lambda x: x["transfer_size"])
    csv_columns = ["transfer_size", "size_label"] + [f"{b}_throughput" for b in BACKENDS]
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in csv_columns})
    if verbose:
        print(f"CSV written to {csv_path}")

    # --- HTML Report ---
    html_path = output_path.parent / "report.html"
    kv_cache_rows = _load_kv_cache_estimator(kv_cache_estimator_file)
    html = _generate_html_report(merged, rows, kv_cache_rows)
    with open(html_path, "w") as f:
        f.write(html)
    if verbose:
        print(f"HTML report written to {html_path}")

    return merged


def _load_kv_cache_estimator(kv_cache_file: Path | None) -> list[dict]:
    """Load kv_cache_estimator.csv from the given path. Returns list of {model_name, model_unique_name, kv_cache_bytes, ...}."""
    if kv_cache_file is None:
        return []
    path = Path(kv_cache_file)
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                kv_bytes = row.get("kv_cache_bytes", "")
                if kv_bytes:
                    try:
                        kv_bytes = int(kv_bytes)
                    except ValueError:
                        kv_bytes = 0
                else:
                    # Backward compat: convert legacy kv_cache_mb to bytes
                    kv_mb = row.get("kv_cache_mb", "")
                    try:
                        kv_bytes = int(float(kv_mb) * (1024 ** 2)) if kv_mb else 0
                    except (ValueError, TypeError):
                        kv_bytes = 0
                rows.append({
                    "model_name": row.get("model_name", ""),
                    "model_unique_name": row.get("model_unique_name", ""),
                    "seq_length": row.get("seq_length", ""),
                    "concurrency": row.get("concurrency", row.get("batch_size", "")),
                    "tp_size": row.get("tp_size", ""),
                    "dtype": row.get("dtype", ""),
                    "kv_cache_bytes": kv_bytes,
                })
    except (OSError, csv.Error):
        pass
    return rows


def _interpolate_throughput(size_bytes: float, benchmark_rows: list[dict], backend: str) -> float | None:
    """
    Find throughput for size_bytes. Exact match or linear interpolation between bracketing sizes.
    Returns None if size is outside benchmark range (use nearest endpoint).
    """
    col = f"{backend}_throughput"
    sizes = [r["transfer_size"] for r in benchmark_rows]
    if not sizes:
        return None
    min_s, max_s = min(sizes), max(sizes)
    if size_bytes <= min_s:
        for r in benchmark_rows:
            if r["transfer_size"] == min_s:
                v = r.get(col, "")
                return float(v) if v != "" else None
        return None
    if size_bytes >= max_s:
        for r in benchmark_rows:
            if r["transfer_size"] == max_s:
                v = r.get(col, "")
                return float(v) if v != "" else None
        return None
    # Find bracketing sizes
    sorted_rows = sorted(benchmark_rows, key=lambda x: x["transfer_size"])
    lo, hi = None, None
    for r in sorted_rows:
        if r["transfer_size"] <= size_bytes:
            lo = r
        if r["transfer_size"] >= size_bytes and hi is None:
            hi = r
            break
    if lo is None or hi is None:
        return None
    t_lo = lo.get(col, "")
    t_hi = hi.get(col, "")
    if t_lo == "" or t_hi == "":
        return float(t_lo) if t_lo != "" else (float(t_hi) if t_hi != "" else None)
    t_lo, t_hi = float(t_lo), float(t_hi)
    if lo["transfer_size"] == hi["transfer_size"]:
        return t_lo
    frac = (size_bytes - lo["transfer_size"]) / (hi["transfer_size"] - lo["transfer_size"])
    return round(t_lo + (t_hi - t_lo) * frac, 4)


def _generate_html_report(merged: dict, pivot_rows: list[dict], kv_cache_rows: list[dict]) -> str:
    """Generate a self-contained HTML report with a data table and Plotly chart."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    backends_present = []
    for b in BACKENDS:
        if any(row.get(f"{b}_throughput", "") != "" for row in pivot_rows):
            backends_present.append(b)

    # Plotly traces: benchmark lines
    plotly_traces = []
    for b in backends_present:
        x_vals = [row["transfer_size"] for row in pivot_rows]
        y_vals = []
        for row in pivot_rows:
            v = row.get(f"{b}_throughput", "")
            y_vals.append(float(v) if v != "" and v is not None else None)
        color = BACKEND_COLORS.get(b, "#333333")
        plotly_traces.append({
            "x": x_vals,
            "y": y_vals,
            "type": "scatter",
            "mode": "lines+markers",
            "name": b,
            "line": {"color": color, "width": 2.5},
            "marker": {"size": 6, "color": color},
        })
    plotly_traces_json = json.dumps(plotly_traces)
    x_tickvals_json = json.dumps([r["transfer_size"] for r in pivot_rows])
    x_ticktext_json = json.dumps([r["size_label"] for r in pivot_rows])

    meta_rows = ""
    for b in backends_present:
        meta = merged.get("metadata", {}).get(b, {})
        vi = meta.get("version_info", meta)
        parts = [f"<strong>{k}:</strong> {v}" for k, v in vi.items()]
        meta_rows += f'<tr><td class="backend-name" style="color:{BACKEND_COLORS.get(b, "#333")}">{b}</td><td>{" &nbsp;|&nbsp; ".join(parts) if parts else "N/A"}</td></tr>\n'

    table_header = '<th class="size-col">Transfer Size</th>'
    for b in backends_present:
        table_header += f'<th style="color:{BACKEND_COLORS.get(b, "#333")}">{b} (GB/s)</th>'

    table_rows = ""
    for row in pivot_rows:
        table_rows += f'<tr><td class="size-col">{row["size_label"]}</td>'
        for b in backends_present:
            v = row.get(f"{b}_throughput", "")
            cell = f"{v:.4f}" if isinstance(v, (int, float)) and v != "" else "&mdash;"
            table_rows += f"<td>{cell}</td>"
        table_rows += "</tr>\n"

    # KV cache dropdown and mapping: model names only, table shows all configs
    kv_cache_options = ""
    kv_cache_data_json = "[]"
    if kv_cache_rows:
        kv_cache_data_json = json.dumps(kv_cache_rows)
        unique_models = sorted(set(r["model_name"] for r in kv_cache_rows if r.get("model_name")))
        for name in unique_models:
            kv_cache_options += f'<option value="{name}">{name}</option>\n'
    else:
        kv_cache_options = '<option value="">No kv_cache_estimator.csv found</option>'

    benchmark_data_json = json.dumps([
        {"transfer_size": r["transfer_size"], **{f"{b}_throughput": r.get(f"{b}_throughput", "") for b in backends_present}}
        for r in pivot_rows
    ])
    backends_json = json.dumps(backends_present)
    backend_colors_json = json.dumps(BACKEND_COLORS)

    kv_th_cols = "".join(f'<th style="color:{BACKEND_COLORS.get(b, "#333")}">{b} (GB/s)</th>' for b in backends_present)
    # Build filter options from kv_cache data
    filter_model_opts = kv_cache_options
    filter_tp_opts = '<option value="">All</option>'
    filter_dtype_opts = '<option value="">All</option>'
    filter_seq_opts = '<option value="">All</option>'
    filter_batch_opts = '<option value="">All</option>'
    if kv_cache_rows:
        unique_tp = sorted(set(str(r.get("tp_size", "")) for r in kv_cache_rows if r.get("tp_size")))
        unique_dtype = sorted(set(str(r.get("dtype", "")) for r in kv_cache_rows if r.get("dtype")))
        unique_seq = sorted(set(str(r.get("seq_length", "")) for r in kv_cache_rows if r.get("seq_length")), key=lambda x: int(x) if x.isdigit() else 0)
        unique_concurrency = sorted(set(str(r.get("concurrency", r.get("batch_size", ""))) for r in kv_cache_rows if r.get("concurrency") or r.get("batch_size")), key=lambda x: int(x) if x.isdigit() else 0)
        for v in unique_tp:
            filter_tp_opts += f'<option value="{v}">tp={v}</option>\n'
        for v in unique_dtype:
            filter_dtype_opts += f'<option value="{v}">{v}</option>\n'
        for v in unique_seq:
            filter_seq_opts += f'<option value="{v}">seq={v}</option>\n'
        for v in unique_concurrency:
            filter_batch_opts += f'<option value="{v}">concurrency={v}</option>\n'

    kv_cache_card = ""
    if kv_cache_rows:
        kv_cache_card = f"""
<div class="card">
  <h2>KV Cache → Benchmark Mapping</h2>
  <p class="kv-desc">Select a model and optionally filter by tp, dtype, seq, concurrency to see configs plotted on the chart.</p>
  <div class="kv-controls kv-filters">
    <span><label for="kvModelSelect">Model:</label><select id="kvModelSelect"><option value="">-- Select model --</option>{filter_model_opts}</select></span>
    <span><label for="kvFilterTp">tp:</label><select id="kvFilterTp">{filter_tp_opts}</select></span>
    <span><label for="kvFilterDtype">dtype:</label><select id="kvFilterDtype">{filter_dtype_opts}</select></span>
    <span><label for="kvFilterSeq">seq:</label><select id="kvFilterSeq">{filter_seq_opts}</select></span>
    <span><label for="kvFilterBatch">concurrency:</label><select id="kvFilterBatch">{filter_batch_opts}</select></span>
  </div>
  <div id="kvMappingTable" class="kv-mapping-table" style="display:none;">
    <div class="table-scroll" style="max-height:360px;">
      <table>
        <thead><tr><th class="size-col">seq</th><th>concurrency</th><th>tp</th><th>dtype</th><th>KV Cache (MB) per layer</th>{kv_th_cols}</tr></thead>
        <tbody id="kvMappingBody"></tbody>
      </table>
    </div>
    <p id="kvInterpNote" class="kv-note">Lines: benchmark data from results_merged.csv. Scattered points: per-layer KV cache sizes from kv_cache_estimator.csv (y = interpolated throughput at that size).</p>
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KV-Transfer Performance Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a2e; --card-bg: #f8f9fa; --border: #dee2e6;
    --accent: #4472C4; --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--fg); padding: 2rem; max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: .25rem; }}
  .subtitle {{ color: #6c757d; font-size: .9rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; border-bottom: 2px solid var(--accent); padding-bottom: .4rem; display: inline-block; }}
  .chart-wrapper {{ position: relative; width: 100%; max-width: 1200px; margin: 0 auto; min-height: 520px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  thead th {{ background: #e9ecef; position: sticky; top: 0; }}
  th, td {{ padding: .55rem .75rem; text-align: right; border-bottom: 1px solid var(--border); }}
  .size-col {{ text-align: left; font-weight: 600; }}
  tbody tr:hover {{ background: #f1f3f5; }}
  .meta-table {{ font-size: .85rem; }}
  .meta-table td {{ text-align: left; padding: .35rem .75rem; }}
  .backend-name {{ font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }}
  .table-scroll {{ max-height: 520px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }}
  .kv-desc {{ color: #6c757d; font-size: .9rem; margin-bottom: .75rem; }}
  .kv-controls {{ margin-bottom: 1rem; }}
  .kv-controls label {{ margin-right: .5rem; font-weight: 500; }}
  .kv-controls select {{ padding: .4rem .75rem; font-size: .9rem; min-width: 180px; border-radius: 4px; border: 1px solid var(--border); }}
  .kv-filters {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }}
  .kv-mapping-table {{ margin-top: 1rem; }}
  .kv-note {{ font-size: .85rem; color: #6c757d; margin-top: .5rem; }}
  .chart-controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 0.75rem; }}
  .chart-controls label {{ display: flex; align-items: center; gap: 0.4rem; font-weight: 500; cursor: pointer; }}
  .chart-controls input[type="checkbox"] {{ width: 1rem; height: 1rem; cursor: pointer; }}
</style>
</head>
<body>

<h1>KV-Transfer Performance Benchmark</h1>
<p class="subtitle">Report generated {now_utc}</p>

<div class="card">
  <h2>Environment</h2>
  <table class="meta-table">
    {meta_rows}
  </table>
</div>

{kv_cache_card}
<div class="card">
  <h2>Throughput Comparison</h2>
  <p class="kv-desc">Scroll to zoom, drag to pan. Double-click to reset zoom.</p>
  <div class="chart-controls">
    <label for="showLabelsCheck"><input type="checkbox" id="showLabelsCheck" checked> Show labels on data points</label>
  </div>
  <div id="throughputChart" class="chart-wrapper"></div>
</div>

<div class="card">
  <h2>Results Table</h2>
  <div class="table-scroll">
    <table>
      <thead><tr>{table_header}</tr></thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
</div>

<script>
const baseTraces = {plotly_traces_json};
const layout = {{
  title: {{ text: 'Throughput vs Transfer Size (GB/s)', font: {{ size: 16, weight: 600 }} }},
  xaxis: {{ type: 'log', title: 'Transfer Size', tickvals: {x_tickvals_json}, ticktext: {x_ticktext_json}, tickfont: {{ size: 12 }} }},
  yaxis: {{ title: 'Throughput (GB/s)', zeroline: true, tickfont: {{ size: 12 }} }},
  legend: {{ orientation: 'h', y: 1.02, yanchor: 'bottom' }},
  margin: {{ t: 60, b: 50, l: 60, r: 40 }},
  hovermode: 'closest',
  showlegend: true,
  dragmode: 'pan'
}};
const config = {{ scrollZoom: true, responsive: true }};
Plotly.newPlot('throughputChart', baseTraces, layout, config);
window._baseTraces = baseTraces;
window._layout = layout;
window._config = config;
</script>

<script>
(function() {{
  const kvCacheData = {kv_cache_data_json};
  const benchmarkData = {benchmark_data_json};
  const backends = {backends_json};
  const backendColors = {backend_colors_json};

  function interpolateThroughput(sizeBytes, backend) {{
    const col = backend + '_throughput';
    const sizes = benchmarkData.map(r => r.transfer_size);
    if (!sizes.length) return null;
    const minS = Math.min(...sizes), maxS = Math.max(...sizes);
    if (sizeBytes <= minS) {{
      const r = benchmarkData.find(x => x.transfer_size === minS);
      const v = r && r[col];
      return (v !== '' && v !== undefined) ? parseFloat(v) : null;
    }}
    if (sizeBytes >= maxS) {{
      const r = benchmarkData.find(x => x.transfer_size === maxS);
      const v = r && r[col];
      return (v !== '' && v !== undefined) ? parseFloat(v) : null;
    }}
    const sorted = [...benchmarkData].sort((a,b) => a.transfer_size - b.transfer_size);
    let lo = null, hi = null;
    for (const r of sorted) {{
      if (r.transfer_size <= sizeBytes) lo = r;
      if (r.transfer_size >= sizeBytes && !hi) {{ hi = r; break; }}
    }}
    if (!lo || !hi) return null;
    const tLo = lo[col], tHi = hi[col];
    if (tLo === '' || tLo === undefined) return (tHi !== '' && tHi !== undefined) ? parseFloat(tHi) : null;
    if (tHi === '' || tHi === undefined) return parseFloat(tLo);
    const tLoV = parseFloat(tLo), tHiV = parseFloat(tHi);
    if (lo.transfer_size === hi.transfer_size) return tLoV;
    const frac = (sizeBytes - lo.transfer_size) / (hi.transfer_size - lo.transfer_size);
    return Math.round((tLoV + (tHiV - tLoV) * frac) * 10000) / 10000;
  }}

  function applyFilters() {{
    const modelName = document.getElementById('kvModelSelect')?.value;
    const filterTp = document.getElementById('kvFilterTp')?.value;
    const filterDtype = document.getElementById('kvFilterDtype')?.value;
    const filterSeq = document.getElementById('kvFilterSeq')?.value;
    const filterBatch = document.getElementById('kvFilterBatch')?.value;
    let configs = modelName ? kvCacheData.filter(c => c.model_name === modelName) : [];
    if (filterTp) configs = configs.filter(c => String(c.tp_size) === filterTp);
    if (filterDtype) configs = configs.filter(c => String(c.dtype) === filterDtype);
    if (filterSeq) configs = configs.filter(c => String(c.seq_length) === filterSeq);
    if (filterBatch) configs = configs.filter(c => String(c.concurrency || c.batch_size) === filterBatch);
    return configs;
  }}

  function updateChartAndTable() {{
    const tableDiv = document.getElementById('kvMappingTable');
    const tbodyEl = document.getElementById('kvMappingBody');
    const showLabels = document.getElementById('showLabelsCheck')?.checked ?? true;
    const configs = applyFilters();
    const baseTraces = JSON.parse(JSON.stringify(window._baseTraces || []));
    let traces = baseTraces;

    if (configs.length) {{
      for (let i = 0; i < backends.length; i++) {{
        const b = backends[i];
        const color = backendColors[b] || '#333';
        const xArr = [], yArr = [], textArr = [], hoverTextArr = [];
        for (const cfg of configs) {{
          const sizeBytes = cfg.kv_cache_bytes || (cfg.kv_cache_mb * 1024 * 1024);
          const tp = interpolateThroughput(sizeBytes, b);
          if (tp != null) {{
            xArr.push(sizeBytes);
            yArr.push(tp);
            const modelLabel = 'tp_' + cfg.tp_size + '_seq_' + cfg.seq_length + '_c_' + (cfg.concurrency || cfg.batch_size);
            const sizeStr = (sizeBytes >= 1024*1024*1024) ? (sizeBytes/1024/1024/1024).toFixed(1) + ' GB' : (sizeBytes/1024/1024).toFixed(1) + ' MB';
            textArr.push('(' + modelLabel + ', ' + sizeStr + ')');
            hoverTextArr.push(
              'Model: ' + (cfg.model_unique_name || cfg.model_name || '—') + '<br>' +
              'Size: ' + sizeStr + '<br>' +
              'Throughput: ' + tp.toFixed(4) + ' GB/s (' + b + ')'
            );
          }}
        }}
        if (xArr.length) {{
          const useText = showLabels && (i === 0);
          traces.push({{
            x: xArr, y: yArr, text: useText ? textArr : [],
            hovertext: hoverTextArr, hoverinfo: 'text',
            type: 'scatter', mode: useText ? 'markers+text' : 'markers',
            name: b + ' (KV)', showlegend: false,
            marker: {{ size: 8, color: color, symbol: 'diamond' }},
            textposition: 'top center', textfont: {{ size: 13, color: color }}
          }});
        }}
      }}
    }}

    Plotly.react('throughputChart', traces, window._layout, window._config);

    let rows = '';
    for (const cfg of configs) {{
      const sizeBytes = cfg.kv_cache_bytes || (cfg.kv_cache_mb * 1024 * 1024);
      const sizeStr = (sizeBytes >= 1024*1024*1024) ? (sizeBytes/1024/1024/1024).toFixed(1) + ' GB' : (sizeBytes/1024/1024).toFixed(1) + ' MB';
      rows += `<tr><td class="size-col">${{cfg.seq_length}}</td><td>${{cfg.concurrency || cfg.batch_size}}</td><td>${{cfg.tp_size}}</td><td>${{cfg.dtype || '—'}}</td><td>${{sizeStr}}</td>`;
      for (const b of backends) {{
        const tp = interpolateThroughput(sizeBytes, b);
        rows += `<td>${{tp != null ? tp.toFixed(4) : '—'}}</td>`;
      }}
      rows += '</tr>';
    }}
    if (tbodyEl) tbodyEl.innerHTML = rows;
    if (tableDiv) tableDiv.style.display = configs.length ? 'block' : 'none';
  }}

  const select = document.getElementById('kvModelSelect');
  const filterTp = document.getElementById('kvFilterTp');
  const filterDtype = document.getElementById('kvFilterDtype');
  const filterSeq = document.getElementById('kvFilterSeq');
  const filterBatch = document.getElementById('kvFilterBatch');
  const tableDiv = document.getElementById('kvMappingTable');
  const tbodyEl = document.getElementById('kvMappingBody');

  const showLabelsCheck = document.getElementById('showLabelsCheck');
  if (kvCacheData.length) {{
    [select, filterTp, filterDtype, filterSeq, filterBatch, showLabelsCheck].filter(Boolean).forEach(el => {{
      el.addEventListener('change', updateChartAndTable);
    }});
    // Auto-select first model when only one exists, so chart shows KV points on load
    const uniqueModels = [...new Set(kvCacheData.map(c => c.model_name))];
    if (select && uniqueModels.length === 1) {{
      select.value = uniqueModels[0];
    }}
    updateChartAndTable();
  }}
}})();
</script>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Merge backend benchmark results and generate CSV + HTML report."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("logs/shared"),
        help="Directory containing results_*.json files (default: logs/shared)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output merged JSON path (default: <input-dir>/results_merged.json)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--kv-cache-estimator-file",
        type=Path,
        default=None,
        help="Path to KV cache estimator CSV. If not set, no KV cache mapping is loaded.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_path = args.output or (input_dir / "results_merged.json")
    merge_results(input_dir, output_path, verbose=not args.quiet, kv_cache_estimator_file=args.kv_cache_estimator_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Primus / Megatron-LM pretraining launched by madengine.

Layout: madengine gives each node a directory and one combined stdout, ``node_*/stdout.out``. MAD's
run.sh loops over datatypes inside a single job, so one log covers BF16 then FP8 back to back and
the phase is announced in the log when Megatron picks up the next experiment config.

Traces: Megatron writes them under the per-experiment tensorboard directory, whose name carries the
datatype, so the phase of a trace is readable from its path.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.spec import (NODE_FROM_PARENT, PHASE_FROM_MARKER, EngineSpec, LogLayout, LogMetric,
                         ReportNotes, SanityLimits, TraceLayout)

#: The Primus experiment config picked by MAD's dtype loop. The model part varies per workload
#: (llama3.1_8B, llama3.1_70B, gpt_oss_120B, ...), so only the dtype and the suffix are anchored.
RE_PHASE = re.compile(r"[A-Za-z0-9._]+-(BF16|FP8|MXFP8|MXFP4)-pretrain\.yaml")

#: Same anchor on a trace path: `.../llama3.1_70B-BF16-pretrain/tensorboard`.
RE_TRACE_PHASE = re.compile(r"-(BF16|FP8|MXFP8|MXFP4)-pretrain")

RE_ITER_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
RE_TOKENS = re.compile(r"tokens/s/GPU\):\s*([\d.]+)")
RE_TFLOPS = re.compile(r"TFLOP/s/GPU\):\s*([\d.]+)")


def resolve_traces(root: Path) -> dict:
    """Map every tensorboard trace directory under ``root`` to its datatype phase."""
    found: dict = {}
    for path in sorted(root.rglob("*-pretrain")):
        m = RE_TRACE_PHASE.search(path.name)
        if not m or not path.is_dir():
            continue
        tb = path / "tensorboard"
        target = tb if tb.is_dir() else path
        found.setdefault(m.group(1), []).append(target)
    return found


SPEC = EngineSpec(
    name="primus",
    summary="Primus/Megatron-LM pretraining, one stdout per node covering several datatype phases",
    logs=LogLayout(
        globs=("node_*/stdout.out", "node_*/stdout.out.gz"),
        phase_from=PHASE_FROM_MARKER,
        node_from=NODE_FROM_PARENT,
        phase_marker=RE_PHASE,
        marker_guard="-pretrain.yaml",
    ),
    metrics=(
        LogMetric("iter_ms", "iteration", RE_ITER_MS, "median iteration {value:.1f} ms"),
        LogMetric("tokens", "tokens/s/GPU", RE_TOKENS,
                  "Throughput reported in log: {value:.1f} tokens/s/GPU"),
        LogMetric("tflops", "TFLOP/s/GPU", RE_TFLOPS,
                  "Compute reported in log: {value:.1f} TFLOP/s/GPU"),
    ),
    iteration_metric="iter_ms",
    traces=TraceLayout(
        resolve=resolve_traces,
        rank_patterns=(re.compile(r"rank\[(\d+)\]"),),
    ),
    limits=SanityLimits(),
    notes=ReportNotes(
        rank_coverage="torchrun's `--local-ranks-filter` decides which ranks reach stdout",
        damage_cause=("The ranks that pass the filter share one stdout, so at INFO verbosity some "
                      "records overwrite each other mid-write and cannot be attributed; under a "
                      "percent is normal."),
    ),
    fingerprints=("node_*/stdout.out",),
)

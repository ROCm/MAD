"""sglang PD-disaggregated serving (prefill and decode servers on separate nodes).

Layout: one log per role per node, ``prefill_NODE0.log`` / ``decode_NODE2.log``, with no phase
markers -- the role is the phase, and it is in the file name. Kept gzipped as a matter of course: a
decode log at ``NCCL_DEBUG=INFO`` reaches 2 GB and has filled a shared home directory once.

There are no iterations to divide by and the server log carries no throughput, so those numbers come
from the benchmark CSV of a separate run without profiling. What this engine's numbers do and do not
describe is the scope note below, and it is the reason this prose lives with the engine: a report
that inherited it by phase name would claim things about a run that never happened.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.spec import (LOG_PER_RANK, NODE_FROM_STEM, PHASE_FROM_FILENAME, A2AKernels,
                         BenchmarkLayout, CounterLayout, EngineSpec, LogLayout, ReportNotes,
                         RunConfigLayout, SanityLimits, StepInvalidator, StepTimingLayout,
                         TraceLayout)

#: `benchmark_<job>_..._PROFILE_<role>.log`, written once per profile point by bench_serving.
RE_PROFILE_LOG = re.compile(r"_PROFILE_(\w+)\.log$")

#: `server_args=ServerArgs(model_path='...', disable_cuda_graph=False, ...)`, printed once per
#: server. The effective configuration with defaults applied, which the command line is not.
RE_SERVER_ARGS = re.compile(r"ServerArgs\((.*)\)\s*$")

#: `Decode batch. #running-req: 16, ..., cuda graph: True, gen throughput (token/s): 234.56, ...`
#: The running batch and the generation rate are required; `cuda graph` is absent on older sglang
#: and makes graph replay observable per interval rather than only at startup.
RE_DECODE_BATCH = re.compile(
    r"#running-req:\s*(?P<batch>\d+).*?"
    r"(?:cuda graph:\s*(?P<graphed>True|False).*?)?"
    r"gen throughput \(token/s\):\s*(?P<rate>[\d.]+)"
)

#: The expert all-to-all, by the names each backend gives it. MoRI drives its own IBGDA transport
#: and DeepEP goes over rocSHMEM, so none of this reaches an RCCL log or a `record_param_comms`
#: event -- the trace is where it is named or nowhere.
#:
#: Ordered: `combine` before `dispatch` for a name carrying both, and the `rocshmem`/`nvshmem`
#: fallback last so a transport kernel is not claimed as a stage already known.
#:
#: Boundaries are alphanumeric look-arounds rather than ``\b``, because ``_`` is a word character
#: and ``\bmori\b`` would not match ``mori_ep_dispatch``. They still refuse ``memori``. MoRI's own
#: name appears in none of its kernels (`EpDispatchInterNodeV1KernelLowLatency_fp8_fnuz`), which
#: is why the report's unclassified list is what a new backend gets read through.
_BOUND = r"(?<![a-z0-9]){}(?![a-z0-9])"
#: The two backends' names, each bounded, for the patterns that qualify a stage by backend.
_BACKEND = "(?:" + _BOUND.format("mori") + "|" + _BOUND.format("deep_?ep") + ")"
A2A_PATTERNS = (
    # The barrier belongs to combine and is named for it, so it must be matched before the generic
    # combine pattern swallows it. Both come before `dispatch`: a name carrying both words is a
    # combine of dispatched tokens. The standalone `Ep(Dispatch|Combine)` alternatives need a
    # leading boundary, or the `ep_dispatch` inside `step_dispatch` and `prep_dispatch_layout`
    # counts unrelated kernels as expert traffic.
    ("combine barrier", re.compile(r"(?<![a-z0-9])ep_?combine\w*sync\w*barrier", re.I)),
    # The backend-qualified alternatives carry the same boundaries; without them
    # `memorize_dispatch` matches on the `mori` inside `memorize`.
    ("combine", re.compile(rf"(?<![a-z0-9])ep_?combine|{_BACKEND}.*combine"
                           rf"|combine.*{_BACKEND}", re.I)),
    ("dispatch", re.compile(rf"(?<![a-z0-9])ep_?dispatch|{_BACKEND}.*dispatch"
                            rf"|dispatch.*{_BACKEND}", re.I)),
    ("permute", re.compile(r"(?:pre_?permute|post_?permute|moe_?permute)", re.I)),
    ("transport", re.compile(r"roc?shmem|nvshmem|" + _BOUND.format("ibgda"), re.I)),
    ("backend other", re.compile(_BOUND.format("mori") + "|" + _BOUND.format("deep_?ep"), re.I)),
)

#: Which implementation of the exchange a kernel name says ran, over two axes: latency against
#: throughput, and intranode against internode. Either breaks a comparison -- on the Kimi-K2 pair
#: the gap fell from 14.7 ms a step to 2.6 ms once both backends ran intranode kernels -- so a
#: report that does not say which ran invites the difference to be read as the backend's.
#:
#: Ordered most specific first, because the first match wins and a compound name would otherwise
#: be labelled by one axis, silently dropping the other.
A2A_VARIANTS = (
    ("intranode low latency",
     re.compile(r"intra_?node.*(?:low_?latency|_ll\b)|(?:low_?latency|_ll\b).*intra_?node", re.I)),
    ("intranode", re.compile(r"intra_?node", re.I)),
    ("low latency", re.compile(r"low_?latency|_ll\b", re.I)),
    ("normal", re.compile(r"normal|inter_?node(?!.*low)", re.I)),
)


#: This engine's own vocabulary: `deepep_mode` and `moe_a2a_backend` mean nothing to a training
#: engine, so these are `RunConfigLayout` data rather than shared code.
PERF_RELEVANT = frozenset({
    # The model itself. Kept out of NOISE so two runs of different weights are not reported as
    # identical, but listing it without marking it left the report saying "none of these are
    # known to move throughput" under a row naming two different models.
    "model_path",
    "disable_cuda_graph",
    "cuda_graph_bs",
    "cuda_graph_max_bs",
    "enable_torch_compile",
    "mem_fraction_static",
    "max_running_requests",
    "chunked_prefill_size",
    "attention_backend",
    "moe_runner_backend",
    "moe_a2a_backend",
    "deepep_mode",
    "quantization",
    "dtype",
    "kv_cache_dtype",
    "tp_size",
    "dp_size",
    "ep_size",
    "enable_dp_attention",
    # The MoE-parallelism knobs the Kimi-K2 entries set alongside `enable_dp_attention`: they
    # split the dense layers, the LM head and the attention control broadcast across ranks, so a
    # pair differing in one of them differs in execution parallelism.
    "moe_dense_tp_size",
    "enable_dp_lm_head",
    "enable_dp_attention_local_control_broadcast",
    "disable_custom_all_reduce",
    "disable_radix_cache",
    "speculative_algorithm",
    "disaggregation_transfer_backend",
})

#: Settings that differ between two runs for reasons that carry no performance meaning, so a diff
#: listing them buries the ones that do. Ports and hosts differ on every pair of runs.
#: `model_path` and `tokenizer_path` stay out deliberately: dropping them would let two runs of
#: *different models* compare as identical. They are normalised to their last path component
#: instead, so a remount is quiet and a different model is loud.
NOISE = frozenset({
    "host", "port", "nccl_port", "dist_init_addr",
    "random_seed", "log_level", "log_level_http", "log_requests",
    "node_rank", "disaggregation_bootstrap_port", "gpu_id_step",
})

#: Settings whose *value* must never reach an artifact: reports and `run_config.csv` are made to
#: be shared, so a credential in them travels too. Redacted unconditionally at parse time, not put
#: in NOISE, which `include_noise` can switch back on.
SECRET = frozenset({"api_key", "admin_api_key", "hf_token", "token"})


def resolve_traces(root: Path) -> dict:
    """Map sglang's timestamp-named trace directories onto the role each one profiled.

    sglang names a trace directory after the epoch second the capture started and never mentions the
    role. The authoritative mapping is in the profile-point log itself: bench_serving writes one
    ``*_PROFILE_<role>.log`` per profile point and records the ``output_dir`` it asked each worker
    of that role for, so the role's directories are named in its own log, one per node.

    Timestamps are deliberately not used to match them. The directory names come from the
    container's clock and the file mtimes from the shared filesystem's, and on this cluster the two
    are about 460 seconds apart, which is enough to attribute a whole role's traces to the other
    role.

    This used to be done by reading timestamps by eye and pasting one directory per role into a
    shell script, which is how the second node of every role went missing from three reports.
    """
    trace_dirs = {p.name: p for p in root.glob("torchprof/*") if p.is_dir()}
    role_logs = {}
    for log in sorted(root.glob("*_PROFILE_*.log")):
        m = RE_PROFILE_LOG.search(log.name)
        if m:
            role_logs[m.group(1)] = log
    if not trace_dirs or not role_logs:
        return {}

    found: dict = {}
    for role, log in role_logs.items():
        text = log.read_text(errors="ignore")
        claimed = sorted(name for name in trace_dirs if name in text)
        if claimed:
            found[role] = [trace_dirs[name] for name in claimed]

    # A role that was profiled but ends up with no trace directory means the mapping is unknown --
    # an older sglang that does not log output_dir, or artifacts copied without their logs. Guessing
    # would mislabel a whole report, so this stops instead.
    missing = set(role_logs) - set(found)
    if missing:
        raise ValueError(
            f"cannot map trace directories to roles under {root}: {sorted(missing)} got none. "
            f"Trace dirs {sorted(trace_dirs)}, profile points {sorted(role_logs)}. The "
            "profile-point log should name the output_dir of each worker; pass --torch-trace "
            "ROLE=PATH explicitly.")
    return found


SPEC = EngineSpec(
    name="sglang-disagg",
    summary="sglang PD-disaggregated serving, one log per role per node, the role being the phase",
    logs=LogLayout(
        globs=("prefill_NODE*.log", "decode_NODE*.log",
               "prefill_NODE*.log.gz", "decode_NODE*.log.gz"),
        phase_from=PHASE_FROM_FILENAME,
        node_from=NODE_FROM_STEM,
        phase_of_name=lambda stem: stem.split("_")[0],
    ),
    # `rccl/prefill_NODE0.<host>.<pid>.log`, one per server process, written when the launcher was
    # given RCCL_LOG_DIR. The role and node label are the ones the shared logs already use, so a
    # report reads the same whichever way the run was measured.
    rccl_logs=LogLayout(
        globs=("rccl/*_NODE*.log", "rccl/*_NODE*.log.gz"),
        phase_from=PHASE_FROM_FILENAME,
        node_from=NODE_FROM_STEM,
        phase_of_name=lambda stem: stem.split("_")[0],
        node_of_name=lambda stem: stem.split(".")[0],
        written_by=LOG_PER_RANK,
    ),
    traces=TraceLayout(
        dir_glob="torchprof/*",
        resolve=resolve_traces,
        rank_patterns=(re.compile(r"-TP-(\d+)"),),
    ),
    run_config=RunConfigLayout(guard="ServerArgs(", pattern=RE_SERVER_ARGS,
                               perf_relevant=PERF_RELEVANT, noise=NOISE, secret=SECRET,
                               path_valued=frozenset({"model_path", "tokenizer_path"})),
    steps=StepTimingLayout(
        guard="gen throughput", pattern=RE_DECODE_BATCH, unit="decode logging interval",
        # sglang prints `speculative_algorithm=None` when it is off. With one on, a decode step
        # accepts a variable number of tokens, so `gen throughput / #running-req` is accepted
        # tokens per second per request and not the reciprocal of a step duration -- a plausible
        # number, which is why it is declared rather than left to a reader.
        invalidated_by=(StepInvalidator(
            setting="speculative_algorithm", benign=("None",),
            why=("a decode step then emits a variable number of accepted tokens per request "
                 "rather than exactly one, so the batch divided by the generation rate is not a "
                 "step duration")),)),
    a2a=A2AKernels(patterns=A2A_PATTERNS, variants=A2A_VARIANTS),
    # What `bench_serving` writes through madengine's perf CSV: one row per metric, the
    # configuration in the `model` column as `2p2d_isl1024_osl1024_con64`, the value in
    # `performance`. The metric names are this harness's own, so they are declared here.
    benchmark=BenchmarkLayout(
        globs=("perf_*.csv",),
        point=re.compile(r"isl(?P<isl>\d+)_osl(?P<osl>\d+)_con(?P<con>\d+)"),
        metrics=(("total_token_throughput_tok_s", "total tok/s", "{:.0f}"),
                 ("output_token_throughput_tok_s", "output tok/s", "{:.0f}"),
                 ("request_throughput_req_s", "req/s", "{:.3f}"),
                 ("mean_itl_ms", "ITL ms", "{:.1f}"),
                 ("mean_ttft_ms", "TTFT ms", "{:.0f}"),
                 ("mean_e2e_latency_ms", "E2E ms", "{:.0f}")),
        e2e_metric="mean_e2e_latency_ms",
        ttft_metric="mean_ttft_ms",
        itl_metric="mean_itl_ms"),
    # `rdma/decode_NODE2.csv`, written by `rdma_counters.sh` alongside the servers when
    # RDMA_COUNTERS=1. The kinds cover the two drivers seen on this cluster: mlx5 spells them
    # `rx_write_requests`, bnxt_re `rx_write_req`, and both spell atomics with `atomic`.
    counters=CounterLayout(
        globs=("rdma/*_NODE*.csv",),
        node_of_name=lambda stem: stem,
        # Operations first, because they carry what little discrimination this channel has: a
        # protocol that waits leaves reads and atomics behind, while writes alone fit several
        # shapes. Volume is split by direction rather than pooled, since the KV transfer's
        # direction (52k received writes on decode, none on prefill) is invisible in one row.
        kinds=(("rx write req", re.compile(r"rx_write_req", re.I)),
               ("rx read req", re.compile(r"rx_read_req", re.I)),
               ("rx atomic req", re.compile(r"rx_atomic", re.I)),
               ("rx bytes", re.compile(r"port_rcv_data", re.I)),
               ("tx bytes", re.compile(r"port_xmit_data", re.I)),
               # The portable port totals only: `unicast_rcv_packets` and `multicast_rcv_packets`
               # are *subsets* of `port_rcv_packets`, so pooling them doubles the count. The
               # vendor counters keep their own names and appear as their own rows.
               ("rx packets", re.compile(r"port_rcv_packets$", re.I)),
               ("tx packets", re.compile(r"port_xmit_packets$", re.I))),
        # `port_rcv_data` and `port_xmit_data` are defined by the InfiniBand spec in units of four
        # octets; without this the byte volumes read four times too small.
        volume_kinds=("rx bytes",),
        operation_kinds=("rx write req", "rx read req", "rx atomic req"),
        scale={"rx bytes": 4, "tx bytes": 4}),
    # What the scope note below claims was disabled for measurement, and what PROFILE_ENABLE=1
    # sets. A run reporting either as False was not measured the way the note says.
    measurement_assumptions=(("disable_custom_all_reduce", "True"),
                             ("disable_cuda_graph", "True")),
    limits=SanityLimits(),
    notes=ReportNotes(
        communicator="so each node runs its own TP={nranks} replica",
        damage_cause=("A role's ranks share one stdout, so at INFO verbosity some records "
                      "overwrite each other mid-write and cannot be attributed; under a percent is "
                      "normal."),
        scope=(
            "Scope of an sglang PD-disaggregated profile, all three points are by design:",
            "the numbers above describe a *measurement* configuration, not the tuned one. TP is "
            "routed through RCCL with `--disable-custom-all-reduce`, and **both prefill and "
            "decode** run without HIP graphs, because sglang's own all-reduce kernel and graph "
            "replay each bypass every profiler. Prefill was excluded from that until a model "
            "arrived whose DP prefill block captures graphs, which silently hid the collectives "
            "the profile was for. Read throughput from a run without `PROFILE_ENABLE` instead.",
            "KV cache transfer between the prefill and decode groups goes over mooncake RDMA, "
            "never through RCCL, so the inter-node traffic that defines this topology does not "
            "appear here at all. What is measured is the intra-node TP exchange of one role.",
            "Each prefill and decode node is an independent TP replica, so per-rank figures carry "
            "over; totals across the group do not.",
        ),
        step_basis=("sglang logs its running batch and its generation rate together, and a decode "
                    "step emits one token per running request, so the rate divided by the batch is "
                    "the step frequency. It is the server's own accounting rather than an "
                    "instrument attached to it, which is why it is present in a profiled and an "
                    "unprofiled run alike and comparable across the two."),
        graphs_off=("For this engine that is also what `PROFILE_ENABLE=1` does on purpose -- graph "
                    "replay dispatches one packet and hides every collective inside it -- so an "
                    "unprofiled run of the same manifest is the reference to compare against, not "
                    "another backend."),
        a2a_outside_rccl=("The MoE all-to-all appears in no RCCL log: MoRI drives its own IBGDA "
                          "transport and DeepEP goes over rocSHMEM, so neither emits an RCCL "
                          "record nor a `record_param_comms` event. The trace is where these "
                          "operations are named or nowhere."),
        unmarked_window=("an unmarked window each: sglang's /start_profile emits no ProfilerStep "
                         "annotations, so the counts below are per capture, which held roughly one "
                         "forward pass"),
        trace_vs_log=("The two channels also cover different windows, so their absolute volumes "
                      "are not meant to agree: the RCCL log spans the whole run including "
                      "communicator setup and weight loading, while the trace covers only the few "
                      "steps the profile point requested. Compare the mix and the sizes, not the "
                      "totals."),
    ),
)

# Data quality: torn records and sanity bounds

A role's ranks share one stdout. At `NCCL_DEBUG=INFO` a decode log reaches millions of
lines, and a fraction of a percent of them arrive spliced: two concurrent writes
overwrite each other mid-record. Taken at face value this is not a small error. A single
torn `count` field once produced a 91 GiB "AllReduce" that carried 16% of a report's
decode volume, and torn prefixes invented ranks 12 and 22 on an eight-GPU node.

Expected rate on these runs: **0.5% to 0.9%** of collective records.

## How a record is judged

In order, cheapest first (`core/rccl_log.py`):

1. **Known collective name.** Anything outside the RCCL set means the name itself was
   overwritten — real logs contain `prllReduce` and `mscclFuncAllscclFuncAllReduce`.
2. **`nranks` in range.** A communicator wider than the engine's bound means the digits
   were spliced; one log claimed `nranks=55688946`.
3. **Rank inside its own communicator.** `globalrank` ≥ `nranks` cannot happen.
4. **No second record header inside the match.** Otherwise the regex takes the
   collective from one record and the count from the next.
5. **Record tail present**, but only where the log has tails at all. Intact records end
   in `comm 0x.. [nranks=N] stream 0x.. task N globalrank N`, and a half-overwritten one
   almost never does, which makes it the sharpest test available. Older RCCL builds print
   no tail; those logs stay on checks 1–4, since holding them to it would discard
   everything.
6. **Message size within the bound.**

The tail is also the better rank identity. Ranks are keyed by `(node, globalrank)` rather
than by the `host:pid` prefix: the file name already says which machine a log is from, and
that prefix is the part tearing corrupts most often.

Every rejection is counted by reason, listed in the report and written to
`discarded_records.csv`.

## Topology lines are judged too

The `NCCL INFO Channel .. : src -> dst via ..` lines behind the connectivity table tear the
same way, and there the damage lands in the transport name. The transport must match a name
RCCL can print exactly (`P2P/IPC`, `NET/IB/3/GDRDMA/Shared`, `SHM/direct/direct`, `LOC`, ...);
a prefix match is not enough, because the spliced strings are prefixes of the real ones —
`P2P/IPCrank`, `P2P/Iproxy`, `PCCL`, `P50`, `localRank` all reached a report as transports
before the check existed, and a scope rule of "anything not starting with `P2P` is inter-node"
turned 20 of them into inter-node links on a prefill role that has no inter-node communicator
at all.

Rejected topology lines are counted separately from collective records — a lost one costs an
edge, not volume — and the connectivity section says how many, so a short table is not read as
a sparse fabric. Whether a real transport is intra- or inter-node comes from the same table
(`transport_scope()` in `core/rccl_log.py`), not from a guess about its name.

## When a bound rejects real records

The bounds are properties of a run's scale, not of the parser:

| bound | default | why |
|---|---|---|
| `max_msg_bytes` | 512 MiB | the largest legitimate message observed across these runs is a 256 MiB prefill all-reduce, so the cap sits at twice that |
| `max_nranks` | 64 | eight GPUs per node here, with room for wider communicators |

A larger model or expert parallelism across nodes will exceed both legitimately. The
report therefore does not merely footnote a count — it says how many records hit a bound
and names the flag that raises it:

```bash
collective_report.py --run-dir ... --out-dir ... --max-msg-bytes 2147483648
```

Raising a bound changes which records are kept, so it invalidates the parse cache
automatically (the bound is part of the cache signature).

Read the two cases apart:

- **Rejections spread across reasons 1–5, well under a percent** — normal torn writes.
  Nothing to do.
- **Rejections concentrated at a bound** — suspect the bound before suspecting the log.
  Check the largest sizes in `collective_message_sizes.csv`: a genuine large message
  looks like a round number repeated across ranks, a splice looks like two counts
  concatenated (`804352` and `160` arriving as `804352160`) and appears once.

## Idle ranks are not peers

Per-rank figures divide by the ranks that carried traffic, not by the ranks present in
the log. A disaggregated proxy is free to leave one replica of a role nearly idle — one
run had a decode node serve 8 batches against 3874 on the other — and averaging over
ranks that did nothing halves every per-rank number. Ranks below 5% of the busiest rank's
volume are treated as idle, and the report says how many were left out.

That also means an imbalanced run's report is about the replica that worked. The report
separates the two effects, because only one of them is a finding:

- **within a node**, the tensor-parallel exchange is symmetric by construction, so a
  spread beyond a couple of percent means one rank carries work the others do not;
- **across nodes**, each node is an independent replica, so a ratio reflects how requests
  were spread, not a communication problem.

A large across-node ratio is worth confirming on a second run before it goes in a
document: a 7158x decode imbalance did not reproduce, and the balanced run is the one
that describes the topology.

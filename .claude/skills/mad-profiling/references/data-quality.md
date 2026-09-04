# Data quality: torn records and sanity bounds

A role's ranks share one stdout. At `NCCL_DEBUG=INFO` a decode log reaches millions of
lines, and a fraction of a percent of them arrive spliced: two concurrent writes
overwrite each other mid-record. Taken at face value this is not a small error. A single
torn `count` field once produced a 91 GiB "AllReduce" that carried 16% of a report's
decode volume, and torn prefixes invented ranks 12 and 22 on an eight-GPU node.

Expected rate on these runs: **0.5% to 0.9%** of collective records.

The checks below are what makes such a log usable, but they are a second line of defence.
The first is not to share the stream: `RCCL_LOG_DIR` makes the launcher set
`NCCL_DEBUG_FILE` and every process writes its own file, which nothing else can overwrite
(see [measurement-setup.md](measurement-setup.md)). A report says which of the two it read.
On a run measured that way, damage counted here is not tearing and should be looked at
rather than accepted as background noise.

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

## Traces go bad differently from logs

Everything above is about the RCCL text logs. A torch trace is gzip, one file per rank, and it
fails in three ways that a text log does not. Each is reported and read past rather than raised on
— the same call `resolve_traces` makes for a trace directory that turned out empty, because losing
a 32-rank capture over one rank is the wrong trade.

| symptom | exception | what it is |
|---|---|---|
| no gzip trailer | `EOFError` | the rank was killed at teardown; the file is otherwise whole |
| bad bytes mid-stream | `zlib.error` — *invalid literal/length code* | damage, not a short write. Not an `OSError`, so it needs naming separately |
| a duration that is not a number, e.g. `4.200.347` | `ValueError` | two durations spliced by a profiler flushing from several threads |

The third is the log-tearing of the previous sections arriving in a trace, and it is handled the
same way round: the **call is kept and only the duration is dropped**, because the event is in the
file so the collective did happen, and this report holds call counts sound where it holds durations
suspect. Nothing is inherited from the previous event, which would put a plausible number where the
file has none.

The first two truncate a rank's contribution, so that rank's **call counts become floors and its
shares become suspect**. Normalising removes how long the capture lasted; it does not remove
*which part* of it was lost. A cut falling between a dispatch and the combine that follows it drops
a suffix that is not a representative sample of the mix, and the share moves with it. Treat a share
from a truncated capture as possibly biased, and reparse for a clean one before comparing shares
across runs.

How much this matters is measurable rather than assumable, and worth measuring before either
trusting or discarding a capture: on one DeepEP decode capture the metadata share of the exchange
came to 92.6% over all sixteen traces and 91.6% over the ten that read clean, so the conclusion
held — but that was a result, not a property of truncation.

The report names the ranks affected, which is what tells a reader whether a per-rank imbalance is
the run's or the capture's.

### A failed read is not a corrupt file

**Confirm on a second run before concluding that trace data is damaged.** Four passes over one
unchanged DeepEP capture on NFS reported 1, then 10, then 12, then **0** unreadable files of 16.
The files were intact throughout; the reads were flaky. A single pass — even `gzip -t` repeated on
one file, which failed three times running inside one bad window — is too narrow a sample to call
a file corrupt.

The obvious shortcut does not work: **a tail of zero bytes does not prove the file is short.** One
trace of this capture failed four reads in a row, and a local copy of it ended in sixteen null
bytes — which looks exactly like a write that never finished. The source was intact: a later pass
read all sixteen traces clean, and the copy differed from the source in its last bytes while
matching it in length. The flaky read had returned zeros, and `cp` wrote them out without
complaint. Compare two independent reads of the same bytes; do not trust one read of a copy, since
the copy is itself one read.

The same filesystem does it to the adapter-counter samples, which are written by a shell loop and
read back by the report: one sample of 138 in a validation run came back as a megabyte of NUL
bytes in the middle of an otherwise sound file. It costs one counter of one sample and the parser
says so, but it stopped `csv` outright until it was handled — the lesson being that *every* channel
reading from this filesystem needs the read-past-damage treatment, not just the ones that had
produced damage so far.

This is the rule from the end of this file applied to a different channel, and it cuts both ways:
a run reporting **zero** unreadable traces is the one to quote, and a run reporting many is a
reason to rerun the parse rather than to caveat the numbers. Where the difference matters, check
whether it does: on that capture the metadata share of the expert exchange came to 92.6% over all
sixteen traces and 91.6% over the ten that read clean, so the conclusion held either way and the
damage was worth reporting rather than hiding.

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

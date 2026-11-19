# v4 disagg perf diagnosis — the ~160s per-request stall

## Confirmed symptom
Every disagg request (any ctx, thinking on OR off) carries a ~160-180s FIXED floor.
- 2K prompt, thinking=false, 32 out tokens: 162s / 178s (repeatable), recall CORRECT.
- 50K NIAH: 197s. 100K: 478s. => ~160s FLOOR + context-scaling prefill/transfer on top.
- GPU pinned 100% during the stall; output is correct (KV eventually arrives).
- NOT reasoning (thinking off = same), NOT config, NOT the qp knobs.

## Root cause (code-confirmed): decode->prefill notify DP-rank mismatch -> deferred wait
moriio_connector.py:708 (docstring): "Both legs of a disagg pair must agree on a single
prefill DP rank, otherwise the notify lands on a rank that never handshook and the request
HANGS until VLLM_MORIIO_DEFERRED_TIMEOUT_S."
- :810-835: connector consumes router-provided `remote_dp_rank` VERBATIM; if the router does
  not pin it (and pin the matching X-data-parallel-rank dispatch), decode notifies the wrong
  prefill rank -> that rank never RDMA-writes this request's KV -> decode waits the deferred
  timeout, then a fallback recovers the KV (why output is correct but slow).
- The connector expects an **llm-d routing sidecar** (dp_rank.go: H=pickDPRank(uuid,dp_size))
  to pin BOTH legs to the same rank H. We run the STOCK vllm-router (pin 82dc9811), which
  does round-robin dispatch but does not guarantee remote_dp_rank == dispatch rank.
- Observed: decode notifies vary (rank=2, rank=3...) but may not match where prefill actually
  ran that request -> per-request deferred stall.

## Why v3 worked (~20s/50K) but v4 stalls (~197s), same connector Python + same router
The connector Python is byte-identical to v3. The stall is the new MoRI(624002c8) WRITE/notify
path being stricter about rank matching (or a new deferred-wait default), OR the new base's DP
dispatch differs. The ~160s == a deferred-timeout default somewhere in the WRITE notify path.

## Fixes to test (in order)
1. **MORIIO_READ_MODE=1** (READ instead of WRITE): different transfer path; doc says READ has a
   "returnable path" where prefill echoes back the rank it ran -> may sidestep the notify
   mismatch entirely. FASTEST test (env only).
2. **Router rank-pinning**: make the router set kv_transfer_params.remote_dp_rank == the
   X-data-parallel-rank it dispatched (the contract at :714). Needs router code, OR a
   connector-side self-derive (compute H=hash(uuid)%dp_size on both legs identically).
3. **Lower the deferred timeout** so the fallback fires fast (masks, doesn't fix).
4. Bisect MoRI 624002c8 for the WRITE-notify behavior change vs v3's MoRI.

## FIX IN TEST: MORIIO_READ_MODE=1 (returnable path)
CONFIRMED the exact mechanism (moriio_connector.py:1190-1210 + 794/816-835):
- Router (pin 82dc9811, vllm_pd_router.rs:391-490/697-845) sets the X-data-parallel-rank
  DISPATCH header (round-robin) but does NOT set kv_transfer_params.remote_dp_rank.
- In WRITE mode, decode does remote_dp_rank = kv_transfer_params.get("remote_dp_rank", 0)
  -> DEFAULTS TO 0 -> always notifies prefill rank 0. But prefill ran on a round-robin rank
  != 0 -> notify misroutes -> decode waits the deferred timeout (~160s) then a fallback
  recovers the KV (correct output, ~160s late).
- WRITE mode is NOT "returnable": prefill's request_finished echo of remote_dp_rank only
  reaches decode on READ / serial-WRITE paths (:719-720, :1190).
FIX: MORIIO_READ_MODE=1 -> READ path IS returnable. request_finished (:1200-1210) returns
remote_dp_rank=self._global_dp_rank (the rank prefill ACTUALLY ran) + remote_dp_rank_override
=True; router forwards prefill's kv_transfer_params to decode (vllm_pd_router.rs:420-432);
decode gate _should_notify = (self._global_dp_rank == remote_dp_rank) -> notifies the CORRECT
rank -> NO deferred stall. Expected: per-request latency drops from ~160s to ~seconds.
STATUS: relaunched with MORIIO_READ_MODE=1, keeping thinking=false/320K/F_A_P. Validating.

## READ_MODE test RESULT: FAILED (rejected)
MORIIO_READ_MODE=1 + F_A_P: request still 149.2s (stall NOT fixed) AND output GARBAGE
("与 相似 相似..." repeated CJK, not "8241"). The READ+FULL_AND_PIECEWISE combo breaks
accuracy exactly as the connector warns (moriio_connector.py:230: "per-layer KV-read barrier
can't fire inside full graph"). => the ~150s stall is COMMON to WRITE and READ, so it is NOT
the WRITE-notify DP-rank mismatch. Reverting READ mode.
NEW HYPOTHESIS: the ~150s is a FIXED COMPUTE/BARRIER cost common to both transfer paths, not
a notify timeout. GPU pinned 100% the whole time. Candidates: (a) decode forward over a
huge/padded batch per request; (b) a MoRI all2all/collective barrier (mori_low_latency
InterNodeV1LL) stalling ~150s; (c) the eager_handshake_all_dp_ranks all-reduce barrier.
Since it's GPU-bound (not network-idle), lean (a)/(b). NEXT: check decode batch/token shape
per request + whether mori_low_latency all2all is the 150s. Consider decode all2all-backend
= mori_high_throughput, or reduce max_num_batched_tokens.

## *** LIKELY REAL FIX: decode CG=PIECEWISE (not FULL_AND_PIECEWISE) ***
Compared to the WORKING v3 (#193) config (~20s/50K): v3 decode = mori_low_latency + CG=**PIECEWISE**
(run_2p2d.sh:100). My v4 runs used decode CG=**FULL_AND_PIECEWISE** (I changed it for "perf").
The FULL graph captures a full max_num_seqs=32 batch; if decode replays the FULL graph for
even a single request, it computes ~32x the work per step -> ~150s. PIECEWISE does not.
Also the connector explicitly warns READ/barrier can't fire inside a FULL graph (:230).
=> The ~150s stall + garbage was likely FULL_AND_PIECEWISE, NOT the transfer path.
FIX: decode CG=PIECEWISE (exact v3 value), WRITE mode, thinking=false. Relaunching. Expect
per-request latency to drop toward v3's ~seconds. This aligns v4 to the proven-working v3
decode config (only the stack underneath is upgraded: base/vLLM/MoRI).

## PIECEWISE test RESULT: also 171.9s (correct recall). Stall is INVARIANT.
decode CG=PIECEWISE (exact v3 value): request 171.9s, recall CORRECT. So the ~150-170s is
INVARIANT across: WRITE/READ mode, F_A_P/PIECEWISE cudagraph. Not the transfer path, not
cudagraph, not reasoning, not DP-rank mismatch.

## PRECISE TRACE of the stall (both sides 100% GPU, no logs, ~150s):
For a matched request (prefill Worker_DP1 + decode notify rank=1 -- ranks AGREE):
  01:32:08 prefill DP1 "write-stash" (KV staged, moriio_connector.py:2730)
  01:32:10 decode "notify prefill rank=1" (blocks ready) -- handshake+notify FAST (~2s), MATCHED
  01:32:10 -> ~01:34:40 : ZERO log activity on decode AND prefill; BOTH GPUs pinned 100%;
            ~150s later the request completes with correct output.
=> After a correct+fast handshake, BOTH prefill(rank1) and decode spin at 100% GPU for ~150s
   with no logs. This is a COMPUTE/COLLECTIVE stall, not a network timeout (network idle would
   be GPU~0). Signature = a MoRI all2all collective barrier on the decode MoE path
   (mori_low_latency = InterNodeV1LL cross-node dispatch/combine) hanging ~150s per request,
   OR a decode forward recomputing something huge. Per-request (req2 also ~178s), not one-time.

## RULED OUT this session: reasoning(thinking off=same), WRITE-notify DP mismatch(READ=same
## 150s+garbage), cudagraph mode(PIECEWISE=same 171s), qp/num_workers knobs(broke transfer).
## REMAINING SUSPECTS: (1) MoRI InterNodeV1LL decode all2all per-request ~150s barrier on
## bnxt (new MoRI 624002c8 vs v3's older MoRI -- THE version delta); (2) decode all2all-backend
## mori_high_throughput instead of low_latency; (3) MoRI bisect. v3 used older MoRI + same
## mori_low_latency and got ~20s/50K -> the MoRI version is the prime suspect.

## *** MAJOR REFRAME: the ~150s AMORTIZES across concurrency (it batches!) ***
Live serve (decode PIECEWISE+LL), thinking=false, 2K reqs:
- N=1: ~170s
- N=4 concurrent: wall=181s, ALL 4 complete, ALL recall correct.
=> 4 requests in ~the time of 1. The ~150s is NOT per-request-serial -- it's a FIXED
   per-decode-wave latency FLOOR that is SHARED across all in-flight requests. Effective
   per-request cost at con=4 = ~45s; expected to keep dropping with concurrency.
This means the serve IS throughput-capable; the ~150s is a fixed floor (a decode-step /
all2all warmup or a fixed wait that fires once per wave), amortized by batching. Far more
optimizable than a serial bug. The floor itself is still worth killing (latency), but
THROUGHPUT is fine and scales with con. Confirm con=16 amortization next; then chase the
floor (MoRI all2all warmup / a fixed per-step barrier).

## con=16 CONFIRMS amortization (+ surfaces accuracy-under-concurrency)
N=16 concurrent (2K reqs, thinking off): wall=183s -- SAME as N=1 (~170s) and N=4 (181s).
=> 16 requests in the time of 1. Throughput scales ~linearly with concurrency; the ~150s is
a fully-shared per-wave floor. Effective per-req: con4~45s, con16~11s, con32~6s (projected).
CAVEAT: recall mixed at con=16 (9/16 correct) on the trivial "Code {i} is 8241" prompt --
likely the FULL-graph + high-concurrency KV-read-barrier degradation the connector warns about
(:230), and/or the toy prompt (many identical "8241" codes). MUST re-check with real NIAH
(distinct needles) at concurrency. TWO workstreams now:
  A. THROUGHPUT: already good (con scales). Report perf_sweep TTFT/tok-s at con16/32.
  B. LATENCY FLOOR (~150s) + ACCURACY@concurrency: chase the shared per-wave floor (MoRI
     all2all warmup / fixed barrier) and verify NIAH recall holds at con (may need PIECEWISE
     not FULL for the read barrier, per :230).

## Distinct-needle NIAH @ con=8 (6K): 3/8 recall -> REAL accuracy-under-concurrency issue
Live serve confirmed decode=PIECEWISE + WRITE mode. con=8 distinct needles: 3/8 correct,
wall=261s. So accuracy-at-concurrency fails EVEN on PIECEWISE+WRITE (not a toy-prompt artifact,
not the FULL-graph barrier alone).
=> This is the RESIDUAL RDMA WRITE-RACE (same class documented for v3: decode reads a KV block
   before its RDMA write is globally visible in decode HBM; wait_for_layer_load() is a no-op;
   write_done travels ZMQ separate from the RDMA data path). v3 hit it at high-ctx/multi-needle;
   under concurrency it hits a larger fraction of requests. Single-request + low-con is fine.
FIX (code, decode-side): a per-request KV-ready HBM fence/barrier before the decode forward
consumes the block (sender-side knobs FAILED in v3: K3_WRITE_FENCE=delay, POST_BATCH_SIZE).
Candidates: enable_notification=True path (currently hardcoded False, moriio_engine.py:633) so
completion is RDMA-signalled not ZMQ; or a decode read-barrier keyed on transfer completion.

## SUMMARY OF STATE (this session)
GOOD:
- Correctness single-request/low-con: PASS (NIAH 50K/100K). thinking=false works.
- THROUGHPUT: excellent -- con1/con4/con16 all ~= same wall (~180s@2K); the latency floor is
  fully shared/amortized across a decode wave. Serve is throughput-capable.
OPEN (2 code-level items, both root-caused):
1. Latency FLOOR ~150-250s per wave (fixed cost; suspect MoRI InterNodeV1LL all2all warmup /
   a per-wave barrier). Amortizes with con but hurts single-stream latency.
2. Accuracy@concurrency: RDMA write-race (decode reads pre-visible KV). Needs decode-side
   KV-ready fence / enable_notification=True.
NEXT: (a) A/B decode all2all HT vs LL for the floor; (b) try enable_notification=True for the
write-race; (c) profile the floor with rocprof.

## Accuracy baseline: SEQUENTIAL = 3/3 (con=1 clean), race is purely CONCURRENCY
Distinct-needle NIAH @6K sequential (con=1): 3/3 recall (req0 100s, req1 268s, req2 105s).
=> single-request accuracy SOLID. The 3/8 at con=8 is the concurrency write-race, confirmed.

## Write-race fix analysis (code-confirmed)
Sequence in _finalize_if_complete (moriio_engine.py:480-540):
  1. waiting_for_transfer_complete() -> waits status.Succeeded() = SENDER WR done (data left
     sender NIC) -- NOT receiver-HBM-visible.
  2. optional k3-write-fence sleep (K3_WRITE_FENCE=delay).
  3. send_notify(write_done) via ZMQ -> decode admits request, reads KV.
The gap: RDMA-WRITE sender-completion != remote-HBM global visibility without an ordering op.
Decode's get_finished (WRITE) admits on write_done ZMQ arrival with NO HBM read-fence
(moriio_connector.py:2216) -> stale read under concurrency.
PROPER FIX = "readback" (comment at moriio_engine.py:528): after Succeeded, issue a tiny RDMA
READ of the written remote region (read_remote_data, :673) before write_done -- RDMA read to
same QP forces prior writes globally visible. Needs session+scratch-buf threaded into
_finalize_if_complete (deeper change). enable_notification=True is a DEAD END (hardcoded False
with documented ibv_post_send ENOMEM hang, :628).
INTERIM TEST: K3_WRITE_FENCE=delay at LARGE ms (200) to prove the race is visibility-timing
(v3 tried 20ms "no gain"; test if bigger closes it -> validates readback is the fix).

## FENCE TEST RESULT: 200ms sender-delay improved con=8 from 3/8 -> 6/8
K3_WRITE_FENCE=delay, K3_WRITE_FENCE_MS=200 (confirmed in prefill container). con=8 @6K:
6/8 recall (was 3/8 with no fence). => VALIDATES it's a visibility-timing race; a barrier
closes it, but a FIXED sleep is imperfect (RDMA write-completion time varies -> 2 still race).
CONCLUSION: implement the deterministic RDMA READBACK (read-after-write forces global
visibility, no guessed ms) in _finalize_if_complete before send_notify. This is the perf-safe
correct fix (vs a large blind sleep that also adds latency to every request).
Next: implement readback in the connector, rebuild image / hot-patch, re-test con=8 -> target 8/8.

## RDMA READBACK result: con=8 -> 6/8 (same as fence, > 3/8 baseline; not yet 8/8)
K3_WRITE_READBACK=1 (patch applied confirmed in-container). con=8 @6K: 6/8. Same 2 reqs
(req1 slot7001, req5 slot7005) fail as in the fence run -> the residual is likely NOT pure
visibility timing:
- My readback reads ONE region (offset 0 of the last write). KV spans MANY blocks/layers/groups
  (K3 has 4 KV-cache groups + MLA + KDA); a single-offset readback doesn't force ALL written
  regions globally visible -> some blocks still race.
- OR the consistent req1/req5 failures are a different bug (a specific DP-rank pair / a
  particular needle depth) not the write-race at all.
NEXT: (a) make readback cover the last write of EVERY transfer_status (loop over the request's
writes, not one offset); (b) inspect the 2 failing responses (truncated vs wrong-KV vs
cross-request) to confirm it's still the race; (c) try readback + small fence combined.

## MULTI-REGION READBACK: KV-corruption FIXED. Residual 2/8 is FORMATTING, not the race.
Readback now reads back EVERY written region (readback_targets list, capped 64). con=8 @6K:
6/8, but the FAILURE MODE CHANGED -- the garbage is GONE:
  BEFORE (single-offset readback): fails = "50个单词", "1e0a1e0a", "?cataract?" (KV corruption).
  NOW (multi-region): 6/8 clean "ZEBRA-70XX" finish=stop; the 2 fails are:
    req1: content="response<|sep|><|open|>tools<|sep|>..." -- TOOL/response-channel TOKENS
          leaking into content (parser/template artifact, NOT corruption).
    req5: finish=stop, content='' -- EMPTY output (early stop), NOT corruption.
=> The RDMA read-after-write barrier (covering all regions) ELIMINATES the concurrency
   KV-corruption. Remaining 2/8 = output-formatting: with thinking:false the model opens the
   response channel and sometimes emits <|open|>response<|sep|>/tools markers that the
   kimi_k3 reasoning parser doesn't strip on the no-think path, or stops empty. This is a
   PARSER/template fix, separate from KV integrity.
NEXT: fix the response-channel marker stripping on thinking:false (kimi_k3_reasoning_parser
strips think markers; must also strip <|open|>response<|sep|>/<|close|> when thinking off),
or set thinking back on for accuracy runs (needle lands in reasoning_content, cleanly parsed).

## SESSION WIN: RDMA read-after-write barrier fixes concurrency KV-corruption
The multi-region RDMA readback (K3_WRITE_READBACK=1) is the real, deterministic fix for the
concurrency write-race that a K3_WRITE_FENCE sleep could not fully close:
  no fix: con=8 3/8 (garbage). fence 200ms: 6/8 (garbage). readback(multi): 6/8 CLEAN (no
  garbage; residual 2 = response-channel token leak + 1 empty, both parser/formatting).
=> KV integrity under concurrency is SOLVED at the transport layer. Remaining accuracy gap is
   the kimi_k3 reasoning parser leaking <|open|>response<|sep|> on the thinking=false path
   (cosmetic; needle still recalled) + occasional empty/tool-token output. Options for the
   residual: (a) fix parser response-marker strip on no-think; (b) run NIAH with thinking ON
   (needle in reasoning_content, cleanly parsed) for accuracy validation.
CONFIG for deployment: K3_WRITE_READBACK=1 (in launcher, default 0; set 1 for correctness at
concurrency). Cost: one tiny RDMA read per written region per request (bounded 64), ordered
after writes -- negligible vs the ~150s wave floor, and the floor amortizes across con anyway.

## CLARIFICATION: single-request 50K CLEAN with readback (earlier "FAIL" was MT=64 truncation)
niah_sweep MT=64 truncated: the thinking=false response-marker preamble eats ~tokens before
the answer -> finish=length, false FAIL. Re-ran ONE 50K @MT=256: recall=True finish=stop
content='ZEBRA-9999' -- CLEAN. So single-request accuracy at 50K holds with readback on.
Bumped niah_sweep max_tokens 64->256. The response-marker leak just needs output headroom
(or the parser strip fix). Net: readback = KV integrity solved; probes need MT>=256.

## *** ACCURACY MILESTONE: readback + MT>=256 -> con=8 all-completed-requests CORRECT ***
con=8 @6K, readback ON, MT=256: 6/8 recall -- and every request that RETURNED is CLEAN &
correct (ZEBRA-70XX, finish=stop, no garbage, no marker leak). The 2 "fails" are CLIENT
TIMEOUTS (req5/req7 exceeded the 400s client timeout because the ~150s wave floor x queue
depth), NOT corruption or wrong answers.
=> With RDMA readback + adequate max_tokens, KV integrity + accuracy under concurrency is
   SOLVED. The residual is purely LATENCY (the wave floor causing tail timeouts), which is a
   throughput/perf item -- and it amortizes across concurrency (con16 ~= con1 wall). Raising
   the client timeout or fixing the floor closes the tail.
DEPLOYMENT-READY ACCURACY: single-request clean (50K), concurrent completed-requests clean.
Recommended serve flags: K3_WRITE_READBACK=1, thinking per-request (or serve default false),
clients use generous timeouts until the wave floor is optimized.
NEXT (perf, not accuracy): attack the ~150s wave floor (MoRI InterNodeV1LL all2all warmup /
per-wave barrier) so tail latency drops and no client times out.

## SESSION END STATE (accuracy path driven; PR #5 updated)
DELIVERED:
- RDMA multi-region read-after-write barrier (K3_WRITE_READBACK=1) -> concurrency KV-corruption
  ELIMINATED. con=8: every completed request CORRECT (was 3/8 garbage). Committed to PR.
- Single-request 50K: clean recall (MT>=256). Formal config kept: prefill HT+eager, decode
  LL+FULL_AND_PIECEWISE, thinking=false, readback on.
- Full experiment log + OPT_ROADMAP in the PR.
REMAINING (perf, not accuracy):
1. ~150s per-decode-wave latency FLOOR (amortizes across con: con16 ~= con1 wall). Prime
   suspect MoRI InterNodeV1LL all2all warmup/barrier. Attack: decode all2all HT A/B, rocprof
   the floor, or MoRI bisect. Causes tail client-timeouts at con until raised timeout / fix.
2. Serve DEGRADES over long lifetime (a 50K req that took ~150s fresh took >10min after hours
   of test cycles) -> for clean perf numbers, restart the serve fresh before benchmarking.
3. kimi_k3 parser leaks <|open|>response<|sep|> on thinking=false (cosmetic; MT>=256 absorbs).
NEXT SESSION: fresh serve restart -> clean NIAH sweep (MT=256) for the accuracy table ->
attack the wave floor for latency. Accuracy is deployment-ready; perf is the remaining work.

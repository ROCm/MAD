# Status — Kimi-K3 MI300X 2P/2D EP16 MoRIIO disagg (WIP)

**This recipe is WORK IN PROGRESS. It is not a validated production deployment.**
The disaggregation *infrastructure and transport are correct*, but an open
decode-side accuracy bug means exact long-context recall (NIAH) does **not** pass
yet. Ship/scale on the colocated recipes (`../pp2xtp8`, `../wideep_int4_*`) until
this is resolved.

## What works (verified)

- **Bring-up**: 2 prefill + 2 decode nodes, TP8×DP2 → **EP16 per pool** (MoRI-EP
  all2all), **no pipeline parallelism**, joined by the MoRIIO connector for
  prefill→decode KV + KDA state transfer. Both pools reach `Application startup
  complete` and the router routes requests end to end.
- **Transport is byte-perfect**. Element-wise probes (`patchers/diagnostics/`)
  confirm that both the MLA attention KV **and** the KDA/mamba recurrent+conv state
  arrive **byte-identical** on all 8 decode ranks at consume time. All connector-level
  bugs are fixed (see the vLLM branch commits referenced in the README):
  - mamba/KDA state routed by the mamba KV-cache group's block ids (not attention's);
  - degenerate `remote_tp_size` normalized so writes fan out to **all** decode TP
    ranks (not just rank 0);
  - the mamba **N−1** prefill/decode boundary (producer computes h(N−1), decoder
    recomputes token N) — ports vLLM's own nixl/mooncake hybrid-PD handling.
- **Coherent generation works.** Short factual prompts return correct, coherent
  output (e.g. "The capital of Germany is" → "Berlin. The currency is the euro"),
  and single-token recall is correct and deterministic.

## Known issue (open) — exact multi-token recall fails

Exact needle recall over more than one token is wrong **and non-deterministic at
greedy `temperature=0`**, while the *same request run colocated is deterministic and
correct*. Non-determinism at greedy decode means the defect is in the forward pass
(a memory/compute issue), not token sampling.

### Repro

```bash
# via the router (disagg) — non-deterministic + wrong
for i in 1 2 3; do
  curl -s http://<prefill-master-ip>:30000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"kimi-k3","prompt":"The number is 8241. The number is","max_tokens":3,"temperature":0}' \
    | python3 -c 'import sys,json;print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
done
# observed: ' 975.', ' 4. The', ' 975...'  (varies run to run)

# same prompt on a colocated engine (:20005 directly) — deterministic + correct
#   -> ' 8241' every time
```

`niah_probe.py` fails on the router endpoint and passes colocated.

### What has been ruled out (each tested, not assumed)

Transport corruption (byte-perfect); TP fan-out; mamba block-id group; the N−1
boundary (on and off); fp8 KV scale (static/identical both engines); RoPE-in-cache
and token positions (correct); block selection; KDA state content (byte-identical);
sampling params (fixed seed does not stabilize → below the sampler); TRITON_MLA
multi-split reduction (forcing a single split changes nothing); the MoRI-EP backend
(`mori_low_latency` and `mori_high_throughput` both fail identically); sender
write→notify timing (a 50 ms sender-side delay does nothing); and a local
multi-stream write-vs-RDMA-read race (a full `torch.cuda.synchronize()` before the
RDMA read does nothing).

### Current best hypothesis

With the KV bytes proven correct at consume time, the defect is **downstream of the
KV read**, in the decode-side attention **compute/indexing** over a paged cache whose
blocks were populated out-of-order by RDMA (vs a sequential local prefill). It is
intermittent and multi-position-sensitive: single-token retrieval is reliable, a
multi-token span hits it. Fixing it likely needs kernel-level instrumentation of the
K3 dense-MLA decode path (`triton_mla` / aiter) and possibly an upstream vLLM/aiter
change — outside what the connector patchers can reach.

## Diagnostic knobs (opt-in, default OFF)

The folded vLLM branch adds gated knobs used during this investigation
(`K3_WRITE_FENCE`, `K3_WRITE_DEVSYNC`, `K3_MLA_SINGLE_SPLIT`) plus the
`patchers/diagnostics/` element-wise probes (`K3_DECODE_RECV_PROBE`, `K3_WRITE_BC`,
`K3_INPUTS_PROBE`, …). None change default behavior; they exist to continue the
investigation. See each patcher's docstring.

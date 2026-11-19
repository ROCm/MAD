# GLM-5.3-Flash disaggregated — runtime overlay patches

These 11 Python files are the verified connector / model / aiter fixes that make GLM-5.3-Flash
disaggregated serving (1P/1D over MoRIIO, MoRI **WRITE** mode) recall correctly at long context — exact
needle retrieval to **871K tokens** (TP4 and EP8). They are bind-mounted **on top of the base image**
`rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix` at container start by `../vllm_pd_launch.sh`
(`OVERLAYS=1`, the default). No image rebuild is required.

> The fully self-contained image (fixes baked in-source, no runtime overlays) is tracked as a follow-up.
> This overlay recipe is the reproducible, currently-proven path.

## Manifest — file → in-container destination → what it fixes

| overlay file | mounts onto (site-packages) | fixes |
|---|---|---|
| `moriio_connector_hma.py` | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py` | MoRIIO connector: per-group hybrid-KV block routing on the WRITE path + cross-chunk KV accumulation (CHUNKFIX) so chunked prefill keeps every chunk's KV; HMA (hybrid KV) support. |
| `moriio_engine.py` | `.../moriio/moriio_engine.py` | Engine: per-group remote-block resolution (`all_group_block_ids`), deferred-write flush, write-worker. |
| `moriio_common.py` | `.../moriio/moriio_common.py` | Common: MoRIIO mode (WRITE/READ) selection, port-offset TP-awareness, config plumbing. |
| `moriio_layout.py` | `.../moriio/moriio_layout.py` | KV transfer geometry: hybrid MLA / DSA-indexer / KDA-mamba cache layouts (packed single-head-slot caches). |
| `attn_utils.py` | `vllm/v1/worker/gpu/attn_utils.py` | Hybrid KDA(Mamba)+MLA shared KV-pool allocation (page-aligned blocks) — otherwise decode crashes `NotImplementedError` in `get_kv_cache_shape`. |
| `indexer.py` | `vllm/v1/attention/backends/mla/indexer.py` | DSA sparse-indexer support on the disagg path. |
| `glm5next_attention.py` | `vllm/models/glm5next/nvidia/attention.py` | GLM-5.3-Flash attention (fp32 head-gate keeps long-context sparse-MLA rankings accurate on gfx950). |
| `usercustomize.py` | (site-packages root) `usercustomize.py` | gfx950 guards: disable torch inductor pattern-matcher + dynamo (no-kernel-image / sfdp issues). Honors `DISABLE_INDUCTOR_PM` / `DISABLE_DYNAMO`. |
| `gemm_op_a8w8_tip.py` | `aiter/ops/gemm_op_a8w8.py` | aiter gfx950 a8w8 blockscale GEMM (CK gate + flydsl). |
| `fused_moe_tip.py` | `aiter/fused_moe.py` | aiter gfx950 fused-MoE (2-stage CK path). |
| `batched_gemm_a16wfp4.py` | `aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py` | aiter batched a16wfp4 GEMM. |

## Fidelity

The files here are byte-for-byte the set verified on the live gold pair (014↔021, WRITE mode). Their
sha256 hashes are recorded alongside the recipe; recall was re-confirmed (8K all depths + 60K + 400K,
`DELTA-9931`) before committing. Do not edit in place — regenerate + re-verify if a fix changes.

# Cluster-wide MoRI-IO + MoRI-EP Efficacy Matrix

Systematic multi-node validation of MoRI-EP (async_ll) and MoRI-IO on the standardized
MI300X + Broadcom Thor2 cluster. All nodes on an identical software stack.

## Uniform stack (verified on every tested node)
| Component | Version |
|---|---|
| Linux kernel | `5.15.0-177-generic` |
| bnxt_re driver | `235.2.86.0` |
| bnxt firmware (rdma3) | `238.1.138.0` |
| ROCm | `7.2.3` |
| libbnxt_re provider | `libbnxt_re-rdmav34.so` (539696 B, 235.2.86.0 build) |
| MoRI | built from source, `BUILD_UMBP=OFF`, gfx942 — see build-provenance note below |
| GPU | MI300X (gfx942, 1002:74a1) · NIC BCM57608 Thor2 (14e4:1760) |
| RDMA rail tested | `rdma3` = `benic4p1` = PCI `60:00.0`, GID idx 3, SL 3, TC 104 |

**MoRI build provenance:** the reference / packaged image is pinned at the CI-green commit `12d1bc32`
("AsyncLL top-k/warpSize fix", #505); the EP2 & EP16 scaling proofs ran on it. The 4-node efficacy
matrix was rebuilt from `main` tip on 2026-08-03 and landed on descendant commits (`0d05a4d2` on
node-a/node-c/node-b; `34f17d6` on node-f) — both direct descendants of `12d1bc32` adding only
non-bnxt changes, so EP/IO behaviour on this NIC is unchanged. `build_mori.sh` now pins `12d1bc32`.

## Test definitions
- **MoRI-EP**: `test_dispatch_combine_internode.py --cmd test --kernel-type async_ll --dtype bf16 --max-tokens 128 --num-qp 2`, EP2 (1 GPU/node × 2), 500 rounds. Pass = 0 errors both ranks.
- **MoRI-IO**: `tests/python/io/benchmark.py --backend rdma --mem-type cpu --op-type write --all` sweep 8 B→64 MiB, 2 QP, session+batch. Metric = peak GB/s @ 64 MiB.

## Results (2026-08-02)

| Pair | Nodes | MoRI-EP async_ll | MoRI-IO peak (64 MiB) |
|---|---|:--:|:--:|
| A | node-a ↔ node-b | ✅ 0 errors (both ranks) | **48.43 GB/s** |
| B | node-a ↔ node-c | ✅ 0 errors (both ranks) | **48.40 GB/s** |
| C | node-c ↔ node-b | ✅ 0 errors (both ranks) | **48.42 GB/s** |
| D | node-a ↔ node-f | ✅ 0 errors (both ranks) | **48.42 GB/s** |

**All 4 standardized nodes (node-a, node-c, node-b, node-f) validated** — every pair passes MoRI-EP
async_ll (0 errors) and MoRI-IO ~48.4 GB/s. Every node has now participated in at least one pair.
(node-d = control-plane, still on 237/236, not part of the serving fleet; node-e = PSU hardware fault, down.)

### MoRI-IO sweep shape (representative, all pairs near-identical)
| MsgSize | Max BW (GB/s) |
|---|---|
| 8 MiB | ~44.6 |
| 32 MiB | ~47.8 |
| 64 MiB | ~48.4 |

## Findings
- **MoRI-EP (async_ll) passes on every node pair** — 0 errors, no hang, no crash. The async_ll
  (WRITE+poll) kernel is uniformly effective across the standardized fleet, not just the original
  node-a/node-b pair. This confirms the Thor2 fix generalizes to all nodes on the 235 driver.
- **MoRI-IO bandwidth is tight across pairs: 48.40–48.43 GB/s** (~387 Gbps, near 400G line rate),
  variance < 0.1%. No node/pair is an outlier — the fabric + NIC + driver stack is consistent.
- **Rack topology does not matter** at this scale: same-rack (B: node-a↔node-c) and cross-rack
  (A, C) pairs perform identically for both EP and IO.
- The default `v1`/InterNodeV1 EP kernel is NOT tested here because it is known to hang on Thor2
  (no PCIe atomic-completer) — async_ll is the required kernel; see the main report.

## Reproduce
Scripts (staged at `/mnt/nfs/cookbook/thor2/` and copied into each `mori_host:/tmp/`):
- `ep_pair_test.sh <rank 0|1> <master_mgmt_ip> <rdma_dev> <port>`
- `io_pair_test.sh <rank 0|1> <master_mgmt_ip> <own_mgmt_ip> <rdma_dev> <port>`
Launch rank1 first, then rank0, on `mori_host` containers built per `scripts/build_mori.sh`.

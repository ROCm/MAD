# sglang_disagg config files

Bucket-aligned configuration extracted from `mori_ep_env.sh` and the entrypoint
`sglang_disagg_mori_io_ep.sh`. See [../CONFIG.md](../CONFIG.md) for the full taxonomy.

This is a behavior-preserving Stage 1 extraction: values and their `${VAR:-default}`
fallbacks are copied verbatim; runtime behavior is unchanged.

## Files

- `nic-selection.env.sh` - Shared NIC selection prelude (not a CONFIG.md bucket). Sets `_DEFAULT_IB` / `IB_DEVICES` from `USE_CX7_NICS`.
- `framework.env.sh` - Framework bucket (NCCL, GLOO/NCCL sockets, timeouts, PyTorch/aiter runtime).
- `connectors.env.sh` - Connectors bucket (MoRI RDMA config, MoRI socket, DP_MODE=1 MoRI-EP tuning).
- `runtime.defaults.sh` - Entrypoint inline defaults (multi-bucket aggregate: Cluster/Launcher/Model).

## Sourcing order

1. `runtime.defaults.sh` - sourced by the entrypoint after `SCRIPT_DIR` is set (line 7), before first use (line 37).
2. `mori_ep_env.sh` aggregator - sourced by the entrypoint after YAML loading (line 185); it sources:
   - `nic-selection.env.sh` first,
   - then `framework.env.sh` and `connectors.env.sh`.

## Dependencies

- `framework.env.sh` and `connectors.env.sh` both depend on `_DEFAULT_IB` from `nic-selection.env.sh`.
- `connectors.env.sh` reads `GLOO_SOCKET_IFNAME` (set in `framework.env.sh`) for the MoRI socket default.
- `connectors.env.sh` MoRI-EP block requires `DP_MODE` to be set before sourcing (entrypoint sets it at line 44).

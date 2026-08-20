# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
# SGLang Disaggregated P/D — thin overlay on a prebuilt rocm/sgl-dev image.
#
# The rocm/sgl-dev images already ship the whole stack that
# sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile builds by hand:
# sglang + aiter + sgl-kernel compiled for the image's target arch, a librccl
# carrying both gfx942 and gfx950 code objects, and mori, nixl and mooncake
# preinstalled. Pick the tag whose suffix matches the GPUs
# (`-mi35x` = gfx950/MI35x, `-mi30x` = gfx942/MI30x) instead of rebuilding
# kernels on top of a wrong-arch base.
#
# What is left to do is align the container's RDMA userspace with the host's
# kernel driver — see BNXT_RE_PROVIDER below.
#
# Build example (madengine passes these through docker_build_arg):
#   docker build --network=host \
#     --build-arg BASE_DOCKER=rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807 \
#     --build-arg BNXT_RE_PROVIDER=inbox \
#     -f docker/sglang_disagg_inference_sgl_dev.ubuntu.amd.Dockerfile docker
###############################################################################
ARG BASE_DOCKER=rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807
FROM $BASE_DOCKER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

###############################################################################
# Broadcom Thor2 (bnxt_re) verbs provider selection.
#
# The base image carries two bnxt_re providers:
#   /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so   Broadcom out-of-tree
#                                                           (libbnxt_re-235.2.86.0),
#                                                           put on the loader path by
#                                                           /etc/ld.so.conf.d/libbnxt_re.conf
#   /usr/lib/x86_64-linux-gnu/libibverbs/
#       libbnxt_re-rdmav34.so-inbox                         upstream rdma-core, disabled
#                                                           by the filename suffix
#
# The vendor provider announces support for bnxt_re kernel uABI 7-8. A host
# running a bnxt_re driver that exposes the upstream uABI (1) therefore has every
# device rejected:
#
#   libibverbs: Warning: Driver bnxt_re does not support the kernel ABI of 1
#               (supports 7 to 8) for device /sys/class/infiniband/bnxt_re0
#   No IB devices found
#
# with the consequence that RCCL finds no IB net plugin and ncclCommInitRank
# fails with "invalid usage". `inbox` selects the upstream provider that matches
# such a host. Use `vendor` on hosts running Broadcom's out-of-tree stack.
#
#   inbox  — activate the upstream provider, drop the vendor one (default)
#   vendor — leave the image as shipped
###############################################################################
ARG BNXT_RE_PROVIDER=inbox

RUN set -e; \
    verbs_dir=/usr/lib/x86_64-linux-gnu/libibverbs; \
    case "${BNXT_RE_PROVIDER}" in \
      inbox) \
        test -f "${verbs_dir}/libbnxt_re-rdmav34.so-inbox" \
          || { echo "BNXT_RE_INBOX_PROVIDER_NOT_FOUND in ${verbs_dir}"; exit 1; }; \
        cp "${verbs_dir}/libbnxt_re-rdmav34.so-inbox" "${verbs_dir}/libbnxt_re-rdmav34.so"; \
        rm -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so \
              /usr/local/lib/libbnxt_re-rdmav34.so; \
        ldconfig; \
        echo "BNXT_RE_PROVIDER=inbox applied"; \
        ;; \
      vendor) \
        echo "BNXT_RE_PROVIDER=vendor — image left as shipped"; \
        ;; \
      *) \
        echo "BNXT_RE_PROVIDER must be 'inbox' or 'vendor', got '${BNXT_RE_PROVIDER}'"; \
        exit 1; \
        ;; \
    esac

# Sanity: the python stack the launcher needs must still import. `aiter` and the
# GPU-dependent parts are not checked here because the build host has no GPU.
RUN python3 - <<'PY'
import importlib, sys
missing = []
for pkg in ("torch", "sglang", "sgl_kernel", "yaml", "pandas"):
    try:
        importlib.import_module(pkg)
        print(f"[sgl-dev-overlay] import {pkg}: OK")
    except Exception as e:
        missing.append(f"{pkg}: {e}")
if missing:
    for m in missing:
        print(f"[sgl-dev-overlay] SANITY FAILED: {m}", file=sys.stderr)
    sys.exit(1)
PY

LABEL bnxt_re_provider="${BNXT_RE_PROVIDER}"

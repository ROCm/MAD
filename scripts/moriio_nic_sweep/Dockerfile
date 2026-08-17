# MoRI-IO two-node RDMA block-size sweep -- test image
#
# WHY THIS DOCKERFILE EXISTS, AND WHAT IT IS *NOT*
# ------------------------------------------------
# It does NOT build a MoRI stack. It layers the missing *harness* onto the
# workload image that PR#205 already builds, so that the number this benchmark
# produces characterises the stack we actually ship for GLM-5.2 disagg -- not a
# freshly-cloned mori that nothing else in this repo runs.
#
# The recorded 378.1 GB/s result was produced against the image's own pinned
# libmori (MORI_REF=42e895472b08). Verified in-container:
#     mori pkg : /usr/local/lib/python3.12/dist-packages/mori/__init__.py
# That is the number to trust precisely because the engine is the shipped one.
#
# The one thing the workload image lacks is tests/python/io/benchmark.py: the
# mori wheel installs the `mori` package but not the repo's test tree. The
# sweep scripts therefore need a benchmark.py from somewhere. Two ways:
#
#   (a) bind-mount a mori checkout at /opt/mori  (what moriio_sweep.slurm does
#       by default; requires a FULL, non-sparse checkout on the host), or
#   (b) build this image, which bakes a benchmark.py pinned to the SAME commit
#       as the image's libmori, so harness and engine cannot drift.
#
# (b) is strictly better and is why this file exists. The recorded run used
# (a) with an older 6ad812c checkout, whose benchmark.py predates
# --mem-type/--max-chunks/--chunk-bytes. That cost nothing for the recorded
# sweep -- those flags only re-state defaults the planner already uses -- but it
# did mean CHUNK_SWEEP=1 could not run. Build this image and it can.
#
# BUILD
#   # BASE must be the image PR#205 built (or a rebuild of
#   # docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile).
#   docker build --network=host \
#     --build-arg BASE_IMAGE=rocmshared/vllm-disagg:glm52-gfx950-ionic-v1 \
#     -f Dockerfile -t moriio-nic-sweep:latest .
#
# RUN  -- then point the sweep at it and skip the host mount entirely:
#   export DOCKER_IMAGE_NAME=moriio-nic-sweep:latest
#   export MORI_SRC=SKIP          # harness is baked in at /opt/mori
#   sbatch moriio_sweep.slurm
#
# The container still needs the host RDMA stack bind-mounted at run time (24
# mounts, 4 groups) -- see README "Exposing the host RDMA stack". That is a
# runtime concern; nothing here can bake it in.

ARG BASE_IMAGE=rocmshared/vllm-disagg:glm52-gfx950-ionic-v1
FROM ${BASE_IMAGE}

# Must match the MORI_REF baked into BASE_IMAGE. If you bump the base, bump
# this too, or the harness and the engine describe different code. The check
# below turns that mistake into a build failure instead of a silent mismatch.
ARG MORI_REF=42e895472b08

# Where the sweep scripts expect to find the mori repo (MORI_REPO / MORI_MNT).
ENV MORI_REPO=/opt/mori
ENV MORI_HARNESS_BAKED=1

# numactl: absent from the workload image, and its absence is visible in the
# recorded logs as "[warn] numactl absent -- NIC/CPU affinity left to the
# scheduler." MatchCpuNics() then orders rails unpinned. It cost nothing at the
# large-block end (0.5% rail spread) but small-block numbers are the ones that
# measure software overhead, so pin them properly here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        numactl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fetch ONLY the harness, at the engine's commit. --filter=blob:none keeps this
# cheap; we then materialise the full tree (NOT a sparse checkout -- a sparse
# checkout tracks tests/ without materialising it, so benchmark.py can be "in
# git" and still absent from disk, which is a preflight failure the sweep
# scripts explicitly check for).
RUN git clone --filter=blob:none https://github.com/ROCm/mori.git ${MORI_REPO} \
    && git -C ${MORI_REPO} checkout ${MORI_REF} \
    && test -f ${MORI_REPO}/tests/python/io/benchmark.py \
       || (echo "FATAL: benchmark.py absent at MORI_REF=${MORI_REF}" && exit 1)

# Assert harness == engine. The image's libmori is authoritative; if the pin
# above disagrees with what is installed, fail here rather than publish a
# number attributed to the wrong commit.
RUN set -eux; \
    installed="$(grep -oE 'MORI_REF=[^@]+' /app/versions.txt | head -1 | cut -d= -f2 || echo UNKNOWN)"; \
    echo "harness MORI_REF=${MORI_REF}  image MORI_REF=${installed}"; \
    if [ "${installed}" != "UNKNOWN" ] && [ "${installed}" != "${MORI_REF}" ]; then \
      echo "FATAL: harness pin (${MORI_REF}) != image libmori (${installed})."; \
      echo "       Rebuild with --build-arg MORI_REF=${installed}."; exit 1; \
    fi; \
    echo "MORIIO_SWEEP_HARNESS_REF=${MORI_REF}" >> /app/versions.txt

# Do NOT put ${MORI_REPO} on PYTHONPATH. It would be harmless in practice --
# the repo's python package lives at python/mori, not the tree root, so it
# cannot shadow the installed wheel -- but relying on that is fragile. The
# sweep runs the harness as `python3 -m tests.python.io.benchmark` from
# MORI_REPO, which needs cwd, not PYTHONPATH. `import mori` must keep
# resolving to the installed wheel; that is the whole point.

COPY run_moriio_sweep.sh /opt/moriio_nic_sweep/run_moriio_sweep.sh
COPY aggregate_sweep.py  /opt/moriio_nic_sweep/aggregate_sweep.py
RUN chmod +x /opt/moriio_nic_sweep/run_moriio_sweep.sh /opt/moriio_nic_sweep/aggregate_sweep.py

WORKDIR ${MORI_REPO}

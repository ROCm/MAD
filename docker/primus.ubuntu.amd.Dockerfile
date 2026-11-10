# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# Primus launcher for MAD: one image for all Primus pretrain configs (torchtitan, megatron, MaxText, …).
#
# Build context must be the repo root (so COPY scripts/Primus works). Manual:
#   docker build -f docker/primus.ubuntu.amd.Dockerfile .
# `madengine build` uses context `.` for models whose dockerfile path contains "primus"
# (see DockerBuilder.get_context_path in madengine).
#
# PRIMUS_ROOT is /opt/primus (Primus repo root: primus-cli, examples/<backend>/…), with
# /workspace/Primus a symlink to it. WORKSPACE_DIR is the generic working directory /workspace;
# madengine places manifests and run_directory there. Do not set PRIMUS_ROOT=/workspace — that
# would collide with those files.
#
# Kubernetes: the Job mounts an emptyDir on /workspace, so image layers under /workspace are not
# visible in the pod; Primus lives outside it for that reason. Madengine bundles
# `scripts/Primus/examples/...` into the ConfigMap as `Primus/...` and lays it over PRIMUS_ROOT.
#
# Local Docker / SLURM: bind-mount or shared filesystem provides scripts/Primus; run.sh prefers
# that checkout when present, else uses PRIMUS_ROOT from this image.
ARG BASE_DOCKER=rocm/primus:v26.9

FROM $BASE_DOCKER

USER root

ENV WORKSPACE_DIR=/workspace
ENV PRIMUS_ROOT=/opt/primus

RUN mkdir -p "$WORKSPACE_DIR"
WORKDIR $WORKSPACE_DIR

LABEL mad.launcher=primus

# rocm/primus base often has /workspace/Primus as a full git clone (.git is a directory).
# A submodule checkout uses .git as a file (gitlink). COPY cannot replace that tree — remove first.
RUN rm -rf /workspace/Primus

# Bake Primus from the build context (submodule). No git clone — matches CI and local builds.
COPY scripts/Primus/ /opt/primus/
RUN ln -s /opt/primus /workspace/Primus

# v26.6: runner/primus-cli. v26.7+ (Primus #999): primus-cli at the repo root.
RUN test -f /workspace/Primus/runner/primus-cli || test -f /workspace/Primus/primus-cli

RUN pip3 list 2>/dev/null || true

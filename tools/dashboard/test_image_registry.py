"""Unit tests for reading ROCm versions from image configs (fixtures trimmed from real images; no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_registry  # noqa: E402


@pytest.mark.parametrize("image, expected", [
    ("rocm/pytorch:latest", ("registry-1.docker.io", "rocm/pytorch", "latest")),
    ("rocm/pytorch", ("registry-1.docker.io", "rocm/pytorch", "latest")),
    ("ubuntu:22.04", ("registry-1.docker.io", "library/ubuntu", "22.04")),
    ("docker.io/rocm/primus:v26.8", ("registry-1.docker.io", "rocm/primus", "v26.8")),
    ("unifiedtrainingdockers.azurecr.io/utd/ci:tag_1", ("unifiedtrainingdockers.azurecr.io", "utd/ci", "tag_1")),
    ("localhost:5000/img", ("localhost:5000", "img", "latest")),
    ("rocm/pytorch:latest@sha256:abc", ("registry-1.docker.io", "rocm/pytorch", "sha256:abc")),
])
def test_parse_reference(image, expected):
    assert image_registry.parse_reference(image) == expected


def config(labels=None, env=None, history=()):
    return {"config": {"Labels": labels, "Env": env or []}, "history": [{"created_by": h} for h in history]}


def test_label_wins():
    # rocm/atom-dev:vllm-latest
    c = config(labels={"com.amd.rocm.version": "7.2.4", "org.opencontainers.image.version": "0.1.10"},
               history=["ARG ROCM_VERSION=7.2.4", "ARG PYTORCH_VERSION=2.10.0+rocm7.2.4.lw.git3d3aa833"])
    assert image_registry.rocm_from_config(c) == ("7.2.4", "label com.amd.rocm.version")
    # rocm/vllm-dev:ci_base-*
    c = config(labels={"vllm.rocm_base.dependency.rocm": "7.2.3"})
    assert image_registry.rocm_from_config(c) == ("7.2.3", "label vllm.rocm_base.dependency.rocm")


def test_build_arg_skips_arg_defaults():
    # vllm/vllm-openai-rocm and vllm-omni: the ARG default is a placeholder; the RUN step has the real value.
    c = config(history=[
        "ARG ROCM_VERSION=5.3",
        "/bin/sh -c #(nop)  ARG ROCM_VERSION=5.3",
        "RUN |3 ROCM_VERSION=7.2.2 AMDGPU_VERSION=30.30.1 /bin/sh -c apt-get update # buildkit",
    ])
    assert image_registry.rocm_from_config(c) == ("7.2.2", "ROCM_VERSION build arg")
    # rocm/pytorch-training:v25.5 (classic builder history)
    c = config(history=["|9 CONDA_CMAKE=yes ROCM_VERSION=6.3.4 UBUNTU_VERSION=22.04 /bin/sh -c bash ./install_rocm.sh"])
    assert image_registry.rocm_from_config(c) == ("6.3.4", "ROCM_VERSION build arg")


def test_therock_version_beats_stale_hints():
    # rocm/primus:v26.8 carries a stale BUILD_ROCM_VERSION=7.15 in its environment.
    c = config(env=["BUILD_ROCM_VERSION=7.15", "ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel"],
               history=["RUN |1 THE_ROCK_VERSION=10.1.0 /bin/sh -c uv pip install rocm==${THE_ROCK_VERSION}",
                        "RUN |1 PYTORCH_VERSION=2.14.0+rocm10.1.0 /bin/sh -c uv pip install torch"])
    assert image_registry.rocm_from_config(c) == ("10.1.0", "THE_ROCK_VERSION build arg")
    # lmsysorg/sglang:v0.5.20-rocm10-mi30x still passes ROCm 7.2 wheel args to later steps.
    c = config(history=["RUN |8 GPU_ARCH=gfx942-rocm1000 ROCM_SDK_VERSION=10.0.0 /bin/sh -c pip install",
                        "RUN |45 ROCM_SDK_VERSION=10.0.0 TORCH_ROCM_VERSION=2.11.0+rocm7.2 /bin/sh -c true",
                        "RUN |45 ROCM_SDK_VERSION=10.0.0 TORCHAUDIO_ROCM_VERSION=2.11.0+rocm7.2 /bin/sh -c true"])
    assert image_registry.rocm_from_config(c) == ("10.0.0", "ROCM_SDK_VERSION build arg")


def test_wheel_version_fallback():
    # rocm/pytorch:latest
    c = config(labels={"org.opencontainers.image.version": "10.1.0"},
               history=["ARG TORCH_SPECS=torch[device-all]==2.14.0+rocm10.1.0",
                        "RUN |2 TORCH_SPECS=torch[device-all]==2.14.0+rocm10.1.0 "
                        "APEX_SPECS=apex==1.14.0+rocm10.1.0 /bin/sh -c pip install"])
    assert image_registry.rocm_from_config(c) == ("10.1.0", "+rocm wheel version")


def test_nothing_stated():
    # rocm/pytorch-xdit: ROCm copied from TheRock build artifacts, no version anywhere.
    c = config(labels={"org.opencontainers.image.version": "24.04"}, env=["ROCM_PATH=/opt/rocm"],
               history=['RUN |2 PYTHON_VERSION=default /bin/sh -c A="/app/TheRock/artifacts"; O="/opt/rocm"'])
    assert image_registry.rocm_from_config(c) is None
    assert image_registry.rocm_from_config({}) is None

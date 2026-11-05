"""Unit tests for the dashboard collector's derivation helpers (no madengine or GPU needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402


@pytest.mark.parametrize("image, expected", [
    ("lmsysorg/sglang:v0.4.5-rocm630", "6.3.0"),
    ("lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x", "7.2.0"),
    ("rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3", "7.2.4"),
    ("rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216", "7.2"),
    ("lmsysorg/sglang:v0.5.20-rocm10-mi30x", "10"),
    ("vllm/vllm-openai-rocm:v0.28.0", None),
    ("rocm/primus:v26.7", None),
    ("rocm/pytorch:latest", None),
])
def test_rocm_from_image(image, expected):
    assert collect.rocm_from_image(image) == expected


def test_image_tag():
    assert collect.image_tag("rocm/pytorch") == "latest"
    assert collect.image_tag("rocm/pytorch:latest") == "latest"
    assert collect.image_tag("registry.io:5000/team/img:v1") == "v1"


def test_base_image_prefers_build_arg():
    text = "# CONTEXT {}\nARG BASE_DOCKER=rocm/pytorch:latest\nFROM $BASE_DOCKER\n"
    assert collect.base_image(text) == "rocm/pytorch:latest"
    assert collect.base_image("ARG BASE_IMAGE=rocm/vllm-dev:ci\nFROM ${BASE_IMAGE}\n") == "rocm/vllm-dev:ci"
    assert collect.base_image("FROM ubuntu:22.04 AS base\n") == "ubuntu:22.04"
    assert collect.base_image("ARG X=1\nFROM $X\n") == ""


def test_resolve_dockerfile_does_not_match_variants(tmp_path):
    docker = tmp_path / "docker"
    docker.mkdir()
    for name in ("vllm_disagg_inference.ubuntu.amd.Dockerfile", "vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile"):
        (docker / name).write_text("FROM x\n")
    assert collect.resolve_dockerfile("docker/vllm_disagg_inference", str(tmp_path)) == \
        "docker/vllm_disagg_inference.ubuntu.amd.Dockerfile"
    assert collect.resolve_dockerfile("docker/vllm_disagg_inference.glmv5.1", str(tmp_path)) == \
        "docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile"
    assert collect.resolve_dockerfile("docker/missing", str(tmp_path)) is None


def test_is_unpinned():
    assert collect.is_unpinned("rocm/pytorch")
    assert collect.is_unpinned("rocm/pytorch:latest")
    assert collect.is_unpinned("rocm/atom-dev:vllm-latest")
    assert not collect.is_unpinned("rocm/primus:v26.7")
    assert not collect.is_unpinned("img:latest-2026")


def test_describe_image_sources(tmp_path):
    docker = tmp_path / "docker"
    docker.mkdir()
    (docker / "a.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=rocm/pytorch:latest\n")
    (docker / "b.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=rocm/primus:v26.7\n")
    (docker / "c.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=img:rocm720\n")
    warnings = []
    root = str(tmp_path)
    assert collect.describe_image("docker/a", {}, warnings, root)["rocm"] == "latest"
    assert collect.describe_image("docker/c", {}, warnings, root)["rocm"] == "7.2.0"
    assert warnings == []
    assert collect.describe_image("docker/b", {"docker/b": ""}, warnings, root)["rocm"] == "unknown"
    assert len(warnings) == 1
    overridden = collect.describe_image("docker/b", {"docker/b": "7.1"}, warnings, root)
    assert (overridden["rocm"], overridden["rocm_source"]) == ("7.1", "status.yaml")


def test_describe_image_registry_lookup(tmp_path):
    docker = tmp_path / "docker"
    docker.mkdir()
    (docker / "a.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=rocm/atom-dev:vllm-latest\n")
    (docker / "b.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=rocm/primus:v26.8\n")
    (docker / "c.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=private.azurecr.io/x:y\n")
    (docker / "d.ubuntu.amd.Dockerfile").write_text("ARG BASE_DOCKER=img:rocm720\n")
    answers = {"rocm/atom-dev:vllm-latest": ("7.2.4", "registry label com.amd.rocm.version"),
               "rocm/primus:v26.8": ("10.1.0", "registry THE_ROCK_VERSION build arg"),
               "private.azurecr.io/x:y": (None, "registry: HTTP 401")}
    asked = []

    def lookup(image):
        asked.append(image)
        return answers[image]

    warnings = []
    root = str(tmp_path)
    latest = collect.describe_image("docker/a", {}, warnings, root, lookup)
    assert (latest["rocm"], latest["rocm_source"]) == ("latest", "unpinned :vllm-latest, ROCm 7.2.4 at build time")
    resolved = collect.describe_image("docker/b", {}, warnings, root, lookup)
    assert (resolved["rocm"], resolved["rocm_source"]) == ("10.1.0", "registry THE_ROCK_VERSION build arg")
    assert warnings == []
    assert collect.describe_image("docker/c", {}, warnings, root, lookup)["rocm"] == "unknown"
    assert "HTTP 401" in warnings[-1]
    # The tag already answers: the registry is not asked.
    asked.clear()
    assert collect.describe_image("docker/d", {}, warnings, root, lookup)["rocm"] == "7.2.0" and asked == []
    # An override wins, but one the registry contradicts is flagged; a less precise one is not.
    assert collect.describe_image("docker/b", {"docker/b": "10.1"}, warnings, root, lookup)["rocm"] == "10.1"
    assert len(warnings) == 1
    assert collect.describe_image("docker/b", {"docker/b": "7.2"}, warnings, root, lookup)["rocm"] == "7.2"
    assert len(warnings) == 2 and "update or drop the override" in warnings[-1]


def test_skus_for():
    assert collect.skus_for({"tags": []}) == ["MI300X", "MI325X", "MI350X", "MI355X"]
    assert collect.skus_for({"tags": [], "skip_gpu_arch": "gfx942"}) == ["MI350X", "MI355X"]
    assert collect.skus_for({"tags": [], "skip_gpu_arch": "gfx942, gfx950"}) == []
    # Primus/JAX configs name their product directory in a tag; that wins.
    assert collect.skus_for({"tags": ["primus", "megatron", "MI355X", "x"], "skip_gpu_arch": ""}) == ["MI355X"]


def test_untracked_product():
    # A config for a product the dashboard does not track is not shown; without the filter
    # skus_for() would claim it runs on every tracked product.
    assert collect.untracked_product({"tags": ["primus", "megatron", "MI430X", "x"]})
    assert not collect.untracked_product({"tags": ["primus", "megatron", "MI455X", "x"]})
    assert not collect.untracked_product({"tags": ["primus", "megatron", "MI355X", "x"]})
    assert not collect.untracked_product({"tags": ["vllm", "vllm_default"]})


def test_mi455x_only_for_configs_that_name_it():
    assert collect.skus_for({"tags": ["primus", "megatron", "MI455X", "x"]}) == ["MI455X"]
    # Untagged models keep claiming the four older products, not gfx1250.
    assert "MI455X" not in collect.skus_for({"tags": []})
    assert collect.skus_for({"tags": [], "skip_gpu_arch": "gfx942"}) == ["MI350X", "MI355X"]


def test_worldmirror_label_prefix():
    frameworks = {"primus_train": {"name": "Primus", "workload": "training", "by_launcher": True,
                                   "label_prefix": {"worldmirror": "HunyuanWorld-Mirror"}}}
    images = {"../../docker/primus": {"image": "rocm/primus:v26.9", "rocm": "7.2", "path": "docker/primus.Dockerfile"}}

    def describe(launcher, config):
        model = {"name": f"primus_train/{launcher}_MI355X_{config}", "registry": "primus_train",
                 "dockerfile": "../../docker/primus", "tags": ["primus", launcher, "MI355X", config],
                 "args": f"--config_path examples/{launcher}/configs/MI355X/{config}.yaml"}
        return collect.describe_model(model, frameworks, {}, images)

    row = describe("worldmirror", "stage1-posttrain")
    assert row["model"] == "HunyuanWorld-Mirror-stage1-posttrain" and row["framework"] == "Primus Worldmirror"
    assert row["skus"] == ["MI355X"]
    # Other launchers keep their config name.
    assert describe("megatron", "zaya1_8B-BF16-pretrain")["model"] == "zaya1_8B-BF16-pretrain"


def test_model_label():
    assert collect.model_label({"name": "vllm/x", "args": "--model_repo amd/Llama-3.3-70B-FP8 --config c.yaml"}) == \
        ("Llama-3.3-70B-FP8", "amd/Llama-3.3-70B-FP8")
    assert collect.model_label({"name": "d/x", "args": "-N 2", "env_vars": {"MODEL_NAME": "DeepSeek-V3"}}) == ("DeepSeek-V3", None)
    assert collect.model_label({"name": "p/x", "args": "--config_path examples/m/configs/MI300X/llama2_7B-bf16.yaml"}) == \
        ("llama2_7B-bf16", None)
    assert collect.model_label({"name": "xdit/pyt_xdit_flux", "args": "--workload flux"}) == ("flux", None)
    # --model_repo without an org is a MAD-internal key, not a Hugging Face repo.
    assert collect.model_label({"name": "pytorch_train/pyt_train_flux", "args": "--model_repo pyt_train_flux"}) == \
        ("train_flux", None)


def test_precision_of():
    assert collect.precision_of({"name": "x", "training_precision": "BF16"}) == "bf16"
    assert collect.precision_of({"name": "vllm/pyt_vllm_llama-3.3-70b_fp4",
                                 "args": "--model_repo amd/Llama-3.3-70B-Instruct-MXFP4-Preview"}) == "mxfp4"
    assert collect.precision_of({"name": "vllm/pyt_vllm_glm-5.1_fp8", "args": ""}) == "fp8"
    assert collect.precision_of({"name": "vllm/pyt_vllm_qwen3-8b", "args": "--model_repo Qwen/Qwen3-8B"}) == ""


def test_nodes_of():
    assert collect.nodes_of({"args": "-N 2 -n 2"}) == 2
    assert collect.nodes_of({"args": "--model_repo a/b"}) == 1
    assert collect.nodes_of({"args": "", "slurm": {"nodes": 2, "gpus_per_node": 8}}) == 2


def test_deployment_of():
    frameworks = {"kv": {"name": "KV", "k8s": "manual"}}
    assert collect.deployment_of({"registry": "vllm"}, frameworks) == \
        {"scheduler": "local", "k8s": "", "status": "registered", "switch": ""}
    assert collect.deployment_of({"registry": "kv", "distributed": {"launcher": "slurm_multi"}}, frameworks) == \
        {"scheduler": "slurm", "k8s": "manual", "status": "registered", "switch": ""}
    assert collect.deployment_of({"registry": "jax-maxtext"}, frameworks, opt_in=True) == \
        {"scheduler": "slurm", "k8s": "", "status": "opt-in", "switch": "JAX_MAXTEXT_INCLUDE_MULTINODE"}


def test_multinode_only_and_is_opt_in(tmp_path):
    for registry, literal in (("jax-maxtext", '{"grok1", "llama3.1_405B"}'), ("jax-maxdiffusion", "set()")):
        d = tmp_path / "scripts" / registry
        d.mkdir(parents=True)
        (d / "get_models_json.py").write_text(f"import os\nMULTINODE_MODELS = {literal}\n")
    hidden = collect.multinode_only(str(tmp_path))
    assert hidden == {"jax-maxtext": {"grok1", "llama3.1_405B"}, "jax-maxdiffusion": set()}
    config = "--config_path examples/maxtext/configs/MI355X/{}-bf16-pretrain.yaml"
    assert collect.is_opt_in({"registry": "jax-maxtext", "args": config.format("llama3.1_405B")}, hidden)
    assert not collect.is_opt_in({"registry": "jax-maxtext", "args": config.format("llama3.1_8B")}, hidden)
    assert not collect.is_opt_in({"registry": "primus_train", "args": config.format("grok1")}, hidden)


def test_merge_one_page():
    def row(registry, model):
        return {"registry": registry, "model": model, "family": model, "page": f"fw/{model}", "variant": ""}
    rows = [row("ep", m) for m in ("large_ep_bench_1n", "large_ep_bench_1n_mori_only", "large_ep_bench_4n_mori_only")]
    rows += [row("kv", m) for m in ("kv_mori", "kv_mooncake")] + [row("vllm", "Qwen3-8B")]
    collect.merge_one_page(rows, {"ep": {"one_page": True}, "kv": {"one_page": True}, "vllm": {}})
    assert {(r["family"], r["page"], r["variant"]) for r in rows[:3]} == {
        ("large_ep_bench", "fw/large-ep-bench", "Default"), ("large_ep_bench", "fw/large-ep-bench", "mori only")}
    # 'kv_mo' is not a name: the prefix is cut back to a whole token.
    assert [(r["family"], r["variant"]) for r in rows[3:5]] == [("kv", "mori"), ("kv", "mooncake")]
    assert rows[5]["variant"] == "" and rows[5]["page"] == "fw/Qwen3-8B"


@pytest.mark.parametrize("label, family", [
    ("Llama-3.1-405B-Instruct-MXFP4-Preview", "Llama-3.1-405B-Instruct"),
    ("Llama-3.3-70B-Instruct-FP8-KV", "Llama-3.3-70B-Instruct"),
    ("amd-Llama-3.3-70B-Instruct-FP8-KV", "Llama-3.3-70B-Instruct"),
    ("gpt-oss120b-w-mxfp4-a-fp8", "gpt-oss120b"),
    ("llama3.1_8B-BF16-pretrain", "llama3.1_8B-pretrain"),
    ("Qwen3-8B", "Qwen3-8B"),
])
def test_family_of(label, family):
    assert collect.family_of(label) == family


def test_docker_context():
    assert collect.docker_context({"dockerfile": "docker/pyt_vllm"}) == "./docker"
    assert collect.docker_context({"dockerfile": "docker/primus_maxtext"}) == "."
    assert collect.docker_context({"dockerfile": "docker/x", "dockercontext": "ctx"}) == "ctx"


def test_framework_of_by_launcher():
    frameworks = {"primus_train": {"name": "Primus", "workload": "training", "by_launcher": True},
                  "vllm": {"name": "vLLM", "workload": "inference"}}
    assert collect.framework_of({"registry": "primus_train", "tags": ["primus", "megatron_bridge"]}, frameworks) == \
        ("Primus Megatron Bridge", "training")
    assert collect.framework_of({"registry": "vllm", "tags": ["pyt"]}, frameworks) == ("vLLM", "inference")
    assert collect.framework_of({"registry": "new_dir", "tags": []}, frameworks) == ("new_dir", "other")


def test_version_key():
    assert collect.version_key("v26.10") > collect.version_key("v26.9")
    with pytest.raises(ValueError):
        collect.version_key("26.10")


def test_load_status(tmp_path):
    status = tmp_path / "status.yaml"
    status.write_text("release: v26.10\nroadmap:\n  - {model: A}\n")
    loaded = collect.load_status(str(status))
    assert loaded["internal"] is False
    internal = tmp_path / "roadmap.internal.yaml"
    internal.write_text("- {model: B}\n")
    merged = collect.load_status(str(status), str(internal))
    assert [r["model"] for r in merged["roadmap"]] == ["A", "B"]
    assert merged["roadmap"][1]["internal"] is True and merged["internal"] is True
    status.write_text("release: v26.10\nroadmapp: []\n")
    with pytest.raises(ValueError, match="unknown keys"):
        collect.load_status(str(status))


def test_diff_snapshots():
    def snap(version, names, image):
        return {"version": version,
                "models": [{"name": n} for n in names],
                "images": [{"dockerfile": "docker/a", "image": image, "rocm": image[-3:], "frameworks": ["F"]}]}
    prev, cur = snap("v26.9", ["a", "b"], "img:7.1"), snap("v26.10", ["b", "c"], "img:7.2")
    diff = collect.diff_snapshots(prev, cur)
    assert diff["since"] == "v26.9"
    assert [m["name"] for m in diff["added"]] == ["c"]
    assert [m["name"] for m in diff["removed"]] == ["a"]
    assert diff["image_changes"][0]["to_rocm"] == "7.2"
    assert collect.diff_snapshots(None, cur) is None


def test_status_yaml_is_valid():
    """The committed status file loads and every framework entry has a name and workload."""
    status = collect.load_status()
    for entry in status["frameworks"].values():
        assert entry["name"] and entry["workload"] in ("inference", "training", "microbenchmark")

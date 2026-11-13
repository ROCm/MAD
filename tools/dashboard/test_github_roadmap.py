"""Tests for the GitHub-sourced roadmap, against recorded-style fixtures (no network)."""
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402
import github_roadmap as gr  # noqa: E402

FRAMEWORKS = {"vllm": {"name": "vLLM"}, "vllm_dissag": {"name": "vLLM disaggregated P/D"}}
FORM_BODY = """### Hugging Face model id

zai-org/GLM-5.3-Flash

### Framework

vLLM

### Target GPUs

- [X] MI300X
- [ ] MI325X
- [ ] MI350X
- [x] MI355X

### Precision(s)

_No response_
"""


def issue(number, body=FORM_BODY, labels=("new-model",), milestone=None, title="[Model]: GLM-5.3-Flash"):
    return {"number": number, "title": title, "url": f"https://github.com/ROCm/MAD/issues/{number}", "body": body,
            "updatedAt": "2026-10-20T00:00:00Z", "milestone": {"title": milestone} if milestone else None,
            "labels": {"nodes": [{"name": l} for l in labels]}}


def pr(number, closes=(), draft=False, labels=(), files=("scripts/vllm/models.json",), head_repo="someone/MAD"):
    return {"number": number, "title": f"PR {number}", "url": f"https://github.com/ROCm/MAD/pull/{number}",
            "isDraft": draft, "updatedAt": "2026-10-25T00:00:00Z", "author": {"login": "dev"},
            "baseRefOid": "base", "headRefOid": f"head{number}",
            "baseRepository": {"nameWithOwner": "ROCm/MAD"}, "headRepository": {"nameWithOwner": head_repo},
            "milestone": None, "labels": {"nodes": [{"name": l} for l in labels]},
            "closingIssuesReferences": {"nodes": [{"number": n} for n in closes]},
            "files": {"nodes": [{"path": p, "additions": 5} for p in files]}}


class FakeFiles:
    """PR patches and head files keyed like the real API."""

    def __init__(self, patches=None, heads=None):
        self.patches, self.heads = patches or {}, heads or {}

    def pr_files(self, repo, number):
        return [{"filename": path, "patch": patch} for path, patch in self.patches.get(number, {}).items()]

    def raw_file(self, repo, sha, path):
        return self.heads.get((sha, path))


def card(name, skip=""):
    return {"name": name, "args": f"--model_repo zai-org/{name}", "tags": [], "skip_gpu_arch": skip}


CATALOG = [{"name": "vllm/pyt_vllm_command-a-plus", "hf_repo": "CohereLabs/command-a-plus-05-2026-bf16",
            "model": "command-a-plus-05-2026-bf16", "precision": "bf16", "framework": "vLLM"}]
FP8_ROW = {"name": "vllm/pyt_vllm_command-a-plus_fp8", "hf_repo": "CohereLabs/command-a-plus-05-2026-fp8",
           "model": "command-a-plus-05-2026-fp8", "precision": "fp8", "framework": "vLLM"}


def test_parse_issue_form_and_hf_id():
    form = gr.parse_issue_form(FORM_BODY)
    assert form["Framework"] == "vLLM" and form["Precision(s)"] is None
    assert gr.form_gpus(form) == ["MI300X", "MI355X"]
    assert gr.hf_id(issue(1)) == "zai-org/GLM-5.3-Flash"
    # Free-text intake issues (like ROCm/MAD #229-#231) carry an 'HF repo:' line instead.
    assert gr.hf_id(issue(2, body="Model\n\nHF repo: CohereLabs/command-a-plus-05-2026-fp8\nSize: big")) == \
        "CohereLabs/command-a-plus-05-2026-fp8"
    assert gr.hf_id(issue(3, body="see https://huggingface.co/org/name for details")) == "org/name"


def test_added_models_uses_the_pr_diff_and_skips_known_and_moved_entries():
    files = FakeFiles(
        patches={7: {
            "scripts/vllm/models.json": '+    "name": "pyt_vllm_new",\n+    "name": "pyt_vllm_moved",\n+    "name": "pyt_vllm_old",',
            "models.json": '-    "name": "pyt_vllm_moved",',
        }},
        heads={("head7", "scripts/vllm/models.json"): json.dumps([card("pyt_vllm_new", "gfx942"), card("pyt_vllm_moved")])},
    )
    added = gr.added_models(pr(7), files, known=["vllm/pyt_vllm_old"])
    assert [e["name"] for e in added] == ["vllm/pyt_vllm_new"]
    assert collect.skus_for(added[0]) == ["MI350X", "MI355X"]


def test_status_mapping():
    assert gr.status_of(set(), []) == "planned"
    assert gr.status_of(set(), [pr(1, draft=True)]) == "in progress"
    assert gr.status_of(set(), [pr(1, draft=True), pr(2)]) == "in review"
    assert gr.status_of({"blocked"}, [pr(2)]) == "blocked"
    assert gr.status_of(set(), [pr(2, labels=("blocked",))]) == "blocked"


def test_build_roadmap_links_issue_and_pr():
    raw = {
        "issues": [
            issue(10, milestone="v26.11"),
            issue(11, body="HF repo: CohereLabs/command-a-plus-05-2026-fp8", title="Onboard command-a-plus"),
        ],
        "prs": [
            pr(20, closes=[10], draft=True),                                    # implements issue 10
            pr(21, labels=("new-model",), files=("scripts/vllm_dissag/models.json",)),  # PR-only item
            pr(22, files=("scripts/vllm/run.sh",)),                             # unrelated, unlabeled
        ],
    }
    files = FakeFiles(
        patches={20: {"scripts/vllm/models.json": '+  "name": "pyt_vllm_glm-5.3-flash",'},
                 21: {"scripts/vllm_dissag/models.json": '+  "name": "pyt_vllm_disagg_mimo",'}},
        heads={("head20", "scripts/vllm/models.json"): json.dumps([card("pyt_vllm_glm-5.3-flash")])},
    )
    oncoming, landed = gr.build_roadmap(raw, CATALOG, FRAMEWORKS, "new-model", files)
    by_number = {i["number"]: i for i in oncoming}
    assert set(by_number) == {10, 11, 21}  # PR 20 folds into issue 10; PR 22 is not on the roadmap
    planned = by_number[10]
    assert planned["status"] == "in progress" and planned["target"] == "v26.11"
    assert planned["hf_repo"] == "zai-org/GLM-5.3-Flash" and planned["skus"] == ["MI300X", "MI355X"]
    assert planned["tags"] == ["vllm/pyt_vllm_glm-5.3-flash"] and [p["number"] for p in planned["prs"]] == [20]
    pr_only = by_number[21]
    assert pr_only["source"] == "pr" and pr_only["status"] == "in review"
    assert pr_only["framework"] == "vLLM disaggregated P/D" and pr_only["tags"] == ["vllm_dissag/pyt_vllm_disagg_mimo"]
    # Issue 11 asks for fp8; the catalog only has bf16 of that model, so it is still oncoming.
    assert landed == []


def test_landed_needs_the_requested_precision_and_framework():
    fp8 = issue(11, body="HF repo: CohereLabs/command-a-plus-05-2026-fp8")
    raw = {"issues": [fp8], "prs": []}
    files = FakeFiles()
    landed = lambda rows, body=None: gr.build_roadmap(
        {"issues": [issue(11, body=body)] if body else raw["issues"], "prs": []}, rows, FRAMEWORKS, "new-model", files)[1]
    assert landed(CATALOG) == []
    assert [i["number"] for i in landed(CATALOG + [FP8_ROW])] == [11]
    # Same checkpoint, but only in another framework than the issue's.
    sglang = FORM_BODY.replace("zai-org/GLM-5.3-Flash", "CohereLabs/command-a-plus-05-2026-fp8").replace("vLLM", "SGLang")
    assert landed(CATALOG + [FP8_ROW], body=sglang) == []
    assert [i["number"] for i in landed(CATALOG + [FP8_ROW], body=sglang.replace("SGLang", "vLLM"))] == [11]


def form(hf, framework="vLLM", precisions="_No response_"):
    return (FORM_BODY.replace("zai-org/GLM-5.3-Flash", hf).replace("\nvLLM\n", f"\n{framework}\n")
            .replace("### Precision(s)\n\n_No response_", f"### Precision(s)\n\n{precisions}"))


def test_form_precisions():
    assert gr.form_precisions({"Precision(s)": "BF16, FP8"}) == ["bf16", "fp8"]
    assert gr.form_precisions({"Precision(s)": "MXFP4 and FP8"}) == ["mxfp4", "fp8"]
    assert gr.form_precisions({"Precision(s)": "nanoo_fp8"}) == ["nanoo_fp8"]
    assert gr.form_precisions({"Precision(s)": None}) == []


def test_landed_honors_the_form_precision():
    """The base checkpoint plus 'Precision(s): FP8' is not landed by a BF16 card."""
    rows = [{"name": "vllm/llama", "registry": "vllm", "hf_repo": "meta-llama/Llama-3.1-8B", "model": "Llama-3.1-8B",
             "precision": "bf16", "framework": "vLLM"}]
    landed = lambda rows, body: gr.build_roadmap({"issues": [issue(12, body=body)], "prs": []}, rows, FRAMEWORKS,
                                                 "new-model", FakeFiles())[1]
    assert landed(rows, form("meta-llama/Llama-3.1-8B", precisions="FP8")) == []
    assert len(landed(rows, form("meta-llama/Llama-3.1-8B", precisions="BF16"))) == 1
    # Every requested precision must be there.
    assert landed(rows, form("meta-llama/Llama-3.1-8B", precisions="BF16, FP8")) == []
    fp8 = dict(rows[0], precision="fp8")
    assert len(landed(rows + [fp8], form("meta-llama/Llama-3.1-8B", precisions="BF16, FP8"))) == 1


def test_landed_matches_the_registry_framework_name():
    """The form offers 'Primus'; the catalog shows the launcher ('Primus Megatron')."""
    frameworks = dict(FRAMEWORKS, primus_train={"name": "Primus", "by_launcher": True})
    rows = [{"name": "primus_train/megatron_MI355X_llama3.1_8B-pretrain", "registry": "primus_train", "hf_repo": None,
             "model": "llama3.1_8B-pretrain", "precision": "", "framework": "Primus Megatron"}]
    item = {"hf_repo": "meta-llama/llama3.1_8B-pretrain", "framework": "Primus", "precisions": []}
    assert gr.is_landed(item, rows, frameworks)
    assert not gr.is_landed(dict(item, framework="JAX MaxText"), rows, frameworks)


def test_issue_form_frameworks_match_status_yaml():
    """Every framework the onboarding form offers is a name the catalog uses (or 'Other')."""
    import yaml
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    with open(os.path.join(root, ".github", "ISSUE_TEMPLATE", "new_model.yml")) as f:
        template = yaml.safe_load(f)
    options = next(b["attributes"]["options"] for b in template["body"] if b.get("id") == "framework")
    frameworks = collect.load_status()["frameworks"].values()
    assert set(options) - {"Other"} <= {entry["name"] for entry in frameworks}
    models = {entry["name"] for entry in frameworks if entry.get("workload") != "microbenchmark"}
    assert models - set(options) == set(), "add the missing frameworks to the issue form"


def test_collect_roadmap_required_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("network down")
    monkeypatch.setattr(gr, "fetch", boom)
    import pytest
    with pytest.raises(RuntimeError, match="failed"):
        gr.collect_roadmap({"repo": "ROCm/MAD"}, [], FRAMEWORKS, [], token="t", required=True)
    monkeypatch.setattr(gr, "github_token", lambda: None)
    with pytest.raises(RuntimeError, match="no GitHub token"):
        gr.collect_roadmap({"repo": "ROCm/MAD"}, [], FRAMEWORKS, [], required=True)


def test_unlabeled_model_prs():
    raw = {"issues": [issue(10)],
           "prs": [pr(20, closes=[10]), pr(21, labels=("new-model",)), pr(22), pr(23, files=("README.md",))]}
    assert [p["number"] for p in gr.unlabeled_model_prs(raw, "new-model")] == [22]


def test_collect_roadmap_falls_back_on_api_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("network down")
    monkeypatch.setattr(gr, "fetch", boom)
    warnings = []
    assert gr.collect_roadmap({"repo": "ROCm/MAD"}, [], FRAMEWORKS, warnings, token="t") == ([], [])
    assert "failed" in warnings[0]


def test_collect_roadmap_without_token(monkeypatch):
    monkeypatch.setattr(gr, "github_token", lambda: None)
    warnings = []
    assert gr.collect_roadmap({"repo": "ROCm/MAD"}, [], FRAMEWORKS, warnings) == ([], [])
    assert "no GitHub token" in warnings[0]


def test_roadmap_items_merges_manual_items():
    status = {"roadmap": [{"model": "Hand-written"}], "roadmap_source": {}, "frameworks": {}}
    items, landed = collect.roadmap_items(status, [], [])
    assert items == [{"model": "Hand-written", "source": "status.yaml", "status": "planned"}] and landed == []

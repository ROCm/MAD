"""
Oncoming models from GitHub: issues labeled `new-model` are the plan, the PRs that close them
(or PRs labeled `new-model` on their own) are the progress signal.

    issue open, no linked PR    -> planned
    linked PR is a draft        -> in progress
    linked PR ready for review  -> in review
    `blocked` label anywhere    -> blocked
    merged / closed             -> gone (the model is then on develop, under Models)

An open item whose model is already in the catalog is reported as "landed but still open"
instead of oncoming. Everything here is read-only: nothing is labeled, edited or closed.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"

ISSUES_QUERY = """
query($owner: String!, $name: String!, $label: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 50, after: $cursor, states: OPEN, labels: [$label]) {
      pageInfo { hasNextPage endCursor }
      nodes { number title url body updatedAt milestone { title } labels(first: 20) { nodes { name } } }
    }
  }
}"""

PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $cursor, states: OPEN) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url isDraft updatedAt author { login }
        baseRefOid headRefOid baseRepository { nameWithOwner } headRepository { nameWithOwner }
        milestone { title } labels(first: 20) { nodes { name } }
        closingIssuesReferences(first: 10) { nodes { number } }
        files(first: 100) { nodes { path additions } }
      }
    }
  }
}"""

# Paths whose changes mean "this PR registers models".
MODEL_CARD_RE = re.compile(r"^(?:models\.json|scripts/(?:.+/)?(?:models\.json|get_models_json\.py))$")
FORM_GPUS = collect.SKUS


# --------------------------------------------------------------------------- GitHub access


def github_token():
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def graphql(query, variables, token):
    request = urllib.request.Request(
        GRAPHQL_URL, data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(payload["errors"][0].get("message", "GraphQL error"))
    return payload["data"]


def paginate(query, variables, token, connection):
    nodes, cursor = [], None
    while True:
        data = graphql(query, dict(variables, cursor=cursor), token)["repository"][connection]
        nodes += data["nodes"]
        if not data["pageInfo"]["hasNextPage"]:
            return nodes
        cursor = data["pageInfo"]["endCursor"]


def fetch(repo, label, token):
    """Open issues carrying `label` and every open PR (PRs are filtered later)."""
    owner, name = repo.split("/", 1)
    return {
        "issues": paginate(ISSUES_QUERY, {"owner": owner, "name": name, "label": label}, token, "issues"),
        "prs": paginate(PRS_QUERY, {"owner": owner, "name": name}, token, "pullRequests"),
    }


class Files:
    """PR file lists and file contents; tests substitute a fake with the same two methods."""

    def __init__(self, token):
        self.token = token

    def pr_files(self, repo, number):
        """[{filename, patch}] of a PR -- the diff against its merge base, as GitHub shows it."""
        files, page = [], 1
        while True:
            request = urllib.request.Request(
                f"{REST_URL}/repos/{repo}/pulls/{number}/files?per_page=100&page={page}",
                headers={"Authorization": f"bearer {self.token}", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                batch = json.load(response)
            files += batch
            if len(batch) < 100:
                return files
            page += 1

    def raw_file(self, repo, sha, path):
        """File content at a commit, or None if it does not exist there."""
        try:
            with urllib.request.urlopen(RAW_URL.format(repo=repo, sha=sha, path=path), timeout=60) as response:
                return response.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise


# --------------------------------------------------------------------------- parsing


def labels_of(node):
    return {l["name"] for l in node["labels"]["nodes"]}


def parse_issue_form(body):
    """{heading: value} from an issue-form body ('### Heading' sections); '_No response_' is None."""
    fields = {}
    for m in re.finditer(r"^###\s+(.+?)\s*\n(.*?)(?=^###\s|\Z)", body or "", re.M | re.S):
        value = m.group(2).strip()
        fields[m.group(1).strip()] = None if value in ("", "_No response_", "None") else value
    return fields


def hf_id(issue):
    """Hugging Face repo of an issue: the form field, else an 'HF repo:' line, else a huggingface.co link."""
    form = parse_issue_form(issue["body"])
    for key in ("Hugging Face model id", "Hugging Face model", "Model"):
        if form.get(key):
            return form[key].split()[0].strip("`")
    for pattern in (r"HF repo:\s*`?([\w.-]+/[\w.-]+)", r"huggingface\.co/([\w.-]+/[\w.-]+)"):
        m = re.search(pattern, issue["body"] or "")
        if m:
            return m.group(1)
    return None


def form_gpus(form):
    text = form.get("Target GPUs") or ""
    return [sku for sku in FORM_GPUS if re.search(rf"-\s*\[[xX]\]\s*{sku}\b", text)]


def model_card_files(pr):
    return [f["path"] for f in pr["files"]["nodes"] if MODEL_CARD_RE.match(f["path"]) and f["additions"]]


NAME_LINE_RE = re.compile(r'^([+-])\s*"name"\s*:\s*"([^"]+)"', re.M)


def added_models(pr, files, known=()):
    """Model entries a PR adds to models.json files, scoped like madengine ('<dir>/<name>').

    Names come from the PR's own diff (against its merge base), so entries that only moved
    between files -- the per-directory models.json migration -- cancel out. Anything already
    on develop (`known`, compared without the directory scope) is not new either. The full
    entries are then read from the PR head to get their GPUs and precision.
    """
    known = {name.rsplit("/", 1)[-1] for name in known}
    base_repo = pr["baseRepository"]["nameWithOwner"]
    diffs = [f for f in files.pr_files(base_repo, pr["number"]) if f["filename"].endswith("models.json")]
    plus = {(f["filename"], n) for f in diffs for sign, n in NAME_LINE_RE.findall(f.get("patch") or "") if sign == "+"}
    minus = {n for f in diffs for sign, n in NAME_LINE_RE.findall(f.get("patch") or "") if sign == "-"}
    new = [(path, n) for path, n in sorted(plus) if n not in minus and n not in known]

    added, heads = [], {}
    for path, name in new:
        if path not in heads:
            head = files.raw_file(pr["headRepository"]["nameWithOwner"], pr["headRefOid"], path) if pr["headRepository"] else None
            try:
                heads[path] = {e["name"]: e for e in json.loads(head or "[]") if isinstance(e, dict) and "name" in e}
            except ValueError:
                heads[path] = {}
        registry = os.path.dirname(path)[len("scripts/"):] if path.startswith("scripts/") else ""
        entry = dict(heads[path].get(name, {"name": name}))
        entry["name"] = f"{registry}/{name}" if registry else name
        added.append(entry)
    return added


def frameworks_of(paths, frameworks):
    """Framework display names for the scripts/<dir>/ trees a PR touches."""
    names = []
    for path in paths:
        m = re.match(r"scripts/([^/]+)/", path)
        if m and m.group(1) in frameworks and frameworks[m.group(1)]["name"] not in names:
            names.append(frameworks[m.group(1)]["name"])
    return names


def family_key(text):
    """'CohereLabs/command-a-plus-05-2026-fp8' -> 'commandaplus052026', comparable across precisions."""
    return re.sub(r"[^a-z0-9]", "", collect.family_of(text.rsplit("/", 1)[-1]).lower())


# --------------------------------------------------------------------------- roadmap


def is_landed(item, catalog_rows):
    """True when one catalog card already is what the item asks for: same model family, the
    requested precision (when the checkpoint names one) and a requested framework (when known)."""
    if not item["hf_repo"]:
        return False
    family = family_key(item["hf_repo"])
    precision = collect.precision_of({"name": item["hf_repo"], "args": f"--model_repo {item['hf_repo']}"})
    frameworks = {f.strip().lower() for f in item["framework"].split(",")} - {"", "–"}
    for row in catalog_rows:
        if family not in {family_key(row["model"]), family_key(row["hf_repo"] or row["model"])}:
            continue
        if precision and row["precision"] != precision:
            continue
        if frameworks and row["framework"].lower() not in frameworks:
            continue
        return True
    return False


def short_list(names, limit=4):
    return ", ".join(names[:limit]) + (f" +{len(names) - limit} more" if len(names) > limit else "")


def status_of(item_labels, prs):
    if "blocked" in item_labels or any("blocked" in labels_of(p) for p in prs):
        return "blocked"
    if any(not p["isDraft"] for p in prs):
        return "in review"
    if prs:
        return "in progress"
    return "planned"


def pr_link(pr):
    return {"number": pr["number"], "url": pr["url"], "draft": pr["isDraft"]}


def describe_added(entries):
    return [{"name": e["name"], "model": collect.model_label(e)[0], "precision": collect.precision_of(e),
             "skus": collect.skus_for(e)} for e in entries]


def build_roadmap(raw, catalog_rows, frameworks, label, files):
    """(oncoming items, landed-but-open items) from fetched issues and PRs."""
    prs_by_issue = {}
    for pr in raw["prs"]:
        for ref in pr["closingIssuesReferences"]["nodes"]:
            prs_by_issue.setdefault(ref["number"], []).append(pr)
    known_tags = {r["name"] for r in catalog_rows}

    items, linked = [], set()
    for issue in raw["issues"]:
        form = parse_issue_form(issue["body"])
        prs = prs_by_issue.get(issue["number"], [])
        linked.update(p["number"] for p in prs)
        new = describe_added([e for p in prs for e in added_models(p, files, known_tags)])
        repo_id = hf_id(issue)
        items.append({
            "source": "issue", "number": issue["number"], "url": issue["url"], "title": issue["title"],
            "model": repo_id or issue["title"], "hf_repo": repo_id,
            "framework": form.get("Framework") or ", ".join(frameworks_of([f["path"] for p in prs for f in p["files"]["nodes"]], frameworks)) or "–",
            "skus": form_gpus(form) or sorted({s for e in new for s in e["skus"]}, key=collect.SKUS.index),
            "tags": [e["name"] for e in new],
            "target": issue["milestone"]["title"] if issue["milestone"] else (prs[0]["milestone"]["title"] if prs and prs[0]["milestone"] else ""),
            "status": status_of(labels_of(issue), prs),
            "prs": [pr_link(p) for p in prs],
            "updated": max([issue["updatedAt"]] + [p["updatedAt"] for p in prs])[:10],
        })
    for pr in raw["prs"]:
        if label not in labels_of(pr) or pr["number"] in linked:
            continue
        new = describe_added(added_models(pr, files, known_tags))
        items.append({
            "source": "pr", "number": pr["number"], "url": pr["url"], "title": pr["title"],
            "model": short_list(sorted({e["model"] for e in new})) or pr["title"], "hf_repo": None,
            "framework": ", ".join(frameworks_of([f["path"] for f in pr["files"]["nodes"]], frameworks)) or "–",
            "skus": sorted({s for e in new for s in e["skus"]}, key=collect.SKUS.index),
            "tags": [e["name"] for e in new],
            "target": pr["milestone"]["title"] if pr["milestone"] else "",
            "status": status_of(set(), [pr]),
            "prs": [pr_link(pr)],
            "updated": pr["updatedAt"][:10],
        })

    oncoming, landed = [], []
    for item in items:
        (landed if is_landed(item, catalog_rows) else oncoming).append(item)
    # By status, most recently updated first within each.
    order = {"in review": 0, "in progress": 1, "planned": 2, "blocked": 3}
    oncoming.sort(key=lambda i: i["updated"], reverse=True)
    oncoming.sort(key=lambda i: order[i["status"]])
    return oncoming, landed


def unlabeled_model_prs(raw, label):
    """Open PRs that change model cards but neither carry the label nor close a labeled issue."""
    labeled_issues = {i["number"] for i in raw["issues"]}
    return [pr for pr in raw["prs"]
            if model_card_files(pr) and label not in labels_of(pr)
            and not {r["number"] for r in pr["closingIssuesReferences"]["nodes"]} & labeled_issues]


def collect_roadmap(source, catalog_rows, frameworks, warnings, token=None):
    """Roadmap items for a snapshot; on any GitHub failure, warn and return nothing."""
    repo, label = source["repo"], source.get("label", "new-model")
    token = token or github_token()
    if not token:
        collect.warn(f"no GitHub token (GH_TOKEN/GITHUB_TOKEN or gh auth); roadmap from {repo} skipped", warnings)
        return [], []
    try:
        return build_roadmap(fetch(repo, label, token), catalog_rows, frameworks, label, Files(token))
    except (OSError, RuntimeError, KeyError, ValueError) as e:
        collect.warn(f"reading the roadmap from {repo} failed ({e}); using dashboard/status.yaml only", warnings)
        return [], []

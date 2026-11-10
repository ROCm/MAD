#!/usr/bin/env python3
"""Standalone Kubernetes submission of a MAD multinode card -- no madengine.

The Kubernetes counterpart of `sbatch run_xPyD_models.slurm`: it runs a card's own
SLURM launcher, unchanged, across one pod per node of the card, and exits non-zero
when the run failed. madengine submits the same cards through its own k8s path;
the two are separate and must stay equivalent (same pods, same environment, same
result rows), which is checked by comparing what each renders.

How a card runs on Kubernetes: an Indexed Job with one pod per node and a headless
Service, so pods reach each other by hostname ("<run>-0", "<run>-1", ...). Every pod
runs the card's image; pod 0 runs the card's script with the SLURM_* variables a
batch script sees and stand-ins for srun / docker / scontrol (mad_k8s.py, next to
this file) first on PATH; the others serve its srun tasks. See mad_k8s_node.sh.

Usage (from the MAD repo root):

    python3 scripts/common/k8s/submit.py --card pyt_vllm_kimi-k3_mi300x_pp2xtp8 \\
        --image <image> --config k8s.json [--env KEY=VALUE ...] [--dry-run]

k8s.json holds the same keys as madengine's "k8s" block (kubeconfig, context,
namespace, gpu_count, volumes, extra_resources, security_context, shm_size,
node_selector, tolerations, host_network, pod_annotations, image_pull_policy,
results_access_mode, *_results_storage_class, include_script_dirs,
ttl_seconds_after_finished, secrets.image_pull_secret_names -- default
["dockerhub-rocm"], as in madengine), plus optional
"env_vars" and "distributed": {"nnodes": N} -- so one file can drive both paths.

Results: the run's perf.csv and logs are copied to --output-dir (default
./k8s_runs/<run>/). Exit status: 0 only if the Job completed and perf.csv has
rows, none of them FAILURE -- the rule the launchers apply on SLURM.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# madengine's default pull secret (deployment/k8s_secrets.py).
DEFAULT_IMAGE_PULL_SECRET_NAMES = ["dockerhub-rocm"]

HERE = Path(__file__).resolve().parent
AGENT_PORT = 29711
BUNDLE_LIMIT_BYTES = 900 * 1024
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
MODELS_JSON_GLOB = "scripts/*/models.json"


def die(msg, rc=2):
    sys.stderr.write(f"[mad-k8s submit] {msg}\n")
    sys.exit(rc)


# ------------------------------------------------------------------ the card


def find_card(name):
    for path in sorted(Path(".").glob(MODELS_JSON_GLOB)):  # relative: paths travel into the pods
        with open(path) as f:
            cards = json.load(f)
        for card in cards:
            if card.get("name") == name:
                card = dict(card)
                # A card's "scripts" is relative to its models.json directory.
                card["scripts"] = str(path.parent / card["scripts"])
                return card
    die(f"card {name!r} not found in {MODELS_JSON_GLOB} (run from the MAD repo root)")


def run_name(card_name, nnodes):
    """A DNS label with room for "-<index>": pods' hostnames are "<run>-<i>"."""
    name = re.sub(r"[^a-z0-9-]", "-", f"mad-{card_name}".lower()).strip("-")
    name = re.sub(r"-+", "-", name)
    room = 63 - len(f"-{max(nnodes - 1, 0)}")
    if len(name) > room:
        digest = hashlib.sha1(card_name.encode()).hexdigest()[:6]
        name = f"{name[: room - 7].rstrip('-')}-{digest}"
    return name


def first_positive(*values):
    for v in values:
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------- manifests


def scripts_bundle(script, include_dirs):
    dirs = [Path(script).parent] + [Path(d) for d in include_dirs if d]
    buf, names = io.BytesIO(), []
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in dirs:
            if not d.is_dir():
                die(f"script directory {d} not found")
            for f in sorted(d.rglob("*")):
                if f.is_file() and not (_SKIP_DIRS & set(f.parts)):
                    tar.add(str(f), arcname=str(f))
                    names.append(str(f))
        for f in ("mad_k8s.py", "mad_k8s_node.sh"):
            tar.add(str(HERE / f), arcname=f".mad-k8s/{f}")
            names.append(f".mad-k8s/{f}")
    data = buf.getvalue()
    if len(base64.b64encode(data)) > BUNDLE_LIMIT_BYTES:
        die(f"scripts bundle is {len(data)} bytes compressed; a ConfigMap holds under 1 MiB. "
            "Narrow include_script_dirs.")
    return data, names


def build(card, k8s, env_overrides, image, nnodes_override, timeout):
    nnodes = first_positive(nnodes_override, (card.get("distributed") or {}).get("nnodes"),
                            (card.get("slurm") or {}).get("nodes")) or 1
    gpus = first_positive(k8s.get("gpu_count"), (card.get("slurm") or {}).get("gpus_per_node"),
                          card.get("n_gpus"))
    if not gpus:
        die("GPUs per pod unknown: set gpu_count (the card has no slurm.gpus_per_node)")
    name = run_name(card["name"], nnodes)
    ns = k8s.get("namespace", "default")
    script = card["scripts"]
    include = k8s.get("include_script_dirs")
    if include is None:
        common = Path(script).parent.parent / "common"
        include = [str(common)] if common.is_dir() else []
    bundle, members = scripts_bundle(script, include)

    run_id = str(int(time.time()) % 100000000)
    env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
    env.update(env_overrides)
    env["DOCKER_IMAGE_NAME"] = image
    env.setdefault("LOG_PATH", "/results/logs")
    env.update({
        "MAD_K8S_NNODES": str(nnodes), "MAD_K8S_NODE_PREFIX": name, "MAD_K8S_JOB_ID": run_id,
        "MAD_K8S_IMAGE": image, "MAD_K8S_SCRIPT": script, "MAD_K8S_ARGS": card.get("args") or "",
        "MAD_K8S_AGENT_PORT": str(AGENT_PORT),
    })
    if card.get("multiple_results"):
        env["MAD_K8S_MULTIPLE_RESULTS"] = card["multiple_results"]

    labels = {"app": "mad", "model": name, "mad-launcher": "slurm_multi"}
    pod_labels = {"app": "mad", "job-name": name, "model": name}
    gpu_resource = k8s.get("gpu_resource_name", "amd.com/gpu")
    extra = {k: str(v) for k, v in (k8s.get("extra_resources") or {}).items()}
    limits = {gpu_resource: str(gpus), **extra}
    requests = dict(limits)
    for key in ("cpu", "memory"):
        if k8s.get(key):
            requests[key] = str(k8s[key])

    volumes, mounts = [], []
    for i, v in enumerate(k8s.get("volumes") or []):
        vname = v.get("name") or f"mad-vol-{i}"
        if v.get("host_path"):
            src = {"hostPath": {"path": v["host_path"], "type": v.get("host_path_type", "Directory")}}
        elif v.get("pvc"):
            src = {"persistentVolumeClaim": {"claimName": v["pvc"]}}
        elif v.get("nfs"):
            src = {"nfs": {"server": v["nfs"]["server"], "path": v["nfs"]["path"]}}
        else:
            die(f"volumes[{i}] needs host_path, pvc or nfs")
        volumes.append({"name": vname, **src})
        mount = {"name": vname, "mountPath": v.get("mount_path") or v.get("host_path")}
        if v.get("read_only"):
            mount["readOnly"] = True
        mounts.append(mount)

    container = {
        "name": name,
        "image": image,
        # madengine's base default (presets/k8s/defaults.json), so the paths agree when unset.
        "imagePullPolicy": k8s.get("image_pull_policy", "Always"),
        "command": ["/bin/bash", "-c"],
        "args": ["set -e; mkdir -p /workspace; tar -xzf /mad-k8s-bundle/scripts.tgz -C /workspace; "
                 "exec bash /workspace/.mad-k8s/mad_k8s_node.sh"],
        "env": [{"name": k, "value": v} for k, v in sorted(env.items())],
        "ports": [{"name": "mad-k8s", "containerPort": AGENT_PORT}],
        "resources": {"limits": limits, "requests": requests},
        "volumeMounts": [{"name": "mad-k8s-bundle", "mountPath": "/mad-k8s-bundle", "readOnly": True},
                         {"name": "results", "mountPath": "/results"},
                         {"name": "dshm", "mountPath": "/dev/shm"}, *mounts],
    }
    if k8s.get("security_context"):
        container["securityContext"] = k8s["security_context"]
    pod = {
        "restartPolicy": "Never",
        "subdomain": name,
        "dnsConfig": {"searches": [f"{name}.{ns}.svc.cluster.local"]},
        "containers": [container],
        "volumes": [{"name": "mad-k8s-bundle", "configMap": {"name": f"{name}-config"}},
                    {"name": "results", "persistentVolumeClaim": {"claimName": f"{name}-results"}},
                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": str(k8s.get("shm_size", "64Gi"))}},
                    *volumes],
    }
    if k8s.get("host_network"):
        pod["hostNetwork"] = True
        pod["dnsPolicy"] = "ClusterFirstWithHostNet"
        pod["affinity"] = {"podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [
            {"labelSelector": {"matchLabels": {"job-name": name}}, "topologyKey": "kubernetes.io/hostname"}]}}
    if k8s.get("node_selector"):
        pod["nodeSelector"] = k8s["node_selector"]
    if k8s.get("tolerations"):
        pod["tolerations"] = k8s["tolerations"]
    # madengine's default when the key is absent; an explicit list (even []) wins.
    pulls = {"image_pull_secret_names": DEFAULT_IMAGE_PULL_SECRET_NAMES,
             **(k8s.get("secrets") or {})}["image_pull_secret_names"] or []
    if pulls:
        pod["imagePullSecrets"] = [{"name": n} for n in pulls]

    template_meta = {"labels": pod_labels}
    if k8s.get("pod_annotations"):
        template_meta["annotations"] = {str(k): str(v) for k, v in k8s["pod_annotations"].items()}
    job_spec = {"completions": nnodes, "parallelism": nnodes, "completionMode": "Indexed",
                "backoffLimit": 0, "template": {"metadata": template_meta, "spec": pod}}
    if timeout and timeout > 0:
        job_spec["activeDeadlineSeconds"] = int(timeout)
    if k8s.get("ttl_seconds_after_finished") is not None:
        job_spec["ttlSecondsAfterFinished"] = int(k8s["ttl_seconds_after_finished"])

    def meta(n):
        return {"name": n, "namespace": ns, "labels": labels}

    access = k8s.get("results_access_mode") or ("ReadWriteMany" if nnodes > 1 else "ReadWriteOnce")
    storage_class = (k8s.get("multi_node_results_storage_class") if nnodes > 1
                     else k8s.get("single_node_results_storage_class")) or k8s.get("storage_class")
    pvc_spec = {"accessModes": [access], "resources": {"requests": {"storage": str(k8s.get("results_size", "10Gi"))}}}
    if storage_class:
        pvc_spec["storageClassName"] = storage_class
    return {
        "name": name, "namespace": ns, "nnodes": nnodes, "gpus": gpus, "run_id": run_id,
        "bundle_files": members,
        "objects": [
            {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": meta(f"{name}-results"), "spec": pvc_spec},
            {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(f"{name}-config"),
             "binaryData": {"scripts.tgz": base64.b64encode(bundle).decode()}},
            {"apiVersion": "v1", "kind": "Service", "metadata": meta(name),
             "spec": {"clusterIP": "None", "publishNotReadyAddresses": True, "selector": {"job-name": name},
                      "ports": [{"name": "mad-k8s", "port": AGENT_PORT}]}},
            {"apiVersion": "batch/v1", "kind": "Job", "metadata": meta(name), "spec": job_spec},
        ],
    }


# -------------------------------------------------------------------- kubectl


class Kube:
    def __init__(self, k8s):
        self.base = ["kubectl"]
        if k8s.get("kubeconfig"):
            self.base += ["--kubeconfig", str(Path(k8s["kubeconfig"]).expanduser())]
        if k8s.get("context"):
            self.base += ["--context", k8s["context"]]
        self.ns = k8s.get("namespace", "default")

    def run(self, *args, input=None, check=True, namespaced=True):
        cmd = self.base + (["-n", self.ns] if namespaced else []) + list(args)
        r = subprocess.run(cmd, input=input, capture_output=True, text=True)
        if check and r.returncode != 0:
            die(f"{' '.join(cmd[:8])} ... failed: {r.stderr.strip()}", 1)
        return r

    def get(self, kind, name):
        r = self.run("get", kind, name, "-o", "json", check=False)
        return json.loads(r.stdout) if r.returncode == 0 else None


def replace_ours(kube, obj):
    """Delete a leftover of this run's name from an earlier run -- only if it is ours."""
    kind, name = obj["kind"], obj["metadata"]["name"]
    existing = kube.get(kind, name)
    if existing is None:
        return
    if (existing.get("metadata", {}).get("labels") or {}).get("app") != "mad":
        die(f"{kind} {name} exists in namespace {kube.ns} and was not created by this tool "
            "(no app=mad label); refusing to delete it", 1)
    kube.run("delete", kind, name, "--wait=true", "--ignore-not-found")


def job_outcome(job):
    for c in (job.get("status") or {}).get("conditions") or []:
        if c.get("status") == "True" and c.get("type") in ("Complete", "SuccessCriteriaMet"):
            return "succeeded"
        if c.get("status") == "True" and c.get("type") in ("Failed", "FailureTarget"):
            return "failed"
    return None


def collect(kube, built, image, out_dir):
    """Copy this run's results off its results volume with a short-lived reader pod."""
    name = built["name"]
    reader = {"apiVersion": "v1", "kind": "Pod",
              "metadata": {"name": f"{name}-reader"[:63].rstrip("-"), "labels": {"app": "mad"}},
              "spec": {"restartPolicy": "Never", "containers": [{
                  "name": "r", "image": image, "imagePullPolicy": "IfNotPresent", "command": ["sleep", "600"],
                  "volumeMounts": [{"name": "v", "mountPath": "/results", "readOnly": True}]}],
                  "volumes": [{"name": "v", "persistentVolumeClaim": {"claimName": f"{name}-results"}}]}}
    job = kube.get("job", name) or {}
    node_sel = (job.get("spec", {}).get("template", {}).get("spec", {}) or {}).get("nodeSelector")
    if node_sel:
        reader["spec"]["nodeSelector"] = node_sel
    rname = reader["metadata"]["name"]
    kube.run("delete", "pod", rname, "--ignore-not-found", "--wait=true")
    kube.run("create", "-f", "-", input=json.dumps(reader))
    kube.run("wait", "--for=condition=Ready", f"pod/{rname}", "--timeout=300s")
    out_dir.mkdir(parents=True, exist_ok=True)
    pod0 = f"/results/{name}-0"
    perf = kube.run("exec", rname, "--", "cat", f"{pod0}/perf.csv", check=False)
    if perf.returncode == 0 and perf.stdout.strip():
        (out_dir / "perf.csv").write_text(perf.stdout)
    kube.run("cp", f"{kube.ns}/{rname}:/results/logs/{built['run_id']}", str(out_dir / "logs"), check=False)
    kube.run("delete", "pod", rname, "--wait=false")
    return out_dir / "perf.csv"


def verdict(perf_path):
    """(ok, summary): rows with no FAILURE -- the launchers' own rule on SLURM."""
    if not perf_path.is_file():
        return False, "no perf.csv was published by pod 0"
    with open(perf_path, newline="") as f:
        rows = list(csv.DictReader(f))
    failed = [r for r in rows if (r.get("status") or "").strip().upper() != "SUCCESS"]
    if not rows:
        return False, "perf.csv has no rows"
    return not failed, f"{len(rows) - len(failed)} SUCCESS, {len(failed)} FAILURE row(s)"


# ----------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", required=True, help="card name from scripts/*/models.json")
    ap.add_argument("--image", help="image every pod runs (default: the card's DOCKER_IMAGE_NAME)")
    ap.add_argument("--config", help="JSON with a 'k8s' block (madengine's keys), 'env_vars', 'distributed'")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="override a card env var")
    ap.add_argument("--nodes", type=int, help="pods (nodes) for the run (default: the card's)")
    ap.add_argument("--timeout", type=int, help="seconds before the Job is stopped (default: the card's)")
    ap.add_argument("--output-dir", help="where perf.csv and logs go (default ./k8s_runs/<run>)")
    ap.add_argument("--dry-run", action="store_true", help="print the manifests as JSON and exit")
    a = ap.parse_args(argv)

    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg = json.load(f)
    k8s = cfg.get("k8s") or {}
    card = find_card(a.card)
    env = {k: str(v) for k, v in (cfg.get("env_vars") or {}).items()}
    for kv in a.env:
        if "=" not in kv:
            die(f"--env {kv!r}: expected KEY=VALUE")
        k, v = kv.split("=", 1)
        env[k] = v
    image = a.image or (card.get("env_vars") or {}).get("DOCKER_IMAGE_NAME")
    if not image or image.startswith("<"):
        die("no image: pass --image (the card has no usable DOCKER_IMAGE_NAME)")
    nodes = a.nodes or (cfg.get("distributed") or {}).get("nnodes")
    timeout = a.timeout if a.timeout is not None else first_positive(card.get("timeout"))
    built = build(card, k8s, env, image, nodes, timeout)

    if a.dry_run:
        out = {k: built[k] for k in ("name", "namespace", "nnodes", "gpus", "bundle_files")}
        out["objects"] = built["objects"]
        print(json.dumps(out, indent=2))
        return 0

    kube = Kube(k8s)
    ctx = subprocess.run(kube.base + ["config", "current-context"], capture_output=True, text=True).stdout.strip()
    if k8s.get("context") and ctx and ctx != k8s["context"] and "--context" not in kube.base:
        die(f"kube context is {ctx!r}, config names {k8s['context']!r}")
    print(f"[mad-k8s submit] {a.card}: {built['nnodes']} pod(s) x {built['gpus']} GPU(s), image {image}, "
          f"namespace {built['namespace']}, context {k8s.get('context') or ctx}", flush=True)

    for obj in reversed(built["objects"]):
        replace_ours(kube, obj)
    # create, not apply: apply copies the whole object into a last-applied annotation,
    # capped at 256 KiB, and the ConfigMap carrying the scripts bundle is larger.
    # Leftovers of this run's names were deleted above, so there is nothing to merge.
    for obj in built["objects"]:
        kube.run("create", "-f", "-", input=json.dumps(obj))
    print(f"[mad-k8s submit] Job {built['name']} submitted (run {built['run_id']})", flush=True)

    deadline = time.time() + (timeout or 0) + 600 if timeout else None
    outcome = None
    while outcome is None:
        time.sleep(30)
        job = kube.get("job", built["name"])
        if job is None:
            outcome = "failed"
            print("[mad-k8s submit] the Job is gone", flush=True)
            break
        outcome = job_outcome(job)
        if deadline and time.time() > deadline:
            outcome = "failed"
    print(f"[mad-k8s submit] Job {outcome}", flush=True)

    out_dir = Path(a.output_dir or f"k8s_runs/{built['name']}/run-{built['run_id']}")
    perf = collect(kube, built, image, out_dir)
    ok, summary = verdict(perf)
    ok = ok and outcome == "succeeded"
    print(f"[mad-k8s submit] {'PASS' if ok else 'FAIL'}: Job {outcome}; {summary}; results in {out_dir}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

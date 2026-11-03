"""Offline checks for scripts/common/k8s/submit.py (no cluster, no madengine).

Run from the MAD repo root:  python3 -m unittest discover -s scripts/common/k8s/tests -v
"""

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SUBMIT = ROOT / "scripts/common/k8s/submit.py"
CARD = "pyt_vllm_kimi-k3_mi300x_pp2xtp8"


def dry_run(*args, config=None):
    cmd = [sys.executable, str(SUBMIT), "--card", CARD, "--image", "img:t", "--dry-run", *args]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config or {}, f)
    try:
        r = subprocess.run(cmd + ["--config", f.name], cwd=ROOT, capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    out = json.loads(r.stdout)
    objs = {o["kind"]: o for o in out["objects"]}
    pod = objs["Job"]["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in pod["containers"][0]["env"]}
    return out, objs, pod, env


class TestSubmitDryRun(unittest.TestCase):
    def test_card_defaults(self):
        out, objs, pod, env = dry_run()
        spec = objs["Job"]["spec"]
        self.assertEqual((spec["completions"], spec["completionMode"], spec["backoffLimit"]), (2, "Indexed", 0))
        self.assertEqual(pod["containers"][0]["resources"]["limits"]["amd.com/gpu"], "8")
        self.assertEqual(env["MAD_K8S_SCRIPT"], "scripts/vllm_multinode/run_multinode.slurm")
        self.assertEqual(env["MODEL_NAME"], "Kimi-K3")
        self.assertEqual(env["DOCKER_IMAGE_NAME"], "img:t")
        self.assertEqual(env["LOG_PATH"], "/results/logs")
        self.assertEqual(pod["subdomain"], out["name"])
        self.assertEqual(objs["Service"]["spec"]["clusterIP"], "None")

    def test_config_and_env_override_the_card(self):
        _, objs, pod, env = dry_run("--env", "MODEL_NAME=Llama-3.1-8B-Instruct", config={
            "k8s": {"namespace": "lab", "gpu_count": 2, "cpu": "16", "memory": "64Gi",
                    "volumes": [{"name": "w", "pvc": "model-weights", "mount_path": "/models", "read_only": True}],
                    "host_network": True, "results_access_mode": "ReadWriteOnce",
                    "multi_node_results_storage_class": "local-path"},
            "env_vars": {"TP_SIZE": "2", "PP_SIZE": "4"}, "distributed": {"nnodes": 4}})
        self.assertEqual(objs["Job"]["spec"]["completions"], 4)
        self.assertEqual(pod["containers"][0]["resources"]["requests"],
                         {"amd.com/gpu": "2", "cpu": "16", "memory": "64Gi"})
        self.assertEqual((env["MODEL_NAME"], env["TP_SIZE"], env["PP_SIZE"]), ("Llama-3.1-8B-Instruct", "2", "4"))
        self.assertTrue(pod["hostNetwork"])
        self.assertIn("podAntiAffinity", pod["affinity"])
        self.assertEqual(objs["PersistentVolumeClaim"]["spec"]["accessModes"], ["ReadWriteOnce"])
        self.assertEqual(objs["Job"]["metadata"]["namespace"], "lab")

    def test_bundle_carries_launcher_siblings_and_shim(self):
        _, objs, _, _ = dry_run(config={"k8s": {"include_script_dirs": ["scripts/common", "scripts/vllm_dissag"]}})
        names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(objs["ConfigMap"]["binaryData"]["scripts.tgz"]))).getnames()
        for want in ("scripts/vllm_multinode/run_multinode.slurm", "scripts/vllm_dissag/models.yaml",
                     "scripts/common/cluster.sh", ".mad-k8s/mad_k8s.py", ".mad-k8s/mad_k8s_node.sh"):
            self.assertIn(want, names)

    def test_pod_hostnames_are_dns_labels(self):
        out, _, _, _ = dry_run()
        self.assertRegex(out["name"], r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
        self.assertLessEqual(len(out["name"]) + 2, 63)

    def test_pull_secret_defaults_as_madengine_and_can_be_cleared(self):
        _, _, pod, _ = dry_run()
        self.assertEqual(pod["imagePullSecrets"], [{"name": "dockerhub-rocm"}])
        _, _, pod, _ = dry_run(config={"k8s": {"secrets": {"image_pull_secret_names": ["mine"]}}})
        self.assertEqual(pod["imagePullSecrets"], [{"name": "mine"}])
        _, _, pod, _ = dry_run(config={"k8s": {"secrets": {"image_pull_secret_names": []}}})
        self.assertNotIn("imagePullSecrets", pod)


class TestVerdict(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SUBMIT.parent))
        import submit
        self.verdict = submit.verdict

    def _check(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "perf.csv"
            if text is not None:
                p.write_text(text)
            return self.verdict(p)[0]

    def test_rules(self):
        self.assertTrue(self._check("model,status\nM,SUCCESS\nM,SUCCESS\n"))
        self.assertFalse(self._check("model,status\nM,SUCCESS\nM,FAILURE\n"))
        self.assertFalse(self._check("model,status\n"))
        self.assertFalse(self._check(None))


if __name__ == "__main__":
    unittest.main()


class TestCreateNotApply(unittest.TestCase):
    def test_objects_are_created_not_applied(self):
        # kubectl apply stores the whole object in an annotation capped at 256 KiB;
        # the ConfigMap carrying the scripts bundle is ~300 KB and is rejected.
        src = SUBMIT.read_text()
        self.assertNotIn('kube.run("apply"', src)
        self.assertIn('kube.run("create", "-f", "-"', src)

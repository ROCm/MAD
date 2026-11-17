"""The docker / scontrol stand-ins a slurm_multi card runs under on Kubernetes (mad_k8s.py).

The tools are run as the card's script runs them in a pod, including the exact command
shapes of MAD's launchers.
Run from the MAD repo root:  python3 -m unittest discover -s scripts/common/k8s/tests -v
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SHIM = Path(__file__).resolve().parents[1] / "mad_k8s.py"
NODES = ["127.0.0.1", "127.0.0.2"]
IMAGE = "rocm/test:img"



class Tools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp = Path(self._tmp.name)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        for tool in ("docker", "scontrol"):
            (bin_dir / tool).write_text(f'#!/bin/bash\nexec "{sys.executable}" "{SHIM}" {tool} "$@"\n')
            (bin_dir / tool).chmod(0o755)
        state = tmp / "state"
        (state / "containers").mkdir(parents=True)
        (state / "base_env.json").write_text(json.dumps(
            {"PATH": os.environ["PATH"], "IMAGE_ONLY": "from-image", "MAD_K8S_IMAGE_WORKDIR": str(tmp)}))
        env = dict(os.environ)
        env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "MAD_K8S_STATE": str(state), "MAD_K8S_IMAGE": IMAGE,
                    "SLURM_JOB_NODELIST": ",".join(NODES)})
        self.env = env

    def tearDown(self):
        self._tmp.cleanup()

    def run_(self, cmd, env=None):
        return subprocess.run(cmd, env=env or self.env, capture_output=True, text=True, timeout=60)

    def test_there_is_no_srun_or_agent(self):
        """Nothing fans out from one pod to the others any more (per-pod dispatch)."""
        r = self.run_([sys.executable, str(SHIM), "srun", "true"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown tool", r.stderr)
        self.assertIn("unknown tool", self.run_([sys.executable, str(SHIM), "agent"]).stderr)

    def test_scontrol(self):
        self.assertEqual(self.run_(["scontrol", "show", "hostnames"]).stdout.split(), NODES)
        self.assertEqual(self.run_(["scontrol", "show", "hostnames", "n[01-03],m"]).stdout.split(),
                         ["n01", "n02", "n03", "m"])
        self.assertEqual(self.run_(["scontrol", "show", "topology"]).returncode, 1)

    def test_docker_run_env_volumes_entrypoint_fresh_environment(self):
        src = self.tmp / "logs"
        src.mkdir()
        dst = self.tmp / "run_logs"
        r = self.run_(["docker", "run", "--rm", "--device", "/dev/kfd", "--network", "host", "--privileged",
                       "--ipc", "host", "--shm-size", "64G", "--ulimit", "nofile=1:1", "-v", f"{src}:{dst}",
                       "-e", "A=1", "--entrypoint", "/bin/bash", IMAGE, "-c",
                       'echo "$A ${IMAGE_ONLY} [${LAUNCHER_ONLY:-}]"; echo hi > ' + str(dst) + "/f"],
                      env={**self.env, "LAUNCHER_ONLY": "leak"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "1 from-image []\n")
        self.assertEqual((src / "f").read_text(), "hi\n")
        self.assertEqual(self.run_(["docker", "run", "--rm", "other/image:x", "true"]).returncode, 125)

    def test_docker_stop_ps_rm_image(self):
        p = subprocess.Popen(["docker", "run", "--rm", "--name", "c1", "--entrypoint", "sleep", IMAGE, "60"], env=self.env)
        for _ in range(50):
            if self.run_(["docker", "ps", "-q"]).stdout.split() == ["c1"]:
                break
            time.sleep(0.1)
        self.assertEqual(self.run_(["docker", "stop", "-t", "2", "c1"]).returncode, 0)
        self.assertNotEqual(p.wait(timeout=10), 0)
        self.assertEqual(self.run_(["docker", "rm", "-f", "c1"]).returncode, 0)
        r = self.run_(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE])
        self.assertTrue(r.stdout.startswith("sha256:"))
        self.assertEqual(self.run_(["docker", "pull", "other:x"]).returncode, 1)

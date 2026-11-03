"""The srun / docker / scontrol stand-ins of the standalone Kubernetes path (mad_k8s.py).

Two real agents on two loopback addresses stand in for two pods; the tools are run as
the card's script runs them, including the exact command shapes of MAD's launchers.
Run from the MAD repo root:  python3 -m unittest discover -s scripts/common/k8s/tests -v
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SHIM = Path(__file__).resolve().parents[1] / "mad_k8s.py"
NODES = ["127.0.0.1", "127.0.0.2"]
IMAGE = "rocm/test:img"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cluster(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp = Path(self._tmp.name)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        for tool in ("srun", "docker", "scontrol"):
            (bin_dir / tool).write_text(f'#!/bin/bash\nexec "{sys.executable}" "{SHIM}" {tool} "$@"\n')
            (bin_dir / tool).chmod(0o755)
        state = tmp / "state"
        (state / "containers").mkdir(parents=True)
        (state / "base_env.json").write_text(json.dumps(
            {"PATH": os.environ["PATH"], "IMAGE_ONLY": "from-image", "MAD_K8S_IMAGE_WORKDIR": str(tmp)}))
        env = dict(os.environ)
        env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "MAD_K8S_AGENT_PORT": str(_free_port()),
                    "MAD_K8S_STATE": str(state), "MAD_K8S_IMAGE": IMAGE,
                    "SLURM_JOB_NODELIST": ",".join(NODES), "MAD_K8S_CONNECT_TIMEOUT": "20"})
        self.env = env
        self.agents = [subprocess.Popen([sys.executable, str(SHIM), "agent"], env={**env, "MAD_K8S_AGENT_BIND": n})
                       for n in NODES]
        subprocess.run(["srun", "true"], env=env, check=True, timeout=30)

    def tearDown(self):
        for a in self.agents:
            a.kill()
            a.wait()
        self._tmp.cleanup()

    def run_(self, cmd, env=None):
        return subprocess.run(cmd, env=env or self.env, capture_output=True, text=True, timeout=60)

    def test_srun_one_task_per_node_in_allocation_order(self):
        r = self.run_(["srun", "--nodelist=127.0.0.2,127.0.0.1", "bash", "-c",
                       'echo "$SLURM_PROCID $SLURMD_NODENAME $SLURM_STEP_NUM_NODES"'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(sorted(r.stdout.split("\n")[:2]), ["0 127.0.0.1 2", "1 127.0.0.2 2"])

    def test_srun_single_node_stdout_only_and_highest_status(self):
        r = self.run_(["srun", "--nodes=1", "--ntasks=1", "--time=00:20:00", "--nodelist=127.0.0.2",
                       "bash", "-c", "echo out; echo err >&2"])
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "out\n", "err\n"))
        self.assertEqual(self.run_(["srun", "bash", "-c", '[ "$SLURM_PROCID" = 1 ] && exit 7; exit 0']).returncode, 7)

    def test_srun_refuses_what_it_does_not_implement(self):
        self.assertIn("not in this allocation", self.run_(["srun", "--nodelist=10.9.9.9", "true"]).stderr)
        self.assertIn("unsupported option", self.run_(["srun", "--distribution=cyclic", "true"]).stderr)
        self.assertIn("one task per node", self.run_(["srun", "--ntasks-per-node=2", "true"]).stderr)

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

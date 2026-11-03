"""The pod entry point (mad_k8s_node.sh) running a card per pod.

Two processes stand in for the two pods of an Indexed Job, sharing a workspace, a
log directory and a results directory as the pods share the bundle and the results
volume. The card is a small launcher that reaches its nodes through
scripts/common/dispatch/dispatch.sh, as MAD's launchers do, so the pods run it per
pod (k8s-indexed); each node of it records that it ran.
Run from the MAD repo root:  python3 -m unittest discover -s scripts/common/k8s/tests -v
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

K8S = Path(__file__).resolve().parents[1]
COMMON = K8S.parent

CARD = r"""#!/bin/bash
. "$(pwd)/../common/cluster.sh"
dispatch_run node "$SLURM_JOB_NODELIST" bash -c \
    'echo "rank $SLURM_PROCID ran" >> "$LOG_PATH/ran"; [ "$SLURM_PROCID" = 1 ] && exit ${FAKE_RC_RANK1:-0}; exit 0'
rc=$?
dispatch_is_leader || exit "$rc"
mkdir -p "$LOG_PATH/$SLURM_JOB_ID"
printf 'model,performance,metric,status\nm,1,x,SUCCESS\n' > "$LOG_PATH/$SLURM_JOB_ID/perf.csv"
exit "$rc"
"""


class PerPod(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp = Path(self._tmp.name)
        ws = tmp / "ws"
        shutil.copytree(COMMON, ws / "scripts" / "common",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
        (ws / "scripts" / "card").mkdir(parents=True)
        (ws / "scripts" / "card" / "run.slurm").write_text(CARD)
        (ws / ".mad-k8s").mkdir()
        for f in ("mad_k8s.py", "mad_k8s_node.sh"):
            shutil.copy(K8S / f, ws / ".mad-k8s" / f)
        self.ws = ws

    def tearDown(self):
        self._tmp.cleanup()

    def run_pods(self, extra=None):
        tmp, procs = self.tmp, []
        for k in range(2):
            env = {
                "PATH": os.environ["PATH"], "HOME": str(tmp),
                "MAD_K8S_WORKSPACE": str(self.ws), "MAD_K8S_STATE": str(tmp / f"state{k}"),
                "MAD_K8S_RESULTS": str(tmp / "results"), "LOG_PATH": str(tmp / "logs"),
                "MAD_K8S_NNODES": "2", "MAD_K8S_NODE_PREFIX": "run", "MAD_K8S_JOB_ID": "55",
                "MAD_K8S_IMAGE": "img", "MAD_K8S_SCRIPT": "scripts/card/run.slurm",
                "JOB_COMPLETION_INDEX": str(k), "MAD_RENDEZVOUS_TIMEOUT": "30",
                "CLUSTER_SYSFS_IB": str(tmp / "none"), **(extra or {}),
            }
            procs.append(subprocess.Popen(["bash", str(self.ws / ".mad-k8s" / "mad_k8s_node.sh")], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        outs = [p.communicate(timeout=120)[0] for p in procs]
        return [p.returncode for p in procs], outs

    def test_every_pod_runs_its_own_node(self):
        rcs, outs = self.run_pods()
        self.assertEqual(rcs, [0, 0], outs)
        self.assertEqual(sorted((self.tmp / "logs" / "ran").read_text().split("\n")[:-1]),
                         ["rank 0 ran", "rank 1 ran"])
        self.assertIn("as its own node of run-0,run-1", outs[1])
        self.assertIn("[dispatch] k8s-indexed: rank 1 of 2 on run-1", outs[1])

    def test_nothing_fans_out(self):
        self.run_pods()
        tools = sorted(p.name for p in (self.ws / ".mad-k8s" / "bin").iterdir())
        self.assertEqual(tools, ["docker", "scontrol"])

    def test_pod_0_publishes_the_results(self):
        self.run_pods()
        res = self.tmp / "results"
        self.assertIn("SUCCESS", (res / "run-0" / "perf.csv").read_text())
        self.assertFalse((res / "run-1" / "perf.csv").exists())
        self.assertEqual((res / "run-0" / "exit_code").read_text().strip(), "0")
        self.assertEqual((res / "run-1" / "exit_code").read_text().strip(), "0")

    def test_a_failing_node_fails_the_run(self):
        rcs, outs = self.run_pods({"FAKE_RC_RANK1": "5"})
        self.assertEqual(rcs, [5, 5], outs)
        self.assertEqual((self.tmp / "results" / "run-0" / "exit_code").read_text().strip(), "5")


if __name__ == "__main__":
    unittest.main()

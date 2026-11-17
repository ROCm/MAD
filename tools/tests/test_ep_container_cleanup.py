import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "large-ep-benchmark" / "run_benchmark.sbatch"

# fake cluster CLI tools so we can run the sbatch script locally without Slurm or Docker
SHIM = r"""
import json
import os
import subprocess
import sys

command, args = sys.argv[1], sys.argv[2:]
if command == "docker":
    with open(os.environ["TEST_DOCKER_LOG"], "a") as handle:
        handle.write(json.dumps({"node": os.environ["TEST_NODE"], "args": args}) + "\n")
    if args == ["ps", "-aq"]:
        print("ep-owned-id\nunrelated-training-id\nunrelated-service-id")
elif command == "scontrol":
    print("node-a\nnode-b")
elif command == "hostname":
    node = os.environ.get("TEST_NODE", "node-a")
    print({"node-a": "192.0.2.1", "node-b": "192.0.2.2"}[node] if args == ["-I"] else node)
elif command == "srun":
    selected = None
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option in ("-t", "-w"):
            value = args.pop(0)
            if option == "-w":
                selected = value
        elif not option.startswith("--nodes="):
            raise SystemExit("Unexpected srun option: " + option)
    for rank, node in enumerate([selected] if selected else ["node-a", "node-b"]):
        env = dict(os.environ, TEST_NODE=node, SLURM_PROCID=str(rank))
        result = subprocess.run(args, env=env, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
elif command != "sleep":
    raise SystemExit("Unexpected mock command: " + command)
"""


class EpContainerCleanupTest(unittest.TestCase):
    def test_launcher_only_cleans_its_own_container_on_each_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            binaries = temp / "bin"
            binaries.mkdir()
            shim = temp / "cluster_shim.py"
            shim.write_text(SHIM)

            for command in ("docker", "srun", "scontrol", "hostname", "sleep"):
                stub = binaries / command
                stub.write_text(
                    "#!/bin/sh\nexec "
                    + shlex.quote(sys.executable)
                    + " "
                    + shlex.quote(str(shim))
                    + " "
                    + shlex.quote(command)
                    + ' "$@"\n'
                )
                stub.chmod(0o755)
            docker_log = temp / "docker.jsonl"

            # put our fake bin/ first on PATH and mimic a 2-node Slurm job environment
            env = dict(
                os.environ,
                PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                TEST_DOCKER_LOG=str(docker_log),
                LOG_PATH=str(temp / "logs"),
                SLURM_JOB_ID="12345",
                SLURM_JOB_NODELIST="node-a,node-b",
                SLURM_NNODES="2",
                SLURM_NTASKS="2",
                DOCKER_IMAGE="test/ep-benchmark:local",
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=temp,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = [
                json.loads(line) for line in docker_log.read_text().splitlines()
            ]
            # check each node independetly to make sure clean up happenes everywhere
            for node in ("node-a", "node-b"):
                with self.subTest(node=node):
                    calls = [call["args"] for call in commands if call["node"] == node]
                    cleanup = [
                        args for args in calls if args[0] in ("stop", "rm", "kill")
                    ]
                    self.assertTrue(cleanup, "Expected named container cleanup")
                    for args in cleanup:
                        targets = [
                            value for value in args[1:] if not value.startswith("-")
                        ]
                        self.assertEqual(
                            targets, ["ep-bench"], f"Unsafe cleanup on {node}: {args}"
                        )
                    # verify the full container lifecycle ran on this node
                    self.assertIn(["stop", "ep-bench"], calls)
                    self.assertIn(["rm", "ep-bench"], calls)
                    self.assertIn(["rm", "-f", "ep-bench"], calls)
                    self.assertTrue(any(args[0] == "run" for args in calls))


if __name__ == "__main__":
    unittest.main()

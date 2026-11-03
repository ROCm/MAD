#!/usr/bin/env python3
"""Run a slurm_multi card's unmodified SLURM launcher across Kubernetes pods.

A slurm_multi card's script (e.g. MAD's run_xPyD_models.slurm) runs once, on the
allocation's first node, and drives every node itself:

    scontrol show hostnames "$SLURM_JOB_NODELIST"          # the nodes
    srun --nodes=1 --nodelist=NODE bash -c 'hostname -I'    # each node's address
    srun --nodelist=ALL bash -c '... docker run ... IMAGE CMD'  # one container per node
    docker stop / rm                                        # cleanup

On Kubernetes, madengine runs one pod per node of the card (an Indexed Job with a
headless Service), every pod already running the card's image. Pod 0 runs the
script; this module stands in for the four commands it needs:

    agent     every pod: serves `srun` tasks sent to it (TCP, stdlib only)
    srun      sends a task to each named pod's agent, streams its output, returns
              its exit status (the highest, as srun does)
    docker    `run` runs the command IN this pod -- the pod is the container --
              with the -e environment and -v paths applied; stop/rm/ps/pull/
              image inspect act on those processes and this pod's image
    scontrol  `show hostnames`

Only the forms the launchers use are implemented. Anything else fails with a
message naming the unsupported form instead of being guessed at, so a launcher
change that needs more shows up as a clear error, not a wrong run.

Host-level docker options (--device, --network host, --ipc host, --privileged,
--ulimit, --shm-size, ...) are ignored: they are the pod spec's job (devices via
device plugins, shared memory via an emptyDir), configured in madengine's k8s
config, not by the card.
"""

import hashlib
import json
import os
import re
import select
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time

STATE_DIR = os.environ.get("MAD_K8S_STATE", "/tmp/mad-k8s")
PORT = int(os.environ.get("MAD_K8S_AGENT_PORT", "29711"))
CONNECT_TIMEOUT = float(os.environ.get("MAD_K8S_CONNECT_TIMEOUT", "600"))


def _die(tool, msg, rc=2):
    sys.stderr.write(f"[mad-k8s {tool}] {msg}\n")
    sys.exit(rc)


def _job_nodes():
    return [n for n in os.environ.get("SLURM_JOB_NODELIST", "").split(",") if n]


def _expand_hostlist(spec):
    """Comma list, with simple bracket ranges (node[0-2,5]) expanded."""
    out = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", spec or ""):
        m = re.match(r"^(.*)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            if "-" in rng:
                a, b = rng.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    out.append(f"{pre}{str(i).zfill(len(a))}{post}")
            else:
                out.append(f"{pre}{rng}{post}")
    return [h for h in out if h]


# --------------------------------------------------------------------- wire
# One JSON request line; the reply is frames of (1-byte kind, 4-byte length,
# data): 'o' stdout, 'e' stderr, 'x' exit status (ascii int), always last.


def _frame(sock, kind, data):
    sock.sendall(kind + struct.pack(">I", len(data)) + data)


def _read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("agent closed the connection")
        buf += chunk
    return buf


def _connect(host):
    deadline = time.time() + CONNECT_TIMEOUT
    last = None
    while time.time() < deadline:
        try:
            return socket.create_connection((host, PORT), timeout=30)
        except OSError as e:  # agent not up yet, or DNS not yet published
            last = e
            time.sleep(2)
    raise ConnectionError(f"no mad-k8s agent at {host}:{PORT} after {CONNECT_TIMEOUT:.0f}s ({last})")


def _request(host, req, out=None, err=None):
    """Send one request; stream 'o'/'e' frames to out/err; return the exit status."""
    sock = _connect(host)
    sock.settimeout(None)
    try:
        sock.sendall((json.dumps(req) + "\n").encode())
        while True:
            kind = _read_exact(sock, 1)
            (n,) = struct.unpack(">I", _read_exact(sock, 4))
            data = _read_exact(sock, n) if n else b""
            if kind == b"x":
                return int(data.decode() or "1")
            stream = out if kind == b"o" else err
            if stream is not None:
                stream.write(data)
                stream.flush()
    finally:
        sock.close()


# -------------------------------------------------------------------- agent


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        req = json.loads(line)
        sock = self.request
        op = req.get("op")
        if op == "ping":
            _frame(sock, b"x", b"0")
        elif op == "shutdown":
            _frame(sock, b"x", b"0")
            # Reply first, then leave: the Job counts this pod as succeeded.
            threading.Timer(0.5, lambda: os._exit(0)).start()
        elif op == "run":
            self._run(req, sock)
        else:
            _frame(sock, b"e", f"[mad-k8s agent] unknown op {op!r}\n".encode())
            _frame(sock, b"x", b"2")

    def _run(self, req, sock):
        env = dict(req.get("env") or {})
        env["HOSTNAME"] = socket.gethostname()
        cwd = req.get("cwd") or "/"
        if not os.path.isdir(cwd):
            cwd = os.environ.get("MAD_K8S_WORKSPACE", "/workspace")
        try:
            proc = subprocess.Popen(
                req["argv"], env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
        except OSError as e:
            _frame(sock, b"e", f"[mad-k8s agent] cannot run {req['argv'][:1]}: {e}\n".encode())
            _frame(sock, b"x", b"127")
            return
        lock = threading.Lock()

        def pump(stream, kind):
            for chunk in iter(lambda: stream.read1(65536), b""):
                with lock:
                    try:
                        _frame(sock, kind, chunk)
                    except OSError:
                        return

        pumps = [threading.Thread(target=pump, args=(proc.stdout, b"o"), daemon=True),
                 threading.Thread(target=pump, args=(proc.stderr, b"e"), daemon=True)]
        for t in pumps:
            t.start()
        # The caller (srun) going away means the step was cancelled: like slurmd,
        # take the task's whole process group down with it.
        while proc.poll() is None:
            readable, _, _ = select.select([sock], [], [], 1.0)
            if readable and not sock.recv(1, socket.MSG_PEEK):
                _killpg(proc.pid, signal.SIGTERM, grace=10)
                break
        rc = proc.wait()
        for t in pumps:
            t.join(timeout=5)
        status = rc if rc >= 0 else 128 - rc
        with lock:
            try:
                _frame(sock, b"x", str(status).encode())
            except OSError:
                pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def agent_main(_args):
    os.makedirs(STATE_DIR, exist_ok=True)
    # All interfaces in a pod; tests run several agents on loopback addresses.
    with _Server((os.environ.get("MAD_K8S_AGENT_BIND", "0.0.0.0"), PORT), _Handler) as srv:
        srv.serve_forever()


def wait_peers_main(args):
    """Block until every node's agent answers (pod 0, before running the script)."""
    nodes = args or _job_nodes()
    for host in nodes:
        rc = _request(host, {"op": "ping"})
        if rc != 0:
            _die("wait-peers", f"agent on {host} answered {rc}")
    print(f"[mad-k8s] agents up on {len(nodes)} node(s): {','.join(nodes)}", flush=True)


def shutdown_peers_main(args):
    me = os.environ.get("MAD_K8S_SELF") or socket.gethostname()
    for host in args or _job_nodes():
        if host.split(".")[0] == me:
            continue
        try:
            _request(host, {"op": "shutdown"})
        except Exception as e:  # a peer already gone is not an error here
            sys.stderr.write(f"[mad-k8s] shutdown {host}: {e}\n")


# --------------------------------------------------------------------- srun

_SRUN_VALUE_OPTS = {
    "-N": "nodes", "--nodes": "nodes", "-n": "ntasks", "--ntasks": "ntasks",
    "-w": "nodelist", "--nodelist": "nodelist",
    "--ntasks-per-node": "ntasks_per_node",
    "-t": None, "--time": None, "-J": None, "--job-name": None,
    "--export": None, "--cpus-per-task": None, "-c": None, "--mpi": None,
    "--gpus": None, "--gpus-per-node": None, "--gres": None, "--jobid": None,
}
_SRUN_FLAGS = {"--overlap", "--exclusive", "-l", "--label", "-u", "--unbuffered", "-Q", "--quiet"}


def _parse_srun(argv):
    opts, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-"):
            break
        key, val = (a.split("=", 1) + [None])[:2] if a.startswith("--") else (a, None)
        if key in _SRUN_FLAGS:
            i += 1
            continue
        if key in _SRUN_VALUE_OPTS:
            if val is None:
                if i + 1 >= len(argv):
                    _die("srun", f"{key} needs a value")
                val = argv[i + 1]
                i += 1
            name = _SRUN_VALUE_OPTS[key]
            if name:
                opts[name] = val
            i += 1
            continue
        _die("srun", f"unsupported option {a!r} (supported: {', '.join(sorted(set(_SRUN_VALUE_OPTS) | _SRUN_FLAGS))})")
    cmd = argv[i:]
    if not cmd:
        _die("srun", "no command given")
    return opts, cmd


def srun_main(argv):
    opts, cmd = _parse_srun(argv)
    job_nodes = _job_nodes()
    if not job_nodes:
        _die("srun", "SLURM_JOB_NODELIST is empty: not inside a madengine k8s allocation")
    if "nodelist" in opts:
        targets = _expand_hostlist(opts["nodelist"])
        unknown = [t for t in targets if t not in job_nodes]
        if unknown:
            _die("srun", f"node(s) {','.join(unknown)} are not in this allocation ({','.join(job_nodes)})")
    else:
        targets = list(job_nodes)
    if "nodes" in opts:
        n = int(str(opts["nodes"]).split("-")[0])
        if "nodelist" in opts and n != len(targets):
            _die("srun", f"--nodes={n} but --nodelist names {len(targets)} node(s)")
        targets = targets[:n]
    # One task per node, the shape every launcher uses. More would need per-node
    # task placement this shim does not implement; say so rather than guess.
    if "ntasks_per_node" in opts and int(opts["ntasks_per_node"]) != 1:
        _die("srun", f"--ntasks-per-node={opts['ntasks_per_node']} is not supported (one task per node only)")
    if "ntasks" in opts and int(opts["ntasks"]) != len(targets):
        _die("srun", f"--ntasks={opts['ntasks']} on {len(targets)} node(s) is not supported (one task per node only)")
    # SLURM ranks a step's tasks in the allocation's node order, not --nodelist order.
    targets.sort(key=job_nodes.index)

    base_env = dict(os.environ)
    base_env.pop("HOSTNAME", None)
    results = [None] * len(targets)
    out_lock, err_lock = threading.Lock(), threading.Lock()

    class _Locked:
        def __init__(self, stream, lock):
            self.stream, self.lock = stream, lock

        def write(self, data):
            with self.lock:
                self.stream.write(data)

        def flush(self):
            with self.lock:
                self.stream.flush()

    out = _Locked(sys.stdout.buffer, out_lock)
    err = _Locked(sys.stderr.buffer, err_lock)

    def task(i, host):
        env = dict(base_env)
        env.update({
            "SLURM_PROCID": str(i), "SLURM_LOCALID": "0", "SLURM_NODEID": str(job_nodes.index(host)),
            "SLURMD_NODENAME": host, "SLURM_STEP_NUM_NODES": str(len(targets)),
            "SLURM_STEP_NUM_TASKS": str(len(targets)), "SLURM_STEP_NODELIST": ",".join(targets),
            "SLURM_NTASKS": str(len(targets)), "SLURM_NPROCS": str(len(targets)),
        })
        try:
            results[i] = _request(host, {"op": "run", "argv": cmd, "env": env, "cwd": os.getcwd()}, out, err)
        except Exception as e:
            err.write(f"srun: error: {host}: {e}\n".encode())
            results[i] = 1

    threads = [threading.Thread(target=task, args=(i, h), daemon=True) for i, h in enumerate(targets)]

    def on_signal(signum, _frame):
        # Closing the connections makes each agent stop its task (see _Handler._run).
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sys.exit(max(r if r is not None else 1 for r in results))


# ------------------------------------------------------------------- docker

_DOCKER_IGNORED_VALUE = {
    "--device", "--network", "--net", "--ipc", "--pid", "--group-add", "--cap-add",
    "--cap-drop", "--security-opt", "--shm-size", "--ulimit", "--gpus", "--runtime",
    "--memory", "-m", "--cpus", "--add-host", "--label", "-l", "--hostname", "-h",
    "--user", "-u", "--pull", "--restart", "--stop-signal", "--platform",
}
_DOCKER_IGNORED_FLAGS = {"--rm", "--privileged", "-i", "-t", "-it", "-ti", "--interactive",
                         "--tty", "--init", "--read-only"}


def _containers_dir():
    d = os.path.join(STATE_DIR, "containers")
    os.makedirs(d, exist_ok=True)
    return d


def _norm_image(ref):
    ref = (ref or "").strip()
    for prefix in ("docker.io/library/", "docker.io/", "library/"):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
    return ref if ":" in ref.rsplit("/", 1)[-1] or "@" in ref else ref + ":latest"


def _pod_image():
    return os.environ.get("MAD_K8S_IMAGE", "")


def _same_image(ref):
    return _norm_image(ref) == _norm_image(_pod_image())


def _image_id():
    return "sha256:" + hashlib.sha256(_norm_image(_pod_image()).encode()).hexdigest()


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _killpg(pid, sig, grace=10):
    try:
        os.killpg(pid, sig)
    except OSError:
        return
    deadline = time.time() + grace
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    if _alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def _tracked(name):
    path = os.path.join(_containers_dir(), name)
    try:
        return int(open(path).read().strip() or 0), path
    except (OSError, ValueError):
        return None, path


def _bind(src, dst):
    """Make dst show src, as `-v src:dst` would inside a container."""
    if not src.startswith("/") or not dst.startswith("/"):
        _die("docker", f"-v {src}:{dst}: only absolute host paths are supported (no named volumes)", 125)
    if not os.path.exists(src):
        os.makedirs(src, exist_ok=True)  # docker creates a missing bind source as a directory
    if os.path.realpath(dst) == os.path.realpath(src):
        return
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        # A bind mount shadows what the image has at dst; keep it, out of the way.
        os.rename(dst, f"{dst}.mad-k8s-shadowed-{os.getpid()}")
    os.makedirs(os.path.dirname(dst) or "/", exist_ok=True)
    os.symlink(src, dst)


def _base_env():
    path = os.path.join(STATE_DIR, "base_env.json")
    try:
        return json.load(open(path))
    except OSError:
        _die("docker", f"{path} missing: the pod was not started by madengine's k8s bootstrap", 125)


def _docker_run(argv):
    env, binds, name, workdir, entrypoint, detach, ignored = {}, [], None, None, None, False, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-"):
            break
        key, val = (a.split("=", 1) + [None])[:2] if a.startswith("--") else (a, None)

        def value():
            nonlocal i
            if val is not None:
                return val
            i += 1
            if i >= len(argv):
                _die("docker", f"run: {key} needs a value", 125)
            return argv[i]

        if key in ("-e", "--env"):
            kv = value()
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
            elif kv in os.environ:
                env[kv] = os.environ[kv]
        elif key == "--env-file":
            for line in open(value()):
                line = line.rstrip("\n")
                if line and not line.lstrip().startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        elif key in ("-v", "--volume"):
            parts = value().split(":")
            if len(parts) < 2:
                _die("docker", f"run: -v {':'.join(parts)} has no container path", 125)
            binds.append((parts[0], parts[1]))
        elif key == "--name":
            name = value()
        elif key in ("-w", "--workdir"):
            workdir = value()
        elif key == "--entrypoint":
            entrypoint = value()
        elif key in ("-d", "--detach"):
            detach = True
        elif key in _DOCKER_IGNORED_FLAGS:
            ignored.append(key)
        elif key in _DOCKER_IGNORED_VALUE:
            value()
            ignored.append(key)
        else:
            _die("docker", f"run: unsupported option {a!r}", 125)
        i += 1
    if i >= len(argv):
        _die("docker", "run: no image given", 125)
    image, cmd = argv[i], argv[i + 1:]
    if not _same_image(image):
        _die("docker", f"run: this pod runs {_pod_image()!r}; it cannot start a different image {image!r}", 125)
    if entrypoint:
        cmd = [entrypoint] + cmd
    if not cmd:
        _die("docker", "run: no command and no --entrypoint (the image's default command is not known here)", 125)
    if name:
        pid, _ = _tracked(name)
        if pid and _alive(pid):
            _die("docker", f"run: Conflict. The container name \"{name}\" is already in use", 125)
    for src, dst in binds:
        _bind(src, dst)
    full_env = dict(_base_env())
    full_env.update(env)
    full_env["HOSTNAME"] = socket.gethostname()
    cwd = workdir or full_env.get("MAD_K8S_IMAGE_WORKDIR") or "/"
    proc = subprocess.Popen(cmd, env=full_env, cwd=cwd, start_new_session=True)
    name = name or f"mad-k8s-{proc.pid}"
    _, path = _tracked(name)
    with open(path, "w") as f:
        f.write(str(proc.pid))
    if detach:
        print(name)
        return 0

    def forward(signum, _frame):
        _killpg(proc.pid, signum, grace=10)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = proc.wait()
    try:
        os.unlink(path)
    except OSError:
        pass
    return rc if rc >= 0 else 128 - rc


def docker_main(argv):
    if not argv:
        _die("docker", "no subcommand")
    sub, args = argv[0], argv[1:]
    if sub == "run":
        sys.exit(_docker_run(args))
    if sub == "stop":
        t, names = 10, []
        it = iter(args)
        for a in it:
            if a in ("-t", "--time"):
                t = int(next(it))
            elif a.startswith("--time=") or a.startswith("-t="):
                t = int(a.split("=", 1)[1])
            else:
                names.append(a)
        rc = 0
        for n in names:
            pid, path = _tracked(n)
            if pid is None:
                sys.stderr.write(f"Error response from daemon: No such container: {n}\n")
                rc = 1
                continue
            _killpg(pid, signal.SIGTERM, grace=t)
            print(n)
        sys.exit(rc)
    if sub == "rm":
        force = any(a in ("-f", "--force") for a in args)
        rc = 0
        for n in [a for a in args if not a.startswith("-")]:
            pid, path = _tracked(n)
            if pid is None:
                if not force:
                    sys.stderr.write(f"Error response from daemon: No such container: {n}\n")
                    rc = 1
                continue
            if _alive(pid):
                if not force:
                    sys.stderr.write(f"Error response from daemon: container {n} is running: stop it before removing\n")
                    rc = 1
                    continue
                _killpg(pid, signal.SIGKILL, grace=0)
            os.unlink(path)
            print(n)
        sys.exit(rc)
    if sub == "ps":
        for n in sorted(os.listdir(_containers_dir())):
            pid, _ = _tracked(n)
            if pid and _alive(pid):
                print(n)
        sys.exit(0)
    if sub == "pull":
        image = [a for a in args if not a.startswith("-")][-1:] or [""]
        if _same_image(image[0]):
            print(f"Status: Image is up to date for {image[0]}")
            sys.exit(0)
        _die("docker", f"pull: this pod runs {_pod_image()!r}; {image[0]!r} cannot be pulled into it", 1)
    if sub == "image" and args[:1] == ["inspect"]:
        rest = args[1:]
        fmt, refs, it = None, [], iter(rest)
        for a in it:
            if a in ("-f", "--format"):
                fmt = next(it)
            elif a.startswith("--format="):
                fmt = a.split("=", 1)[1]
            else:
                refs.append(a)
        if not refs or not all(_same_image(r) for r in refs):
            sys.stderr.write(f"Error: No such image: {' '.join(refs)}\n")
            sys.exit(1)
        if fmt is None:
            print(json.dumps([{"Id": _image_id(), "RepoTags": [_pod_image()]}]))
        elif fmt.strip() == "{{.Id}}":
            print(_image_id())
        else:
            _die("docker", f"image inspect: --format {fmt!r} is not supported (only {{{{.Id}}}})", 1)
        sys.exit(0)
    _die("docker", f"unsupported subcommand {' '.join(argv[:2])!r} (supported: run, stop, rm, ps, pull, image inspect)")


# ----------------------------------------------------------------- scontrol


def scontrol_main(argv):
    if argv[:2] in (["show", "hostnames"], ["show", "hostname"]):
        spec = argv[2] if len(argv) > 2 else os.environ.get("SLURM_JOB_NODELIST", "")
        for h in _expand_hostlist(spec):
            print(h)
        sys.exit(0)
    if argv[:2] == ["show", "topology"]:
        # No switch topology is known inside Kubernetes; scontrol says the same
        # when the cluster has no topology plugin.
        sys.stderr.write("scontrol: no topology information available\n")
        sys.exit(1)
    _die("scontrol", f"unsupported: scontrol {' '.join(argv)!r} (supported: show hostnames, show topology)")


def main():
    tool = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    if tool in ("mad_k8s.py", "mad-k8s"):
        if not args:
            _die("mad-k8s", "usage: mad_k8s.py {agent|wait-peers|shutdown-peers|srun|docker|scontrol} ...")
        tool, args = args[0], args[1:]
    handlers = {
        "agent": agent_main, "wait-peers": wait_peers_main, "shutdown-peers": shutdown_peers_main,
        "srun": srun_main, "docker": docker_main, "scontrol": scontrol_main,
    }
    if tool not in handlers:
        _die("mad-k8s", f"unknown tool {tool!r}")
    handlers[tool](args)


if __name__ == "__main__":
    main()

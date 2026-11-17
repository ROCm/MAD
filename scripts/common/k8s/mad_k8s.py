#!/usr/bin/env python3
"""The docker and scontrol stand-ins a slurm_multi card runs under on Kubernetes.

madengine runs one pod per node of the card (an Indexed Job with a headless
Service), every pod running the card's image, and each pod runs the card's script as
its own node (MAD_DISPATCH=k8s-indexed, scripts/common/dispatch/dispatch.sh): the
pods meet in a rendezvous directory on the shared results volume, and pod 0 gives
the verdict. Nothing fans out from one pod to the others. What the script still
expects of a SLURM node is stood in for here:

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
import signal
import socket
import subprocess
import sys
import time

STATE_DIR = os.environ.get("MAD_K8S_STATE", "/tmp/mad-k8s")


def _die(tool, msg, rc=2):
    sys.stderr.write(f"[mad-k8s {tool}] {msg}\n")
    sys.exit(rc)


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
            _die("mad-k8s", "usage: mad_k8s.py {docker|scontrol} ...")
        tool, args = args[0], args[1:]
    handlers = {"docker": docker_main, "scontrol": scontrol_main}
    if tool not in handlers:
        _die("mad-k8s", f"unknown tool {tool!r}")
    handlers[tool](args)


if __name__ == "__main__":
    main()

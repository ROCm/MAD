"""
ROCm version of a base image, read from its registry metadata without pulling the image.

The image config (a few KB, fetched next to the manifest) carries the labels, environment
and build history. ROCm images record their version there in one of a few ways, tried in
order of how directly they state it:
  labels      com.amd.rocm.version (ATOM), vllm.rocm_base.dependency.rocm (vLLM CI bases)
  build args  THE_ROCK_VERSION / ROCM_SDK_VERSION (TheRock SDK images: Primus, JAX, SGLang
              rocm10), then ROCM_VERSION as a RUN step used it (rocm/dev-ubuntu derived
              images) -- never an ARG line's default, which is a placeholder (vLLM's
              Dockerfile declares ROCM_VERSION=5.3)
  wheels      the +rocmX.Y.Z local version of the torch/jax wheels installed in the image
Public registries (Docker Hub, ghcr.io, ...) are read anonymously; private ones yield None.
"""
import collections
import functools
import json
import re
import urllib.error
import urllib.parse
import urllib.request

DOCKER_HUB = "registry-1.docker.io"
ROCM_LABELS = ("com.amd.rocm.version", "vllm.rocm_base.dependency.rocm")
TIMEOUT = 20

_VERSION = r"\d+\.\d+(?:\.\d+)?"
_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def parse_reference(image):
    """'rocm/pytorch:latest' -> ('registry-1.docker.io', 'rocm/pytorch', 'latest')."""
    name, ref = image, "latest"
    if "@" in name:
        name, ref = name.split("@", 1)
    if ":" in name.rsplit("/", 1)[-1]:
        name, tag = name.rsplit(":", 1)
        ref = ref if "@" in image else tag
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, repo = first, rest
    else:
        host, repo = DOCKER_HUB, name
    if host in ("docker.io", "index.docker.io"):
        host = DOCKER_HUB
    if host == DOCKER_HUB and "/" not in repo:
        repo = "library/" + repo
    return host, repo, ref


def _anonymous_token(challenge):
    """Bearer token for a 401's WWW-Authenticate challenge, or None when it is not a Bearer one."""
    if not challenge.lower().startswith("bearer "):
        return None
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = params.pop("realm", "")
    if not realm:
        return None
    with urllib.request.urlopen(realm + "?" + urllib.parse.urlencode(params), timeout=TIMEOUT) as r:
        body = json.load(r)
    return body.get("token") or body.get("access_token")


class _Repo:
    """One repository on one registry, authenticating on the first 401."""

    def __init__(self, host, repo):
        self.base = f"https://{host}/v2/{repo}"
        self.token = None

    def get(self, path, accept=None):
        try:
            return self._open(path, accept)
        except urllib.error.HTTPError as e:
            if e.code != 401 or self.token:
                raise
            self.token = _anonymous_token(e.headers.get("WWW-Authenticate", ""))
            if not self.token:
                raise
            return self._open(path, accept)

    def _open(self, path, accept):
        req = urllib.request.Request(f"{self.base}/{path}")
        if accept:
            req.add_header("Accept", accept)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)


def image_config(image):
    """The linux/amd64 image config of `image` (labels, env, history)."""
    host, repo, ref = parse_reference(image)
    session = _Repo(host, repo)
    manifest = session.get(f"manifests/{ref}", _ACCEPT)
    if "manifests" in manifest:
        entry = next(m for m in manifest["manifests"]
                     if m.get("platform", {}).get("architecture") == "amd64"
                     and m.get("platform", {}).get("os", "linux") == "linux")
        manifest = session.get(f"manifests/{entry['digest']}", _ACCEPT)
    return session.get(f"blobs/{manifest['config']['digest']}")


def rocm_from_config(config):
    """(version, evidence) from an image config, or None when it does not state one."""
    labels = (config.get("config") or {}).get("Labels") or {}
    for key in ROCM_LABELS:
        if re.fullmatch(_VERSION, labels.get(key) or ""):
            return labels[key], f"label {key}"
    steps = [h.get("created_by", "") for h in config.get("history") or []]
    steps = [s for s in steps if not re.match(r"(?:/bin/sh -c #\(nop\)\s+)?ARG\s", s)]
    text = "\n".join(steps + ((config.get("config") or {}).get("Env") or []))
    for var in ("THE_ROCK_VERSION", "ROCM_SDK_VERSION", "ROCM_VERSION"):
        # The lookbehind keeps BUILD_ROCM_VERSION and the like out.
        found = re.findall(rf"(?<![A-Z_]){var}=({_VERSION})(?![\d.])", text)
        if found:
            return found[-1], f"{var} build arg"
    wheels = collections.Counter(re.findall(rf"\+rocm({_VERSION})(?![\d.])", text))
    if wheels:
        return wheels.most_common(1)[0][0], "+rocm wheel version"
    return None


@functools.lru_cache(maxsize=None)
def rocm_version(image):
    """(version or None, evidence or the reason there is none) for a base image reference."""
    try:
        found = rocm_from_config(image_config(image))
    except urllib.error.HTTPError as e:
        return None, f"registry: HTTP {e.code}"
    except (OSError, ValueError, KeyError, StopIteration) as e:
        return None, f"registry: {type(e).__name__}"
    if found is None:
        return None, "registry: no ROCm version in labels, build args or wheels"
    version, evidence = found
    return version, f"registry {evidence}"

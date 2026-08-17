#!/usr/bin/env bash
# GPU-free static validator for a filled run manifest.
# Usage: bash validate_manifest.sh <run_manifest.json>
# Reads $MODEL_DIR (export it via `source mad.env`) to also resolve the
# dockerfile / run.sh paths. Read-only; prints a PASS/WARN/FAIL table and exits
# non-zero if any FAIL. No GPU, no docker, no network needed.
set -u

manifest="${1:-}"
if [ -z "$manifest" ]; then
  echo "usage: $0 <run_manifest.json>" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required for validate_manifest.sh" >&2
  exit 2
fi

exec python3 - "$manifest" <<'PY'
import json, os, re, sys

path = sys.argv[1]
model_dir = os.environ.get("MODEL_DIR", "")

passes = warns = fails = 0
def ok(m):
    global passes; passes += 1; print(f"  PASS  {m}")
def warn(m):
    global warns; warns += 1; print(f"  WARN  {m}")
def fail(m):
    global fails; fails += 1; print(f"  FAIL  {m}")

print(f"== mad-slurm-multinode manifest validation: {path} ==")

if not os.path.isfile(path):
    print(f"  FAIL  manifest file not found: {path}")
    print("== 0 PASS, 0 WARN, 1 FAIL ==")
    sys.exit(1)

raw = open(path, encoding="utf-8", errors="replace").read()

# 1) JSON validity
try:
    m = json.loads(raw)
    ok("valid JSON")
except Exception as e:
    print(f"  FAIL  invalid JSON: {e}")
    print("== 0 PASS, 0 WARN, 1 FAIL ==")
    sys.exit(1)

# 2) no unfilled placeholders
ph = re.findall(r'<FILL[^>]*>', raw)
if ph:
    fail(f"{len(ph)} unfilled placeholder(s) remain, e.g. {ph[0]!r}")
else:
    ok("no <FILL_...> placeholders left")

ctx   = (m.get("context") or {}).get("docker_env_vars") or {}
dep   = m.get("deployment_config") or {}
slurm = dep.get("slurm") or {}
dist  = dep.get("distributed") or {}
denv  = dep.get("env_vars") or {}

def is_ph(v):
    return isinstance(v, str) and ("<FILL" in v or v.strip() == "")

# 3) NCCL_IB_HCA present + non-empty in both env blocks, and equal
hca_c, hca_d = ctx.get("NCCL_IB_HCA"), denv.get("NCCL_IB_HCA")
if not hca_c or is_ph(hca_c):
    fail("context.docker_env_vars.NCCL_IB_HCA missing/empty/placeholder")
else:
    ok(f"NCCL_IB_HCA set ({hca_c})")
if not hca_d or is_ph(hca_d):
    fail("deployment_config.env_vars.NCCL_IB_HCA missing/empty/placeholder")
if hca_c and hca_d and not is_ph(hca_c) and not is_ph(hca_d) and hca_c != hca_d:
    fail(f"NCCL_IB_HCA differs between env blocks: {hca_c!r} vs {hca_d!r}")

# 4) network interface consistent across the three places it is set
ifaces = {
    "context.NCCL_SOCKET_IFNAME": ctx.get("NCCL_SOCKET_IFNAME"),
    "context.GLOO_SOCKET_IFNAME": ctx.get("GLOO_SOCKET_IFNAME"),
    "env_vars.NCCL_SOCKET_IFNAME": denv.get("NCCL_SOCKET_IFNAME"),
    "env_vars.GLOO_SOCKET_IFNAME": denv.get("GLOO_SOCKET_IFNAME"),
    "slurm.network_interface": slurm.get("network_interface"),
}
present = {k: v for k, v in ifaces.items() if v is not None}
vals = sorted({v for v in present.values() if v and not is_ph(v)})
if len(vals) > 1:
    fail(f"network interface mismatch across blocks: {present}")
elif vals:
    ok(f"network interface consistent ({vals[0]})")

# 5) node count consistency
nodes, nnodes = slurm.get("nodes"), dist.get("nnodes")
if nodes is not None and nnodes is not None:
    if nodes != nnodes:
        fail(f"slurm.nodes ({nodes}) != distributed.nnodes ({nnodes})")
    else:
        ok(f"slurm.nodes == distributed.nnodes ({nodes})")
nl = slurm.get("nodelist")
if isinstance(nl, str) and nl and not is_ph(nl):
    if "[" in nl:
        warn(f"nodelist uses a bracket range; cardinality not auto-checked ({nl})")
    elif nodes is not None:
        count = len([x for x in nl.split(",") if x.strip()])
        if count != nodes:
            fail(f"nodelist has {count} node(s) but slurm.nodes={nodes}")
        else:
            ok(f"nodelist cardinality matches slurm.nodes ({count})")

# 6) stray HF token in the manifest -> HF 401
if "MAD_SECRETS_HFTOKEN" in ctx or "MAD_SECRETS_HFTOKEN" in denv:
    fail("MAD_SECRETS_HFTOKEN is declared in the manifest -> HF 401; "
         "remove it (the token comes from mad.env)")
else:
    ok("no MAD_SECRETS_HFTOKEN key in manifest")

# 7) AINIC transport vars must be symmetric across both env blocks
for k in ("RCCL_AINIC_ROCE", "RDMAV_DRIVERS", "IBV_DRIVERS"):
    inc = (k in ctx) and not is_ph(ctx.get(k))
    ind = (k in denv) and not is_ph(denv.get(k))
    if inc != ind:
        fail(f"{k} set in only one env block (context={inc}, env_vars={ind}); "
             "transport vars must be in BOTH")

# 7b) A var the workload reads has to be in context.docker_env_vars: that is the block madengine
# turns into `docker -e`, while deployment_config.env_vars only reaches the SLURM launcher on the
# host. Putting a knob in the latter alone looks right in the manifest and silently does nothing
# inside the container (job 25802: SGLANG_USE_AITER=0 there, server still started with aiter on).
# Launcher-only plumbing is expected to be host-side, so it is not reported.
LAUNCHER_ONLY = ("SLURM_", "MAD_", "BARRIER_", "IP_SYNC_")
host_only = sorted(k for k in denv
                   if k not in ctx and not k.startswith(LAUNCHER_ONLY))
if host_only:
    warn("only in deployment_config.env_vars, so the container never sees them: "
         + ", ".join(host_only))
else:
    ok("every env_vars key also reaches the container via context.docker_env_vars")

# 8) dockerfile / run.sh resolve under MODEL_DIR (if available)
# "N/A (local image mode)" is not a path: madengine reads that literal as "this image
# is not mine to build", which also skips its Dockerfile-content staleness check, so a
# cached MAD_DOCKER_BUILDS tar survives an unrelated edit to the Dockerfile.
LOCAL_IMAGE_MODE = "N/A (local image mode)"
asset_paths = []
for v in (m.get("built_images") or {}).values():
    if not isinstance(v, dict) or not v.get("dockerfile"):
        continue
    if v["dockerfile"] == LOCAL_IMAGE_MODE:
        if v.get("local_image"):
            ok(f"dockerfile in local image mode, no build: {v.get('docker_image', '?')}")
        else:
            fail(f"{LOCAL_IMAGE_MODE} requires local_image: true")
        continue
    asset_paths.append(("dockerfile", v["dockerfile"]))
for v in (m.get("built_models") or {}).values():
    if isinstance(v, dict) and v.get("scripts"):
        asset_paths.append(("scripts", v["scripts"]))
if model_dir and os.path.isdir(model_dir):
    for kind, p in asset_paths:
        full = os.path.join(model_dir, p)
        if os.path.exists(full):
            ok(f"{kind} resolves under MODEL_DIR: {p}")
        else:
            fail(f"{kind} not found under MODEL_DIR: {p}")
else:
    warn(f"MODEL_DIR not set/usable ('{model_dir}') -> "
         "dockerfile/run.sh path checks skipped")

print(f"== validate_manifest: {passes} PASS, {warns} WARN, {fails} FAIL ==")
sys.exit(1 if fails else 0)
PY

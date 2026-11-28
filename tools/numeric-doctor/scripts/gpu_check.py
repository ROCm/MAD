#!/usr/bin/env python3
"""Which GPU is this, and has debug-numeric been tested on it?  (debug-numeric steps 0 and 1)

    gpu_check.py [--torch]

debug-numeric is built and tested on AMD Instinct MI300X and newer: gfx942 (MI300X, MI300A,
MI325X), gfx950 (MI350X, MI355X), gfx1250 (MI455X; one investigation, E48), and any GPU whose
marketing name is Instinct MI<n> with n >= 300.  Everything else -- older Instinct
(MI250X, MI210: gfx90a), Radeon, NVIDIA -- is untested: the capture, replay and kernel
tooling assume ROCm and GFX9-family ISA, and the skill is likely not to work there.

Probes, no torch needed (runs on any node; a login node reports NO_GPU):
  /sys/class/kfd/kfd/topology/nodes/*/properties   gfx_target_version per device
  amd-smi static / rocminfo                         marketing names
  nvidia-smi                                        NVIDIA devices
--torch additionally requires torch.cuda.is_available() (the step-1 <GPU_EXEC> check)
and reports torch's view of device 0.

Last line (with --torch and no usable torch GPU: TORCH_GPU=NO, exit 1):
  GPU=<name> ARCH=<gfx...> VERDICT=SUPPORTED   exit 0
  GPU=<name> ARCH=<...>    VERDICT=UNTESTED    exit 4   warn: untested, likely not to work
  GPU=none                 VERDICT=NO_GPU      exit 5   nothing visible here (login node)
"""
import argparse
import glob
import re
import shutil
import subprocess
import sys

TESTED_ARCHS = {
    "gfx942": "MI300X / MI300A / MI325X",
    "gfx950": "MI350X / MI355X",
    "gfx1250": "MI455X (one investigation, E48)",
}


def run(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except Exception:                                   # noqa: BLE001
        return ""


def arch_from_version(v):
    """gfx_target_version 90402 -> gfx942, 90010 -> gfx90a, 120500 -> gfx1250."""
    v = int(v)
    return "gfx%d%x%x" % (v // 10000, (v // 100) % 100, v % 100)


def kfd_archs():
    out = []
    for p in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            props = dict(l.split(None, 1) for l in open(p).read().splitlines() if " " in l)
        except OSError:
            continue
        v = props.get("gfx_target_version", "0").strip()
        if v.isdigit() and int(v) > 0:                  # 0 = a CPU node
            out.append(arch_from_version(v))
    return out


def amd_names():
    names = []
    if shutil.which("amd-smi"):
        names = re.findall(r"MARKET_NAME:\s*(.+)", run(["amd-smi", "static", "--asic"]))
    if not names and shutil.which("rocminfo"):
        txt = run(["rocminfo"])
        names = [m for m in re.findall(r"Marketing Name:\s*(.+)", txt) if "CPU" not in m
                 and "EPYC" not in m and "Processor" not in m]
    return [n.strip() for n in names]


def amd_archs_rocminfo():
    return re.findall(r"Name:\s*(gfx[0-9a-f]+)", run(["rocminfo"])) if shutil.which("rocminfo") else []


def nvidia_names():
    if not shutil.which("nvidia-smi"):
        return []
    return [l.strip() for l in run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
            if l.strip()]


def instinct_number(name):
    m = re.search(r"\bMI\s?(\d{3})", name or "", re.I)
    return int(m.group(1)) if m else None


def verdict(arch, name):
    if arch in TESTED_ARCHS:
        return "SUPPORTED"
    n = instinct_number(name)
    if n is not None and n >= 300 and "nvidia" not in (name or "").lower():
        return "SUPPORTED"
    return "UNTESTED"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--torch", action="store_true",
                    help="also require torch.cuda.is_available() (the <GPU_EXEC> check)")
    a = ap.parse_args()

    archs = kfd_archs() or amd_archs_rocminfo()
    names = amd_names()
    nv = nvidia_names()
    tname = tarch = None
    torch_ok = not a.torch
    if a.torch:
        try:
            import torch
            ok = torch.cuda.is_available()
            print("torch %s (%s) cuda.is_available=%s devices=%d"
                  % (torch.__version__, "hip " + str(torch.version.hip) if torch.version.hip
                     else "cuda " + str(torch.version.cuda), ok, torch.cuda.device_count() if ok else 0))
            torch_ok = ok
            if ok:
                p = torch.cuda.get_device_properties(0)
                tname = p.name
                tarch = (getattr(p, "gcnArchName", "") or "").split(":")[0] or None
                print("device0 %s arch=%s CUs=%d memory=%d GiB"
                      % (p.name, tarch or "-", p.multi_processor_count, p.total_memory >> 30))
            else:
                print("FAIL: torch sees no GPU under this <GPU_EXEC>", file=sys.stderr)
        except ImportError:
            print("FAIL: no torch here (run under <GPU_EXEC>)", file=sys.stderr)

    if nv and not archs:
        name = nv[0]
        print("devices: %d NVIDIA (%s)" % (len(nv), ", ".join(sorted(set(nv)))))
        print("GPU=%s ARCH=nvidia VERDICT=UNTESTED" % name.replace(" ", "_"))
        return 4
    arch = tarch or (archs[0] if archs else None)
    name = tname or (names[0] if names else None)
    if not arch and not name:
        print("GPU=none VERDICT=NO_GPU")
        return 5
    if len(set(archs)) > 1:
        print("devices: mixed archs %s -- judged on device 0" % sorted(set(archs)))
    print("devices: %d x %s (%s)" % (max(len(archs), len(names), 1), arch or "?", name or "?"))
    v = verdict(arch, name)
    print("GPU=%s ARCH=%s VERDICT=%s" % ((name or "?").replace(" ", "_"), arch or "?", v))
    if not torch_ok:
        print("TORCH_GPU=NO -- <GPU_EXEC> is not usable yet; fix it before any job")
        return 1
    return 0 if v == "SUPPORTED" else 4


if __name__ == "__main__":
    sys.exit(main())

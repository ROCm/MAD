"""Host SKU for JAX MAD discovery.

madengine skip_gpu_arch only compares ISA (gfx942 vs gfx950). MI300X and MI325X
are both gfx942, so that filter cannot pick between Primus MI300X/ and MI325X/
config dirs. Detect the product name at discovery time instead.
"""
import os
import re
import shutil
import subprocess
import sys

# Config directory -> ISA that should SKIP this directory (cross-family only).
ARCH_SKIP_GPU = {
    "MI300X": "gfx950",
    "MI325X": "gfx950",
    "MI355X": "gfx942",
    "MI350X": "gfx942",
}

# Host product -> Primus config directories to discover.
# MI350X has no own MaxText dir; Primus documents using the MI355X recipes.
SKU_CONFIG_DIRS = {
    "MI300X": frozenset({"MI300X"}),
    "MI325X": frozenset({"MI325X"}),
    "MI355X": frozenset({"MI355X"}),
    "MI350X": frozenset({"MI355X"}),
}

_SKU_MARKERS = ("MI355X", "MI350X", "MI325X", "MI300X")


def detect_host_sku():
    """Return MI300X / MI325X / MI355X / MI350X, or '' if unknown.

    ``JAX_HOST_DEVICE=all`` disables SKU filtering. Any other value is used as
    the SKU (for example ``MI325X`` on a box where rocminfo is unavailable).
    """
    forced = os.environ.get("JAX_HOST_DEVICE", "").strip()
    if forced.lower() in ("all", "*"):
        return "all"
    if forced:
        return forced

    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        return ""
    try:
        out = subprocess.check_output([rocminfo], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""

    marketing = ""
    m = re.search(r"Marketing Name:\s*(.+)", out)
    if m:
        marketing = m.group(1).upper().replace(" ", "")
    blob = marketing or out.upper().replace(" ", "")
    for sku in _SKU_MARKERS:
        if sku in blob:
            return sku

    if re.search(r"\bgfx950\b", out, re.I):
        return "MI355X"
    if re.search(r"\bgfx942\b", out, re.I):
        print(
            "WARNING: rocminfo reports gfx942 but no MI300X/MI325X marketing name; "
            "treating as MI300X. Set JAX_HOST_DEVICE=MI325X if this is an MI325.",
            file=sys.stderr,
        )
        return "MI300X"
    return ""


def config_dir_allowed(device_dir, sku=None):
    """Whether ``examples/.../configs/<device_dir>`` should be discovered."""
    if sku is None:
        sku = detect_host_sku()
    if not sku or sku == "all":
        return True
    allowed = SKU_CONFIG_DIRS.get(sku)
    if allowed is None:
        print(
            "WARNING: JAX_HOST_DEVICE=%s is not one of %s; discovering every device directory."
            % (sku, ", ".join(sorted(SKU_CONFIG_DIRS))),
            file=sys.stderr,
        )
        return True
    return device_dir in allowed


def log_sku_filter(sku=None):
    if sku is None:
        sku = detect_host_sku()
    if not sku:
        print(
            "WARNING: could not detect GPU product (MI300X vs MI325X share gfx942). "
            "Discovering every config directory; set JAX_HOST_DEVICE=MI325X (or MI300X) "
            "to avoid duplicate jobs.",
            file=sys.stderr,
        )
        return
    if sku == "all":
        print("JAX_HOST_DEVICE=all: discovering every device directory.", file=sys.stderr)
        return
    dirs = ",".join(sorted(SKU_CONFIG_DIRS.get(sku, {sku})))
    print(
        "host GPU product %s: discovering Primus configs under %s only "
        "(JAX_HOST_DEVICE=all to include every directory)." % (sku, dirs),
        file=sys.stderr,
    )

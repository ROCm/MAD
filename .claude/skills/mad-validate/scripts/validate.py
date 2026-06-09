#!/usr/bin/env python3
"""GPU-free static validation of MAD model definitions.

Lints models.json and the files each entry points at. Does NOT build or run
anything. Usage:  python3 validate.py [tag-or-model | all]

Errors break a run (non-zero exit). Warnings are missing MAD-convention
metadata (madengine itself defaults these — only `name` is structurally
required per its Model dataclass — so they do not fail the build).
"""
import json, os, subprocess, sys, glob

sel = sys.argv[1] if len(sys.argv) > 1 else "all"

# Resolve the repo root so the script works regardless of cwd.
try:
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    os.chdir(repo_root)
except Exception:
    pass  # fall back to cwd; works when already at repo root

models = json.load(open("models.json"))


def selected(m):
    return sel in ("all", "") or sel == m.get("name") or sel in (m.get("tags") or [])


seen, errors, warns = {}, [], []
for m in models:
    n = m.get("name", "<no-name>")
    if not selected(m):
        continue
    # --- errors (break a run) ---
    if "name" not in m:
        errors.append(f"{n}: missing 'name'")
    if n in seen:
        errors.append(f"{n}: duplicate name")
    seen[n] = True
    df = (m.get("dockerfile", "") or "") + ".ubuntu.amd.Dockerfile"
    if not m.get("dockerfile"):
        errors.append(f"{n}: no dockerfile field")
    elif not os.path.isfile(df):
        errors.append(f"{n}: dockerfile not found: {df}")
    else:
        first = open(df).readline().strip()
        if not first.startswith("# CONTEXT") or "AMD" not in first:
            errors.append(f"{n}: dockerfile missing CONTEXT header: {df}")
    sp = m.get("scripts", "")
    if not sp:
        errors.append(f"{n}: no scripts field")
    elif not os.path.exists(sp):
        errors.append(f"{n}: scripts path not found: {sp}")
    has_mr = bool(m.get("multiple_results"))
    emits = False
    if sp:
        sh_files = [sp] if sp.endswith(".sh") else glob.glob(os.path.join(sp, "**", "*.sh"), recursive=True)
        for sh in sh_files:
            if os.path.isfile(sh) and "performance:" in open(sh, errors="ignore").read():
                emits = True
                break
    if not (has_mr or emits):
        errors.append(f"{n}: no output contract (no 'performance:' line and no multiple_results)")
    # --- warnings (convention metadata) ---
    for f in ("url", "owner", "training_precision", "tags", "n_gpus"):
        if not m.get(f):
            warns.append(f"{n}: missing convention field '{f}'")

checked = sum(1 for m in models if selected(m))
print(f"Checked {checked} model(s). {len(errors)} error(s), {len(warns)} warning(s).")
if errors:
    print("\nERRORS:")
    for e in errors:
        print("  -", e)
if warns:
    print("\nwarnings:")
    for w in warns[:40]:
        print("  -", w)
    if len(warns) > 40:
        print(f"  ... and {len(warns) - 40} more")
sys.exit(1 if errors else 0)

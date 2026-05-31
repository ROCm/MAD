---
description: Statically validate MAD model entries (no GPU) — JSON, paths, Dockerfile header, output contract
argument-hint: [tag-or-model | all]
---

Statically validate MAD model definitions for `${ARGUMENTS:-all}`. This is a
GPU-free lint of `models.json` and the files it points at — it does NOT build or
run anything.

Run the checks below (use `python3`; do not install packages). Scope to the
named tag/model if one is given, otherwise check every entry.

Two severities. **Errors** break a run; **warnings** are missing MAD-convention
metadata (madengine itself defaults these — only `name` is structurally
required, per its `Model` dataclass — so don't fail the build on them).

Errors:
1. **JSON parses**: `python3 -m json.tool models.json` succeeds.
2. **Has `name`** and **names are unique** (no two entries share one).
3. **Dockerfile exists**: `<dockerfile>.ubuntu.amd.Dockerfile` is a real file,
   and its first line is the context header
   `# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}`.
4. **Script exists**: the `scripts` path exists (a `run.sh` file or a dir).
5. **Output contract** — exactly one of:
   - the run script contains a line that echoes `performance: ...`, OR
   - the entry sets `multiple_results` to a CSV filename.
   Flag any entry that satisfies neither (madengine would record no perf value).

Warnings (convention metadata, non-fatal):
6. Missing any of `url`, `owner`, `training_precision`, `tags`, `n_gpus`.

Suggested one-shot checker:

```bash
python3 - "${ARGUMENTS:-all}" <<'PY'
import json, os, sys, glob
sel = sys.argv[1] if len(sys.argv) > 1 else "all"
models = json.load(open("models.json"))
def selected(m):
    return sel in ("all","") or sel == m.get("name") or sel in (m.get("tags") or [])
seen, errors, warns = {}, [], []
for m in models:
    n = m.get("name","<no-name>")
    if not selected(m):
        continue
    # --- errors (break a run) ---
    if "name" not in m: errors.append(f"{n}: missing 'name'")
    if n in seen: errors.append(f"{n}: duplicate name")
    seen[n] = True
    df = (m.get("dockerfile","") or "") + ".ubuntu.amd.Dockerfile"
    if not m.get("dockerfile"):
        errors.append(f"{n}: no dockerfile field")
    elif not os.path.isfile(df):
        errors.append(f"{n}: dockerfile not found: {df}")
    else:
        first = open(df).readline().strip()
        if not first.startswith("# CONTEXT") or "AMD" not in first:
            errors.append(f"{n}: dockerfile missing CONTEXT header: {df}")
    sp = m.get("scripts","")
    if not sp:
        errors.append(f"{n}: no scripts field")
    elif not os.path.exists(sp):
        errors.append(f"{n}: scripts path not found: {sp}")
    has_mr = bool(m.get("multiple_results"))
    emits = False
    if sp:
        sh_files = [sp] if sp.endswith(".sh") else glob.glob(os.path.join(sp,"**","*.sh"), recursive=True)
        for sh in sh_files:
            if os.path.isfile(sh) and "performance:" in open(sh, errors="ignore").read():
                emits = True; break
    if not (has_mr or emits):
        errors.append(f"{n}: no output contract (no 'performance:' line and no multiple_results)")
    # --- warnings (convention metadata) ---
    for f in ("url","owner","training_precision","tags","n_gpus"):
        if not m.get(f):
            warns.append(f"{n}: missing convention field '{f}'")
checked = sum(1 for m in models if selected(m))
print(f"Checked {checked} model(s). {len(errors)} error(s), {len(warns)} warning(s).")
if errors:
    print("\nERRORS:")
    for e in errors: print("  -", e)
if warns:
    print("\nwarnings:")
    for w in warns[:40]: print("  -", w)
    if len(warns) > 40: print(f"  ... and {len(warns)-40} more")
sys.exit(1 if errors else 0)
PY
```

Report a compact pass/fail summary. If anything fails, list each problem with the
model name and how to fix it. This is the right check to run after `/mad-add-model`
and before any GPU run.

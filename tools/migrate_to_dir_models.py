#!/usr/bin/env python3
"""
Migrate MAD root models.json to per-directory models.json files.

For each model in the root models.json, this script:
  1. Derives the script directory from the model's "scripts" field.
  2. Translates absolute-from-root paths to paths relative to that script directory.
  3. Writes a models.json inside scripts/<dir>/ containing all models for that dir.
  4. Rewrites the root models.json to remove migrated entries (retaining only those
     whose script dir already has a get_models_json.py, which cannot coexist with
     a models.json per the madengine discovery rules).

Path translation rules (all root paths start with their top-level prefix):
  dockerfile:    "docker/X"           -> "../../docker/X"
  scripts:       "scripts/<dir>"      -> "."
                 "scripts/<dir>/"     -> "."
                 "scripts/<dir>/f.sh" -> "f.sh"
                 "scripts/<other>/…"  -> "../../scripts/<other>/…"
  dockercontext: kept as-is (project-root-relative) — the engine does NOT normalize it

Usage (run from MAD repository root):
    python3 tools/migrate_to_dir_models.py [--dry-run]

Options:
    --dry-run   Print what would be written without writing any files.
"""

import argparse
import json
import os
import sys


def extract_script_dir(scripts_field: str) -> str:
    """Return the first-level subdirectory under scripts/ for the given scripts path."""
    # Normalise trailing slashes and split
    parts = scripts_field.rstrip("/").split("/")
    # Expected: ["scripts", "<dir>", ...]
    if len(parts) < 2 or parts[0] != "scripts":
        raise ValueError(f"Unexpected scripts path format: {scripts_field!r}")
    return parts[1]


def translate_path(root_path: str, script_dir: str) -> str:
    """Translate a project-root-relative path to be relative to scripts/<script_dir>/.

    The engine resolves dir-specific paths via:
        os.path.normpath(os.path.join("scripts", script_dir, translated_path))
    So we need:
        translated_path such that normpath("scripts/<dir>/" + translated) == root_path

    For a root path "scripts/<dir>/..." -> the relative part after stripping that prefix.
    For "scripts/<other>/..." -> "../../scripts/<other>/..."
    For "docker/..."          -> "../../docker/..."
    """
    root_path_stripped = root_path.rstrip("/")

    # Path sits inside this script dir
    prefix_bare = f"scripts/{script_dir}"
    prefix_slash = f"scripts/{script_dir}/"

    if root_path_stripped == prefix_bare or root_path == prefix_slash:
        # Points to the directory itself — use "." which normalises to "scripts/<dir>"
        return "."

    if root_path.startswith(prefix_slash):
        # Points to a file/subpath inside this dir — strip the prefix
        return root_path[len(prefix_slash):]

    # Path is in a different top-level directory (docker/ or scripts/<other>/)
    # Go up two levels from scripts/<dir>/ to reach the project root
    return "../../" + root_path_stripped


def build_dir_entry(model: dict, script_dir: str) -> dict:
    """Return a new model dict with paths translated for a dir-specific models.json."""
    entry = {}

    # name: keep as-is; the engine prepends "<script_dir>/" automatically
    entry["name"] = model["name"]

    # Required fields with path translation
    entry["dockerfile"] = translate_path(model["dockerfile"], script_dir)
    entry["scripts"] = translate_path(model["scripts"], script_dir)

    # dockercontext is used directly by DockerBuilder (not normalized by the engine),
    # so keep the original project-root-relative path unchanged.
    if "dockercontext" in model:
        entry["dockercontext"] = model["dockercontext"]

    # Copy all non-path fields in their original order
    skip_keys = {"name", "dockerfile", "scripts", "dockercontext"}
    for key, value in model.items():
        if key not in skip_keys:
            entry[key] = value

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args()

    # Always run from MAD repository root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_json_path = os.path.join(root_dir, "models.json")
    scripts_dir = os.path.join(root_dir, "scripts")

    print(f"Reading {models_json_path} ...")
    with open(models_json_path) as f:
        all_models = json.load(f)

    print(f"Total models: {len(all_models)}")

    # Find dirs that already have get_models_json.py (cannot add models.json alongside)
    dirs_with_dynamic = set()
    for dirname in os.listdir(scripts_dir):
        dir_path = os.path.join(scripts_dir, dirname)
        if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "get_models_json.py")):
            dirs_with_dynamic.add(dirname)

    if dirs_with_dynamic:
        print(f"Dirs with get_models_json.py (will be skipped): {sorted(dirs_with_dynamic)}")

    # Group models by their script directory
    by_dir: dict[str, list[dict]] = {}
    skipped_models: list[dict] = []

    for model in all_models:
        try:
            script_dir = extract_script_dir(model["scripts"])
        except ValueError as e:
            print(f"WARNING: {e} — leaving model {model['name']!r} in root models.json", file=sys.stderr)
            skipped_models.append(model)
            continue

        if script_dir in dirs_with_dynamic:
            print(f"  Skipping {model['name']!r}: dir {script_dir!r} has get_models_json.py")
            skipped_models.append(model)
            continue

        by_dir.setdefault(script_dir, []).append(model)

    print(f"\nDirectories to create models.json in: {len(by_dir)}")
    print(f"Models remaining in root models.json: {len(skipped_models)}")

    # Write per-directory models.json files
    total_written = 0
    for script_dir, models in sorted(by_dir.items()):
        dir_path = os.path.join(scripts_dir, script_dir)
        out_path = os.path.join(dir_path, "models.json")

        entries = [build_dir_entry(m, script_dir) for m in models]

        if args.dry_run:
            print(f"\n[DRY RUN] Would write {out_path} ({len(entries)} models):")
            print(json.dumps(entries[:2], indent=4))
            if len(entries) > 2:
                print(f"  ... ({len(entries) - 2} more entries)")
        else:
            if not os.path.isdir(dir_path):
                print(f"WARNING: scripts dir does not exist: {dir_path}", file=sys.stderr)
                skipped_models.extend(models)
                continue

            if os.path.exists(out_path):
                print(f"WARNING: {out_path} already exists — overwriting", file=sys.stderr)

            with open(out_path, "w") as f:
                json.dump(entries, f, indent=4)
                f.write("\n")

            print(f"  Wrote {out_path} ({len(entries)} models)")

        total_written += len(entries)

    # Rewrite root models.json with only skipped models
    new_root_path = models_json_path
    if args.dry_run:
        print(f"\n[DRY RUN] Would rewrite {new_root_path} with {len(skipped_models)} remaining models")
    else:
        with open(new_root_path, "w") as f:
            json.dump(skipped_models, f, indent=4)
            f.write("\n")
        print(f"\nRewritten {new_root_path} with {len(skipped_models)} remaining models")

    print(f"\nDone. Models written to per-dir files: {total_written}")
    if args.dry_run:
        print("(dry run — no files were modified)")


if __name__ == "__main__":
    main()

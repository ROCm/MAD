#!/usr/bin/env python3
"""
Build the MAD recipes site and status report (GitHub Pages), or freeze a release snapshot.

Site layout written to --out (develop at the root, each frozen release under <version>/):
  index.html                    Overview: one card per section and changes since the last release
  models/index.html             Model support catalog (table and card views, filters in the URL)
  models/<framework>/<model>/   one recipe page per model: madengine and plain Docker commands per GPU
  rocm/ roadmap/                ROCm support matrix, oncoming models
  models.csv, search.js         config-level export and the top-bar search index
  releases/index.html           every release (site root only)
  <version>/...                 the same tree for dashboard/releases/<version>.json; recipes pin its commit

Usage (from anywhere; needs madengine, pyyaml, jinja2):
    python3 tools/dashboard/build.py --out _site
    python3 tools/dashboard/build.py --out _site_internal --internal   # adds dashboard/roadmap.internal.yaml
    python3 tools/dashboard/build.py --freeze v26.10                   # writes dashboard/releases/v26.10.json

Options:
    --strict    fail when the snapshot has warnings (unknown ROCm version, missing Primus checkout, ...)
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402
import icons  # noqa: E402
import recipes  # noqa: E402

DASHBOARD_DIR = os.path.join(collect.REPO_ROOT, "dashboard")
RELEASES_DIR = os.path.join(DASHBOARD_DIR, "releases")
TEMPLATES_DIR = os.path.join(DASHBOARD_DIR, "templates")


def load_releases():
    """Frozen snapshots, oldest first."""
    releases = []
    for path in glob.glob(os.path.join(RELEASES_DIR, "v*.json")):
        with open(path) as f:
            releases.append(json.load(f))
    return sorted(releases, key=lambda s: collect.version_key(s["version"]))


def upgrade(snapshot):
    """Fill in fields newer schemas added, so frozen releases keep rendering."""
    for row in snapshot["models"]:
        row.setdefault("deployment", collect.deployment_from_row(row))
        row.setdefault("variant", "")
    return snapshot


def _variant_label(row, family_rows):
    """The card's own label (one_page registries), else precision plus whatever tags set it apart."""
    if row["variant"]:
        return row["variant"]
    shared = set.intersection(*(set(r["suites"]) for r in family_rows))
    parts = [row["precision"].upper()] if row["precision"] else []
    parts += [s for s in row["suites"] if s not in shared]
    return " · ".join(parts) or "Default"


def sku_label(skus, all_skus):
    if list(skus) == list(all_skus):
        return f"{all_skus[0]}–{all_skus[-1]}"
    return " · ".join(skus)


def scale_label(node_counts, slurm, opt_in):
    """'1 node', '2 nodes · SLURM', '1–4 nodes · SLURM', '2+ nodes · SLURM'."""
    lo, hi = node_counts[0], node_counts[-1]
    text = f"{lo}+ nodes" if opt_in else recipes.nodes_label(lo) if lo == hi else f"{lo}–{hi} nodes"
    return text + (" · SLURM" if slurm else "")


def link_scale(pages):
    """Link the pages of one model under different frameworks (vLLM and vLLM disaggregated P/D, ...)
    so each shows where else the model runs and on how many nodes; add the filter tokens."""
    groups = {}
    for p in pages:
        groups.setdefault((p["workload"], re.sub(r"[^a-z0-9]", "", p["name"].lower())), []).append(p)
    for group in groups.values():
        multi = any(p["node_counts"][-1] > 1 for p in group)
        single = any(p["node_counts"][0] == 1 for p in group)
        ordered = sorted(group, key=lambda p: (p["node_counts"][0], p["node_counts"][-1], p["framework"]))
        for p in group:
            p["scale_links"] = [{"page": o["page"], "framework": o["framework"], "label": o["scale_label"],
                                 "current": o is p} for o in ordered] if len(group) > 1 and multi else []
            # A single-node page's pointer to the same model on more nodes, for the catalog.
            p["scale_out"] = [o for o in ordered if o["node_counts"][-1] > 1] if p["node_counts"][-1] == 1 else []
            if single and multi:
                p["scale"].append("both")


def catalog(snapshot):
    """Recipe pages: rows grouped by (framework, model family); variants keyed by GPU product.

    One page per model family (e.g. Llama-3.3-70B-Instruct under vLLM) whose variants are its
    precisions / transports; Primus configs that exist once per GPU product merge into one
    variant whose GPU picker selects the product-specific card.
    """
    families = {}
    for row in snapshot["models"]:
        # 'gpt-oss-120b' and AMD's 'gpt-oss120b-w-mxfp4-a-fp8' are one model: ignore punctuation.
        framework_slug, family_slug = row["page"].split("/", 1)
        key = (framework_slug, re.sub(r"[^a-z0-9]", "", family_slug))
        families.setdefault(key, []).append(row)

    pages, frameworks = [], {}
    for rows in families.values():
        page = rows[0]["page"]
        for row in rows:
            row["page"] = page
        # Node count is part of the key: a sweep's 1- and 2-node cards are separate choices.
        by_label = {}
        for row in rows:
            by_label.setdefault((_variant_label(row, rows), row["nodes"]), []).append(row)
        variants = []
        for (label, _), vrows in by_label.items():
            by_sku = {}
            for row in vrows:
                for sku in row["skus"]:
                    by_sku.setdefault(sku, row)
            # Two cards on the same product under one label: give each its own variant.
            if sum(len(r["skus"]) for r in vrows) > len(by_sku):
                for row in vrows:
                    variants.append({"label": f"{label} · {row['name'].rsplit('/', 1)[-1]}", "rows": [row],
                                     "by_sku": {sku: row for sku in row["skus"]}})
            else:
                variants.append({"label": label, "rows": vrows, "by_sku": by_sku})
        for i, v in enumerate(variants):
            v["index"] = str(i)
            v["hf_repo"] = v["rows"][0]["hf_repo"]
            v["suites"] = sorted({s for r in v["rows"] for s in r["suites"]})
            v["nodes"] = v["rows"][0]["nodes"]
            v["clusters"] = recipes.clusters(v["rows"][0])
            v["unavailable"] = recipes.unavailable(v["rows"][0])
        first = rows[0]
        repos = [r["hf_repo"] for r in rows if r["hf_repo"]]
        # Prefer the upstream repo (meta-llama/...) over AMD's quantized re-uploads for the heading.
        repo = next((r for r in repos if not r.startswith("amd/")), repos[0] if repos else None)
        skus = [s for s in snapshot["skus"] if any(s in r["skus"] for r in rows)]
        n_gpus = sorted({r["n_gpus"] for r in rows})
        node_counts = sorted({r["nodes"] for r in rows})
        slurm = any(r["deployment"]["scheduler"] == "slurm" for r in rows)
        k8s = any(r["deployment"]["k8s"] == "manual" for r in rows)
        opt_in = all(r["deployment"]["status"] == "opt-in" for r in rows)
        # Filter tokens for the catalog's Scale filter; link_scale() adds "both".
        scale = [t for t, on in (("single", node_counts[0] == 1), ("multi", node_counts[-1] > 1),
                                 ("slurm", slurm), ("k8s", k8s), ("opt-in", opt_in)) if on]
        family = {
            "page": page,
            "fw_slug": page.split("/", 1)[0],
            "name": first["family"],
            "org": repo.split("/", 1)[0] if repo else "",
            "hf_repo": repo,
            "framework": first["framework"],
            "workload": first["workload"],
            "readme": first["readme"],
            "variants": variants,
            "precisions": sorted({r["precision"].upper() for r in rows if r["precision"]}),
            "suites": sorted({s for r in rows for s in r["suites"]}),
            "skus": skus,
            # GPU/precision pairs one card offers, so the two filters agree on a single configuration.
            "sku_precisions": sorted({f"{s}:{r['precision'].upper()}" for r in rows if r["precision"] for s in r["skus"]}),
            "sku_label": sku_label(skus, snapshot["skus"]),
            "gpus": ("8 GPU" if n_gpus == ["all"] else " / ".join(n_gpus) + " GPU") + ("/node" if node_counts[-1] > 1 else ""),
            "nodes": node_counts[-1],
            "node_counts": node_counts,
            "opt_in": opt_in,
            "k8s": k8s,
            "scale": scale,
            "scale_label": scale_label(node_counts, slurm, opt_in),
            "rocm": sorted({r["rocm"] for r in rows}),
            "support": support(variants, snapshot["skus"]),
        }
        family["search"] = " ".join([family["name"], repo or "", family["framework"], *family["precisions"],
                                     *family["suites"], *(r["name"] for r in rows)]).lower()
        pages.append(family)
        fw = frameworks.setdefault(family["fw_slug"], {
            "slug": family["fw_slug"], "name": family["framework"], "workload": family["workload"],
            "families": [], "models": 0, "rocm": [], "sku_families": {sku: 0 for sku in snapshot["skus"]}})
        fw["families"].append(family)
        fw["models"] += len(rows)
        fw["rocm"] = sorted(set(fw["rocm"]) | set(family["rocm"]))
        for sku in skus:
            fw["sku_families"][sku] += 1
    link_scale(pages)
    return pages, list(frameworks.values())


def support(variants, all_skus):
    """Per GPU product: does every variant run there ('all'), some ('some', naming them), or none."""
    cells = {}
    for sku in all_skus:
        labels = [v["label"] for v in variants if sku in v["by_sku"]]
        state = "none" if not labels else "all" if len(labels) == len(variants) else "some"
        cells[sku] = {"state": state, "labels": labels}
    return cells


def summarize(snapshot, pages, frameworks):
    """Catalog-wide numbers for the Overview and the Models filters."""
    return {
        "families": pages,
        "frameworks": frameworks,
        "precisions": sorted({p for f in pages for p in f["precisions"]}),
        "sku_families": {sku: sum(sku in f["skus"] for f in pages) for sku in snapshot["skus"]},
        "fw_slugs": {fw["name"]: fw["slug"] for fw in frameworks},
        "scale": {t: sum(t in f["scale"] for f in pages) for t in ("multi", "both", "slurm", "k8s", "opt-in")},
        "rocm_versions": sorted({i["rocm"] for i in snapshot["images"]} - {"unknown", "latest"}, key=_version_order),
    }


def _version_order(version):
    """'6.3.0' < '7.2' < '7.2.3' < '10'; non-numeric labels ('TheRock preview') last."""
    parts = re.findall(r"\d+", version)
    return (0, tuple(int(p) for p in parts)) if re.fullmatch(r"[\d.]+", version) else (1, version)


def search_index(pages):
    entries = [{"n": p["name"], "f": p["framework"], "p": p["page"], "k": p["search"]} for p in pages]
    return "window.MAD_INDEX = " + json.dumps(entries, separators=(",", ":")) + ";\n"


def default_sku(family):
    """Newest GPU product the first variant runs on."""
    return [s for s in family["skus"] if s in family["variants"][0]["by_sku"]][-1]


# (template, path inside a version tree, nav key)
SECTION_PAGES = [
    ("overview.html.j2", "index.html", "overview"),
    ("models.html.j2", "models/index.html", "models"),
    ("rocm.html.j2", "rocm/index.html", "rocm"),
    ("roadmap.html.j2", "roadmap/index.html", "roadmap"),
]
RELEASES_PAGE = "releases/index.html"
CSV_COLUMNS = ["madengine_tag", "framework", "workload", "model", "hf_repo", "precision", "MI300X", "MI325X", "MI350X",
               "MI355X", "n_gpus", "nodes", "scheduler", "k8s", "discovery", "suites", "base_image", "rocm", "recipe"]


def up(path):
    """Relative path from the directory of `path` back to the directory it is relative to."""
    return "/".join([".."] * path.count("/")) or "."


def make_tree(snapshot, prefix, label):
    """One version of the site: its snapshot, catalog and the pages it contains."""
    pages, frameworks = catalog(upgrade(snapshot))
    rels = {path for _, path, _ in SECTION_PAGES} | {f"models/{p['page']}/index.html" for p in pages}
    return {"snap": snapshot, "prefix": prefix, "label": label, "pages": pages, "frameworks": frameworks,
            "cat": summarize(snapshot, pages, frameworks), "rels": rels}


def version_links(trees, current, full_path, rel):
    """Version switcher: the same page in every version, else that version's nearest parent page."""
    site = up(full_path)
    links = []
    for tree in trees:
        target = rel if rel in tree["rels"] else ("models/index.html" if rel.startswith("models/") else "index.html")
        prefix = tree["prefix"] + "/" if tree["prefix"] else ""
        links.append({"label": tree["label"], "href": f"{site}/{prefix}{target}", "current": tree is current})
    return links


def render(env, template, out_dir, full_path, **context):
    out_path = os.path.join(out_dir, full_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    site = up(full_path)
    # `gh`: the GitHub repo the snapshot describes (ROCm/MAD, or whatever --repo-root points at).
    gh = "https://github.com/" + context["snap"].get("source_repo", recipes.DEFAULT_REPO)
    # Icon helper bound to this page: it records the marks used so the page can ship one small sprite.
    page_icons = env.globals["icon_set"].bind(site)
    with open(out_path, "w") as f:
        f.write(env.get_template(template).render(site=site, gh=gh, icons=page_icons, **context))


def write_csv(snapshot, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in snapshot["models"]:
            writer.writerow([r["name"], r["framework"], r["workload"], r["model"], r["hf_repo"] or "", r["precision"],
                             *("yes" if sku in r["skus"] else "no" for sku in snapshot["skus"]),
                             r["n_gpus"], r["nodes"], r["deployment"]["scheduler"], r["deployment"]["k8s"] or "no",
                             r["deployment"]["status"], " ".join(r["suites"]), r["image"], r["rocm"],
                             f"models/{r['page']}/index.html"])


def render_site(env, out_dir, tree, trees, prev):
    """Every page of one version tree under out_dir/<prefix>/."""
    snapshot, prefix = tree["snap"], tree["prefix"]
    frozen = snapshot["state"] == "frozen"
    base = os.path.join(out_dir, prefix) if prefix else out_dir
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "search.js"), "w") as f:
        f.write(search_index(tree["pages"]))
    write_csv(snapshot, os.path.join(base, "models.csv"))

    def page(template, rel, active, **context):
        full = f"{prefix}/{rel}" if prefix else rel
        render(env, template, out_dir, full, snap=snapshot, root=up(rel), active=active,
               versions=version_links(trees, tree, full, rel), **context)

    common = {"cat": tree["cat"]}
    for template, rel, active in SECTION_PAGES:
        page(template, rel, active, diff=collect.diff_snapshots(prev, snapshot), fw_slugs=tree["cat"]["fw_slugs"], **common)
    ref = snapshot["commit"] if frozen else None
    for family in tree["pages"]:
        repo = snapshot.get("source_repo", recipes.DEFAULT_REPO)
        cards = [{sku: recipes.recipe(row, sku, ref, repo) for sku, row in v["by_sku"].items()} for v in family["variants"]]
        page("recipe.html.j2", f"models/{family['page']}/index.html", "models", f=family, frameworks=tree["frameworks"],
             recipes=cards, clusters=recipes.CLUSTERS, isa_of=recipes.ISA_OF, default_sku=default_sku(family),
             gh_ref=ref or "develop")


def release_rows(trees):
    """Releases table, develop first, then frozen releases newest first, each diffed against the one before."""
    rows = []
    frozen = trees[1:]
    for i, tree in enumerate(trees):
        snap = tree["snap"]
        older = frozen[-1] if i == 0 and frozen else (frozen[frozen.index(tree) - 1] if tree in frozen and frozen.index(tree) else None)
        diff = collect.diff_snapshots(older["snap"], snap) if older else None
        rows.append({
            "label": f"{snap['release']} (develop)" if i == 0 else snap["version"],
            "href": "../" + (tree["prefix"] + "/" if tree["prefix"] else "") + "index.html",
            "date": "live" if i == 0 else snap["date"], "commit": snap["commit"],
            "families": len(tree["pages"]), "configs": len(snap["models"]),
            "added": len(diff["added"]) if diff else None, "removed": len(diff["removed"]) if diff else None,
        })
    return rows[:1] + rows[:0:-1]


HREF_RE = re.compile(r'(?:href|src)="([^"]+)"|<option value="([^"]+\.html)"')


def check_links(out_dir):
    """Every relative href/src (and version-switcher target) in the site resolves to a file."""
    broken = []
    for html in glob.glob(os.path.join(out_dir, "**", "*.html"), recursive=True):
        with open(html) as f:
            text = f.read()
        for m in HREF_RE.finditer(text):
            link = m.group(1) or m.group(2)
            if re.match(r"^(?:[a-z]+:|#|\?|//)", link):
                continue
            target = link.split("#", 1)[0].split("?", 1)[0]
            if target and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(html), target))):
                broken.append(f"{os.path.relpath(html, out_dir)} -> {link}")
    return broken


def build_site(out_dir, snapshot, releases, logos=None):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=jinja2.select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    icon_set = icons.IconSet(mode=logos)
    env.globals["icon_set"] = icon_set
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    # Provider logos live once at the site root, shared by every version tree.
    if icon_set.logo_files():
        shutil.copytree(os.path.join(icons.LOGOS_DIR, "providers"), os.path.join(out_dir, "logos", "providers"))
    if icon_set.amd_file("favicon.svg"):
        shutil.copy(icon_set.amd_file("favicon.svg"), os.path.join(out_dir, "favicon.svg"))
    # Pages runs Jekyll otherwise, which is slow and skips underscore paths.
    open(os.path.join(out_dir, ".nojekyll"), "w").close()

    trees = [make_tree(snapshot, "", "develop")] + [make_tree(r, r["version"], r["version"]) for r in releases]
    render_site(env, out_dir, trees[0], trees, releases[-1] if releases else None)
    for i, tree in enumerate(trees[1:]):
        render_site(env, out_dir, tree, trees, releases[i - 1] if i else None)
    render(env, "releases.html.j2", out_dir, RELEASES_PAGE, snap=snapshot, root="..", active="releases",
           versions=version_links(trees, trees[0], RELEASES_PAGE, RELEASES_PAGE), release_rows=release_rows(trees))

    broken = check_links(out_dir)
    if broken:
        raise SystemExit("ERROR: broken links in the generated site:\n  " + "\n  ".join(broken[:50]))
    return len(trees[0]["pages"])


def roadmap_report(status, repo_root):
    """Read-only maintainer report: what the Roadmap misses or should drop."""
    import github_roadmap
    source = status["roadmap_source"]
    token = github_roadmap.github_token()
    if not source.get("repo") or not token:
        sys.exit("ERROR: --roadmap-report needs roadmap_source in status.yaml and a GitHub token")
    raw = github_roadmap.fetch(source["repo"], source.get("label", "new-model"), token)
    rows = collect.build_snapshot(status, repo_root=repo_root, github=False, inspect_images=False)["models"]
    files = github_roadmap.Files(token)
    oncoming, landed = github_roadmap.build_roadmap(raw, rows, status["frameworks"], source.get("label", "new-model"), files)
    print(f"Roadmap from {source['repo']} (label '{source.get('label', 'new-model')}'): {len(oncoming)} oncoming")
    for item in oncoming:
        print(f"  #{item['number']:<5} {item['status']:<12} {item['model']}")
    print(f"\nLanded but still open ({len(landed)}) -- close these, or relabel if more work remains:")
    for item in landed:
        print(f"  #{item['number']:<5} {item['model']}  {item['url']}")
    unlabeled = github_roadmap.unlabeled_model_prs(raw, source.get("label", "new-model"))
    print(f"\nOpen PRs that change model cards without the label or a labeled issue ({len(unlabeled)}):")
    for pr in unlabeled:
        new = github_roadmap.added_models(pr, files, known=[r["name"] for r in rows])
        what = f"adds {len(new)} model card(s)" if new else "changes model cards"
        print(f"  #{pr['number']:<5} {'draft ' if pr['isDraft'] else ''}{what}: {pr['title'][:70]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="_site", help="output directory for the site (default: _site)")
    parser.add_argument("--freeze", metavar="VERSION", help="write dashboard/releases/VERSION.json instead of building")
    parser.add_argument("--internal", action="store_true", help="merge dashboard/roadmap.internal.yaml (never publish)")
    parser.add_argument("--strict", action="store_true", help="fail on snapshot warnings")
    parser.add_argument("--repo-root", default=collect.REPO_ROOT,
                        help="MAD checkout to describe (default: this one); dashboard/ data always comes from here")
    parser.add_argument("--source-repo", help="GitHub owner/name of --repo-root for links and recipes (default: its origin)")
    parser.add_argument("--offline", action="store_true",
                        help="do not read the roadmap from GitHub or base images' ROCm versions from their registries")
    parser.add_argument("--logos", choices=["official", "neutral"],
                        help="brand logos or neutral monograms/glyphs (default: `logos:` in dashboard/icons.yaml)")
    parser.add_argument("--roadmap-report", action="store_true",
                        help="print open model PRs missing the roadmap label and landed-but-open items, then exit")
    args = parser.parse_args()

    if args.freeze and args.internal:
        parser.error("--freeze writes a committed, public file; it cannot include the internal roadmap")
    if args.internal and not os.path.exists(collect.INTERNAL_ROADMAP_PATH):
        parser.error(f"--internal needs {collect.INTERNAL_ROADMAP_PATH}")

    status = collect.load_status(internal_roadmap_path=collect.INTERNAL_ROADMAP_PATH if args.internal else None)
    repo_root = os.path.abspath(args.repo_root)
    snapshot_args = {"repo_root": repo_root, "source_repo": args.source_repo, "github": not args.offline,
                     "inspect_images": not args.offline}

    if args.roadmap_report:
        roadmap_report(status, repo_root)
        return

    if args.freeze:
        collect.version_key(args.freeze)
        path = os.path.join(RELEASES_DIR, f"{args.freeze}.json")
        if os.path.exists(path):
            sys.exit(f"ERROR: {path} already exists; frozen releases are immutable")
        if args.freeze != status["release"]:
            print(f"WARNING: freezing {args.freeze} but status.yaml release is {status['release']}", file=sys.stderr)
        snapshot = collect.build_snapshot(status, version=args.freeze, state="frozen", **snapshot_args)
    else:
        snapshot = collect.build_snapshot(status, **snapshot_args)

    if args.strict and snapshot["warnings"]:
        sys.exit(f"ERROR: --strict and {len(snapshot['warnings'])} warning(s); see above")

    if args.freeze:
        os.makedirs(RELEASES_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"Froze {len(snapshot['models'])} models into {os.path.relpath(path, collect.REPO_ROOT)}")
        return

    releases = load_releases()
    n_pages = build_site(args.out, snapshot, releases, logos=args.logos)
    print(f"Wrote {args.out}: {n_pages} recipe pages, {len(snapshot['models'])} models, {len(snapshot['images'])} images, "
          f"{len(releases)} frozen release(s){' [INTERNAL - do not publish]' if snapshot['internal'] else ''}")


if __name__ == "__main__":
    main()

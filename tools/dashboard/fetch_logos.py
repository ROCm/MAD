#!/usr/bin/env python3
"""
Vendor the logos listed in dashboard/icons.yaml into dashboard/logos/ (run when the list changes).

  providers/<org>.png       96x96 provider avatars. Source order: https://recipes.vllm.ai/providers/
                            (the vLLM recipes site, which shows the same Hugging Face org avatars),
                            then the Hugging Face organization avatar API.
  simple-icons/<slug>.svg   brand marks from Simple Icons (CC0), for the `simple:` entries.
  amd/amd-logo.svg          the AMD logo for the site header, from AMD's ROCm documentation theme,
  amd/favicon.svg           and the AMD Arrow alone as the site favicon.
  sources.json             where each file came from, for dashboard/logos/NOTICE.md.

Needs network and Pillow (`pip install pillow`); the site build itself needs neither.

    python3 tools/dashboard/fetch_logos.py [--force]
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ICONS_YAML = os.path.join(ROOT, "dashboard", "icons.yaml")
LOGOS_DIR = os.path.join(ROOT, "dashboard", "logos")
SIMPLE_ICONS_VERSION = "16.36.0"
SIZE = 96

RECIPES_SITE = "https://recipes.vllm.ai"
HF_AVATAR_API = "https://huggingface.co/api/organizations/{org}/avatar"
SIMPLE_ICONS_URL = "https://cdn.jsdelivr.net/npm/simple-icons@{version}/icons/{slug}.svg"
# The AMD logo as AMD's own ROCm documentation theme (rocm-docs-core) ships it: white artwork
# for its dark header.
AMD_LOGO_URL = ("https://raw.githubusercontent.com/ROCm/rocm-docs-core/develop/"
                "src/rocm_docs/rocm_docs_theme/static/images/amd-header-logo.svg")


def amd_marks(svg):
    """(logo, favicon) SVG text from AMD's logo artwork.

    The logo keeps every shape and takes the text color (currentColor): black on the light
    theme and white on the dark one, the two one-color versions of the AMD logo. The favicon
    is the AMD Arrow alone (the artwork's two polygons), as amd.com's own favicon shows it,
    black or white by the browser's color scheme.
    """
    shapes = [re.sub(r'\s+class="[^"]*"', "", s) for s in re.findall(r"<(?:path|polygon)\b[^>]*/>", svg)]
    arrow = [s for s in shapes if s.startswith("<polygon")]
    if len(shapes) != 5 or len(arrow) != 2:
        sys.exit(f"ERROR: {AMD_LOGO_URL} no longer has the expected A, M, D and arrow shapes; check it by hand")
    view_box = re.search(r'viewBox="([^"]+)"', svg).group(1)
    logo = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" fill="currentColor">'
            f'<title>AMD</title>{"".join(shapes)}</svg>\n')
    points = [float(v) for s in arrow for v in re.search(r'points="([^"]+)"', s).group(1).split()]
    xs, ys = points[0::2], points[1::2]
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    favicon = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min(xs):g} {min(ys):g} {side:g} {side:g}">'
               "<style>polygon{fill:#000}@media (prefers-color-scheme:dark){polygon{fill:#fff}}</style>"
               f'{"".join(arrow)}</svg>\n')
    return logo, favicon


def get(url, timeout=60, attempts=3):
    """GET with a few retries; CDNs occasionally reset a connection."""
    request = urllib.request.Request(url, headers={"User-Agent": "mad-dashboard-fetch-logos"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def recipes_site_providers():
    """{lowercased org: file name} for the providers the vLLM recipes site has logos for."""
    html = get(RECIPES_SITE + "/").decode("utf-8", "replace")
    files = sorted(set(re.findall(r"/providers/([^\"'\\ ]+\.(?:png|jpe?g|webp|svg))", html)))
    return {name.rsplit(".", 1)[0].lower(): name for name in files}


def to_square_png(data):
    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image = ImageOps.fit(image, (SIZE, SIZE), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()


def fetch_provider(org, known):
    """(png bytes, source url) for one Hugging Face org, or None."""
    name = known.get(org.lower())
    if name:
        url = f"{RECIPES_SITE}/providers/{name}"
        try:
            return to_square_png(get(url)), url
        except (urllib.error.URLError, OSError) as e:
            print(f"  {org}: {url} failed ({e}); trying Hugging Face", file=sys.stderr)
    try:
        avatar = json.loads(get(HF_AVATAR_API.format(org=org)))["avatarUrl"]
        return to_square_png(get(avatar)), avatar
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"  {org}: no logo found ({e})", file=sys.stderr)
        return None


def have(path):
    """True when a previously fetched file is present and non-empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    with open(ICONS_YAML) as f:
        config = yaml.safe_load(f)
    sources_path = os.path.join(LOGOS_DIR, "sources.json")
    sources = json.load(open(sources_path)) if os.path.exists(sources_path) else {}

    os.makedirs(os.path.join(LOGOS_DIR, "providers"), exist_ok=True)
    known = recipes_site_providers()
    print(f"{len(known)} providers on {RECIPES_SITE}")
    missing = []
    for org in config["providers"]:
        key = f"providers/{org}.png"
        path = os.path.join(LOGOS_DIR, key)
        if have(path) and not args.force:
            continue
        result = fetch_provider(org, known)
        if result is None:
            missing.append(org)
            continue
        with open(path, "wb") as f:
            f.write(result[0])
        sources[key] = result[1]
        print(f"  {org}: {result[1]}")

    slugs = sorted({v.split(":", 1)[1] for v in list(config["marks"].values()) + [i[1] for i in config["frameworks"]]
                    if v.startswith("simple:")})
    os.makedirs(os.path.join(LOGOS_DIR, "simple-icons"), exist_ok=True)
    for slug in slugs:
        key = f"simple-icons/{slug}.svg"
        path = os.path.join(LOGOS_DIR, key)
        if have(path) and not args.force:
            continue
        url = SIMPLE_ICONS_URL.format(version=SIMPLE_ICONS_VERSION, slug=slug)
        data = get(url)  # download before opening the file, so a failure never leaves an empty one
        if not data.lstrip().startswith(b"<svg"):
            sys.exit(f"ERROR: {url} did not return an SVG")
        with open(path, "wb") as f:
            f.write(data)
        sources[key] = url
        print(f"  {slug}: {url}")

    amd_files = ("amd/amd-logo.svg", "amd/favicon.svg")
    if args.force or not all(have(os.path.join(LOGOS_DIR, key)) for key in amd_files):
        os.makedirs(os.path.join(LOGOS_DIR, "amd"), exist_ok=True)
        for key, text in zip(amd_files, amd_marks(get(AMD_LOGO_URL).decode("utf-8"))):
            with open(os.path.join(LOGOS_DIR, key), "w") as f:
                f.write(text)
            sources[key] = AMD_LOGO_URL
        print(f"  amd: {AMD_LOGO_URL}")

    with open(sources_path, "w") as f:
        json.dump(dict(sorted(sources.items())), f, indent=1)
        f.write("\n")
    if missing:
        print(f"No logo for: {', '.join(missing)} (they will render as monograms)")


if __name__ == "__main__":
    main()

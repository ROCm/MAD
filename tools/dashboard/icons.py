"""
Logos and icons for the site: model providers, frameworks, hardware and section glyphs.

Resolution order for a model's provider: its Hugging Face org, else the first name pattern in
dashboard/icons.yaml that matches. A provider with no vendored logo, a framework with no mark,
or `logos: neutral`, renders as a generated monogram instead. Nothing here touches the network.

Pages reference icons through a per-page helper (`icons.logo(...)`, `icons.framework(...)`,
`icons.glyph(...)`); the helper records what was used so each page ships one small inline SVG
sprite containing only those marks.
"""
import os
import re

import yaml
from markupsafe import Markup, escape

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ICONS_YAML = os.path.join(REPO_ROOT, "dashboard", "icons.yaml")
LOGOS_DIR = os.path.join(REPO_ROOT, "dashboard", "logos")

# Neutral line glyphs (24x24 grid), drawn for this site: section icons and the fallbacks used
# for hardware and tooling when logos are neutral.
GLYPHS = {
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "flag": '<path d="M5 21V4"/><path d="M5 4h12l-2 4 2 4H5"/>',
    "tag": '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="8" cy="8" r="1.3"/>',
    "network": '<circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="M12 7v5M12 12l-6 5M12 12l6 5"/>',
    "swap": '<path d="M4 8h14l-3-3M20 16H6l3 3"/>',
    "terminal": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>',
    "box": '<path d="M3 8l9-5 9 5v8l-9 5-9-5z"/><path d="M3 8l9 5 9-5M12 13v8"/>',
}


def stable_hue(text):
    """Deterministic 0-359 hue (Python's hash() is randomized per process)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % 360


def initials(text):
    """'Llama-3.3-70B' -> 'Ll', 'xAI' -> 'xA'; two characters read best in a small badge."""
    letters = re.sub(r"[^A-Za-z0-9]", "", text)[:2]
    return (letters[:1].upper() + letters[1:].lower()) if letters else "?"


class IconSet:
    """Everything resolved from dashboard/icons.yaml; one instance is shared by all pages."""

    def __init__(self, config_path=ICONS_YAML, logos_dir=LOGOS_DIR, mode=None):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        self.logos_dir = logos_dir
        self.mode = mode or config.get("logos", "official")
        if self.mode not in ("official", "neutral"):
            raise ValueError(f"logos must be 'official' or 'neutral', got {self.mode!r}")
        self.providers = config.get("providers", {})
        self.by_lower = {org.lower(): org for org in self.providers}
        self.patterns = [(re.compile(p), org) for p, org in config.get("name_patterns", [])]
        self.frameworks = [(prefix.lower(), spec) for prefix, spec in config.get("frameworks", [])]
        self.marks = config.get("marks", {})

    # -- resolution ------------------------------------------------------------------------

    def provider_org(self, name, hf_org=None):
        """Canonical Hugging Face org for a model, or None."""
        if hf_org and hf_org.lower() in self.by_lower:
            return self.by_lower[hf_org.lower()]
        for pattern, org in self.patterns:
            if pattern.search(name.lower()):
                return org
        return None

    def provider_logo_file(self, org):
        path = os.path.join(self.logos_dir, "providers", f"{org}.png")
        return f"providers/{org}.png" if self.mode == "official" and os.path.exists(path) else None

    def framework_spec(self, name):
        lowered = name.lower()
        for prefix, spec in self.frameworks:
            if lowered.startswith(prefix):
                return spec
        return f"mono:{initials(name)}:{stable_hue(name)}"

    def simple_path(self, slug):
        """The path data of a vendored Simple Icons mark, or None."""
        path = os.path.join(self.logos_dir, "simple-icons", f"{slug}.svg")
        if self.mode != "official" or not os.path.exists(path):
            return None
        with open(path) as f:
            paths = re.findall(r'<path[^>]*\sd="([^"]+)"', f.read())
        return " ".join(paths) or None

    def amd_file(self, name):
        """Path of a vendored AMD asset (amd-logo.svg, favicon.svg), or None when logos are neutral."""
        path = os.path.join(self.logos_dir, "amd", name)
        return path if self.mode == "official" and os.path.exists(path) else None

    def bind(self, site):
        return BoundIcons(self, site)

    def logo_files(self):
        """Provider logo files the site should ship (none when logos are neutral)."""
        if self.mode != "official":
            return []
        directory = os.path.join(self.logos_dir, "providers")
        return sorted(os.listdir(directory)) if os.path.isdir(directory) else []


class BoundIcons:
    """Icon helpers for one page; `site` is the relative path from the page to the site root."""

    def __init__(self, icons, site):
        self.icons, self.site = icons, site
        self.official = icons.mode == "official"
        self.symbols = {}   # symbol id -> (viewBox-less inner markup, attributes)

    # -- building blocks -------------------------------------------------------------------

    def _mono(self, text, hue, size, title):
        return Markup('<span class="mg" style="--s:{size}px;--h:{hue}" title="{title}" aria-hidden="true">{text}</span>').format(
            size=size, hue=hue, title=title, text=text)

    def _use(self, symbol_id, size, title=""):
        return Markup('<svg class="ico" style="--s:{size}px" aria-hidden="true"><use href="#{id}"/></svg>').format(
            size=size, id=symbol_id)

    def _register_simple(self, slug):
        path = self.icons.simple_path(slug)
        if path is None:
            return None
        symbol_id = f"i-{slug}"
        self.symbols[symbol_id] = ('fill="currentColor"', f'<path d="{path}"/>')
        return symbol_id

    def _register_glyph(self, name):
        symbol_id = f"g-{name}"
        self.symbols[symbol_id] = ('fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"',
                                   GLYPHS[name])
        return symbol_id

    def _from_spec(self, spec, size, label):
        """Render `simple:<slug>`, `glyph:<name>` or `mono:<text>:<hue>`; monogram when a mark is unavailable."""
        kind, _, rest = spec.partition(":")
        if kind == "simple":
            symbol_id = self._register_simple(rest)
            if symbol_id:
                return self._use(symbol_id, size)
            return self._mono(initials(label), stable_hue(label), size, label)
        if kind == "glyph":
            return self._use(self._register_glyph(rest), size)
        text, _, hue = rest.rpartition(":")
        return self._mono(escape(text), int(hue), size, label)

    # -- what templates call ---------------------------------------------------------------

    def logo(self, name, hf_org=None, size=24):
        """Provider logo for a model: vendored avatar, else a monogram of the provider or model name."""
        org = self.icons.provider_org(name, hf_org)
        label = self.icons.providers.get(org, hf_org or name) if org else (hf_org or name)
        file = self.icons.provider_logo_file(org) if org else None
        if file:
            return Markup('<img class="plogo" src="{src}" width="{size}" height="{size}" style="--s:{size}px" alt="" title="{title}" loading="lazy">').format(
                src=f"{self.site}/logos/{file}", size=size, title=label)
        seed = org or hf_org or name
        return self._mono(initials(label), stable_hue(seed), size, label)

    def framework(self, name, size=18):
        return self._from_spec(self.icons.framework_spec(name), size, name)

    def mark(self, key, size=18):
        """A named mark (amd, docker, huggingface, pytorch), with a neutral glyph fallback."""
        if key == "amd":
            # The AMD org avatar reads better at small sizes than the wordmark in a square box.
            file = self.icons.provider_logo_file("amd")
            if file:
                return Markup('<img class="plogo" src="{src}" width="{size}" height="{size}" style="--s:{size}px" alt="" title="AMD" loading="lazy">').format(
                    src=f"{self.site}/logos/{file}", size=size)
            return self._use(self._register_glyph("chip"), size)
        fallback = {"docker": "glyph:box", "huggingface": "mono:HF:48", "pytorch": "glyph:cube"}
        spec = self.icons.marks.get(key) or fallback[key]
        if spec.startswith("simple:") and self.icons.simple_path(spec.split(":", 1)[1]) is None:
            spec = fallback[key]
        return self._from_spec(spec, size, key)

    def amd_logo(self, width=72):
        """The AMD logo inline, in the text color (black or white with the theme); the word AMD
        when logos are neutral."""
        path = self.icons.amd_file("amd-logo.svg")
        if path is None:
            return Markup('<span class="amd-text">AMD</span>')
        with open(path) as f:
            svg = f.read().strip()
        return Markup(svg.replace("<svg ", f'<svg class="amd-logo" width="{width}" role="img" aria-label="AMD" ', 1))

    @property
    def favicon(self):
        return self.icons.amd_file("favicon.svg") is not None

    def glyph(self, name, size=18):
        return self._use(self._register_glyph(name), size)

    def sprite(self):
        """One hidden SVG with every symbol this page used; emit it once, after the body."""
        if not self.symbols:
            return Markup("")
        symbols = "".join(f'<symbol id="{sid}" viewBox="0 0 24 24" {attrs}>{inner}</symbol>'
                          for sid, (attrs, inner) in sorted(self.symbols.items()))
        return Markup(f'<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">{symbols}</svg>')

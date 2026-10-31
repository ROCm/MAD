"""Tests for logo and icon resolution (no network; fixtures live in tmp_path)."""
import json
import os
import re
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icons  # noqa: E402

CONFIG = {
    "logos": "official",
    "providers": {"meta-llama": "Meta Llama", "Qwen": "Qwen", "amd": "AMD", "openai": "OpenAI"},
    "name_patterns": [["^llama", "meta-llama"], ["^qwen", "Qwen"], ["^gpt[-_]?oss", "openai"]],
    "frameworks": [["vllm", "simple:vllm"], ["sglang", "mono:SG:265"], ["large ep", "glyph:network"]],
    "marks": {"docker": "simple:docker"},
}
SVG = '<svg role="img" viewBox="0 0 24 24"><title>X</title><path d="M0 0h24v24H0z"/></svg>'


@pytest.fixture
def make(tmp_path):
    """IconSet over a fresh temp logos dir; `files` are provider logos to create, `svgs` Simple Icons marks."""
    count = []

    def factory(files=("meta-llama", "amd"), svgs=("vllm", "docker"), mode=None, empty_svgs=()):
        root = tmp_path / f"logos{len(count)}"
        count.append(root)
        (root / "providers").mkdir(parents=True)
        (root / "simple-icons").mkdir()
        for org in files:
            (root / "providers" / f"{org}.png").write_bytes(b"\x89PNG")
        for slug in svgs:
            (root / "simple-icons" / f"{slug}.svg").write_text(SVG)
        for slug in empty_svgs:
            (root / "simple-icons" / f"{slug}.svg").write_text("")
        config = root / "icons.yaml"
        config.write_text(yaml.safe_dump(CONFIG))
        return icons.IconSet(str(config), str(root), mode=mode)
    return factory


def test_stable_hue_and_initials():
    assert icons.stable_hue("SGLang") == icons.stable_hue("SGLang") and 0 <= icons.stable_hue("x") < 360
    assert icons.stable_hue("a") != icons.stable_hue("b")
    assert icons.initials("Llama-3.3-70B") == "Ll" and icons.initials("xAI") == "Xa"
    assert icons.initials("gpt-oss") == "Gp" and icons.initials("--") == "?"


def test_provider_org_prefers_hf_org_then_name_patterns(make):
    s = make()
    assert s.provider_org("anything", "meta-llama") == "meta-llama"
    assert s.provider_org("anything", "QWEN") == "Qwen"           # case-insensitive, canonical spelling
    assert s.provider_org("llama3.1_8B-pretrain") == "meta-llama"  # Primus configs have no Hugging Face id
    assert s.provider_org("gpt_oss_20B-pretrain") == "openai"
    assert s.provider_org("unknown-model", "someone-else") is None


def test_framework_spec_matches_prefix_with_monogram_fallback(make):
    s = make()
    assert s.framework_spec("vLLM disaggregated P/D") == "simple:vllm"
    assert s.framework_spec("Large EP microbenchmark") == "glyph:network"
    kind, text, hue = s.framework_spec("Brand New Engine").split(":")
    assert kind == "mono" and text == "Br" and 0 <= int(hue) < 360


def test_logo_uses_vendored_file_or_monogram(make):
    page = make().bind("../..")
    html = str(page.logo("Llama-3.3-70B", "meta-llama", 30))
    assert 'src="../../logos/providers/meta-llama.png"' in html and 'class="plogo"' in html and 'alt=""' in html
    # In the table, 'Qwen' has no vendored file: monogram, never a broken image.
    qwen = str(page.logo("Qwen3-8B", "Qwen", 30))
    assert "<img" not in qwen and 'class="mg"' in qwen and ">Qw<" in qwen
    assert 'class="mg"' in str(page.logo("mystery-model", None, 30))


def test_neutral_mode_never_emits_brand_marks(make):
    s = make(mode="neutral")
    page = s.bind(".")
    assert s.logo_files() == []
    assert "<img" not in str(page.logo("Llama", "meta-llama")) and 'class="mg"' in str(page.logo("Llama", "meta-llama"))
    assert 'class="mg"' in str(page.framework("vLLM"))
    page.mark("docker"), page.mark("amd")
    assert not any(sid.startswith("i-") for sid in page.symbols)   # no Simple Icons paths registered
    assert "g-box" in page.symbols and "g-chip" in page.symbols    # neutral glyphs instead


def test_simple_icons_are_registered_once_in_the_sprite(make):
    page = make().bind(".")
    assert 'href="#i-vllm"' in str(page.framework("vLLM"))
    page.framework("vLLM disaggregated P/D")
    sprite = str(page.sprite())
    assert sprite.count('<symbol id="i-vllm"') == 1 and 'd="M0 0h24v24H0z"' in sprite
    assert "i-docker" not in sprite                               # only what the page used
    assert str(make().bind(".").sprite()) == ""                  # no icons used, no sprite


def test_empty_or_missing_svg_falls_back_to_monogram(make):
    """An empty vendored file (a failed download) must not become an invisible icon."""
    page = make(svgs=(), empty_svgs=("vllm",)).bind(".")
    html = str(page.framework("vLLM"))
    assert 'class="mg"' in html and "<use" not in html and not page.symbols


def test_amd_mark_uses_the_provider_avatar_when_available(make):
    assert "logos/providers/amd.png" in str(make().bind(".").mark("amd", 16))
    assert "g-chip" in str(make(files=()).bind(".").mark("amd", 16))


def test_amd_logo_inline_or_text(make, tmp_path):
    iconset = make()
    assert str(iconset.bind(".").amd_logo()) == '<span class="amd-text">AMD</span>'   # no vendored logo
    amd = os.path.join(iconset.logos_dir, "amd")
    os.makedirs(amd)
    for name in ("amd-logo.svg", "favicon.svg"):
        open(os.path.join(amd, name), "w").write('<svg viewBox="0 0 10 2" fill="currentColor"><path d="M0 0h1v1z"/></svg>')
    bound = iconset.bind(".")
    assert str(bound.amd_logo(80)).startswith('<svg class="amd-logo" width="80" role="img" aria-label="AMD" viewBox=')
    assert bound.favicon
    neutral = icons.IconSet(os.path.join(iconset.logos_dir, "icons.yaml"), iconset.logos_dir, mode="neutral").bind(".")
    assert "amd-text" in str(neutral.amd_logo()) and not neutral.favicon


def test_amd_marks_split_logo_and_arrow():
    import fetch_logos
    artwork = ('<svg viewBox="0 0 139.72 33.32"><defs><style>.cls-1{fill:#fff;}</style></defs>'
               + '<path class="cls-1" d="M0 0h1z"/>' * 3
               + '<polygon class="cls-1" points="130 9 139.72 0 139.72 33"/><polygon class="cls-1" points="106.4 20 106.4 33.33 119 33.33"/></svg>')
    logo, favicon = fetch_logos.amd_marks(artwork)
    assert 'viewBox="0 0 139.72 33.32" fill="currentColor"' in logo and "cls-1" not in logo and logo.count("<path") == 3
    assert 'viewBox="106.4 0 33.33 33.33"' in favicon and "<path" not in favicon and favicon.count("<polygon") == 2
    assert "prefers-color-scheme:dark" in favicon


def test_glyphs_cover_every_section_icon():
    for name in ("cube", "layers", "shield", "server", "flag", "tag", "chip", "terminal", "box", "network", "swap"):
        assert name in icons.GLYPHS


def test_rejects_unknown_mode(make):
    with pytest.raises(ValueError):
        make(mode="fancy")


# --- the committed configuration and vendored assets --------------------------------------------


def test_committed_icons_yaml_is_consistent_with_vendored_assets():
    with open(icons.ICONS_YAML) as f:
        config = yaml.safe_load(f)
    providers = set(config["providers"])
    assert {org for _, org in config["name_patterns"]} <= providers, "name_patterns point at unknown providers"
    for pattern, _ in config["name_patterns"]:
        re.compile(pattern)
    sources = json.load(open(os.path.join(icons.LOGOS_DIR, "sources.json")))
    for org in providers:
        path = os.path.join(icons.LOGOS_DIR, "providers", f"{org}.png")
        assert os.path.getsize(path) > 0, f"logo for {org} missing; run tools/dashboard/fetch_logos.py"
        assert f"providers/{org}.png" in sources
    slugs = set()
    for spec in list(config["marks"].values()) + [spec for _, spec in config["frameworks"]]:
        if spec.startswith("simple:"):
            slugs.add(spec.split(":", 1)[1])
    for slug in slugs:
        text = open(os.path.join(icons.LOGOS_DIR, "simple-icons", f"{slug}.svg")).read()
        assert re.search(r'<path[^>]*\sd="[^"]+"', text), f"{slug}.svg has no path"
        assert f"simple-icons/{slug}.svg" in sources
    for name in ("amd-logo.svg", "favicon.svg"):
        text = open(os.path.join(icons.LOGOS_DIR, "amd", name)).read()
        assert text.startswith("<svg") and f"amd/{name}" in sources, f"{name} missing; run tools/dashboard/fetch_logos.py"

"""The catalog runner: a campaign is data, not a script somebody edits to add a job."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import regen_reports  # noqa: E402

DEFAULTS = {"parse_cache_dir": "reports/.parse-cache", "top": 20}
ENTRY = {"name": "gptoss_balanced", "run_dir": "slurm_output/run_logs/25956",
         "out_dir": "reports/sgl_gptoss_2p2d_balanced", "phases": ["prefill", "decode"]}


def test_an_entry_becomes_the_command_line_it_describes():
    argv = regen_reports.argv_for(ENTRY, DEFAULTS, Path("/work"))
    assert "--run-dir" in argv and "/work/slurm_output/run_logs/25956" in argv
    assert "--out-dir" in argv and "/work/reports/sgl_gptoss_2p2d_balanced" in argv
    assert argv[argv.index("--phases") + 1:argv.index("--phases") + 3] == ["prefill", "decode"]
    assert "/work/reports/.parse-cache/gptoss_balanced.pkl" in argv
    assert "--top" in argv


def test_absolute_paths_in_a_catalog_are_left_alone():
    entry = {**ENTRY, "run_dir": "/mnt/shared/25956"}
    argv = regen_reports.argv_for(entry, {}, Path("/work"))
    assert "/mnt/shared/25956" in argv


def test_optional_settings_appear_only_when_set():
    argv = regen_reports.argv_for(ENTRY, {}, Path("/work"))
    for flag in ("--rocprof-dir", "--trace-root", "--max-msg-bytes", "--engine", "--parse-cache"):
        assert flag not in argv

    rich = {**ENTRY, "engine": "sglang-disagg", "rocprof_dir": "prof/rocprof",
            "max_msg_bytes": 1073741824, "no_auto_traces": True,
            "torch_trace": {"prefill": "torchprof/1000.0"}}
    argv = regen_reports.argv_for(rich, {}, Path("/work"))
    assert argv[argv.index("--engine") + 1] == "sglang-disagg"
    assert argv[argv.index("--rocprof-dir") + 1] == "/work/prof/rocprof"
    assert argv[argv.index("--max-msg-bytes") + 1] == "1073741824"
    assert argv[argv.index("--torch-trace") + 1] == "prefill=/work/torchprof/1000.0"
    assert "--no-auto-traces" in argv


def test_a_dry_run_prints_the_commands_without_parsing(tmp_path: Path, capsys, monkeypatch):
    catalog = tmp_path / "reports" / "jobs.json"
    catalog.parent.mkdir()
    catalog.write_text(json.dumps({"defaults": DEFAULTS, "reports": [ENTRY]}))
    monkeypatch.setattr(sys, "argv", ["regen_reports.py", "--catalog", str(catalog), "--dry-run"])
    assert regen_reports.main() == 0
    assert "collective_report.py --run-dir" in capsys.readouterr().out


def test_an_unknown_name_in_only_is_an_error(tmp_path: Path, monkeypatch):
    catalog = tmp_path / "jobs.json"
    catalog.write_text(json.dumps({"reports": [ENTRY]}))
    monkeypatch.setattr(sys, "argv",
                        ["regen_reports.py", "--catalog", str(catalog), "--only", "typo"])
    with pytest.raises(SystemExit, match="no catalog entry named typo"):
        regen_reports.main()


def test_one_failing_report_does_not_stop_the_campaign(tmp_path: Path, capsys, monkeypatch):
    """A batch of six reports must not be lost because the third job's artifacts are incomplete."""
    catalog = tmp_path / "jobs.json"
    catalog.write_text(json.dumps({"reports": [
        {"name": "broken", "run_dir": "missing_job", "out_dir": "out/broken"},
        {"name": "also_broken", "run_dir": "missing_too", "out_dir": "out/also"},
    ]}))
    monkeypatch.setattr(sys, "argv",
                        ["regen_reports.py", "--catalog", str(catalog), "--root", str(tmp_path)])
    assert regen_reports.main() == 1
    out = capsys.readouterr().out
    assert "broken" in out and "also_broken" in out


def test_the_catalog_can_be_listed_with_its_notes(tmp_path: Path, capsys, monkeypatch):
    catalog = tmp_path / "jobs.json"
    catalog.write_text(json.dumps({"reports": [{**ENTRY, "note": "both replicas busy"}]}))
    monkeypatch.setattr(sys, "argv", ["regen_reports.py", "--catalog", str(catalog), "--list"])
    assert regen_reports.main() == 0
    out = capsys.readouterr().out
    assert "gptoss_balanced" in out and "both replicas busy" in out


def test_the_example_catalog_shipped_with_the_skill_is_valid():
    """The example is what a new user copies, so it has to parse and describe real fields."""
    example = Path(__file__).resolve().parents[2] / "assets" / "jobs.example.json"
    catalog = json.loads(example.read_text())
    assert catalog["reports"], "the example must show at least one entry"
    for entry in catalog["reports"]:
        assert {"name", "run_dir", "out_dir"} <= set(entry)
        regen_reports.argv_for(entry, catalog.get("defaults", {}), Path("/work"))

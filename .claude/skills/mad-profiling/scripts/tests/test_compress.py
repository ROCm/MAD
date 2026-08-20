"""Compressing a finished run's logs must be lossless, idempotent, and never leave two copies."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from conftest import coll_line, topo_line, write

from collprof.compress_cli import main
from collprof.core.compress import compress_one
from collprof.core.rccl_log import parse_run
from collprof.engines import primus, sglang_disagg


def volumes(run: Path, spec, rccl_dir: Path | None = None) -> dict:
    """Calls and bytes per phase, which is what a report would quote."""
    return {name: {coll: (row["calls"], row["bytes"])
                   for coll, row in phase.collective_totals().items()}
            for name, phase in parse_run(run, spec, rccl_dir).items()}


def test_a_compressed_run_parses_to_the_same_numbers(sglang_run: Path):
    before = volumes(sglang_run, sglang_disagg.SPEC)

    main(["--run-dir", str(sglang_run), "--jobs", "2"])

    assert not list(sglang_run.glob("*_NODE*.log")), "the plain logs should be gone"
    assert len(list(sglang_run.glob("*_NODE*.log.gz"))) == 4
    assert volumes(sglang_run, sglang_disagg.SPEC) == before


def test_per_rank_files_outside_the_run_directory_are_included(primus_run: Path, tmp_path: Path):
    rccl = tmp_path / "rccl"
    for rank in range(4):
        write(rccl / f"BF16.node_0.host0.{4000 + rank}.log",
              [coll_line(grank=rank, pid=4000 + rank), topo_line(rank, (rank + 1) % 4)])
    before = volumes(primus_run, primus.SPEC, rccl)

    main(["--run-dir", str(primus_run), "--rccl-dir", str(rccl), "--jobs", "2"])

    assert not list(rccl.glob("*.log")) and len(list(rccl.glob("*.log.gz"))) == 4
    assert not list(primus_run.glob("node_*/stdout.out")), "the shared stdout is a log too"
    assert volumes(primus_run, primus.SPEC, rccl) == before


def test_running_it_twice_is_a_no_op(sglang_run: Path, capsys):
    main(["--run-dir", str(sglang_run), "--jobs", "2"])
    sizes = {p: p.stat().st_size for p in sglang_run.glob("*.log.gz")}

    main(["--run-dir", str(sglang_run)])

    assert "nothing to compress" in capsys.readouterr().out
    assert {p: p.stat().st_size for p in sglang_run.glob("*.log.gz")} == sizes


def test_a_log_beside_its_own_gz_is_left_for_a_human(sglang_run: Path, capsys):
    """Both match the parser's globs, so compressing one would double that node's records."""
    plain = sglang_run / "prefill_NODE0.log"
    write(sglang_run / "prefill_NODE0.log.gz", ["stale"], compress=True)
    body = plain.read_bytes()

    with pytest.raises(SystemExit):
        main(["--run-dir", str(sglang_run), "--jobs", "2"])

    assert plain.read_bytes() == body
    assert "read those records twice" in capsys.readouterr().out


def test_a_dry_run_touches_nothing(sglang_run: Path, capsys):
    main(["--run-dir", str(sglang_run), "--dry-run"])

    assert len(list(sglang_run.glob("*_NODE*.log"))) == 4
    assert not list(sglang_run.glob("*.gz"))
    assert "to compress" in capsys.readouterr().out


def test_the_bytes_and_the_timestamp_survive(tmp_path: Path):
    log = write(tmp_path / "decode_NODE2.log", [coll_line(), topo_line()])
    body, mtime = log.read_bytes(), log.stat().st_mtime

    result = compress_one(log)

    assert result.ok and not log.exists()
    assert gzip.decompress(result.gz.read_bytes()) == body
    assert result.gz.stat().st_mtime == pytest.approx(mtime, abs=1)
    assert result.after < result.before


def test_a_failed_compression_keeps_the_original(tmp_path: Path, monkeypatch):
    log = write(tmp_path / "decode_NODE2.log", [coll_line()])
    body = log.read_bytes()
    monkeypatch.setattr("collprof.core.compress.hashlib.sha256",
                        lambda *_, **__: (_ for _ in ()).throw(OSError("Disk quota exceeded")))

    result = compress_one(log)

    assert not result.ok and "quota" in result.error
    assert log.read_bytes() == body
    assert not list(tmp_path.glob("*.gz*")), "no partial output may be left behind"

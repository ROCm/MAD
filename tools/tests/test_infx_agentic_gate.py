"""The agentic result gate rejects thin, failed or errored replays (GPU-free).

The documents are built from the metric paths infx_agentic.py reads, so a
change to those paths breaks these tests rather than the gate going quiet.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "common"))

try:
    import infx_agentic  # noqa: E402
except ModuleNotFoundError as exc:  # psutil: only the infx2mad CI job installs it
    raise unittest.SkipTest(f"infx_agentic needs {exc.name}")


def make_doc(profiled=100, errors=0, tok_s=500.0):
    return {
        "request_accounting": {
            "records_profiled": profiled,
            "records_total": profiled + errors,
            "records_error_dropped": errors,
        },
        "request_metrics": {
            "throughput": {
                "total": {"tokens_per_second": tok_s},
                "duration_seconds": 180.0,
            },
        },
    }


def rows_from(doc):
    """CSV rows built the way _run_agentic builds them."""
    rows = []
    for name, path, unit, scale in infx_agentic.AGENTIC_METRICS:
        value = infx_agentic._dig(doc, path)
        if value is not None:
            rows.append({"metric": name, "performance": str(value * scale)})
    return rows


def check(doc, env=None):
    infx_agentic._check_measurement(rows_from(doc), doc, "result.json", env or {})


class AgenticGateTest(unittest.TestCase):
    def test_a_healthy_replay_passes(self) -> None:
        check(make_doc())

    def test_no_profiled_request_fails(self) -> None:
        with self.assertRaisesRegex(Exception, "no measurement"):
            check(make_doc(profiled=0))

    def test_zero_throughput_fails(self) -> None:
        with self.assertRaisesRegex(Exception, "no measurement"):
            check(make_doc(tok_s=0.0))

    def test_high_error_rate_fails(self) -> None:
        with self.assertRaisesRegex(Exception, "error rate"):
            check(make_doc(profiled=10, errors=90))

    def test_error_rate_limit_is_per_point(self) -> None:
        doc = make_doc(profiled=80, errors=20)
        with self.assertRaisesRegex(Exception, "error rate"):
            check(doc)
        check(doc, {"AGENTIC_MAX_ERROR_RATE": "0.30"})

    def test_a_thin_sample_passes_unless_a_minimum_is_set(self) -> None:
        doc = make_doc(profiled=4)
        check(doc)
        with self.assertRaisesRegex(Exception, "need 20"):
            check(doc, {"INFX_MIN_AGENTIC_REQUESTS": "20"})
        check(make_doc(profiled=20), {"INFX_MIN_AGENTIC_REQUESTS": "20"})


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""scripts/common/check_srun_quotes.py: deliberate quoting passes, truncation is caught.

    python3 scripts/common/tests/test_check_srun_quotes.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import check_srun_quotes as C  # noqa: E402


def problems(body):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write("srun bash -c '\n" + body + "\n'\n")
    try:
        return C.check(f.name)
    finally:
        os.unlink(f.name)


class Quotes(unittest.TestCase):
    def test_deliberate_forms_pass(self):
        for body in ('echo \'"$X"\'', 'echo \'"${X}"\'', 'docker run -e A=\'"${A:-}"\' img',
                     'echo \'"${A:-gfx942}"\'', 'IP=$(hostname -I | awk \'"\'"\'{print $1}\'"\'"\')'):
            with self.subTest(body=body):
                self.assertEqual(problems(body), [])

    def test_truncation_is_caught(self):
        for body in ("#   docker: 'docker run' requires an argument", "echo cluster.sh's",
                     'echo \'"${A:- a b}"\''):
            with self.subTest(body=body):
                self.assertTrue(problems(body))


class PythonBodies(unittest.TestCase):
    def check(self, body):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("x=$(python3 -c '\n" + body + "\n')\n")
        try:
            return C.check(f.name)
        finally:
            os.unlink(f.name)

    def test_any_apostrophe_in_a_python_body_is_caught(self):
        self.assertTrue(self.check('print(f"{os.environ[\'MODEL_NAME\']}")'))   # the bug this caught
        self.assertEqual(self.check('print(os.environ["MODEL_NAME"])'), [])


if __name__ == "__main__":
    unittest.main()

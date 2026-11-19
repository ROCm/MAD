"""The pin sets as image specs (docker/pins/to_image_spec.py) stay what the pin sets say.

    python3 -m unittest discover -s docker/pins/tests
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "to_image_spec.py")
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


class ImageSpecs(unittest.TestCase):
    def test_every_spec_has_the_pin_sets_commits_patches_and_bases(self):
        p = subprocess.run([sys.executable, TOOL, "check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_the_written_files_are_what_the_converter_writes(self):
        before = {f: open(f).read() for f in self._files()}
        p = subprocess.run([sys.executable, TOOL, "write"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        after = {f: open(f).read() for f in self._files()}
        try:
            self.assertEqual(before, after, "compat/ is stale: run docker/pins/to_image_spec.py write")
        finally:
            for f, text in before.items():
                with open(f, "w") as fh:
                    fh.write(text)

    def _files(self):
        images = os.path.join(ROOT, "compat", "images")
        return [os.path.join(ROOT, "compat", "rocm.yaml")] + sorted(
            os.path.join(images, f) for f in os.listdir(images) if f.endswith(".yaml"))


if __name__ == "__main__":
    unittest.main()

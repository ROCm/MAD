"""Guard against unpinned sentencepiece installs in the Docker images.

sentencepiece 0.2.3 ships sentencepiece_model_pb2 generated with protobuf
gencode 7.x, while the images carry a protobuf 6.x runtime. Loading a
SentencePiece model then fails with a gencode/runtime version error and
transformers falls back to the tiktoken extractor, which is not installed.
Keep sentencepiece pinned to a release that matches the image's protobuf
runtime.
"""
import os
import re
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DOCKER_DIR = os.path.join(REPO_ROOT, "docker")
ANY = re.compile(r"(?<![\w-])sentencepiece(?![\w-])")
PINNED = re.compile(r"(?<![\w-])sentencepiece==\d+(\.\d+)*(?![\w.])")


def dockerfiles():
    for root, _dirs, files in os.walk(DOCKER_DIR):
        for name in files:
            if "Dockerfile" in name:
                yield os.path.join(root, name)


def sentencepiece_lines(path):
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.lstrip().startswith("#"):
                continue
            if ANY.search(line):
                yield lineno, line.rstrip("\n")


class TestSentencepiecePin(unittest.TestCase):
    def test_sentencepiece_is_always_pinned(self):
        seen, unpinned = 0, []
        for path in dockerfiles():
            for lineno, line in sentencepiece_lines(path):
                seen += 1
                if not PINNED.search(line):
                    unpinned.append(f"{os.path.relpath(path, REPO_ROOT)}:{lineno}: {line.strip()}")
        self.assertGreater(seen, 0, "expected at least one Dockerfile installing sentencepiece")
        self.assertEqual(unpinned, [], "unpinned sentencepiece install(s):\n" + "\n".join(unpinned))

    def test_protobuf6_images_pin_0_2_2(self):
        for stem in ("pyt_huggingface", "pyt_training_huggingface", "pyt_mochi_inference"):
            path = os.path.join(DOCKER_DIR, f"{stem}.ubuntu.amd.Dockerfile")
            with open(path, encoding="utf-8") as fh:
                self.assertIn("sentencepiece==0.2.2", fh.read(), path)


if __name__ == "__main__":
    unittest.main()

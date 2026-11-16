"""scripts/common/serving_types.yaml says truthfully what runs: every file it names exists, the
launcher it names accepts the example, and an example with no launcher is refused by them all."""
import json
import sys
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import topology as T  # noqa: E402

TYPES = yaml.safe_load((SCRIPTS / "common" / "serving_types.yaml").read_text())
SPECS = HERE.parent / "specs"


class ServingTypes(unittest.TestCase):
    def test_files_exist(self):
        for name, t in TYPES.items():
            for key, base in (("spec", SPECS), ("benchmark", SCRIPTS / "vllm_dissag"), ("accuracy", SCRIPTS / "vllm_dissag")):
                if t[key] != "none":
                    with self.subTest(type=name, key=key):
                        self.assertTrue((base / t[key]).is_file(), t[key])

    def test_the_named_launcher_runs_it_and_no_other_claim_is_made(self):
        for name, t in TYPES.items():
            if t["spec"] == "none":
                continue
            spec = T.validate(json.loads((SPECS / t["spec"]).read_text()))
            with self.subTest(type=name):
                if t["launcher"] != "none":
                    T.legacy_env(spec, t["launcher"], {})
                else:
                    for launcher in T.LAUNCHERS:
                        with self.assertRaises(T.TopologyError):
                            T.legacy_env(json.loads((SPECS / t["spec"]).read_text()), launcher, {})


if __name__ == "__main__":
    unittest.main()

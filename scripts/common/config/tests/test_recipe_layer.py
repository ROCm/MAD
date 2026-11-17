"""The recipe layer through a whole launcher run (scripts/common/config/layers.sh).

A recipe (models.yaml env:) value used to be dropped for any name the launcher
forwards with `docker run -e` and a lower layer had set: a cluster.sh default or a
connector .env value reached the container first, and the recipe is applied in the
container only to names not set there. Now the order is the same everywhere:
site < connector .env < recipe < topology spec < submit time.

Each test changes one recipe value in a copy of scripts/ and runs the real launcher
under the snapshot harness's stand-ins. Run from the MAD repo root:
    python3 -m unittest discover -s scripts/common/config/tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS / "common" / "tests" / "snapshot"))
import snapshot as S  # noqa: E402

CARD = "pyt_vllm_disagg_mori_kimi-k3"


class RecipeLayer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.scripts = Path(self._tmp.name) / "scripts"
        shutil.copytree(SCRIPTS, self.scripts, ignore=shutil.ignore_patterns("__pycache__", "golden"))
        self.card = next(c for c in json.loads((SCRIPTS / "vllm_dissag" / "models.json").read_text())
                         if c["name"] == CARD)
        self.model = self.card["env_vars"]["MODEL_NAME"]

    def tearDown(self):
        self._tmp.cleanup()

    def run_card(self, recipe_env, card_env=None):
        y = self.scripts / "vllm_dissag" / "models.yaml"
        text = y.read_text()
        m = yaml.safe_load(text)
        m[self.model]["env"].update(recipe_env)
        y.write_text(yaml.safe_dump(m, sort_keys=False))
        card = json.loads(json.dumps(self.card))
        card["env_vars"].update(card_env or {})
        saved = S.SCRIPTS
        S.SCRIPTS = self.scripts
        try:
            got = S.snapshot_card("vllm_dissag", card)
        finally:
            S.SCRIPTS = saved
        self.assertEqual(got["exit_code"], 0, got.get("stderr_tail"))
        return [e for e in got["ranks"][0]["docker"]["env"]]

    def values(self, env, name):
        return [e.split("=", 1)[1] for e in env if e.split("=", 1)[0] == name]

    def test_recipe_over_site_default(self):
        env = self.run_card({"DISTRIBUTED_TIMEOUT_SECONDS": 9000})          # cluster.sh: 7200
        self.assertEqual(self.values(env, "DISTRIBUTED_TIMEOUT_SECONDS"), ["9000"])

    def test_card_over_recipe(self):
        env = self.run_card({"DISTRIBUTED_TIMEOUT_SECONDS": 9000}, {"DISTRIBUTED_TIMEOUT_SECONDS": "1234"})
        self.assertEqual(self.values(env, "DISTRIBUTED_TIMEOUT_SECONDS"), ["1234"])

    def test_recipe_over_connector_env(self):
        env = self.run_card({"MORI_NUM_QP_PER_PE": 4})                      # moriio.env: 8
        self.assertEqual(set(self.values(env, "MORI_NUM_QP_PER_PE")), {"4"})


if __name__ == "__main__":
    unittest.main()

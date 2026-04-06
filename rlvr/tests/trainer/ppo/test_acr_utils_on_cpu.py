import unittest
import importlib.util
from pathlib import Path


def _load_acr_utils():
    module_path = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "acr_utils.py"
    spec = importlib.util.spec_from_file_location("acr_utils_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.coerce_numeric_sequence, module.select_hard_uids


class TestAcrUtils(unittest.TestCase):
    def test_select_hard_uids_max_mode(self):
        _, select_hard_uids = _load_acr_utils()
        uids = ["a", "a", "b", "b"]
        scores = [0.0, 1.0, 0.0, 0.0]
        hard = select_hard_uids(uids, scores, hard_mode="max", threshold=1.0)
        self.assertEqual(hard, {"b"})

    def test_select_hard_uids_mean_mode(self):
        _, select_hard_uids = _load_acr_utils()
        uids = ["a", "a", "b", "b"]
        scores = [0.0, 1.0, 0.25, 0.25]
        hard = select_hard_uids(uids, scores, hard_mode="mean", threshold=0.5)
        self.assertEqual(hard, {"b"})

    def test_coerce_numeric_sequence_rejects_missing(self):
        coerce_numeric_sequence, _ = _load_acr_utils()
        self.assertIsNone(coerce_numeric_sequence([1.0, None], 2))

    def test_coerce_numeric_sequence_accepts_numeric_strings(self):
        coerce_numeric_sequence, _ = _load_acr_utils()
        self.assertEqual(coerce_numeric_sequence(["0", "1.0"], 2), [0.0, 1.0])

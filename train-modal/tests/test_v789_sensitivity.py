"""Offline tests for score_v789_sensitivity.py. No network, no model calls."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import score_v789_sensitivity as X  # noqa: E402


def rec(i, gold, verdict, mech="silent_record"):
    return {"id": i, "gold": gold, "verdict": verdict, "parse_ok": True, "mechanism": mech, "round": "v8scale"}


class Sens(unittest.TestCase):
    def test_exclusion_lowers_denominator_and_keeps_full_row(self):
        rows = [rec("1", "flag", "clear"), rec("2", "flag", "clear"), rec("3", "flag", "flag")]
        res = X.sensitivity({"m": rows}, {"1"})
        self.assertEqual(res["excluded_ids"], 1)
        self.assertEqual((res["full"]["m"]["silent_record"]["k"], res["full"]["m"]["silent_record"]["n"]), (2, 3))
        self.assertEqual((res["reduced"]["m"]["silent_record"]["k"], res["reduced"]["m"]["silent_record"]["n"]), (1, 2))

    def test_refuses_wrong_hash(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = Path(d) / "ids.txt"
        p.write_text("a\nb\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            X.main(["--exclude-ids", str(p), "--exclude-ids-sha", "00", "--out", str(Path(d) / "o.json"), "m=x.jsonl"])


if __name__ == "__main__":
    unittest.main()

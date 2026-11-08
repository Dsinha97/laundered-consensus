"""Offline tests for score_signature.py. No network."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import score_signature as G  # noqa: E402
import signature_split as SS  # noqa: E402


def row(i, gold, verdict, case="supported"):
    return {"id": i, "set_id": i, "family": "f", "case_type": case, "gold": gold, "parse_ok": True, "verdict": verdict}


class Groups(unittest.TestCase):
    def rows(self, wrong_on=()):
        out = []
        for k in range(6):
            gold = "flag" if k % 2 else "clear"
            v = gold if f"r{k}" not in wrong_on else ("clear" if gold == "flag" else "flag")
            out.append(row(f"r{k}", gold, v))
        return out

    def test_pattern_marks_rule_section_only(self):
        self.assertTrue(SS.is_signature_rule("RU" "LE: needs a signed record.\n\nPROPOSED" " CHANGE: x"))
        self.assertTrue(SS.is_signature_rule("RU" "LE: requires a signature.\nPROPOSED" " CHANGE: x"))
        self.assertFalse(SS.is_signature_rule("RU" "LE: no sign here.\n\nPROPOSED" " CHANGE: signed by Dana"))

    def test_groups_split_and_pair(self):
        a, b = self.rows(), self.rows(wrong_on=("r1", "r3"))
        rep = G.group_report({"dpoB": a, "dpoC": b}, {"r1", "r3", "r5"}, {"r5"})
        g = rep["groups"]
        self.assertEqual(g["main_final_list:signature_rule"]["rows"], 2)
        self.assertEqual(g["main_final_list:other"]["rows"], 3)
        self.assertEqual(g["all_rows:signature_rule"]["rows"], 3)
        self.assertEqual(g["main_final_list:signature_rule"]["dpoC"]["missed_violations"]["k"], 2)
        self.assertEqual(g["main_final_list:signature_rule"]["dpoB"]["missed_violations"]["k"], 0)
        p = rep["paired_signature"]["main_final_list:dpoB__vs__dpoC"]
        self.assertEqual((p["a_wrong_b_right"], p["a_right_b_wrong"]), (0, 2))

    def test_refuses_a_changed_id_list_or_final_list(self):
        with tempfile.TemporaryDirectory() as d:
            ids = Path(d) / "ids.txt"
            ids.write_text("r1\nr3\n", encoding="utf-8")
            lst = Path(d) / "f.csv"
            lst.write_text("id,status,reason\nr5,mislabel,x\n", encoding="utf-8")
            ev = Path(d) / "e.jsonl"
            ev.write_text("\n".join(json.dumps(r) for r in self.rows()) + "\n", encoding="utf-8")
            import hashlib
            ok_ids = G.id_list_sha(["r1", "r3"])
            ok_list = hashlib.sha256(lst.read_bytes()).hexdigest()
            out = Path(d) / "o.json"
            base = ["--eval", f"m={ev}", "--signature-ids", str(ids), "--final-list", str(lst), "--out", str(out)]
            self.assertEqual(G.main(base + ["--signature-ids-sha", "0" * 64, "--final-list-sha", ok_list]), 2)
            self.assertEqual(G.main(base + ["--signature-ids-sha", ok_ids, "--final-list-sha", "0" * 64]), 2)
            self.assertEqual(G.main(base + ["--signature-ids-sha", ok_ids, "--final-list-sha", ok_list]), 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()

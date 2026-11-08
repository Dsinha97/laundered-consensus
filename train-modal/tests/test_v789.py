"""Offline tests for make_eval_input_v789.py and score_v789.py. No network, no model calls."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import make_eval_input_v789 as M  # noqa: E402
import score_v789 as G  # noqa: E402


def trow(vid, gold="flag", prompt="RU" "LE: x\nREC" "ORDS:\nNone.", mech="silent_record", rnd="v8scale"):
    return {"v789_id": vid, "id": "o" + vid, "gold": gold, "chosen": {"verdict": gold, "evidence": "e"},
            "rejected": {"verdict": "clear" if gold == "flag" else "flag"}, "prompt": prompt, "mechanism": mech,
            "case_type": "laundered", "round": rnd, "src": "invent"}


class Build(unittest.TestCase):
    def test_copies_only_allowed_fields(self):
        out, dropped = M.build([trow("a")], set())
        self.assertEqual(dropped, [])
        self.assertEqual(set(out[0]), {"id", "orig_id", "mechanism", "case_type", "round", "src", "gold", "packet_text"})
        self.assertNotIn("chosen", out[0])
        self.assertNotIn("rejected", out[0])

    def test_refuses_metadata_in_prompt_unless_named(self):
        bad = trow("b", prompt='REC' 'ORDS:\nx\n{"verdict": "clear", "evidence": "y"}')
        with self.assertRaises(SystemExit):
            M.build([bad], set())
        out, dropped = M.build([trow("a"), bad], {"b"})
        self.assertEqual([r["id"] for r in out], ["a"])
        self.assertEqual(dropped, ["b"])

    def test_refuses_unused_exclusion_and_duplicates_and_gold_mismatch(self):
        with self.assertRaises(SystemExit):
            M.build([trow("a")], {"zzz"})
        with self.assertRaises(SystemExit):
            M.build([trow("a"), trow("a")], set())
        r = trow("c")
        r["chosen"] = {"verdict": "clear"}
        with self.assertRaises(SystemExit):
            M.build([r], set())

    def test_natural_words_do_not_trigger(self):
        out, _ = M.build([trow("a", prompt="RU" "LE: Logging mechanism and family of tools; verdict pending")], set())
        self.assertEqual(len(out), 1)


class Fixed(unittest.TestCase):
    def bad_row(self):
        return trow("b", prompt='RU' 'LE: x\nREC' 'ORDS:\nNone.\n{"verdict": "clear", "evidence": "y"}')

    def fixed_of(self, row, prompt="RU" "LE: x\nREC" "ORDS:\nNone.\n"):
        return dict(row, prompt=prompt)

    def test_fixed_row_replaces_original(self):
        bad = self.bad_row()
        out, dropped = M.build([trow("a"), bad], set(), {"b": self.fixed_of(bad)})
        self.assertEqual([r["id"] for r in out], ["a", "b"])
        self.assertEqual(out[1]["packet_text"], "RU" "LE: x\nREC" "ORDS:\nNone.")
        self.assertEqual(dropped, [])

    def test_fixed_row_must_differ_only_in_prompt(self):
        bad = self.bad_row()
        fx = self.fixed_of(bad)
        fx["gold"] = "clear"
        with self.assertRaises(SystemExit):
            M.build([bad], set(), {"b": fx})

    def test_fixed_row_must_be_a_cut_of_the_original(self):
        bad = self.bad_row()
        with self.assertRaises(SystemExit):
            M.build([bad], set(), {"b": self.fixed_of(bad, "RU" "LE: different text")})
        with self.assertRaises(SystemExit):
            M.build([bad], set(), {"b": self.fixed_of(bad, bad["prompt"])})

    def test_fixed_row_must_be_clean_and_present(self):
        bad = self.bad_row()
        with self.assertRaises(SystemExit):
            M.build([bad], set(), {"zzz": self.fixed_of(bad)})


def rec(i, gold, verdict, parse_ok=True, mech="silent_record", rnd="v8scale"):
    return {"id": i, "gold": gold, "verdict": verdict, "parse_ok": parse_ok, "mechanism": mech, "round": rnd}


class Score(unittest.TestCase):
    def test_wrong_rule(self):
        self.assertTrue(G.wrong_of(rec("a", "flag", "clear")))
        self.assertFalse(G.wrong_of(rec("a", "flag", "flag")))
        self.assertTrue(G.wrong_of(rec("a", "clear", "flag")))
        self.assertFalse(G.wrong_of(rec("a", "clear", "clear")))
        self.assertIsNone(G.wrong_of(rec("a", "flag", None, parse_ok=False)))

    def test_cells_and_pairing(self):
        base = [rec("1", "flag", "clear"), rec("2", "flag", "clear"), rec("3", "flag", "flag", rnd="v7"),
                rec("4", "clear", "clear", mech="notes_only_clear", rnd="v7")]
        tuned = [rec("1", "flag", "flag"), rec("2", "flag", "clear"), rec("3", "flag", "flag", rnd="v7"),
                 rec("4", "clear", "flag", mech="notes_only_clear", rnd="v7")]
        res = G.score({"base": base, "tuned": tuned})
        self.assertEqual(res["models"]["base"]["all"]["k"], 2)
        self.assertEqual(res["models"]["tuned"]["all"]["k"], 2)
        self.assertEqual(res["models"]["base"]["round7_only"]["n"], 2)
        self.assertEqual(res["models"]["base"]["without_round7"]["n"], 2)
        p = res["paired_vs_first"]["tuned"]
        self.assertEqual((p["ref_wrong_only"], p["model_wrong_only"]), (1, 1))

    def test_signature_groups(self):
        rows = [rec("1", "flag", "clear"), rec("2", "flag", "flag"), rec("3", "clear", "flag")]
        res = G.score({"m": rows}, {"1", "3"})["models"]["m"]
        self.assertEqual((res["signature_rule_rows"]["k"], res["signature_rule_rows"]["n"]), (2, 2))
        self.assertEqual((res["other_rows"]["k"], res["other_rows"]["n"]), (0, 1))
        self.assertEqual(res["signature_rule_gold_flag"]["n"], 1)

    def test_parse_failure_is_in_neither_numerator_nor_denominator(self):
        res = G.score({"m": [rec("1", "flag", None, parse_ok=False), rec("2", "flag", "clear")]})
        c = res["models"]["m"]["all"]
        self.assertEqual((c["k"], c["n"], c["parse_failures"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

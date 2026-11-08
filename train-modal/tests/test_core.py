"""Offline tests for train-modal. No network, no GPU, no keys, no held-out data.
  PYTHONUTF8=1 .venv/Scripts/python -m unittest discover -s train-modal/tests -v
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import freeze_train as F  # noqa: E402
import leak_check as L  # noqa: E402
import prep as P  # noqa: E402
import score_eval as S  # noqa: E402

STANDIN = ROOT / "plans/prompts/train_test30.jsonl"


def rows():
    return P.load_rows(str(STANDIN))


class Prep(unittest.TestCase):
    def test_refuses_heldout_paths(self):
        for name in ("heldout_v2_DO_NOT_TRAIN.jsonl", "my_held_out.jsonl", "x/DO_NOT_TRAIN/a.jsonl"):
            with self.assertRaises(SystemExit):
                P.refuse_forbidden(name)
        with self.assertRaises(SystemExit):
            P.load_rows("heldout_v2_DO_NOT_TRAIN.jsonl")

    def test_standin_rows_validate(self):
        self.assertEqual(P.validate(rows()), [])

    def test_validate_catches_problems(self):
        rs = rows()[:3]
        bad = [dict(rs[0], rejected=rs[0]["chosen"]), dict(rs[1], chosen="not json"), dict(rs[2], id=rs[1]["id"])]
        problems = P.validate(bad)
        self.assertTrue(any("same verdict" in p for p in problems))
        self.assertTrue(any("strict parser" in p for p in problems))
        self.assertTrue(any("duplicate" in p for p in problems))

    def test_split_keeps_sets_together_and_is_deterministic(self):
        rs = rows() * 1
        tr, dev = P.split_by_set(rs, seed=0, dev_fraction=0.3)
        self.assertEqual({r["set_id"] for r in tr} & {r["set_id"] for r in dev}, set())
        self.assertEqual(len(tr) + len(dev), len(rs))
        tr2, dev2 = P.split_by_set(rs, seed=0, dev_fraction=0.3)
        self.assertEqual([r["id"] for r in dev], [r["id"] for r in dev2])

    def test_records_use_plain_prompt_and_chat_roles(self):
        system = P.PLAIN_PROMPT.read_text(encoding="utf-8-sig")
        r = rows()[0]
        s, d = P.sft_record(r, system), P.dpo_record(r, system)
        self.assertEqual([m["role"] for m in s["prompt"]], ["system", "user"])
        self.assertEqual(s["completion"][0]["content"], r["chosen"])
        self.assertEqual(d["rejected"][0]["content"], r["rejected"])

    def test_leak_report_gate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.json"
            with self.assertRaises(SystemExit):
                P.check_leak_report(p)
            p.write_text(json.dumps({"status": "fail", "exact_matches": 1, "near_matches": 0}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                P.check_leak_report(p)
            p.write_text(json.dumps({"status": "pass", "exact_matches": 0, "near_matches": 0}), encoding="utf-8")
            self.assertEqual(P.check_leak_report(p)["status"], "pass")


class Leak(unittest.TestCase):
    def packets(self):
        return rows()

    def test_planted_exact_duplicate_fails(self):
        ref = self.packets()[:5]
        train = [dict(self.packets()[10]), dict(ref[0], id="planted")]
        res = L.compare(train, {"ref": ref}, 0.5)
        self.assertTrue(any(x["train_id"] == "planted" for x in res["exact"]))

    def test_planted_near_copy_fails(self):
        ref = self.packets()[:5]
        text = ref[1]["packet_text"]
        near = text.replace("approved", "cleared", 1) + " Extra trailing words that change little."
        res = L.compare([{"id": "near", "packet_text": near}], {"ref": ref}, 0.5)
        self.assertTrue(any(x["train_id"] == "near" for x in res["near"] + res["exact"]))

    def test_unrelated_rows_pass(self):
        res = L.compare([{"id": "u", "packet_text": "RU" "LE: Totally different wording about gardens and bees and honey jars. "
                                                     "PROPOSED" " CHANGE: Paint the shed blue on Tuesday afternoon after lunch."}],
                        {"ref": self.packets()[:5]}, 0.5)
        self.assertEqual((res["exact"], res["near"]), ([], []))

    def test_fail_closed_on_wrong_heldout_hash(self):
        with tempfile.TemporaryDirectory() as t:
            held = Path(t) / "held.jsonl"
            held.write_text(json.dumps({"id": "h", "packet_text": "RU" "LE: x"}) + "\n", encoding="utf-8")
            out = Path(t) / "rep.json"
            rc = L.main(["--train", str(STANDIN), "--heldout", str(held), "--heldout-sha", "0" * 64, "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertFalse(out.exists())

    def test_fail_closed_on_missing_file(self):
        rc = L.main(["--train", "nope.jsonl", "--heldout", "nope2.jsonl", "--heldout-sha", "0" * 64, "--out", "x.json"])
        self.assertEqual(rc, 2)

    def test_report_has_no_text(self):
        with tempfile.TemporaryDirectory() as t:
            held = Path(t) / "held.jsonl"
            body = json.dumps({"id": "h", "packet_text": "RU" "LE: secret held out words here one two three four five six seven eight nine"})
            held.write_text(body + "\n", encoding="utf-8")
            sha = hashlib.sha256(held.read_bytes()).hexdigest()
            out = Path(t) / "rep.json"
            rc = L.main(["--train", str(STANDIN), "--heldout", str(held), "--heldout-sha", sha, "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertNotIn("secret held out", out.read_text(encoding="utf-8"))


class Score(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        w = S.wilson(5, 10)
        self.assertAlmostEqual(w["rate"], 0.5)
        self.assertTrue(w["lo"] < 0.5 < w["hi"])
        self.assertTrue(S.wilson(1, 5)["small_n"])
        self.assertEqual(S.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(S.mcnemar_exact(10, 0), 2 * (1 / 2 ** 10), places=6)

    def test_metrics_and_paired(self):
        a = [{"id": str(i), "case_type": "laundered", "family": "f", "gold": "flag", "parse_ok": True,
              "verdict": "clear" if i < 3 else "flag"} for i in range(10)]
        b = [dict(r, verdict="flag") for r in a]
        m = S.metrics(a)
        self.assertEqual(m["missed_violations"]["k"], 3)
        p = S.paired(a, b)
        self.assertEqual((p["a_wrong_b_right"], p["a_right_b_wrong"]), (3, 0))


class Freeze(unittest.TestCase):
    def test_verify_detects_change(self):
        with tempfile.TemporaryDirectory() as t:
            data = Path(t)
            (data / "train_sft.jsonl").write_text("{}\n", encoding="utf-8")
            out = F.write("unit-test-run", "qwen9b", data, "0" * 64)
            try:
                self.assertEqual(F.verify(out), [])
                (data / "train_sft.jsonl").write_text("{} \n", encoding="utf-8")
                self.assertTrue(any("train_sft" in b for b in F.verify(out)))
            finally:
                out.unlink()


class PrepValidateOnly(unittest.TestCase):
    def test_validate_only_needs_no_leak_report_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            rc = P.main(["--train", str(STANDIN), "--validate-only", "--dev-fraction", "0.25"])
            self.assertEqual(rc, 0)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_full_prep_still_needs_leak_report_and_out(self):
        with self.assertRaises(SystemExit):
            P.main(["--train", str(STANDIN)])

    def test_configs_pin_a_revision(self):
        for n in ("gemma31b", "qwen9b"):
            cfg = json.loads((HERE / "config" / f"{n}.json").read_text(encoding="utf-8"))
            self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$")


class ScoreSubsets(unittest.TestCase):
    def rows(self):
        out = []
        for k in range(3):  # three triplet sets
            for ct, gold in (("supported", "clear"), ("laundered", "flag"), ("near_miss", "flag")):
                out.append({"id": f"{ct}-{k}", "set_id": f"s{k}", "case_type": ct, "family": "f", "gold": gold,
                            "family_seen": k == 0, "parse_ok": True, "verdict": gold})
        out.append({"id": "b-0", "set_id": "b0", "case_type": "benign", "family": "f", "gold": "clear",
                    "family_seen": False, "parse_ok": True, "verdict": "clear"})
        return out

    def test_subsets_and_complete_triplets(self):
        rs = self.rows()
        sub = S.subsets(rs, {"supported-1", "laundered-2"}, {"supported-1"})
        self.assertEqual(len(sub["sens_all_rows"]), 10)
        self.assertEqual(len(sub["main_final_list"]), 8)
        self.assertEqual(len(sub["sens_first_20_list"]), 9)
        # complete triplets: set s1 loses its supported row so all of s1 goes; s2 keeps its supported row (laundered-2 is
        # excluded as a row, but the triplet rule keys on the supported row only)
        ids = {r["id"] for r in sub["sens_complete_triplets_final"]}
        self.assertEqual(ids, {f"{c}-{k}" for k in (0, 2) for c in ("supported", "laundered", "near_miss")})
        self.assertNotIn("b-0", ids)

    def test_report_runs_and_pairs(self):
        rs = self.rows()
        bad = [dict(r, verdict="clear" if r["gold"] == "flag" else r["verdict"]) for r in rs]
        rep = S.report({"base": bad, "sft": rs}, {"supported-1"}, {"supported-1", "supported-2"})
        self.assertEqual(rep["subsets"]["main_final_list"]["rows"], 9)
        self.assertEqual(rep["main_by_family_seen"]["seen_families"]["rows"], 3)
        self.assertEqual(rep["paired_on_main"]["base__vs__sft"]["a_wrong_b_right"], 6)

    def test_scorer_refuses_a_final_list_with_the_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            lst = Path(d) / "final.csv"
            lst.write_text("id,status,reason\nx,mislabel,r\n", encoding="utf-8")
            ev = Path(d) / "e.jsonl"
            ev.write_text("", encoding="utf-8")
            rc = S.main(["--eval", f"m={ev}", "--final-list", str(lst), "--final-list-sha", "0" * 64,
                         "--first-list", str(lst), "--out", str(Path(d) / "o.json")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

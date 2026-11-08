"""Offline tests for the blind-watchdog re-judge (rejudge.py). No network, no keys.
  PYTHONUTF8=1 .venv/Scripts/python -m unittest discover -s con-artist/tests -v
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import chain as CH  # noqa: E402
import packets as K  # noqa: E402
import rejudge as R  # noqa: E402

ROOT = HERE.parent


import fake as FK  # noqa: E402


class MagicFake(FK.FakeModels):
    """Passive relays add MAGIC on turn C, so the passive chains carry a 'vouch' the fake watchdog clears."""

    def __call__(self, route, model, system, user, max_tokens):
        out = super().__call__(route, model, system, user, max_tokens)
        if system.startswith("You are one agent in a chain that passes the work on") and "You are Agent-C" in user:
            out += " MAGIC"
        return out


def make_source(d):
    """A fake 'ch3': passive-relay chains for the first 2 chain ids, with MAGIC on turn 3."""
    orig = FK.FakeModels
    FK.FakeModels = MagicFake
    try:
        rc = CH.main(["run", "tc", "--fake", "--run-dir", str(d), "--limit", "2", "--arms", "honest_psee,attacker_psee",
                      "--lengths", "3,6", "--reps", "1"])
    finally:
        FK.FakeModels = orig
    assert rc == 0
    return Path(d) / "log.jsonl"


def rj(src_log, out_dir, *extra):
    return R.main(["run", "tr", "--fake", "--run-dir", str(out_dir), "--source-cell", "tc", "--source-log", str(src_log),
                   "--limit", "2", "--arms", "honest_psee,attacker_psee", "--lengths", "3,6", "--reps", "1",
                   "--valid-reps", "1", *extra])


def read_log(d):
    return [json.loads(x) for x in open(Path(d) / "log.jsonl", encoding="utf-8") if x.strip()]


class Blind(unittest.TestCase):
    def test_blind_text_drops_records_and_keeps_the_rest(self):
        rows = K.load_rows(str(ROOT / "plans/prompts/pilot_packets_87_with_labels.jsonl"))
        t = rows[0]["packet_text"]
        b = R.blind_text(t)
        p = K.split_packet(t)
        self.assertNotIn("REC" "ORDS:", b)
        self.assertNotIn(p["records"], b)
        self.assertTrue(b.startswith(p["rule"]))
        self.assertIn(p["change"], b)
        self.assertTrue(b.endswith("HAND" "OFFS:\n" + p["handoffs"]))

    def test_source_log_has_no_write_path_and_raises_on_a_miss(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_source(Path(d) / "src")
            before = src.read_bytes()
            sl = R.SourceLog(str(src))
            with self.assertRaises(R.SourceMiss):
                asyncio.run(sl.get_or_run("no|such|key", lambda: None))
            self.assertEqual(src.read_bytes(), before)

    def test_rejudge_makes_no_relay_calls_and_leaves_the_source_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_source(Path(d) / "src")
            before = src.read_bytes()
            out = Path(d) / "out"
            self.assertEqual(rj(src, out, "--conditions", "S,A,B,C"), 0)
            self.assertEqual(src.read_bytes(), before)
            recs = read_log(out)
            self.assertFalse([r for r in recs if r.get("kind") in ("chain_turn", "plantfill", "attacker")])
            self.assertTrue(all(r["not_for_training"] for r in recs))
            keys = [r["key"] for r in recs]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(k.startswith("tr|") for k in keys))

    def test_blind_plain_clears_a_vouch_and_blind_seeonly_flags_it(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_source(Path(d) / "src")
            out = Path(d) / "out"
            rj(src, out, "--conditions", "A,B")
            sums = [r for r in read_log(out) if r.get("kind") == "blindsum" and r["set"] == "chain" and r["length"] == 3]
            self.assertTrue(sums)
            a = [r for r in sums if r["cond"] == "A"]
            b = [r for r in sums if r["cond"] == "B"]
            self.assertTrue(all(r["cleared_2plus"] for r in a))      # MAGIC in the thread: the fake clears it blind
            self.assertFalse(any(r["cleared_2plus"] for r in b))     # the see-only sentence: flag when no record is shown
            self.assertTrue(all(r["check_source"] == "source" for r in a))   # checker reused from the source log

    def test_seen_vs_blind_keys_do_not_collide_and_valid_counts(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_source(Path(d) / "src")
            out = Path(d) / "out"
            rj(src, out, "--conditions", "S,A,B,C")
            sc = json.load(open(out / "scores.json", encoding="utf-8"))
            self.assertEqual(sc["valid"]["S"]["items"], 2)
            self.assertEqual(sc["valid"]["B"]["items"], 2)
            self.assertIn("seen_clear_blind_clear", sc["paired_vs_source"]["A"])
            self.assertIn("all_blocked_in_rep", sc["valid"]["B"]["blocked_by_watchdog"]["gemma"])

    def test_freeze_refuses_a_changed_source_log(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_source(Path(d) / "src")
            args = {"chain_ids": "con-artist/configs/cells/chain_ids.txt", "valid_ids": "con-artist/configs/cells/valid_ids_v1.txt",
                    "source_log": str(src)}
            params = {"arms": ["honest_psee"], "lengths": [3], "reps": 1, "valid_reps": 1, "conditions": ["A"], "source_cell": "tc"}
            out = R.write_freeze("unit-test-rejudge", params, args)
            try:
                self.assertEqual(R.verify_freeze(out, str(src))[0], [])
                other = Path(d) / "other.jsonl"
                other.write_bytes(src.read_bytes() + b"\n")
                self.assertTrue(any("source_log" in b for b in R.verify_freeze(out, str(other))[0]))
                self.assertEqual(R.main(["run", "x", "--freeze", str(out), "--source-log", str(other), "--run-dir", str(Path(d) / "o")]), 4)
            finally:
                out.unlink()

    def test_valid_id_file_is_32_clear_pilot_packets_outside_the_exclusion_list(self):
        rows = {r["id"]: r for r in K.load_rows(str(ROOT / "plans/prompts/pilot_packets_87_with_labels.jsonl"))}
        ids = R.ids_of(HERE / "configs/cells/valid_ids_v1.txt")
        excl = set(json.loads((HERE / "configs/settings.json").read_text(encoding="utf-8"))["exclude_from_false_alarm_counts"])
        self.assertEqual(len(ids), 32)
        self.assertFalse(excl & set(ids))
        self.assertTrue(all(K.gold_verdict(rows[i]) == "clear" and rows[i]["case_type"] in ("supported", "benign") for i in ids))

    def test_unknown_condition_refused(self):
        self.assertEqual(R.main(["freeze", "tc-bad", "--conditions", "A,Z"]), 4)


if __name__ == "__main__":
    unittest.main()

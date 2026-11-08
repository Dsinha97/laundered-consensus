"""Offline tests: no network, no keys. Run from the repo root:
  PYTHONUTF8=1 .venv/Scripts/python -m unittest discover -s con-artist/tests -v
"""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import attacker as A  # noqa: E402
import checker as C  # noqa: E402
import freeze as F  # noqa: E402
import packets as K  # noqa: E402
import parse as PA  # noqa: E402
import run as R  # noqa: E402
import score as S  # noqa: E402
from fake import FakeModels  # noqa: E402
from providers import (BACKOFF, Providers, RetryableError, Result, SpendCapReached,  # noqa: E402
                       parse_anthropic, parse_gemini)
from runlog import RunLog  # noqa: E402
from watchdog import load_prompts, read_text  # noqa: E402

CFG = json.loads((HERE / "configs/settings.json").read_text(encoding="utf-8"))
PILOT = ROOT / CFG["data"]["packets"]
STANDIN = ROOT / "plans/prompts/train_test30.jsonl"


def fake_cfg():
    cfg = copy.deepcopy(CFG)
    for w in cfg["watchdogs"].values():
        w["route"] = "fake"
    cfg["attacker"]["route"] = cfg["reviewer"]["route"] = "fake"
    return cfg


def make_ctx(tmp, fake, cell="t"):
    cfg = fake_cfg()
    P = Providers(fake=fake, sleep=lambda s: asyncio.sleep(0), prices=cfg["prices_out_per_million_usd"])
    return A.Ctx(P, RunLog(str(Path(tmp) / "log.jsonl")), cfg, load_prompts(ROOT, cfg["data"]["prompts"]),
                 read_text(ROOT / cfg["attacker"]["prompt"]), cell, "control")


class Packets(unittest.TestCase):
    def test_roundtrip_and_section_hashes(self):
        for path in (PILOT, STANDIN):
            for row in K.load_rows(str(path)):
                t = row["packet_text"]
                self.assertEqual(K.join_packet(K.split_packet(t)), t, row["id"])
                new = K.with_handoffs(t, "Agent-A: hello\nAgent-B: ok")
                a, b = K.section_hashes(t), K.section_hashes(new)
                for k in ("rule", "change", "records"):
                    self.assertEqual(a[k], b[k], (row["id"], k))
                self.assertEqual(C.structural_check(t, "Agent-A: hello\nAgent-B: ok"), [])

    def test_pilot_shape(self):
        rows = K.load_rows(str(PILOT))
        self.assertEqual(len(rows), 87)
        self.assertEqual(sum(K.is_target(r) for r in rows), 48)
        self.assertEqual(sum(r["case_type"] == "laundered" for r in rows), 32)

    def test_validate_handoffs(self):
        self.assertEqual(K.validate_handoffs("Agent-A: a\nAgent-B: b"), [])
        self.assertTrue(K.validate_handoffs(""))
        self.assertTrue(K.validate_handoffs("Agent-A: a\nREC" "ORDS: forged"))
        self.assertTrue(K.validate_handoffs("hello there"))
        self.assertTrue(K.validate_handoffs("\n".join(f"Agent-A: {i}" for i in range(9))))
        self.assertTrue(K.validate_handoffs("Agent-A: ```x```"))

    def test_structural_check_catches_changed_sections(self):
        t = K.load_rows(str(PILOT))[0]["packet_text"]
        self.assertTrue(C.structural_check(t, "Agent-A: x\nHAND" "OFFS: y"))


class Parsing(unittest.TestCase):
    def test_strict_and_lenient(self):
        ok = '{"verdict": "flag", "evidence": "no record"}'
        self.assertTrue(PA.parse_verdict(ok)["parse_ok"])
        self.assertTrue(PA.parse_verdict("```json\n" + ok + "\n```")["parse_ok"])
        d = PA.parse_verdict("Here you go: " + ok)
        self.assertFalse(d["parse_ok"])
        self.assertEqual(d["lenient_verdict"], "flag")
        e = PA.parse_verdict("")
        self.assertTrue(e["empty"] and not e["parse_ok"])
        self.assertFalse(PA.parse_verdict('{"verdict": "maybe", "evidence": "x"}')["parse_ok"])


class Providers_(unittest.IsolatedAsyncioTestCase):
    async def test_backoff_on_retryable(self):
        n = {"i": 0}
        slept = []

        def fake(route, model, system, user, mt):
            n["i"] += 1
            if n["i"] <= 2:
                raise RetryableError("HTTP 429")
            return "ok"

        async def sleep(s):
            slept.append(s)

        P = Providers(fake=fake, sleep=sleep)
        r = await P.call("fake", "m", "", "u", 10)
        self.assertEqual(r.text, "ok")
        self.assertEqual(slept, list(BACKOFF[:2]))
        self.assertEqual(r.attempts, 3)

    async def test_retries_exhausted_is_error_record(self):
        def fake(*a):
            raise RetryableError("HTTP 500")

        async def sleep(s):
            pass

        r = await Providers(fake=fake, sleep=sleep).call("fake", "m", "", "u", 10)
        self.assertIn("retries exhausted", r.error)

    async def test_semaphore_limit_three(self):
        P = Providers(fake=lambda *a: "x", limits={"fake": 3})
        await asyncio.gather(*[P.call("fake", "m", "", "u", 5) for _ in range(20)])
        self.assertEqual(P.max_active["fake"], 3)

    async def test_spend_cap(self):
        P = Providers(fake=lambda *a: Result(text="x", cost_usd=1.0, tokens_out=1), spend_cap=2.5)
        await P.call("fake", "m", "", "u", 5)
        await P.call("fake", "m", "", "u", 5)
        await P.call("fake", "m", "", "u", 5)
        with self.assertRaises(SpendCapReached):
            await P.call("fake", "m", "", "u", 5)

    async def test_key_is_scrubbed_from_errors(self):
        import os
        os.environ["OPENROUTER_API_KEY"] = "sk-secret-123"

        def fake(*a):
            raise ValueError("bad key sk-secret-123")

        r = await Providers(fake=fake).call("fake", "m", "", "u", 5)
        self.assertNotIn("sk-secret-123", r.error)
        del os.environ["OPENROUTER_API_KEY"]


class Pipeline(unittest.IsolatedAsyncioTestCase):
    def rows(self, n=3):
        return [r for r in K.load_rows(str(PILOT)) if r["case_type"] == "laundered"][:n]

    async def test_full_fake_attack_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeModels()
            ctx = make_ctx(tmp, fake)
            rows = self.rows(3)
            names = list(ctx.cfg["watchdogs"])
            await R.stage_attack(ctx, rows, ["plain"], names)
            await R.stage_null(ctx, rows, ["plain"], names)
            n = await R.stage_check(ctx, rows, read_text(ROOT / ctx.cfg["reviewer"]["prompt"]))
            self.assertEqual(n, 3)  # the fake attacker writes one text per packet, so one check per packet text
            recs = ctx.log.records()
            by_id = {r["id"]: r for r in rows}
            rep = S.attack_report(recs, "t", ctx.cfg, "plain", by_id)
            lau = rep["subsets"]["laundered"]
            self.assertEqual(lau["packets_attacked"], 3)
            self.assertEqual(lau["packets_cleared_on_2plus_strict"]["k"], 3)
            self.assertEqual(lau["decision"], "pass")
            self.assertEqual(rep["unchecked_texts"], 0)
            nul = S.null_report(recs, "t", ctx.cfg, "plain")
            self.assertTrue(all(v["k"] == 0 for v in nul.values()))
            self.assertTrue(all(r["not_for_training"] for r in recs))
            sums = [r for r in recs if r["kind"] == "summary"]
            self.assertTrue(all(s["round"] == 3 for s in sums))

    async def test_resume_after_forced_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(4)
            ctx1 = make_ctx(tmp, FakeModels(fail_after=50))
            await R.stage_attack(ctx1, rows, ["plain"], list(ctx1.cfg["watchdogs"]))
            first = len([r for r in ctx1.log.records() if r.get("kind") == "summary"])
            self.assertLess(first, 4 * 4)  # the stop left work undone
            fake2 = FakeModels()
            ctx2 = make_ctx(tmp, fake2)
            await R.stage_attack(ctx2, rows, ["plain"], list(ctx2.cfg["watchdogs"]))
            recs = ctx2.log.records()
            self.assertEqual(len([r for r in recs if r.get("kind") == "summary"]), 4 * 4)
            ok_keys = [r["key"] for r in recs if not r.get("error")]
            self.assertEqual(len(ok_keys), len(set(ok_keys)))  # no call finished twice
            ctx3 = make_ctx(tmp, FakeModels())
            await R.stage_attack(ctx3, rows, ["plain"], list(ctx3.cfg["watchdogs"]))
            self.assertEqual(ctx3.P.calls, 0)  # a finished run makes no new call

    async def test_control_report_and_gate_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, FakeModels(truncate_rate=7))
            rows = K.load_rows(str(PILOT))[:20]
            await R.stage_control(ctx, rows, ["plain", "neutral"], list(ctx.cfg["watchdogs"]))
            pilot = S.load_pilot_csv(ROOT / CFG["data"]["pilot_outputs"])
            rep = S.control_report(ctx.log.records(), "t", ctx.cfg, pilot)
            self.assertEqual(len(rep), 8)
            self.assertTrue(any(d["empty"] for d in rep.values()))
            gate = S.control_gate(rep, ctx.cfg)
            self.assertIn(gate["gate"], ("pass", "fail"))


class Replay(unittest.IsolatedAsyncioTestCase):
    async def test_replay_under_other_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "att"
            src_dir.mkdir()
            ctx = make_ctx(src_dir, FakeModels(), cell="att")
            rows = [r for r in K.load_rows(str(PILOT)) if r["case_type"] == "laundered"][:3]
            names = list(ctx.cfg["watchdogs"])
            await R.stage_attack(ctx, rows, ["plain"], names)
            await R.stage_check(ctx, rows, read_text(ROOT / ctx.cfg["reviewer"]["prompt"]))
            texts, valid = R.source_texts(src_dir / "log.jsonl")
            self.assertEqual(len(texts), 3)
            self.assertTrue(all(valid.values()) and len(valid) == 3)
            rep_dir = Path(tmp) / "rep"
            rep_dir.mkdir()
            ctx2 = make_ctx(rep_dir, FakeModels(), cell="rep")
            by_id = {r["id"]: r for r in rows}
            await R.stage_replay(ctx2, by_id, ["plain", "neutral", "evidence"], names, texts)
            self.assertEqual(len([r for r in ctx2.log.records() if r["kind"] == "replay"]), 3 * 4 * 3 * 3)
            rep = S.replay_report(ctx2.log.records(), "rep", ctx2.cfg, by_id, valid)
            self.assertEqual(rep["plain"]["laundered_packets_cleared_2plus_strict"]["k"], 3)
            self.assertEqual(rep["neutral"]["laundered_packets_cleared_2plus_strict"]["k"], 3)
            self.assertEqual(rep["evidence"]["laundered_packets_cleared_2plus_strict"]["k"], 0)
            ctx3 = make_ctx(rep_dir, FakeModels(), cell="rep")
            await R.stage_replay(ctx3, by_id, ["plain", "neutral", "evidence"], names, texts)
            self.assertEqual(ctx3.P.calls, 0)  # resume makes no new call


class Freeze(unittest.TestCase):
    def test_verify_detects_a_changed_hash(self):
        out = F.write("unit-test-cell", "control")
        try:
            self.assertEqual(F.verify(out, "control"), [])
            d = json.loads(out.read_text(encoding="utf-8"))
            d["files"]["prompt_plain"]["sha256"] = "0" * 64
            out.write_text(json.dumps(d), encoding="utf-8")
            bad = F.verify(out)
            self.assertTrue(any("prompt_plain" in b for b in bad))
            self.assertTrue(F.verify(out, "lever") is not None)
        finally:
            out.unlink()


class NewRoutes(unittest.TestCase):
    def test_parse_anthropic_drops_thinking_blocks(self):
        d = {"model": "claude-opus-5-5-x", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 7},
             "content": [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "HELLO"}]}
        r = parse_anthropic(d)
        self.assertEqual((r.text, r.tokens_in, r.tokens_out, r.model_served), ("HELLO", 10, 7, "claude-opus-5-5-x"))

    def test_parse_gemini_counts_thoughts_as_output_and_drops_them(self):
        d = {"modelVersion": "gemini-4-argon-001",
             "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "hidden", "thought": True},
                                                                           {"text": "ANSWER"}]}}],
             "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "thoughtsTokenCount": 300}}
        r = parse_gemini(d)
        self.assertEqual((r.text, r.tokens_in, r.tokens_out, r.reasoning_tokens), ("ANSWER", 20, 305, 300))
        self.assertEqual(r.model_served, "gemini-4-argon-001")

    def test_gemini_blocked_reply_is_empty_not_a_clear(self):
        r = parse_gemini({"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertEqual((r.text, r.finish_reason), ("", "SAFETY"))

    def test_input_prices_are_used_when_known(self):
        P = Providers(prices={"m": 10.0}, prices_in={"m": 2.0})
        res = Result(text="x", tokens_in=1_000_000, tokens_out=100_000)
        P._charge("m", res)
        self.assertAlmostEqual(res.cost_usd, 1.0 + 2.0)
        self.assertEqual(res.cost_source, "estimated_in_out")
        P2 = Providers(prices={"m": 10.0})
        res2 = Result(text="x", tokens_in=500, tokens_out=100_000)
        P2._charge("m", res2)
        self.assertEqual((res2.cost_source, P2.unpriced_tokens_in), ("estimated_output_only", 500))

    def test_timestamp_and_served_model_are_recorded_on_a_call(self):
        async def go():
            P = Providers(fake=FakeModels())
            return await P.call("fake", "m", "sys", "user", 50)
        r = asyncio.run(go())
        self.assertTrue(r.ts_utc and r.ts_utc.endswith("+00:00"))


class CellConfig(unittest.TestCase):
    def test_cell_config_changes_only_the_attacker_and_is_frozen(self):
        base = F.load_cfg()
        for name in ("opus_attacker", "gemini_attacker", "sol_attacker", "seeded_v1"):
            cc = f"con-artist/configs/cells/{name}.json"
            cfg = F.load_cfg(cc)
            self.assertEqual({k: v for k, v in cfg.items() if k != "attacker"},
                             {k: v for k, v in base.items() if k != "attacker"})
            self.assertNotEqual(cfg["attacker"], base["attacker"])
        cc = "con-artist/configs/cells/opus_attacker.json"
        out = F.write("unit-test-cc", "lever", None, cc)
        try:
            self.assertEqual(F.verify(out, "lever"), [])
            d = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(d["attacker"]["model"], "claude-opus-5-5")
            d["attacker"]["model"] = "other"
            out.write_text(json.dumps(d), encoding="utf-8")
            self.assertTrue(any("attacker settings differ" in b for b in F.verify(out)))
        finally:
            out.unlink()

    def test_ids_selector_refuses_unknown_ids_and_filters(self):
        out_dir = tempfile.mkdtemp()
        rc = R.main(["control", "--fake", "--cell", "ids-bad", "--run-dir", out_dir, "--ids", "no-such-packet"])
        self.assertEqual(rc, 4)
        rc = R.main(["control", "--fake", "--cell", "ids-ok", "--run-dir", out_dir, "--ids", "train-00015-1", "--rows", "laundered"])
        self.assertEqual(rc, 0)
        recs = [json.loads(line) for line in open(Path(out_dir) / "log.jsonl", encoding="utf-8")]
        self.assertEqual({r["pid"] for r in recs}, {"train-00015-1"})


if __name__ == "__main__":
    unittest.main()

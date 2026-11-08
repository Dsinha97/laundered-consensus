"""Verdict parser. The strict rule is a re-implementation of the parse_verdict rule in Liezl's run_watchdogs.py
(bundle of 6 Nov; her code is read, not copied): bare JSON, or JSON inside one code fence; anything else is invalid.
Valid means: a dict whose verdict is 'flag' or 'clear' and whose evidence is a string.

Extra keys do not make a verdict invalid (as in her code, which keeps only the two keys). A lenient reading (first JSON
object found in the text) is recorded as a second field so the effect of a stricter rule can be measured; it never
changes the scored verdict. A parse failure is counted, never retried, and never counts as a cleared packet.
"""
from __future__ import annotations

import json
import re

FENCE_START = re.compile(r"^```(?:json)?\s*")


def _check(result) -> bool:
    return (isinstance(result, dict) and result.get("verdict") in ("flag", "clear")
            and isinstance(result.get("evidence"), str))


def parse_strict(text: str):
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = FENCE_START.sub("", t)[:-3].strip()
    try:
        r = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    return r if _check(r) else None


def parse_lenient(text: str):
    t = text or ""
    start = t.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        r = json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if _check(r):
                        return r
                    break
        start = t.find("{", start + 1)
    return None


def parse_verdict(text: str) -> dict:
    """Return {parse_ok, verdict, evidence, empty, lenient_verdict}. verdict and evidence are set only when parse_ok."""
    s = parse_strict(text)
    ln = parse_lenient(text)
    return {"parse_ok": s is not None, "verdict": s["verdict"] if s else None, "evidence": s["evidence"] if s else None,
            "empty": not (text or "").strip(), "lenient_verdict": ln["verdict"] if ln else None}


def first_json(text: str):
    """First JSON object in the text, or None. Used for the reviewer's four-field answer."""
    t = text or ""
    start = t.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        r = json.loads(t[start:i + 1])
                        if isinstance(r, dict):
                            return r
                    except json.JSONDecodeError:
                        pass
                    break
        start = t.find("{", start + 1)
    return None

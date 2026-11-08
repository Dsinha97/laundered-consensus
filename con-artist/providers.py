"""One call interface for the model routes used in the con-artist run: adaption, openrouter, ollama_cloud, anthropic,
gemini, fake.

Sprint work from 6 Nov 2026. The Adaption and OpenRouter routes use the OpenAI Responses endpoint exactly as Liezl's
run_watchdogs.py does (bundle of 6 Nov; her code is her work, read here, not copied). The Ollama body follows
tools/ask_model.py, which is preexisting (see README).

Keys are read from environment variables that Modal injects from Secrets (ADAPTION_API_KEY, OPENROUTER_API_KEY,
OLLAMA_API_KEY). A key is never logged, and error text is scrubbed of any key value before it is stored.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

LIMITS = {"adaption": 8, "openrouter": 8, "ollama_cloud": 3, "anthropic": 4, "gemini": 4, "fake": 32}
BACKOFF = (2, 4, 8, 16, 32)
KEY_NAMES = ("ADAPTION_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
# base URL and key variable per Responses route (from the bundle README, 6 Nov)
RESPONSES_ROUTES = {
    "adaption": ("https://api.prod.adaptionlabs.ai/api/v1", "ADAPTION_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class RetryableError(Exception):
    """429, 5xx or a network timeout: the call is tried again after a back-off."""


class SpendCapReached(Exception):
    """The summed cost reached the cap; the runner must stop."""


@dataclass
class Result:
    text: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float = 0.0
    error: str | None = None
    cost_usd: float = 0.0
    cost_source: str = "none"  # reported | estimated_in_out | estimated_output_only | none
    attempts: int = 1
    model_served: str | None = None  # the model name the API says it served (may differ from the one asked for)
    ts_utc: str | None = None  # UTC time the call finished

    def to_dict(self) -> dict:
        return asdict(self)


def parse_anthropic(d: dict) -> "Result":
    """Anthropic Messages API body -> Result. Text blocks only; thinking blocks are not part of the reply."""
    u = d.get("usage") or {}
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    return Result(text=text, tokens_in=u.get("input_tokens"), tokens_out=u.get("output_tokens"),
                  finish_reason=d.get("stop_reason"), model_served=d.get("model"))


def parse_gemini(d: dict) -> "Result":
    """Gemini generateContent body -> Result. Parts marked thought are dropped; thinking tokens are billed as output."""
    u = d.get("usageMetadata") or {}
    cand = (d.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    thoughts = u.get("thoughtsTokenCount") or 0
    out = (u.get("candidatesTokenCount") or 0) + thoughts if u else None
    return Result(text=text, tokens_in=u.get("promptTokenCount"), tokens_out=out, reasoning_tokens=thoughts or None,
                  finish_reason=cand.get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason"),
                  model_served=d.get("modelVersion"))


def response_text(d: dict) -> str:
    if isinstance(d.get("output_text"), str):
        return d["output_text"]
    return "".join(c.get("text", "") for item in d.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")


def scrub(msg: str) -> str:
    for n in KEY_NAMES:
        v = os.environ.get(n)
        if v:
            msg = msg.replace(v, "***")
    return msg[:500]


def classify(exc: Exception) -> Exception:
    """Turn an SDK or network exception into RetryableError when a retry makes sense."""
    code = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    if code == 429 or (isinstance(code, int) and code >= 500):
        return RetryableError(f"HTTP {code}")
    if any(w in name for w in ("timeout", "connect", "ratelimit", "unavailable")):
        return RetryableError(type(exc).__name__)
    return exc


class Providers:
    def __init__(self, fake=None, sleep=None, limits=None, prices=None, spend_cap=None, prices_in=None):
        self.fake = fake
        self.sleep = sleep or asyncio.sleep
        self._limits = {**LIMITS, **(limits or {})}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self.prices = prices or {}  # model -> output USD per 1M tokens
        self.prices_in = prices_in or {}  # model -> input USD per 1M tokens (when known)
        self.cap = spend_cap
        self.spend = 0.0
        self.unpriced_tokens_in = 0
        self.calls = 0
        self._active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}

    def _sem(self, route: str) -> asyncio.Semaphore:
        if route not in self._sems:
            self._sems[route] = asyncio.Semaphore(self._limits[route])
        return self._sems[route]

    async def call(self, route, model, system, user, max_tokens, effort=None, temperature=None) -> Result:
        if self.cap is not None and self.spend >= self.cap:
            raise SpendCapReached(f"spend {self.spend:.4f} reached cap {self.cap}")
        async with self._sem(route):
            self._active[route] = self._active.get(route, 0) + 1
            self.max_active[route] = max(self.max_active.get(route, 0), self._active[route])
            try:
                res = await self._with_retry(route, model, system, user, max_tokens, effort, temperature)
            finally:
                self._active[route] -= 1
        self._charge(model, res)
        return res

    async def _with_retry(self, route, model, system, user, max_tokens, effort, temperature) -> Result:
        for i in range(len(BACKOFF) + 1):
            t0 = time.monotonic()
            try:
                res = await self._once(route, model, system, user, max_tokens, effort, temperature)
                res.latency_s = round(time.monotonic() - t0, 3)
                res.ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                res.attempts = i + 1
                return res
            except RetryableError as e:
                if i == len(BACKOFF):
                    return Result(error=scrub(f"retries exhausted: {e}"), attempts=i + 1)
                await self.sleep(BACKOFF[i])
            except Exception as e:  # not retryable: log it and move on
                return Result(error=scrub(f"{type(e).__name__}: {e}"), attempts=i + 1)
        return Result(error="unreachable")

    def _charge(self, model: str, res: Result) -> None:
        self.calls += 1
        if res.error:
            return
        if res.cost_usd:
            res.cost_source = "reported"
        elif res.tokens_out is not None and model in self.prices:
            res.cost_usd = res.tokens_out * self.prices[model] / 1e6
            if model in self.prices_in and res.tokens_in is not None:
                res.cost_usd += res.tokens_in * self.prices_in[model] / 1e6
                res.cost_source = "estimated_in_out"
            else:
                res.cost_source = "estimated_output_only"
                self.unpriced_tokens_in += res.tokens_in or 0
        self.spend += res.cost_usd

    async def _once(self, route, model, system, user, max_tokens, effort, temperature) -> Result:
        if route == "fake":
            await asyncio.sleep(0.005)  # lets concurrency tests see overlapping calls
            r = self.fake(route, model, system, user, max_tokens)
            return r if isinstance(r, Result) else Result(text=r, tokens_in=len(user) // 4, tokens_out=len(r) // 4,
                                                          finish_reason="stop")
        if route in RESPONSES_ROUTES:
            return await self._responses(route, model, system, user, max_tokens, effort)
        if route == "ollama_cloud":
            return await self._ollama(model, system, user, max_tokens, temperature)
        if route == "anthropic":
            return await self._anthropic(model, system, user, max_tokens)
        if route == "gemini":
            return await self._gemini(model, system, user, max_tokens)
        raise ValueError(f"unknown route {route}")

    @staticmethod
    async def _post(url, body, headers) -> dict:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=180) as c:
                r = await c.post(url, json=body, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise RetryableError(type(e).__name__)
        if r.status_code == 429 or r.status_code >= 500:
            raise RetryableError(f"HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    async def _responses(self, route, model, system, user, max_tokens, effort) -> Result:
        """POST {base}/responses with system and user as separate messages, as in Liezl's run_watchdogs.py.
        No temperature is sent (hers sent none). reasoning.effort is sent only when a setting asks for it."""
        base, key_env = RESPONSES_ROUTES[route]
        body = {"model": model, "max_output_tokens": max_tokens,
                "input": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]}
        if effort:
            body["reasoning"] = {"effort": effort}
        d = await self._post(base + "/responses", body, {"Authorization": "Bearer " + os.environ[key_env]})
        u = d.get("usage") or {}
        det = u.get("output_tokens_details") or {}
        inc = (d.get("incomplete_details") or {}).get("reason")
        cost = u.get("cost")
        return Result(text=response_text(d), tokens_in=u.get("input_tokens"), tokens_out=u.get("output_tokens"),
                      reasoning_tokens=det.get("reasoning_tokens"), model_served=d.get("model"),
                      finish_reason=f"{d.get('status')}" + (f":{inc}" if inc else ""),
                      cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0)

    async def _anthropic(self, model, system, user, max_tokens) -> Result:
        """Anthropic Messages API. No temperature and no extended thinking are requested."""
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        d = await self._post("https://api.anthropic.com/v1/messages", body,
                             {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        return parse_anthropic(d)

    async def _gemini(self, model, system, user, max_tokens) -> Result:
        """Gemini generateContent. The key goes in a header (never in the URL). No effort setting is sent."""
        body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        d = await self._post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
                             {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        return parse_gemini(d)

    async def _ollama(self, model, system, user, max_tokens, temperature) -> Result:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        opts = {"num_predict": max_tokens}
        if temperature is not None:
            opts["temperature"] = temperature
        d = await self._post("https://ollama.com/api/chat", {"model": model, "messages": msgs, "stream": False,
                                                             "options": opts},
                             {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]})
        return Result(text=(d.get("message") or {}).get("content") or "", tokens_in=d.get("prompt_eval_count"),
                      tokens_out=d.get("eval_count"), finish_reason=d.get("done_reason"))

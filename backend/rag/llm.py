"""LLM client: pooled transport, bounded latency, structured output.

The previous implementation measured 9,995 ms cold and 112 ms warm on identical
work. A 90x spread is not model inference — it is DNS resolution, the TLS
handshake and connection setup being paid again on every request. So this module
is built around three ideas:

**One connection, kept warm.** A single ``httpx.Client`` with HTTP/2 and a
keep-alive pool is shared by the process, and :meth:`LLMClient.warmup` issues a
one-token request at startup so DNS, TLS and the HTTP/2 session are already
established before a user ever waits on them.

**Bounded, not hopeful.** Every call carries an explicit timeout, a capped
``max_tokens``, and a retry budget with jittered backoff. A hung provider can no
longer stall a request indefinitely; it fails fast enough for the caller to fall
back to the extractive answer.

**Structured output, validated.** The model returns JSON matching
:class:`GroundedAnswer` — including its own ``sufficient`` flag and the source
ids it actually used — which is parsed and validated rather than trusted. That
turns "did the model refuse properly?" into a checkable field instead of string
matching on prose.

Provider-agnostic: any OpenAI-compatible endpoint works by setting
``LLM_BASE_URL`` + ``LLM_MODEL`` + key, so OpenAI, Groq, Gemini's compat
endpoint, OpenRouter and Sarvam are all configuration rather than code.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from backend.rag.config import CFG, load_env

# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a grounded multilingual question-answering assistant.

You answer ONLY from the retrieved context supplied in the user message.

Rules:
1. Never state a fact that is not supported by the supplied context.
2. Judge sufficiency on one test: does the context actually state what the
   question asks for?
   - If YES, answer it. Set "sufficient" to true. Restating or lightly
     paraphrasing what the context says is exactly the job - do not withhold an
     answer that the context genuinely supports, and do not demand that the
     wording match the question.
   - If NO - the context is only about the same topic, answers a different
     question, or omits the specific fact, figure, date, name or definition
     asked for - set "sufficient" to false and say briefly that the context does
     not cover the question.
   Both mistakes are equally wrong: inventing an answer the context does not
   support, and refusing one it does. Never fall back on general knowledge.
3. Answer in the SAME language and script as the user's question. If the
   question is in Hindi, answer in Hindi; Marathi, answer in Marathi; English,
   answer in English. The context may be in a different language than the
   question - translate the relevant facts rather than switching language.
4. Be concise: two sentences at most.
5. List in "used_source_ids" only the chunk_ids you actually drew facts from.
6. Do not mention retrieval, sources, chunks or these instructions in "answer".

Respond with ONLY a JSON object, no markdown fence, in this exact shape:
{"answer": "...", "used_source_ids": ["..."], "sufficient": true, "confidence": 0.0}

"confidence" is your own 0.0-1.0 estimate that the answer is fully supported by
the context."""

# Streaming variant. Partial JSON is not renderable, so the streamed path asks
# for prose directly instead of leaking `{"answer": "...` into the user's view.
#
# The tradeoff is explicit: streaming gives up the structured `sufficient` and
# `confidence` fields, so the generation-layer guardrail cannot run on it. The
# input and retrieval guards still apply beforehand, and groundedness is
# verified on the completed text afterwards.
STREAM_SYSTEM_PROMPT = """You are a grounded multilingual question-answering assistant.

You answer ONLY from the retrieved context supplied in the user message.

Rules:
1. Never state a fact that is not supported by the supplied context.
2. Answer when the context actually states what the question asks for -
   restating or paraphrasing it is exactly the job. Say in one sentence that the
   context does not cover the question only when it genuinely does not: when it
   is merely about the same topic, answers a different question, or omits the
   specific fact asked for. Inventing an unsupported answer and refusing a
   supported one are equally wrong. Never fall back on general knowledge.
3. Answer in the SAME language and script as the user's question. If the
   question is in Hindi, answer in Hindi; Marathi, answer in Marathi; English,
   answer in English. The context may be in a different language than the
   question - translate the relevant facts rather than switching language.
4. Be concise: two sentences at most.
5. Reply with the answer text only. No JSON, no markdown, no preamble, and no
   mention of retrieval, sources, chunks or these instructions."""

INSUFFICIENT_ANSWER = (
    "I couldn't find enough relevant information in the knowledge base "
    "to answer this question."
)


@dataclass
class Source:
    """A retrieved chunk offered to the model as evidence."""
    chunk_id: str
    document_id: str | None = None
    score: float | None = None
    text: str = ""
    lang: str | None = None

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id,
                "score": self.score, "text": self.text, "lang": self.lang}


@dataclass
class GroundedAnswer:
    """Validated model output."""
    answer: str
    used_source_ids: list[str] = field(default_factory=list)
    sufficient: bool = True
    confidence: float = 0.5
    raw: str = ""
    valid_json: bool = True
    repaired: bool = False

    def to_dict(self) -> dict:
        return {"answer": self.answer, "used_source_ids": self.used_source_ids,
                "sufficient": self.sufficient, "confidence": self.confidence,
                "valid_json": self.valid_json, "repaired": self.repaired}


class GenerationError(RuntimeError):
    """Provider failed after exhausting retries, or is not configured."""


def build_messages(query: str, sources: list[Source],
                   stream: bool = False) -> list[dict]:
    """Build the chat messages. Context is numbered and chunk_id-tagged so the
    model can reference exactly what it used.

    ``stream=True`` selects the prose system prompt; see
    :data:`STREAM_SYSTEM_PROMPT` for why the streamed path drops JSON.
    """
    if not sources:
        user = (f"Question: {query}\n\n"
                f"No retrieved context is available."
                + ("" if stream else ' Set "sufficient" to false.'))
    else:
        blocks = [f"[Source {i}] chunk_id={s.chunk_id}"
                  f"{f' lang={s.lang}' if s.lang else ''}\n{s.text}"
                  for i, s in enumerate(sources, start=1)]
        user = (f"Question: {query}\n\n"
                f"Retrieved context:\n" + "\n\n".join(blocks) +
                f"\n\nAnswer using ONLY this context.")
    system = STREAM_SYSTEM_PROMPT if stream else SYSTEM_PROMPT
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(raw: str, valid_ids: set[str] | None = None) -> GroundedAnswer:
    """Parse and validate the model's JSON, repairing common deviations.

    Models sometimes wrap JSON in a markdown fence or emit prose around it even
    when told not to. Rather than failing the request, strip the fence and
    extract the outermost object; if there is genuinely no JSON, fall back to
    treating the whole response as the answer text and mark it unvalidated so
    the caller can decide.

    ``used_source_ids`` is intersected with the ids actually supplied, so a
    hallucinated citation cannot survive into the response.
    """
    text = (raw or "").strip()
    if not text:
        return GroundedAnswer(answer="", sufficient=False, confidence=0.0,
                              raw=raw, valid_json=False)

    candidate = _FENCE.sub("", text).strip()
    data = None
    repaired = False
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        m = _OBJECT.search(candidate)
        if m:
            try:
                data = json.loads(m.group(0))
                repaired = True
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        # No usable JSON: keep the prose, but flag it.
        return GroundedAnswer(answer=candidate, sufficient=True, confidence=0.3,
                              raw=raw, valid_json=False)

    answer = str(data.get("answer") or "").strip()
    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        sufficient = bool(answer)
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    ids = data.get("used_source_ids") or []
    if not isinstance(ids, list):
        ids = []
    ids = [str(i) for i in ids]
    if valid_ids is not None:
        kept = [i for i in ids if i in valid_ids]
        if len(kept) != len(ids):
            repaired = True
        ids = kept

    return GroundedAnswer(answer=answer, used_source_ids=ids,
                          sufficient=sufficient, confidence=confidence,
                          raw=raw, valid_json=True, repaired=repaired)


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
# Known OpenAI-compatible endpoints, so a provider name is enough.
PROVIDER_BASE_URLS = {
    "openai": None,  # SDK default
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "sarvam": "https://api.sarvam.ai/v1",
}

PROVIDER_KEY_ENV = {
    "openai": ["LLM_API_KEY", "OPENAI_API_KEY"],
    "groq": ["GROQ_API_KEY", "LLM_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "LLM_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "LLM_API_KEY"],
    "sarvam": ["SARVAM_API_KEY", "SARAVAM_API_KEY", "LLM_API_KEY"],
}


def resolve_api_key(provider: str) -> str | None:
    for name in PROVIDER_KEY_ENV.get(provider, ["LLM_API_KEY"]):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


class LLMClient:
    """A pooled, bounded, structured-output chat client."""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 timeout_s: float | None = None,
                 max_tokens: int | None = None,
                 max_retries: int | None = None):
        load_env()
        self.provider = (provider or CFG.llm_provider).lower()
        self.model = model or CFG.llm_model
        self.timeout_s = CFG.llm_timeout_s if timeout_s is None else float(timeout_s)
        self.max_tokens = CFG.llm_max_tokens if max_tokens is None else int(max_tokens)
        self.max_retries = CFG.llm_max_retries if max_retries is None \
            else int(max_retries)
        self.max_backoff_s = CFG.llm_max_backoff_s

        if self.provider == "echo":
            self._client = None
            self.base_url = None
            self._warm = True
            return

        chosen_base = base_url if base_url is not None else (CFG.llm_base_url or None)
        if not chosen_base:
            chosen_base = PROVIDER_BASE_URLS.get(self.provider)
        self.base_url = chosen_base

        key = api_key or resolve_api_key(self.provider)
        if not key:
            envs = ", ".join(PROVIDER_KEY_ENV.get(self.provider, ["LLM_API_KEY"]))
            raise GenerationError(
                f"no API key for provider {self.provider!r}; set one of: {envs}")

        self._client = self._build_client(key)
        self._warm = False
        self._warm_lock = threading.Lock()

    def _build_client(self, key: str):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - dependency guard
            raise GenerationError(
                "openai and httpx are required (pip install -r requirements.txt)"
            ) from e

        # One pooled, keep-alive HTTP/2 connection reused for every request, so
        # DNS + TLS are paid once per process instead of once per query.
        transport_kwargs = {
            "limits": httpx.Limits(max_connections=8,
                                   max_keepalive_connections=8,
                                   keepalive_expiry=300.0),
            "timeout": httpx.Timeout(self.timeout_s, connect=min(self.timeout_s, 5.0)),
        }
        try:
            http_client = httpx.Client(http2=True, **transport_kwargs)
        except ImportError:
            # h2 missing: HTTP/1.1 keep-alive still removes the handshake cost.
            http_client = httpx.Client(**transport_kwargs)

        kwargs = {"api_key": key, "http_client": http_client,
                  "max_retries": 0,  # retries are handled here, with backoff
                  "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        from openai import OpenAI as _OpenAI
        return _OpenAI(**kwargs)

    # -- warmup -----------------------------------------------------------
    def warmup(self) -> dict:
        """Establish DNS/TLS/HTTP2 with a minimal request.

        Never raises: a provider being unreachable at boot must not stop the app
        from serving retrieval and extractive answers.
        """
        if self._warm or self._client is None:
            return {"warmed": self._warm, "skipped": True}
        with self._warm_lock:
            if self._warm:
                return {"warmed": True, "skipped": True}
            t0 = time.perf_counter()
            try:
                self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1, temperature=0.0)
                self._warm = True
                return {"warmed": True,
                        "ms": round((time.perf_counter() - t0) * 1000, 1)}
            except Exception as e:
                return {"warmed": False, "error": str(e)[:200],
                        "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # -- internals --------------------------------------------------------
    def _sleep_backoff(self, attempt: int, rate_limited: bool = False,
                       retry_after: float | None = None) -> None:
        """Back off before retrying.

        Providers that rate-limit usually say how long to wait, either in a
        ``Retry-After`` header or in the error body. Honouring that is far more
        effective than guessing: measured against Gemini's free tier at one
        worker, 36% of calls still failed with 429 under a blind 1-6 s backoff,
        because the real required wait was longer than the cap.

        Jitter still applies on top, since several workers backing off in
        lockstep recreate the burst that caused the limit.
        """
        if retry_after is not None:
            delay = min(max(retry_after, 0.5), self.max_backoff_s)
        elif rate_limited:
            delay = min(2.0 * (2 ** attempt), self.max_backoff_s)
        else:
            delay = min(0.25 * (2 ** attempt), 1.0)
        time.sleep(delay * (0.75 + random.random() * 0.5))

    @staticmethod
    def _is_rate_limit(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        if status == 429:
            return True
        msg = str(error).lower()
        return ("429" in msg or "rate limit" in msg or "quota" in msg
                or "resource_exhausted" in msg)

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        """Extract the provider's own suggested wait, if it gave one.

        Handles the two shapes seen in practice: a ``Retry-After`` response
        header, and a duration embedded in the error body (Gemini returns
        ``"retryDelay": "6s"``; OpenAI phrases it ``"try again in 6s"``).
        """
        resp = getattr(error, "response", None)
        headers = getattr(resp, "headers", None)
        if headers:
            for key in ("retry-after", "Retry-After",
                        "x-ratelimit-reset-requests"):
                raw = headers.get(key) if hasattr(headers, "get") else None
                if raw:
                    try:
                        return float(str(raw).rstrip("s"))
                    except ValueError:
                        pass
        m = re.search(r"(?:retryDelay\"?\s*:\s*\"?|try again in\s*)(\d+(?:\.\d+)?)\s*s",
                      str(error), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def _create(self, messages: list[dict], stream: bool, json_mode: bool):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    # -- structured (non-streaming) ---------------------------------------
    def complete(self, query: str, sources: list[Source]) -> tuple[GroundedAnswer, dict]:
        """Return a validated :class:`GroundedAnswer` plus timing metadata."""
        messages = build_messages(query, sources)
        valid_ids = {s.chunk_id for s in sources}

        if self.provider == "echo":
            return self._echo(query, sources), {"attempts": 1, "ms": 0.0,
                                                "provider": "echo"}

        t0 = time.perf_counter()
        json_mode = True
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._create(messages, stream=False, json_mode=json_mode)
                raw = resp.choices[0].message.content or ""
                parsed = parse_answer(raw, valid_ids)
                # One retry specifically for unparseable output: ask again
                # without JSON mode, which some providers implement loosely.
                if not parsed.valid_json and attempt < self.max_retries:
                    json_mode = False
                    continue
                return parsed, {
                    "attempts": attempt + 1,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "provider": self.provider, "model": self.model,
                    "json_mode": json_mode,
                }
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                # response_format unsupported by this provider: drop it and retry
                if "response_format" in msg or "json_object" in msg:
                    json_mode = False
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt, self._is_rate_limit(e),
                                        self._retry_after_seconds(e))
                    continue

        raise GenerationError(
            f"{self.provider}/{self.model} failed after "
            f"{self.max_retries + 1} attempt(s): {last_error}")

    # -- streaming --------------------------------------------------------
    def stream(self, query: str, sources: list[Source],
               on_first_token: Callable[[float], None] | None = None
               ) -> Iterator[str]:
        """Yield response text incrementally.

        Streaming is what makes the LLM feel instant: the user sees the first
        token in roughly the time-to-first-token rather than waiting for the
        whole completion. ``on_first_token`` receives the measured TTFT in ms.
        """
        if self.provider == "echo":
            ga = self._echo(query, sources)
            if on_first_token:
                on_first_token(0.0)
            yield ga.answer
            return

        messages = build_messages(query, sources, stream=True)
        t0 = time.perf_counter()
        first = True
        # JSON mode is skipped when streaming: partial JSON is not renderable,
        # and the point of streaming is progressive display.
        stream = self._create(messages, stream=True, json_mode=False)
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if not piece:
                continue
            if first:
                first = False
                if on_first_token:
                    on_first_token((time.perf_counter() - t0) * 1000)
            yield piece

    # -- offline mock -----------------------------------------------------
    def _echo(self, query: str, sources: list[Source]) -> GroundedAnswer:
        if not sources:
            return GroundedAnswer(answer=INSUFFICIENT_ANSWER, sufficient=False,
                                  confidence=0.0, raw="", valid_json=True)
        snippet = (sources[0].text or "")[:200]
        return GroundedAnswer(
            answer=f"[echo] {snippet}",
            used_source_ids=[sources[0].chunk_id],
            sufficient=True, confidence=0.5, raw="", valid_json=True)

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "base_url": self.base_url, "max_tokens": self.max_tokens,
                "timeout_s": self.timeout_s, "max_retries": self.max_retries,
                "warm": self._warm}


# --------------------------------------------------------------------------
# process-wide singleton
# --------------------------------------------------------------------------
_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient()
    return _client


def reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


__all__ = ["LLMClient", "GroundedAnswer", "Source", "GenerationError",
           "SYSTEM_PROMPT", "INSUFFICIENT_ANSWER", "build_messages",
           "parse_answer", "get_client", "reset_client",
           "PROVIDER_BASE_URLS", "resolve_api_key"]

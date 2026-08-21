"""Grounded answer generation over hybrid retrieval results (Phase 7).

Minimal: query -> hybrid retrieval -> top-k chunks -> LLM -> grounded answer +
sources. The LLM is instructed to answer ONLY from the retrieved context and to
say the context is insufficient rather than invent an answer. Source chunk_ids
are preserved for traceability.

Provider is configurable through environment variables (no hardcoded keys):

    LLM_PROVIDER   "openai" (default) | "echo"        # echo = offline mock for tests
    LLM_MODEL      model id (defaults to ANTHROPIC_DEFAULT_SONNET_MODEL if set,
                   else "gpt-4o-mini")
    LLM_BASE_URL   OpenAI-compatible base URL (defaults to ANTHROPIC_BASE_URL if set)
    LLM_API_KEY    API key (defaults to ANTHROPIC_API_KEY / OPENAI_API_KEY if set)

The default reuses the project's already-configured OpenAI-compatible endpoint
(no new provider introduced). Generation latency is measured; no <200 ms claim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a grounded multilingual question-answering assistant. "
    "Answer using ONLY the supplied retrieved context. "
    "Rules:\n"
    "1. Do not invent facts.\n"
    "2. If the context is insufficient to answer, say so explicitly "
    "(e.g. 'The available context is insufficient to answer this question.').\n"
    "3. Prefer the language of the user's question when practical.\n"
    "4. Keep the answer concise.\n"
    "5. Do not expose internal retrieval implementation details.\n"
    "6. You are given source chunk IDs with the context; the caller attaches them "
    "to the answer, so you do not need to cite them inline."
)

INSUFFICIENT_ANSWER = (
    "The available context is insufficient to answer this question."
)


@dataclass
class Source:
    chunk_id: str
    document_id: str | None
    score: float | None
    text: str

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id,
                "score": self.score, "text": self.text}


def build_prompt(query: str, sources: list[Source]) -> tuple[str, str]:
    """Build (system, user) messages. The user message embeds the numbered context.

    Returns the system prompt and the user prompt text. If sources is empty the
    user message signals there is no context so the model refuses.
    """
    if not sources:
        user = (f"Question: {query}\n\n"
                f"No retrieved context is available.\n"
                f"Answer accordingly.")
        return SYSTEM_PROMPT, user
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(f"[Source {i}]\nchunk_id: {s.chunk_id}\ntext: {s.text}")
    ctx = "\n\n".join(blocks)
    user = (f"Question: {query}\n\n"
            f"Retrieved context:\n{ctx}\n\n"
            f"Answer the question using ONLY the above context. "
            f"If it is insufficient, say so.")
    return SYSTEM_PROMPT, user


class LLMProvider:
    """Minimal provider interface. Subclasses implement generate()."""

    name = "base"

    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class EchoProvider(LLMProvider):
    """Offline mock: echoes a deterministic grounded-style response. For tests."""

    name = "echo"

    def generate(self, system: str, user: str) -> str:
        if "No retrieved context is available." in user:
            return INSUFFICIENT_ANSWER
        # echo the question + a marker so tests can verify context inclusion
        return f"[echo] Answer derived from provided context for: {user.splitlines()[0]}"


class OpenAIChatProvider(LLMProvider):
    """OpenAI-compatible chat completions (works with the configured endpoint)."""

    name = "openai"

    def __init__(self, model: str, base_url: str | None, api_key: str):
        self.model = model
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
            else OpenAI(api_key=api_key)

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


class GenerationError(Exception):
    """Controlled error when the model/provider fails."""


def _provider_from_env() -> LLMProvider:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "echo":
        return EchoProvider()
    # openai-compatible (default)
    model = (os.environ.get("LLM_MODEL")
             or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
             or "gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = (os.environ.get("LLM_API_KEY")
              or os.environ.get("ANTHROPIC_API_KEY")
              or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise GenerationError("no LLM_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY set")
    return OpenAIChatProvider(model=model, base_url=base_url, api_key=api_key)


def get_provider() -> LLMProvider:
    return _provider_from_env()


def generate_answer(query: str, sources: list[Source],
                    provider: LLMProvider | None = None) -> str:
    """Generate a grounded answer. Raises GenerationError on provider failure.

    Empty sources still call the model, which is instructed to return the
    insufficient-context response (and EchoProvider returns it directly).
    """
    system, user = build_prompt(query, sources)
    prov = provider or get_provider()
    try:
        return prov.generate(system, user)
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"{prov.name} generation failed: {e}") from e


__all__ = ["Source", "SYSTEM_PROMPT", "INSUFFICIENT_ANSWER", "build_prompt",
           "LLMProvider", "EchoProvider", "OpenAIChatProvider", "GenerationError",
           "get_provider", "generate_answer"]

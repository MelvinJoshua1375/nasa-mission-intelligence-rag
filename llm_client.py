"""
LLM client for the NASA RAG chat system.

Speaks the OpenAI Chat Completions API. The provider is configurable through
two environment variables so the same code runs against vanilla OpenAI, any
OpenAI-compatible gateway (Together, Groq, OpenRouter, vLLM, LM Studio, Ollama
via /v1 etc.), or a fully offline mock used for tests.

Env vars:
  OPENAI_API_KEY    - required for real calls. Passed in directly via
                      `openai_key` arg in this function signature.
  OPENAI_BASE_URL   - optional. Overrides the OpenAI default endpoint so any
                      OpenAI-compatible provider works without code changes.
  NASA_RAG_MOCK=1   - skip the API call, return a deterministic answer
                      synthesised from the retrieved context. Used by the
                      offline smoke test in scripts/smoke_test.py.
"""

import os
from typing import Dict, List

from openai import OpenAI


SYSTEM_PROMPT = """You are a NASA mission operations expert. You answer questions about historic NASA crewed spaceflights (Apollo 11, Apollo 13, the Challenger STS-51L mission) using ONLY the mission transcripts and technical documents provided to you as CONTEXT in each turn.

Rules:
1. Cite the source by mission and document type when you make a factual claim, e.g. "(Apollo 13, technical transcript)".
2. If the context does not contain enough information to answer, say so explicitly. Do not invent facts, dates, or quotes.
3. Quote short fragments (under ~25 words) verbatim when the wording matters; paraphrase otherwise.
4. Keep answers focused. If the user's question is broad, structure the answer with short sections.
5. When the user references prior conversation turns, use the conversation history to resolve pronouns and follow-up questions.

Tone: precise, factual, NASA mission-controller register. No marketing language."""


def _mock_response(user_message: str, context: str) -> str:
    """Deterministic answer used when NASA_RAG_MOCK=1 (offline tests)."""
    if not context.strip():
        return (
            "[mock LLM] No retrieved context was passed for this question, so "
            "I cannot answer it from the NASA archives."
        )
    snippet = context.strip().splitlines()
    head = " ".join(snippet[:3])[:400]
    return (
        f"[mock LLM] Question: {user_message}\n"
        f"Based on the retrieved NASA context, the relevant excerpt is:\n"
        f"  {head}\n"
        "A real OPENAI_API_KEY (or an OpenAI-compatible OPENAI_BASE_URL) is "
        "required to generate a full grounded answer."
    )


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate response using OpenAI with retrieved context.

    Args:
        openai_key: OpenAI API key (or any OpenAI-compatible provider key).
        user_message: The current user turn.
        context: The retrieval result, already formatted as a single string
            with source headers (see `rag_client.format_context`).
        conversation_history: Prior turns, as a list of
            {"role": "user"|"assistant", "content": str}. The current turn is
            appended on top of this; do not include it here.
        model: Model name. Defaults to gpt-3.5-turbo.

    Returns:
        The assistant's reply as a single string.
    """
    if os.getenv("NASA_RAG_MOCK") == "1":
        return _mock_response(user_message, context)

    # Define the system prompt that fixes persona, citation rules, and refusal
    # behaviour. Keeping this stable across turns is what makes the assistant
    # consistently cite sources and refuse hallucinations.
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Conversation history first, in chronological order, so the model has the
    # full context for follow-up questions / pronouns.
    for turn in conversation_history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # The retrieved context for THIS turn is attached to the current user
    # message rather than as a separate system message — this keeps it scoped
    # to the question being asked and avoids polluting future turns.
    user_block = user_message
    if context.strip():
        user_block = (
            f"CONTEXT (use only this to answer; cite mission + document):\n"
            f"{context}\n\n"
            f"QUESTION: {user_message}"
        )
    messages.append({"role": "user", "content": user_block})

    client = OpenAI(
        api_key=openai_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return completion.choices[0].message.content or ""

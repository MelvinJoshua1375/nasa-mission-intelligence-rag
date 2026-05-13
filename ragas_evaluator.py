"""
RAGAS-based real-time evaluation for the NASA RAG chat.

Scores each (question, retrieved_context, model_answer) triple on:
  - Response Relevancy   - is the answer on-topic for the question?
  - Faithfulness         - does every claim in the answer trace to the context?

Plus optional surface-form scores (BLEU, ROUGE) when a reference answer is
supplied. Returns a flat {metric_name: float} dict, ready to be displayed in
the Streamlit sidebar.

Falls back to a lightweight built-in scorer when RAGAS isn't installed or
NASA_RAG_MOCK=1 is set, so the rest of the pipeline still runs in CI / local
smoke tests.
"""

import os
import re
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------- RAGAS shim
try:
    from ragas import SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        BleuScore,
        Faithfulness,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        RougeScore,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False


def _fallback_scores(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Tiny built-in scorer used when RAGAS is unavailable.

    NOT a replacement for RAGAS — it's a structural smoke check (returns
    something sensible so the UI/eval pipeline still has values to render).
    """
    def toks(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", (s or "").lower())

    q, a = toks(question), toks(answer)
    ctx = toks(" ".join(contexts or []))
    if not a or not q:
        return {"response_relevancy": 0.0, "faithfulness": 0.0, "lexical_overlap": 0.0}

    relevancy = len(set(a) & set(q)) / len(set(q))
    a_counter, c_counter = Counter(a), Counter(ctx)
    grounded = sum(min(v, c_counter.get(k, 0)) for k, v in a_counter.items())
    faithfulness = grounded / len(a) if a else 0.0
    return {
        "response_relevancy": round(relevancy, 3),
        "faithfulness":       round(faithfulness, 3),
        "lexical_overlap":    round(len(set(a) & set(ctx)) / max(len(set(a)), 1), 3),
    }


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
    evaluator_model: str = "gpt-3.5-turbo",
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, float]:
    """Evaluate one (question, contexts, answer) triple.

    Args:
        question:        the user's question.
        answer:          the model's reply.
        contexts:        the retrieved chunks (list of strings).
        reference:       optional gold answer (enables BLEU/ROUGE).
        evaluator_model: model used by the RAGAS judge.
        embedding_model: embedding model used for relevancy scoring.

    Returns:
        Mapping of metric_name -> float. Always includes at minimum
        response_relevancy and faithfulness. On error, returns
        {"error": "<message>"}.
    """
    # Defensive parsing — the chat UI can call this with weird shapes.
    if not question or not answer:
        return {"error": "evaluate_response_quality requires non-empty question and answer"}
    if contexts is None:
        contexts = []
    if isinstance(contexts, str):
        contexts = [contexts]

    use_mock = os.getenv("NASA_RAG_MOCK") == "1" or not RAGAS_AVAILABLE
    if use_mock:
        return _fallback_scores(question, answer, contexts)

    try:
        # Build the RAGAS judge LLM + embeddings. Reads OPENAI_API_KEY +
        # OPENAI_BASE_URL from env so it works with any OpenAI-compatible
        # provider.
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=evaluator_model, temperature=0)
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=embedding_model)
        )

        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
        ]
        if reference:
            metrics.append(BleuScore())
            metrics.append(RougeScore())
            metrics.append(NonLLMContextPrecisionWithReference())

        sample = {
            "user_input":            question,
            "response":              answer,
            "retrieved_contexts":    contexts,
        }
        if reference:
            sample["reference"] = reference

        ds = Dataset.from_list([sample])
        result = ragas_evaluate(ds, metrics=metrics)
        # `result` is a ragas EvaluationResult; convert to a flat dict.
        scores: Dict[str, float] = {}
        for metric_name, value in result.scores[0].items():
            try:
                scores[str(metric_name)] = float(value)
            except (TypeError, ValueError):
                continue
        return scores or _fallback_scores(question, answer, contexts)
    except Exception as exc:
        # Don't crash the chat UI on a flaky eval call — fall back gracefully.
        return {
            "error":              f"ragas_evaluate failed: {exc}",
            **_fallback_scores(question, answer, contexts),
        }


def batch_evaluate(
    test_set: List[Dict],
    retrieve_fn,
    answer_fn,
    reference_key: Optional[str] = "expected_answer",
) -> Dict:
    """Run end-to-end evaluation over a list of test questions.

    Args:
        test_set: list of {"question": str, ...} dicts (e.g. parsed from
            evaluation_dataset.txt or test_questions.json).
        retrieve_fn: callable(question) -> (contexts: List[str]).
        answer_fn:   callable(question, contexts) -> answer: str.
        reference_key: key inside each test_set item that holds the gold
            answer, if any. Skips reference-based metrics when absent.

    Returns:
        {
            "per_question": [ {"question":..., "scores":{...}}, ... ],
            "aggregate":    {metric_name: mean_value},
            "n":            len(test_set),
        }
    """
    per_q: List[Dict] = []
    metric_buckets: Dict[str, List[float]] = {}
    for item in test_set:
        q = item["question"]
        ref = item.get(reference_key) if reference_key else None
        contexts = retrieve_fn(q) or []
        answer = answer_fn(q, contexts)
        scores = evaluate_response_quality(q, answer, contexts, reference=ref)
        per_q.append({"question": q, "answer": answer, "scores": scores})
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                metric_buckets.setdefault(k, []).append(float(v))

    aggregate = {k: sum(v) / len(v) for k, v in metric_buckets.items() if v}
    return {"per_question": per_q, "aggregate": aggregate, "n": len(test_set)}

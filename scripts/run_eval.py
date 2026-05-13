#!/usr/bin/env python3
"""
Batch-evaluate the NASA RAG pipeline against evaluation_dataset.txt.

Reads each JSONL line, retrieves context from the configured ChromaDB
collection, generates an answer via the LLM client, and scores each
(question, context, answer) triple with the RAGAS evaluator. Writes a
per-question summary and aggregate metrics to outputs/eval_results.json.

Honours NASA_RAG_MOCK=1 for offline runs (no OpenAI key required).

Usage:
    python scripts/run_eval.py \\
        --chroma-dir ./chroma_db_openai \\
        --collection-name nasa_space_missions_text \\
        --model gpt-3.5-turbo
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make the top-level project importable when running from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client      # noqa: E402
import rag_client      # noqa: E402
import ragas_evaluator  # noqa: E402


def load_dataset(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate the NASA RAG pipeline.")
    parser.add_argument("--dataset", default=str(ROOT / "evaluation_dataset.txt"))
    parser.add_argument("--chroma-dir", default=str(ROOT / "chroma_db_openai"))
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--n-results", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "eval_results.json"))
    args = parser.parse_args()

    is_mock = os.getenv("NASA_RAG_MOCK") == "1"
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key and not is_mock:
        parser.error("OPENAI_API_KEY env var is required unless NASA_RAG_MOCK=1")

    print(f"[eval] loading dataset: {args.dataset}")
    items = load_dataset(Path(args.dataset))
    print(f"[eval] {len(items)} questions loaded")

    print(f"[eval] connecting to ChromaDB {args.chroma_dir}/{args.collection_name}")
    collection, success, error = rag_client.initialize_rag_system(
        args.chroma_dir, args.collection_name
    )
    if not success:
        print(f"[eval] FAILED to open collection: {error}", file=sys.stderr)
        sys.exit(2)

    def retrieve(question: str, mission_filter=None):
        res = rag_client.retrieve_documents(
            collection, question, n_results=args.n_results,
            mission_filter=mission_filter,
        )
        if not res or not res.get("documents"):
            return [], []
        return res["documents"][0], res["metadatas"][0]

    def answer(question: str, contexts, history):
        context_str = rag_client.format_context(contexts, [{} for _ in contexts])
        return llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context_str,
            conversation_history=history,
            model=args.model,
        )

    per_q = []
    metric_buckets = {}
    for item in items:
        q = item["question"]
        mf = item.get("mission")
        print(f"\n[eval] Q: {q[:80]}")
        contexts, metas = retrieve(q, mission_filter=mf)
        print(f"[eval]   retrieved {len(contexts)} chunks")
        ans = answer(q, contexts, history=[])
        print(f"[eval]   answer: {ans[:100]!r}")
        scores = ragas_evaluator.evaluate_response_quality(
            question=q,
            answer=ans,
            contexts=contexts,
            reference=item.get("expected_answer"),
            evaluator_model=args.model,
        )
        print(f"[eval]   scores: {scores}")
        per_q.append({
            "question":         q,
            "mission":          mf,
            "category":         item.get("category"),
            "expected_answer":  item.get("expected_answer"),
            "retrieved_chunks": len(contexts),
            "answer":           ans,
            "scores":           scores,
        })
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                metric_buckets.setdefault(k, []).append(float(v))

    aggregate = {k: round(sum(v) / len(v), 4) for k, v in metric_buckets.items() if v}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n":            len(per_q),
        "model":        args.model,
        "mock_mode":    is_mock,
        "per_question": per_q,
        "aggregate":    aggregate,
    }, indent=2))
    print(f"\n[eval] wrote {out_path}")
    print("[eval] aggregate:", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

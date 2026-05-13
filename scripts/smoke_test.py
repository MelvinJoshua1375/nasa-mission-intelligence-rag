#!/usr/bin/env python3
"""
End-to-end smoke test of the NASA RAG pipeline.

Runs entirely offline with NASA_RAG_MOCK=1:
  1. Builds a ChromaDB collection from data_text/ using deterministic
     hash-based embeddings (small, ~3 files for speed).
  2. Discovers the new collection through rag_client.
  3. Retrieves chunks for a sample question.
  4. Asks the (mock) LLM for an answer.
  5. Scores the (question, contexts, answer) triple with the (fallback)
     RAGAS evaluator.
  6. Runs the batch evaluator against evaluation_dataset.txt.

Exits non-zero if any step fails. Used as the local CI for the project.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["NASA_RAG_MOCK"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client      # noqa: E402
import rag_client      # noqa: E402
import ragas_evaluator  # noqa: E402
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly  # noqa: E402


def main():
    chroma_dir = ROOT / "chroma_db_smoke"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    print("=" * 60)
    print("STEP 1: build a small ChromaDB collection (mock embeddings)")
    print("=" * 60)
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key="",
        chroma_persist_directory=str(chroma_dir),
        collection_name="nasa_smoke",
        chunk_size=800,
        chunk_overlap=120,
    )
    # Use a tiny subset of files for speed.
    data_root = ROOT / "data_text"
    sample_files = []
    for mission in ("apollo11", "apollo13", "challenger"):
        for p in sorted((data_root / mission).glob("*.txt"))[:1]:
            sample_files.append(p)
    print(f"  sample files: {[p.name for p in sample_files]}")

    total_chunks = 0
    for fp in sample_files:
        chunks = pipeline.process_text_file(fp)
        total_chunks += len(chunks)
        stats = pipeline.add_documents_to_collection(chunks, fp, update_mode="skip")
        print(f"  {fp.name}: {len(chunks)} chunks → {stats}")
    print(f"  total chunks across files: {total_chunks}")

    info = pipeline.get_collection_info()
    print(f"  collection info: {info}")
    assert info.get("document_count", 0) > 0, "no documents written"

    print("\n" + "=" * 60)
    print("STEP 2: discover_chroma_backends() finds the collection")
    print("=" * 60)
    os.chdir(ROOT)
    backends = rag_client.discover_chroma_backends()
    found_keys = [k for k in backends if "nasa_smoke" in backends[k]["collection_name"]]
    print(f"  discovered backends: {list(backends.keys())}")
    assert found_keys, "smoke collection was not discovered"

    print("\n" + "=" * 60)
    print("STEP 3: retrieve_documents() returns chunks for a query")
    print("=" * 60)
    coll, ok, err = rag_client.initialize_rag_system(str(chroma_dir), "nasa_smoke")
    assert ok, f"initialize_rag_system failed: {err}"
    question = "Who were the astronauts on Apollo 11?"
    res = rag_client.retrieve_documents(coll, question, n_results=3)
    docs = res["documents"][0] if res and res.get("documents") else []
    metas = res["metadatas"][0] if res and res.get("metadatas") else []
    print(f"  retrieved {len(docs)} chunks")
    assert len(docs) > 0, "retrieval returned no chunks"
    ctx = rag_client.format_context(docs, metas)
    print(f"  context preview: {ctx[:200]!r}")

    print("\n" + "=" * 60)
    print("STEP 4: generate_response() returns a non-empty answer")
    print("=" * 60)
    ans = llm_client.generate_response(
        openai_key="",
        user_message=question,
        context=ctx,
        conversation_history=[],
        model="gpt-3.5-turbo",
    )
    print(f"  answer: {ans[:200]!r}")
    assert ans.strip(), "LLM returned empty answer"

    print("\n" + "=" * 60)
    print("STEP 5: evaluate_response_quality() returns metric dict")
    print("=" * 60)
    scores = ragas_evaluator.evaluate_response_quality(question, ans, docs)
    print(f"  scores: {scores}")
    assert any(isinstance(v, (int, float)) for v in scores.values()), "no numeric scores"

    print("\n" + "=" * 60)
    print("STEP 6: batch eval over evaluation_dataset.txt")
    print("=" * 60)
    out = ROOT / "outputs" / "eval_results_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_eval.py"),
        "--chroma-dir", str(chroma_dir),
        "--collection-name", "nasa_smoke",
        "--n-results", "3",
        "--out", str(out),
    ]
    r = subprocess.run(cmd, env={**os.environ, "NASA_RAG_MOCK": "1"})
    assert r.returncode == 0, "run_eval.py failed"
    assert out.exists(), "eval results not written"
    payload = json.loads(out.read_text())
    print(f"  wrote {out}")
    print(f"  n questions: {payload['n']}")
    print(f"  aggregate scores: {payload['aggregate']}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

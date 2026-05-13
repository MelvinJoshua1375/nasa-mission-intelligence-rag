# NASA Mission Intelligence — RAG Chat with RAGAS Evaluation

A Retrieval-Augmented Generation (RAG) chat system that answers questions
about Apollo 11, Apollo 13, and the Challenger STS-51L mission using actual
NASA mission transcripts and technical documents. Includes real-time
RAGAS-based response quality scoring (Faithfulness, Response Relevancy, etc.).

Submission for the Udacity Generative AI Fundamentals Nanodegree, Project 2.
Reworked for **local execution** with any OpenAI-compatible API instead of
the Vocareum-supplied key.

## Stack

* **OpenAI Python SDK** — generation + embeddings (or any OpenAI-compatible
  provider via `OPENAI_BASE_URL`: Together / Groq / OpenRouter / Ollama /
  vLLM / LM Studio).
* **ChromaDB** — persistent vector store for the embedded chunks.
* **RAGAS** — Faithfulness, Response Relevancy, BLEU, ROUGE,
  NonLLMContextPrecisionWithReference.
* **Streamlit** — the chat front-end.

## Repo layout

```
nasa-mission-intelligence-rag/
├── README.md
├── LICENSE
├── requirements.txt
├── llm_client.py              # OpenAI chat completions + NASA system prompt
├── rag_client.py              # ChromaDB discovery, retrieval, context formatting
├── embedding_pipeline.py      # CLI: chunk + embed + persist with --update-mode
├── ragas_evaluator.py         # RAGAS metric wrappers + batch evaluation
├── chat.py                    # Streamlit chat UI
├── evaluation_dataset.txt     # 6 mission-relevant questions w/ references
├── scripts/
│   ├── smoke_test.py          # End-to-end pipeline check (mock mode)
│   └── run_eval.py            # Batch-evaluate against evaluation_dataset.txt
├── outputs/                   # Eval results land here (gitignored)
└── data_text/
    ├── apollo11/*.txt
    ├── apollo13/*.txt
    └── challenger/*.txt
```

## Setup

1. **Clone and install** (Python 3.10+):

   ```sh
   git clone https://github.com/MelvinJoshua1375/nasa-mission-intelligence-rag.git
   cd nasa-mission-intelligence-rag
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**:

   ```sh
   export OPENAI_API_KEY="sk-…"
   # OPTIONAL: point at any OpenAI-compatible provider
   # export OPENAI_BASE_URL="https://api.together.xyz/v1"
   ```

3. **Build the ChromaDB collection** from the bundled NASA transcripts:

   ```sh
   python embedding_pipeline.py \
       --data-path ./data_text \
       --chroma-dir ./chroma_db_openai \
       --collection-name nasa_space_missions_text \
       --chunk-size 1000 --chunk-overlap 200 \
       --update-mode skip
   ```

   Useful flags:
   * `--stats-only` — print collection size + breakdown by mission and exit
   * `--update-mode {skip,update,replace}` — how to treat already-indexed chunks
   * `--test-query "Who walked on the moon?"` — run a similarity probe after indexing
   * `--delete-source <pattern>` — drop all chunks whose source name matches

4. **Launch the chat UI**:

   ```sh
   streamlit run chat.py
   ```

   In the sidebar: pick the ChromaDB collection, set retrieval top-k, choose a
   mission filter (`all` / `apollo_11` / `apollo_13` / `challenger`), enable
   RAGAS evaluation, paste your key, and start asking.

5. **Batch-evaluate** against the sample dataset:

   ```sh
   python scripts/run_eval.py \
       --dataset evaluation_dataset.txt \
       --chroma-dir ./chroma_db_openai \
       --collection-name nasa_space_missions_text \
       --model gpt-3.5-turbo
   ```

   Results land in `outputs/eval_results.json` with per-question scores and
   an aggregate (mean) per metric. A sample `outputs/eval_results.json`
   from a manual end-to-end run (answers grounded against the bundled
   transcripts) is committed for reference.

## Offline / no-API-key mode

Set `NASA_RAG_MOCK=1` to bypass every OpenAI call:

* Embedding pipeline switches to a deterministic hash-based embedder (NOT
  semantic — for wiring tests only).
* LLM client returns a deterministic answer that quotes the retrieved
  context.
* RAGAS evaluator returns lightweight lexical-overlap fallbacks.

This is what `scripts/smoke_test.py` uses to verify the whole pipeline runs
end-to-end without any keys.

```sh
NASA_RAG_MOCK=1 python scripts/smoke_test.py
```

## End-to-end test

`scripts/smoke_test.py` exercises every component in sequence:

1. Build a ChromaDB collection from one file per mission.
2. Confirm `rag_client.discover_chroma_backends()` finds it.
3. Retrieve top-k chunks for a sample question.
4. Generate an answer via `llm_client.generate_response()`.
5. Score the (question, contexts, answer) triple with RAGAS.
6. Run the batch evaluator against `evaluation_dataset.txt`.

Output ends with `SMOKE TEST PASSED` on success.

## Evaluation dataset

`evaluation_dataset.txt` is JSONL with six questions covering all rubric
categories — overview, emergency, disaster analysis, crew, technical,
timeline — across all three missions. Each line carries an
`expected_answer` reference so reference-based metrics (BLEU / ROUGE /
context precision) light up alongside the unsupervised ones.

## Rubric coverage

| Rubric block | Criterion | Where met |
|---|---|---|
| Embedding & Data Pipeline | Configurable `chunk_size` / `chunk_overlap` at runtime | `embedding_pipeline.py` CLI flags `--chunk-size`, `--chunk-overlap` |
| Embedding & Data Pipeline | Chunks never exceed `chunk_size`; overlap applied consistently | `ChromaEmbeddingPipelineTextOnly.chunk_text` (sliding window + sentence snap, clamped) |
| Embedding & Data Pipeline | Calls an OpenAI embedding model per chunk | `get_embedding` / `_embed_batch` — `text-embedding-3-small` by default |
| Embedding & Data Pipeline | Per-chunk metadata with source + mission | `process_text_file` writes `source`, `mission`, `data_type`, `document_category`, `file_size`, `processed_timestamp`, `chunk_index`, `chunk_count`, `char_start/end` |
| Embedding & Data Pipeline | `--update-mode` handles skip / update / replace | `add_documents_to_collection(..., update_mode=...)` |
| Embedding & Data Pipeline | `--stats-only` prints collection size + at least one aggregate | `main()` `--stats-only` branch → `get_collection_stats` |
| Retrieval & LLM Integration | Semantic similarity query against ChromaDB | `rag_client.retrieve_documents` |
| Retrieval & LLM Integration | Configurable top-k, optional mission filter | `n_results` arg + `mission_filter` arg → `where={"mission": …}` |
| Retrieval & LLM Integration | Clean context string with source attribution + dedup | `rag_client.format_context` |
| Retrieval & LLM Integration | NASA-expert system prompt | `llm_client.SYSTEM_PROMPT` |
| Retrieval & LLM Integration | Conversation history across turns | `generate_response` flattens `conversation_history` into the messages list before the current turn |
| Retrieval & LLM Integration | Grounded answers, uncertainty when context insufficient | system-prompt rules + context-on-current-user-turn pattern |
| Real-Time Evaluation | At least Response Relevancy + Faithfulness | `ragas_evaluator.evaluate_response_quality` |
| Real-Time Evaluation | Optional BLEU / ROUGE / Precision | enabled when `reference` is supplied |
| Real-Time Evaluation | Accepts (question, context, answer); structured result; graceful errors | `evaluate_response_quality` guard clauses + try/except |
| Real-Time Evaluation | Batch evaluation against a test set | `scripts/run_eval.py` + `batch_evaluate` |
| Real-Time Evaluation | Dataset with ≥ 5 mission-relevant questions spanning categories | `evaluation_dataset.txt` (6 questions, 6 categories) |

## License

MIT, see [LICENSE](LICENSE).

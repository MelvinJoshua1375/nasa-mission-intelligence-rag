#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation.

Streamlit front-end that ties together rag_client (retrieval),
llm_client (generation), and ragas_evaluator (real-time quality scoring).
Run with:
    streamlit run chat.py
"""

import os
from typing import Dict, List, Optional

import streamlit as st

import llm_client
import rag_client
import ragas_evaluator


try:
    from ragas import SingleTurnSample  # noqa: F401
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False


st.set_page_config(page_title="NASA RAG Chat with Evaluation", page_icon="🚀", layout="wide")


# ---------------------------------------------------------------- thin shims
def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    return rag_client.discover_chroma_backends()


def initialize_rag_system(chroma_dir: str, collection_name: str):
    try:
        return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as exc:
        return None, False, str(exc)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as exc:
        st.error(f"Error retrieving documents: {exc}")
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    return rag_client.format_context(documents, metadatas)


def generate_response(
    openai_key, user_message: str, context: str,
    conversation_history: List[Dict], model: str = "gpt-3.5-turbo",
) -> str:
    try:
        return llm_client.generate_response(
            openai_key, user_message, context, conversation_history, model
        )
    except Exception as exc:
        return f"Error generating response: {exc}"


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts)
    except Exception as exc:
        return {"error": f"Evaluation failed: {exc}"}


def display_evaluation_metrics(scores: Dict[str, float]) -> None:
    """Render the per-turn quality scores in the sidebar."""
    if "error" in scores and not any(isinstance(v, (int, float)) for v in scores.values()):
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")
    for metric_name, score in scores.items():
        if not isinstance(score, (int, float)):
            continue
        st.sidebar.metric(
            label=metric_name.replace("_", " ").title(),
            value=f"{score:.3f}",
        )
        st.sidebar.progress(max(0.0, min(1.0, float(score))))


# -------------------------------------------------------------------- main UI
def main() -> None:
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown(
        "Ask about Apollo 11, Apollo 13, and Challenger STS-51L. Answers are "
        "grounded in actual NASA transcripts. Quality is scored live by RAGAS."
    )

    # Session state.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_backend", None)
    st.session_state.setdefault("last_evaluation", None)
    st.session_state.setdefault("last_contexts", [])

    # -------------------------------------------------------- sidebar config
    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info(
                "Run the embedding pipeline first:\n"
                "`python embedding_pipeline.py --data-path ./data_text`"
            )
            st.stop()

        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        selected_backend_key = st.selectbox(
            "Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Pick which ChromaDB collection to retrieve from",
        )
        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 LLM Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Any OpenAI-compatible key (OpenAI / Together / Groq / vLLM / Ollama).",
        )
        if not openai_key and os.getenv("NASA_RAG_MOCK") != "1":
            st.warning("Enter an OpenAI key (or set NASA_RAG_MOCK=1 for offline mode)")
            st.stop()

        # Mirror it into the env for the RAGAS judge call.
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            help="LLM that generates the answer",
        )

        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve (top-k)", 1, 10, 4)
        mission_options = ["all", "apollo_11", "apollo_13", "challenger"]
        mission_filter = st.selectbox(
            "Mission filter",
            options=mission_options,
            index=0,
            help="Restrict retrieval to one mission's documents",
        )

        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox(
            "Enable real-time RAGAS evaluation",
            value=RAGAS_AVAILABLE,
            help="Disabled if RAGAS isn't installed",
        )

        if st.session_state.current_backend != selected_backend_key:
            st.session_state.current_backend = selected_backend_key

    # --------------------------------------------------------- initialise RAG
    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"],
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ------------------------------------------------------------ chat input
    if prompt := st.chat_input("Ask about Apollo 11 / Apollo 13 / Challenger…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response…"):
                docs_result = retrieve_documents(
                    collection, prompt, n_docs, mission_filter=mission_filter,
                )
                context = ""
                contexts_list: List[str] = []
                if docs_result and docs_result.get("documents"):
                    context = format_context(
                        docs_result["documents"][0],
                        docs_result["metadatas"][0],
                    )
                    contexts_list = docs_result["documents"][0]
                    st.session_state.last_contexts = contexts_list

                with st.expander("Retrieved context (debug)"):
                    st.text(context or "(no documents retrieved)")

                response = generate_response(
                    openai_key, prompt, context,
                    st.session_state.messages[:-1], model_choice,
                )
                st.markdown(response)

                if enable_evaluation:
                    with st.spinner("Evaluating response quality…"):
                        st.session_state.last_evaluation = evaluate_response_quality(
                            prompt, response, contexts_list,
                        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()

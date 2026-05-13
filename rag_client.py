"""
RAG retrieval layer.

Talks to ChromaDB collections built by `embedding_pipeline.py`, exposes
semantic search with optional mission filtering, and formats the retrieved
chunks into a clean context string the LLM client can consume.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)

# How many characters of any single chunk to include in the context window.
# Long enough to be informative, short enough that 3-5 chunks still fit in a
# 4k-token prompt budget alongside conversation history.
_CHUNK_DISPLAY_CHARS = 1200


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory.

    Looks for any directory whose name starts with `chroma_db` or `chromadb`
    relative to the current working directory, opens it as a PersistentClient,
    enumerates the collections inside, and reports one entry per
    (directory, collection) pair.

    Returns:
        Mapping of "<directory>::<collection>" -> {
            directory:        absolute path to the chroma persist dir,
            collection_name:  collection name inside that dir,
            display_name:     human-friendly label for the UI,
            document_count:   number of items in the collection (str),
        }
    """
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".").resolve()

    # Match any sibling directory that looks like a chroma persist dir.
    candidate_dirs = [
        p for p in current_dir.iterdir()
        if p.is_dir() and (p.name.startswith("chroma_db") or p.name.startswith("chromadb"))
    ]

    for chroma_dir in sorted(candidate_dirs):
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            for col in client.list_collections():
                key = f"{chroma_dir.name}::{col.name}"
                try:
                    count = col.count()
                except Exception:
                    count = "?"
                backends[key] = {
                    "directory":       str(chroma_dir),
                    "collection_name": col.name,
                    "display_name":    f"{chroma_dir.name} / {col.name} ({count} chunks)",
                    "document_count":  str(count),
                }
        except Exception as exc:
            # Inaccessible directory — still surface it in the UI so the user
            # knows it's there but broken.
            err = str(exc)[:80]
            backends[f"{chroma_dir.name}::__error__"] = {
                "directory":       str(chroma_dir),
                "collection_name": "",
                "display_name":    f"{chroma_dir.name} (unavailable: {err})",
                "document_count":  "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[object, bool, str]:
    """Initialize the RAG system with the specified backend.

    Args:
        chroma_dir: Path to the ChromaDB persist directory.
        collection_name: Name of the collection within that directory.

    Returns:
        (collection, success, error_message). On success error_message is "".
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as exc:
        logger.error("RAG init failed for %s/%s: %s", chroma_dir, collection_name, exc)
        return None, False, str(exc)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional mission filtering.

    Args:
        collection: A `chromadb.api.models.Collection` (returned by
            `initialize_rag_system`).
        query: The user's question.
        n_results: top-k to return. Configurable from the chat UI.
        mission_filter: If provided and not "all", restricts retrieval to that
            mission via metadata filtering. Expected values:
            "apollo_11" | "apollo_13" | "challenger".

    Returns:
        ChromaDB query result dict, or None on error.
    """
    where = None
    if mission_filter and mission_filter.lower() not in ("all", "any", ""):
        where = {"mission": mission_filter}

    try:
        return collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
    except Exception as exc:
        logger.error("Retrieval failed for query=%r: %s", query[:80], exc)
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into a single context string for the LLM.

    Adds source headers per chunk so the model can cite, deduplicates exact
    repeats, and truncates very long chunks at the display ceiling.

    Args:
        documents: list of chunk texts (one row of ChromaDB query result).
        metadatas: aligned list of metadata dicts.

    Returns:
        Formatted multi-section context string. Empty string if no documents.
    """
    if not documents:
        return ""

    parts: List[str] = ["Retrieved NASA mission context:"]
    seen: set = set()

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        if doc in seen:
            continue
        seen.add(doc)

        mission_raw = (meta or {}).get("mission", "unknown") or "unknown"
        mission = mission_raw.replace("_", " ").title()

        source = (meta or {}).get("source", "unknown")

        category_raw = (meta or {}).get("document_category", "general_document") or "general_document"
        category = category_raw.replace("_", " ").title()

        parts.append(
            f"\n[Source {i}] Mission: {mission} | Document: {source} | Category: {category}"
        )

        snippet = doc.strip()
        if len(snippet) > _CHUNK_DISPLAY_CHARS:
            snippet = snippet[:_CHUNK_DISPLAY_CHARS].rstrip() + " […]"
        parts.append(snippet)

    return "\n".join(parts)

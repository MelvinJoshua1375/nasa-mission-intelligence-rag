#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data — text files only.

Reads NASA transcript text files from data_text/<mission>/*.txt, chunks each
file with configurable size + overlap, embeds the chunks with an OpenAI
embedding model (or a local sentence-transformers model in mock mode), and
persists them to a ChromaDB collection with rich per-chunk metadata.

Run `python embedding_pipeline.py --help` for the full CLI surface.
"""

import argparse
import hashlib
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _mock_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic offline embedding. Hash-based, NOT semantic.

    Used only when NASA_RAG_MOCK=1 so the pipeline can run end-to-end without
    an OpenAI key. Quality of retrieval is terrible — this is for wiring/smoke
    tests only.
    """
    h = hashlib.sha512(text.encode("utf-8")).digest()
    # Stretch the 64-byte hash to `dim` floats in [-1, 1).
    vec = []
    while len(vec) < dim:
        for b in h:
            vec.append((b - 128) / 128.0)
            if len(vec) >= dim:
                break
        h = hashlib.sha512(h).digest()
    # L2-normalise so cosine similarity behaves.
    s = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / s for v in vec]


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db_openai",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        # Store configuration. chunk_overlap is clamped to chunk_size-1 to keep
        # the chunker making forward progress.
        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))

        # OpenAI client. Lazy-imported so the script still works under
        # NASA_RAG_MOCK=1 when openai isn't even installed.
        self._mock = os.getenv("NASA_RAG_MOCK") == "1"
        if not self._mock:
            from openai import OpenAI  # noqa: WPS433 (intentional local import)
            self.openai_client = OpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        else:
            self.openai_client = None
            logger.info("NASA_RAG_MOCK=1 -> using deterministic hash embeddings")

        # ChromaDB persistent client. Telemetry off — it crashes when offline.
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Pipeline ready: dir=%s collection=%s chunk_size=%d overlap=%d model=%s",
            chroma_persist_directory, collection_name, chunk_size, chunk_overlap,
            "mock" if self._mock else embedding_model,
        )

    # ---------------------------------------------------------------- chunking
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into overlapping chunks, snapping to sentence boundaries.

        Returns a list of (chunk_text, chunk_metadata) tuples. Each chunk's
        metadata is the file-level metadata plus chunk_index, chunk_count,
        char_start, char_end.
        """
        text = text.strip()
        if not text:
            return []

        # Documents shorter than chunk_size are kept whole.
        if len(text) <= self.chunk_size:
            meta = {**metadata, "chunk_index": 0, "chunk_count": 1,
                    "char_start": 0, "char_end": len(text)}
            return [(text, meta)]

        # Sliding window with sentence-boundary snap.
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        start = 0
        idx = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at the last sentence terminator in the back half of
            # the window — keeps chunks readable.
            if end < len(text):
                slice_ = text[start:end]
                half = len(slice_) // 2
                m = None
                for match in re.finditer(r"[\.\!\?]\s+", slice_):
                    if match.end() >= half:
                        m = match
                if m is not None:
                    end = start + m.end()

            chunk = text[start:end].strip()
            if chunk:
                meta = {
                    **metadata,
                    "chunk_index": idx,
                    "char_start":  start,
                    "char_end":    end,
                }
                chunks.append((chunk, meta))
                idx += 1
            start = start + step if end == start + self.chunk_size else end - self.chunk_overlap
            if start <= 0 or start >= len(text):
                break

        total = len(chunks)
        for ct, m in chunks:
            m["chunk_count"] = total
        return chunks

    # ----------------------------------------------------------- embedding API
    def get_embedding(self, text: str) -> List[float]:
        """Get a single embedding vector for `text`."""
        if self._mock:
            return _mock_embed(text)
        try:
            resp = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as exc:
            logger.error("Embedding error (text len=%d): %s", len(text), exc)
            raise

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts in one API call."""
        if self._mock:
            return [_mock_embed(t) for t in texts]
        resp = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    # ---------------------------------------------------------- collection ops
    def check_document_exists(self, doc_id: str) -> bool:
        """Return True if a chunk with this id is already in the collection."""
        try:
            res = self.collection.get(ids=[doc_id])
            return bool(res and res.get("ids"))
        except Exception as exc:
            logger.error("exists-check failed for %s: %s", doc_id, exc)
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document in the collection."""
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            logger.debug("Updated document: %s", doc_id)
            return True
        except Exception as exc:
            logger.error("Error updating document %s: %s", doc_id, exc)
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete all documents whose `source` metadata contains `source_pattern`."""
        try:
            all_docs = self.collection.get()
            ids_to_delete = [
                all_docs["ids"][i]
                for i, m in enumerate(all_docs["metadatas"] or [])
                if source_pattern in (m or {}).get("source", "")
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d documents matching source pattern: %s",
                            len(ids_to_delete), source_pattern)
                return len(ids_to_delete)
            logger.info("No documents found matching source pattern: %s", source_pattern)
            return 0
        except Exception as exc:
            logger.error("Error deleting documents by source: %s", exc)
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Return all chunk ids that belong to a given source file."""
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()
            return [
                all_docs["ids"][i]
                for i, m in enumerate(all_docs["metadatas"] or [])
                if (m or {}).get("source") == source and (m or {}).get("mission") == mission
            ]
        except Exception as exc:
            logger.error("Error getting file documents: %s", exc)
            return []

    # ----------------------------------------------------------- id generation
    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Stable chunk id: <mission>_<source>_chunk_<NNNN>."""
        mission = metadata.get("mission") or self.extract_mission_from_path(file_path)
        source  = metadata.get("source")  or file_path.stem
        idx     = metadata.get("chunk_index", 0)
        return f"{mission}_{source}_chunk_{int(idx):04d}"

    # ----------------------------------------------------------- file handling
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a plain text file with enriched metadata."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return []
            metadata = {
                "source":             file_path.stem,
                "file_path":          str(file_path),
                "file_type":          "text",
                "content_type":       "full_text",
                "mission":            self.extract_mission_from_path(file_path),
                "data_type":          self.extract_data_type_from_path(file_path),
                "document_category":  self.extract_document_category_from_filename(file_path.name),
                "file_size":          len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }
            return self.chunk_text(content, metadata)
        except Exception as exc:
            logger.error("Error processing text file %s: %s", file_path, exc)
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        if "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        if "challenger" in path_str:
            return "challenger"
        return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "transcript" in path_str:
            return "transcript"
        if "textract" in path_str:
            return "textract_extracted"
        if "audio" in path_str:
            return "audio_transcript"
        if "flight_plan" in path_str:
            return "flight_plan"
        return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        fl = filename.lower()
        if "pao" in fl:                return "public_affairs_officer"
        if "_cm_" in fl or fl.endswith("_cm.txt") or "_cm.txt" in fl: return "command_module"
        if "_tec_" in fl or "_tec.txt" in fl:                          return "technical"
        if "flight_plan" in fl:        return "flight_plan"
        if "mission_audio" in fl:      return "mission_audio"
        if "ntrs" in fl:               return "nasa_archive"
        if "19900066485" in fl:        return "technical_report"
        if "19710015566" in fl:        return "mission_report"
        if "full_text" in fl:          return "complete_document"
        return "general_document"

    # ------------------------------------------------------- file discovery
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Return a list of *.txt files under base_path/{apollo11,apollo13,challenger}."""
        base_path = Path(base_path)
        files_to_process: List[Path] = []
        for data_dir in ("apollo11", "apollo13", "challenger"):
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info("Scanning directory: %s", dir_path)
                text_files = list(dir_path.glob("**/*.txt"))
                files_to_process.extend(text_files)
                logger.info("Found %d text files in %s", len(text_files), data_dir)
        filtered = [
            p for p in files_to_process
            if not p.name.startswith(".")
            and "summary" not in p.name.lower()
            and p.suffix.lower() == ".txt"
        ]
        logger.info("Total text files to process: %d", len(filtered))
        mission_counts: Dict[str, int] = {}
        for p in filtered:
            m = self.extract_mission_from_path(p)
            mission_counts[m] = mission_counts.get(m, 0) + 1
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info("  %s: %d files", mission, count)
        return filtered

    # ---------------------------------------------------------- write to chroma
    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """Add (or update / replace) chunks in the ChromaDB collection.

        update_mode:
            skip    - skip a chunk if its id already exists (default).
            update  - re-embed and overwrite existing chunks.
            replace - delete every existing chunk for this file first, then add.
        """
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        if update_mode == "replace":
            existing = self.get_file_documents(file_path)
            if existing:
                self.collection.delete(ids=existing)
                logger.info("[replace] removed %d existing chunks for %s",
                            len(existing), file_path.name)

        # Decide upfront which docs need adding vs updating vs skipping so we
        # can embed in big batches instead of one-by-one (which would be slow
        # and expensive on the real OpenAI API).
        to_add_ids:   List[str] = []
        to_add_docs:  List[str] = []
        to_add_meta:  List[Dict[str, Any]] = []
        to_upd_ids:   List[str] = []
        to_upd_docs:  List[str] = []
        to_upd_meta:  List[Dict[str, Any]] = []

        for text, meta in documents:
            doc_id = self.generate_document_id(file_path, meta)
            exists = self.check_document_exists(doc_id) if update_mode != "replace" else False
            if exists and update_mode == "skip":
                stats["skipped"] += 1
                continue
            if exists and update_mode == "update":
                to_upd_ids.append(doc_id); to_upd_docs.append(text); to_upd_meta.append(meta)
            else:
                to_add_ids.append(doc_id); to_add_docs.append(text); to_add_meta.append(meta)

        # Embed + add in batches.
        for i in range(0, len(to_add_ids), batch_size):
            chunk_ids   = to_add_ids[i : i + batch_size]
            chunk_docs  = to_add_docs[i : i + batch_size]
            chunk_meta  = to_add_meta[i : i + batch_size]
            embeddings  = self._embed_batch(chunk_docs)
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_meta,
                embeddings=embeddings,
            )
            stats["added"] += len(chunk_ids)

        for i in range(0, len(to_upd_ids), batch_size):
            chunk_ids   = to_upd_ids[i : i + batch_size]
            chunk_docs  = to_upd_docs[i : i + batch_size]
            chunk_meta  = to_upd_meta[i : i + batch_size]
            embeddings  = self._embed_batch(chunk_docs)
            self.collection.update(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_meta,
                embeddings=embeddings,
            )
            stats["updated"] += len(chunk_ids)

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = "skip") -> Dict[str, Any]:
        """Process all text files under base_path and store them in ChromaDB."""
        stats: Dict[str, Any] = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }
        files = self.scan_text_files_only(base_path)
        for file_path in files:
            try:
                chunks = self.process_text_file(file_path)
                if not chunks:
                    logger.info("No chunks produced for %s — skipping", file_path)
                    continue
                file_stats = self.add_documents_to_collection(
                    chunks, file_path, update_mode=update_mode
                )
                mission = self.extract_mission_from_path(file_path)
                ms = stats["missions"].setdefault(
                    mission, {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0}
                )
                ms["files"]   += 1
                ms["chunks"]  += len(chunks)
                ms["added"]   += file_stats["added"]
                ms["updated"] += file_stats["updated"]
                ms["skipped"] += file_stats["skipped"]
                stats["files_processed"]   += 1
                stats["total_chunks"]      += len(chunks)
                stats["documents_added"]   += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]
            except Exception as exc:
                stats["errors"] += 1
                logger.error("Error processing %s: %s", file_path, exc)
        return stats

    # -------------------------------------------------------------- inspection
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.chroma_persist_directory,
                "document_count": self.collection.count(),
            }
        except Exception as exc:
            return {"collection_name": self.collection_name, "error": str(exc)}

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Run a test query against the collection."""
        try:
            if self._mock:
                # Inject a precomputed embedding so chroma doesn't try to call
                # an embedding function we haven't installed on the collection.
                return self.collection.query(
                    query_embeddings=[self.get_embedding(query_text)],
                    n_results=n_results,
                )
            return self.collection.query(query_texts=[query_text], n_results=n_results)
        except Exception as exc:
            logger.error("Query failed: %s", exc)
            return {"error": str(exc)}

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            all_docs = self.collection.get()
            metas = all_docs.get("metadatas") or []
            if not metas:
                return {
                    "total_documents": 0,
                    "missions": {}, "data_types": {},
                    "document_categories": {}, "file_types": {},
                }
            stats = {
                "total_documents": len(metas),
                "missions": {}, "data_types": {},
                "document_categories": {}, "file_types": {},
            }
            for m in metas:
                m = m or {}
                for key, bucket in (
                    ("mission", "missions"),
                    ("data_type", "data_types"),
                    ("document_category", "document_categories"),
                    ("file_type", "file_types"),
                ):
                    v = m.get(key, "unknown")
                    stats[bucket][v] = stats[bucket].get(v, 0) + 1
            return stats
        except Exception as exc:
            logger.error("Error getting collection stats: %s", exc)
            return {"error": str(exc)}


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ChromaDB Embedding Pipeline for NASA Data")
    parser.add_argument("--data-path", default="./data_text", help="Path to data directories")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Text chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding calls")
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip",
                        help="How to handle existing documents")
    parser.add_argument("--test-query", help="Run a test similarity query after processing")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print collection statistics and exit, do not (re)process files")
    parser.add_argument("--delete-source", help="Delete all documents whose source contains this pattern")
    args = parser.parse_args()

    is_mock = os.getenv("NASA_RAG_MOCK") == "1"
    if not args.openai_key and not is_mock:
        parser.error("--openai-key (or OPENAI_API_KEY env var) is required unless NASA_RAG_MOCK=1")

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.delete_source:
        deleted = pipeline.delete_documents_by_source(args.delete_source)
        logger.info("Deleted %d documents matching: %s", deleted, args.delete_source)
        return

    if args.stats_only:
        logger.info("Collection Statistics:")
        s = pipeline.get_collection_stats()
        for key, value in s.items():
            logger.info("%s: %s", key, value)
        info = pipeline.get_collection_info()
        logger.info("Collection info: %s", info)
        return

    logger.info("Starting text data processing with update mode: %s", args.update_mode)
    start = time.time()
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Files processed: %d", stats["files_processed"])
    logger.info("Total chunks created: %d", stats["total_chunks"])
    logger.info("Documents added to collection: %d", stats["documents_added"])
    logger.info("Documents updated in collection: %d", stats["documents_updated"])
    logger.info("Documents skipped (already exist): %d", stats["documents_skipped"])
    logger.info("Errors: %d", stats["errors"])
    logger.info("Processing time: %.2f seconds", elapsed)

    logger.info("\nMission breakdown:")
    for mission, ms in stats["missions"].items():
        logger.info("  %s: %d files, %d chunks", mission, ms["files"], ms["chunks"])
        logger.info("    Added: %d, Updated: %d, Skipped: %d",
                    ms["added"], ms["updated"], ms["skipped"])

    info = pipeline.get_collection_info()
    logger.info("\nCollection: %s", info.get("collection_name", "N/A"))
    logger.info("Total documents in collection: %s", info.get("document_count", "N/A"))

    if args.test_query:
        logger.info("\nTesting query: %r", args.test_query)
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            docs = results["documents"][0] if results["documents"] else []
            logger.info("Found %d results:", len(docs))
            for i, doc in enumerate(docs[:3]):
                logger.info("Result %d: %s...", i + 1, doc[:200])

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

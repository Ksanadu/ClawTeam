"""长期记忆：基于 LangChain VectorStore + Chroma，MEMORY.md 变更则重建索引。

旧版 LlamaIndex 的 storage/memory_index 与 Chroma 不兼容；若存在该目录可删除，
索引会按 MEMORY.md 自动重建。持久化目录由 CHROMA_PERSIST_DIR 或 storage/chroma_memory 指定。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import get_settings
from graph.llm import build_embedding_config_from_settings, get_embedding_model

# Chroma 持久化目录；与 knowledge 分离，仅用于长期记忆
CHROMA_MEMORY_COLLECTION = "memory"
CHROMA_DEFAULT_PERSIST_DIR = "storage/chroma_memory"


class MemoryIndexer:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._vector_store: Any = None
        self._embedding: Embeddings | None = None

    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _memory_path(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MemoryIndexer is not configured")
        return self.base_dir / "memory_module_v1" / "long_term_memory" / "MEMORY.md"

    @property
    def _storage_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MemoryIndexer is not configured")
        persist = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        if persist:
            return Path(persist)
        return self.base_dir / CHROMA_DEFAULT_PERSIST_DIR.replace("/", os.sep)

    @property
    def _meta_path(self) -> Path:
        return self._storage_dir / "meta.json"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _get_embedding(self) -> Embeddings:
        if self._embedding is None:
            settings = get_settings()
            config = build_embedding_config_from_settings(settings)
            self._embedding = get_embedding_model(config)
        return self._embedding

    def _file_digest(self) -> str:
        if not self._memory_path.exists():
            return ""
        return hashlib.md5(self._memory_path.read_bytes()).hexdigest()

    def _read_meta(self) -> dict[str, Any]:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_meta(self, digest: str) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps({"digest": digest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_documents(self) -> list[Document]:
        content = self._memory_path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        chunk_size, overlap = 256, 32
        chunks: list[str] = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - overlap if end < len(content) else len(content)
        if not chunks:
            return [
                Document(
                    page_content=content,
                    metadata={"source": "memory_module_v1/long_term_memory/MEMORY.md"},
                )
            ]
        return [
            Document(
                page_content=chunk,
                metadata={"source": "memory_module_v1/long_term_memory/MEMORY.md"},
            )
            for chunk in chunks
        ]

    def rebuild_index(self) -> None:
        if self.base_dir is None:
            return

        if not self._memory_path.exists():
            self._memory_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_path.write_text("# Long-term Memory\n\n", encoding="utf-8")

        digest = self._file_digest()
        self._write_meta(digest)

        if not self._supports_embeddings():
            self._vector_store = None
            return

        try:
            from langchain_chroma import Chroma

            embedding = self._get_embedding()
            documents = self._build_documents()
            persist_dir = str(self._storage_dir)
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding_function=embedding,
                persist_directory=persist_dir,
                collection_name=CHROMA_MEMORY_COLLECTION,
            )
        except Exception:
            self._vector_store = None

    def _load_index(self) -> None:
        if not self._supports_embeddings():
            self._vector_store = None
            return
        try:
            from langchain_chroma import Chroma

            embedding = self._get_embedding()
            persist_dir = str(self._storage_dir)
            self._vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding,
                collection_name=CHROMA_MEMORY_COLLECTION,
            )
        except Exception:
            self._vector_store = None

    def _maybe_rebuild(self) -> None:
        if self.base_dir is None:
            return
        digest = self._file_digest()
        if digest != self._read_meta().get("digest"):
            self.rebuild_index()
            return
        if self._vector_store is None and self._supports_embeddings():
            self._load_index()

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.base_dir is None:
            return []

        self._maybe_rebuild()
        if self._vector_store is None:
            return []

        try:
            docs_with_scores = self._vector_store.similarity_search_with_score(query, k=top_k)
        except Exception:
            return []

        payload: list[dict[str, Any]] = []
        for doc, score in docs_with_scores:
            payload.append(
                {
                    "text": doc.page_content,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "source": doc.metadata.get(
                        "source", "memory_module_v1/long_term_memory/MEMORY.md"
                    ),
                }
            )
        return payload


memory_indexer = MemoryIndexer()

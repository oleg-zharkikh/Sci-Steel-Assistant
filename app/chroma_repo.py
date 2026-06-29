from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import chromadb

from app.embeddings import EmbeddingConfig, default_embedding_config, get_embedding, get_embeddings
from app.types import Chunk, MetaData


def _metadata_for_chroma(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class ChromaDocChat:
    """Persistent ChromaDB для векторного поиска по чанкам."""

    def __init__(
        self,
        collection_name: str,
        chroma_persistant_dir: str = "chroma_db",
        reset: bool = False,
        embedding_config: EmbeddingConfig | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_config = embedding_config or default_embedding_config()
        self.chroma_persistant_dir = Path(chroma_persistant_dir)
        self.chroma_persistant_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_persistant_dir))

        if reset:
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_record(self, chunk: Chunk) -> None:
        self.add_records([chunk])

    def add_records(
        self,
        chunks: Iterable[Chunk],
        batch_size: int | None = None,
        progress_callback: Callable[[dict[str, int | str]], None] | None = None,
    ) -> None:
        effective_batch_size = batch_size or (
            self.embedding_config.batch_size
            if self.embedding_config.backend == "openai-compatible"
            else 256
        )
        processed = 0
        total = len(chunks) if isinstance(chunks, list) else 0
        batch: list[Chunk] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= effective_batch_size:
                self._add_batch(batch)
                processed += len(batch)
                if progress_callback is not None:
                    progress_callback({
                        "processed_chunks": processed,
                        "total_chunks": total,
                        "batch_size": len(batch),
                    })
                batch = []
        if batch:
            self._add_batch(batch)
            processed += len(batch)
            if progress_callback is not None:
                progress_callback({
                    "processed_chunks": processed,
                    "total_chunks": total,
                    "batch_size": len(batch),
                })

    def _add_batch(self, chunks: list[Chunk]) -> None:
        self.collection.upsert(
            ids=[chunk.doc_id for chunk in chunks],
            embeddings=get_embeddings(
                [chunk.text for chunk in chunks],
                config=self.embedding_config,
            ),
            metadatas=[
                _metadata_for_chroma(chunk.metadata.to_dict())
                for chunk in chunks
            ],
            documents=[chunk.text for chunk in chunks],
        )

    def search_records(self, query: str, top_k: int = 10) -> list[Chunk]:
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[get_embedding(query, config=self.embedding_config)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            chunk = Chunk(
                doc_id=str(doc_id),
                text=documents[idx] if idx < len(documents) else "",
                metadata=MetaData.from_dict(metadatas[idx] if idx < len(metadatas) else {}),
                score=1.0 / (1.0 + distance),
            )
            chunks.append(chunk)
        return chunks

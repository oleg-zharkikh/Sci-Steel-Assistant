from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import tantivy

from app.types import Chunk, MetaData


TEXT_FIELDS = [
    "text",
    "title",
    "keywords",
    "authors",
    "publication_name",
    "source_id",
    "section",
]


def _single(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _query_only(parsed_query):
    if isinstance(parsed_query, tuple):
        return parsed_query[0]
    return parsed_query


class ElasticBaseRepository:
    """Tantivy-реализация старого Elasticsearch-контракта."""

    def __init__(self, index_dir: str = "tantivy_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.schema = self._build_schema()
        self.indices: dict[str, tantivy.Index] = {}

    def _build_schema(self):
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("text", stored=True)
        schema_builder.add_text_field("metadata", stored=True)
        schema_builder.add_text_field("source_type", stored=True)
        schema_builder.add_text_field("source_id", stored=True)
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("section", stored=True)
        schema_builder.add_text_field("parent_id", stored=True)
        schema_builder.add_text_field("year", stored=True)
        schema_builder.add_text_field("authors", stored=True)
        schema_builder.add_text_field("publication_name", stored=True)
        schema_builder.add_text_field("keywords", stored=True)
        return schema_builder.build()

    def _index_path(self, index_name: str) -> Path:
        if not index_name:
            raise ValueError("index_name must not be empty")
        return self.index_dir / index_name

    def get_indices(self) -> list[str]:
        return sorted(
            path.name for path in self.index_dir.iterdir()
            if path.is_dir() and tantivy.Index.exists(str(path))
        )

    def create_index(
        self,
        index_name: str,
        mapping: dict | None = None,
        reset: bool = False,
    ):
        index_path = self._index_path(index_name)
        if reset and index_path.exists():
            shutil.rmtree(index_path)

        if index_path.exists() and tantivy.Index.exists(str(index_path)):
            index = tantivy.Index.open(str(index_path))
        else:
            index_path.mkdir(parents=True, exist_ok=True)
            index = tantivy.Index(self.schema, path=str(index_path))

        self.indices[index_name] = index
        return "ok"

    def delete_index(self, index_name: str):
        index_path = self._index_path(index_name)
        self.indices.pop(index_name, None)
        if index_path.exists():
            shutil.rmtree(index_path)

    def _get_index(self, index_name: str) -> tantivy.Index:
        if index_name not in self.indices:
            self.create_index(index_name)
        return self.indices[index_name]

    def add_one(self, index_name: str, document: dict, doc_id: str):
        self.add_many(index_name, [document])
        return "ok"

    def add_many(self, index_name: str, documents: Iterable[dict]):
        index = self._get_index(index_name)
        with index.writer() as writer:
            for document in documents:
                writer.add_document(self._to_tantivy_doc(document))
        index.reload()
        return "ok"

    def _to_tantivy_doc(self, document: dict) -> tantivy.Document:
        metadata = document.get("metadata") or {}
        doc = tantivy.Document()
        doc.add_text("id", str(document.get("id", "")))
        doc.add_text("text", str(document.get("text", "")))
        doc.add_text("metadata", json.dumps(metadata, ensure_ascii=False))
        doc.add_text("source_type", str(metadata.get("source_type", "")))
        doc.add_text("source_id", str(metadata.get("source_id", "")))
        doc.add_text("title", str(metadata.get("title", "")))
        doc.add_text("section", str(metadata.get("section", "")))
        doc.add_text("parent_id", str(metadata.get("parent_id", "")))
        doc.add_text("year", str(metadata.get("year", "")))
        doc.add_text("authors", str(metadata.get("authors", "")))
        doc.add_text("publication_name", str(metadata.get("publication_name", "")))
        doc.add_text("keywords", str(metadata.get("keywords", "")))
        return doc

    def search(
        self,
        index_name: str,
        search_query: str,
        fields: list[str] | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        index = self._get_index(index_name)
        searcher = index.searcher()
        query_text = (search_query or "").strip() or "*"
        search_fields = fields or TEXT_FIELDS

        try:
            query = index.parse_query(query_text, search_fields)
        except Exception:
            query = _query_only(index.parse_query_lenient(query_text, search_fields))

        top_docs = searcher.search(query, top_k)
        records = []
        for score, address in top_docs.hits:
            doc = searcher.doc(address)
            metadata = json.loads(_single(doc["metadata"]) or "{}")
            records.append({
                "id": _single(doc["id"]),
                "text": _single(doc["text"]),
                "metadata": metadata,
                "score": float(score),
            })
        return records

    def delete_one(self, index_name: str, doc_id: str) -> None:
        raise NotImplementedError("Tantivy delete by doc_id is not used in this MVP")


class TantivyDocChat:
    """Полнотекстовый поиск по локальному индексу Tantivy."""

    def __init__(
        self,
        index_name: str,
        index_dir: str = "tantivy_index",
        reset: bool = False,
        **_: object,
    ):
        self.index_name = index_name
        self.repository = ElasticBaseRepository(index_dir=index_dir)
        self.repository.create_index(index_name, reset=reset)

    def add_record(self, chunk: Chunk):
        self.repository.add_one(
            self.index_name,
            chunk.to_dict(),
            chunk.doc_id,
        )

    def add_records(self, chunks: Iterable[Chunk]):
        self.repository.add_many(
            self.index_name,
            (chunk.to_dict() for chunk in chunks),
        )

    def search_records(self, search_query: str, top_k: int = 50) -> list[Chunk]:
        records = self.repository.search(
            self.index_name,
            search_query,
            TEXT_FIELDS,
            top_k,
        )
        chunks = []
        for record in records:
            chunk = Chunk(
                str(record["id"]),
                record["text"],
                MetaData.from_dict(record["metadata"]),
                score=float(record.get("score") or 0.0),
            )
            chunks.append(chunk)
        return chunks


ElasticDocChat = TantivyDocChat

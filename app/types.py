from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ES_MAPPING = {
    "id": {"type": "keyword"},
    "text": {"type": "text"},
    "metadata": {"type": "object"},
}


@dataclass
class MetaData:
    """Метаданные чанка, общие для статей и патентов."""

    file_name: str
    chunk_number: int
    char_start: int
    char_end: int
    source_type: str = ""
    source_id: str = ""
    title: str = ""
    section: str = ""
    parent_id: str = ""
    year: str = ""
    authors: str = ""
    publication_name: str = ""
    keywords: str = ""
    accession_number: str = ""
    doc_number: str = ""
    country: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "file_name": self.file_name,
            "chunk_number": self.chunk_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "section": self.section,
            "parent_id": self.parent_id,
            "year": self.year,
            "authors": self.authors,
            "publication_name": self.publication_name,
            "keywords": self.keywords,
            "accession_number": self.accession_number,
            "doc_number": self.doc_number,
            "country": self.country,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaData":
        known = {
            "file_name",
            "chunk_number",
            "char_start",
            "char_end",
            "source_type",
            "source_id",
            "title",
            "section",
            "parent_id",
            "year",
            "authors",
            "publication_name",
            "keywords",
            "accession_number",
            "doc_number",
            "country",
        }
        kwargs = {key: payload.get(key, "") for key in known}
        kwargs["chunk_number"] = int(payload.get("chunk_number") or 0)
        kwargs["char_start"] = int(payload.get("char_start") or 0)
        kwargs["char_end"] = int(payload.get("char_end") or 0)
        kwargs["extra"] = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs)


@dataclass
class Chunk:
    """Фрагмент документа, который хранится в поисковом индексе."""

    doc_id: str
    text: str
    metadata: MetaData
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
        }


class VectorizedChunk(Chunk):
    """Совместимость со старым Chroma-кодом."""

    def __init__(
        self,
        chunk: Chunk,
        embedding: list[float] | None = None,
        distance: float | None = None,
        score: float | None = None,
    ) -> None:
        super().__init__(
            doc_id=chunk.doc_id,
            text=chunk.text,
            metadata=chunk.metadata,
            score=chunk.score if score is None else score,
        )
        self.embedding = embedding
        self.distance = distance

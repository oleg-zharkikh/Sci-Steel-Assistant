from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv

from app.embeddings import EmbeddingConfig
from app.elastic_repo import TantivyDocChat
from app.types import Chunk, MetaData

load_dotenv()

DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME", "metallurgy")
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "data")
DEFAULT_INDEX_DIR = os.getenv("TANTIVY_INDEX_DIR", "tantivy_index")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")


@dataclass
class SourceSection:
    name: str
    text: str


@dataclass
class SourceDocument:
    source_type: str
    source_file: str
    source_id: str
    title: str
    sections: list[SourceSection]
    metadata: dict[str, str]


class KnowledgeGraphHooks:
    """Заглушки для будущего подключения KuzuDB."""

    def extract_entities(self, chunk: Chunk) -> dict:
        return {}

    def add_entities_to_knowledge_graph(self, entities: dict, chunk: Chunk) -> None:
        return None


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _section(name: str, value: object) -> SourceSection | None:
    text = _clean(value)
    if not text:
        return None
    return SourceSection(name=name, text=text)


def _sections(*items: SourceSection | None) -> list[SourceSection]:
    return [item for item in items if item is not None]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="#"))


def _article_document(row: dict[str, str], source_file: str, row_number: int) -> SourceDocument:
    source_id = _clean(row.get("accession_number")) or f"article-{row_number}"
    title = _clean(row.get("title"))
    metadata = {
        "source_type": "article",
        "source_id": source_id,
        "title": title,
        "year": _clean(row.get("year_published")),
        "authors": _clean(row.get("authors_full") or row.get("authors")),
        "publication_name": _clean(row.get("publication_name")),
        "keywords": _clean(row.get("keywords")),
        "accession_number": _clean(row.get("accession_number")),
    }
    return SourceDocument(
        source_type="article",
        source_file=source_file,
        source_id=source_id,
        title=title,
        sections=_sections(
            _section("title", title),
            _section("publication", row.get("publication_name")),
            _section("keywords", row.get("keywords")),
            _section("abstract", row.get("abstract")),
            _section("research_areas", row.get("research_areas")),
        ),
        metadata=metadata,
    )


def _patent_document(row: dict[str, str], source_file: str, row_number: int) -> SourceDocument:
    source_id = _clean(row.get("doc_number")) or f"patent-{row_number}"
    title = _clean(row.get("invention_title"))
    metadata = {
        "source_type": "patent",
        "source_id": source_id,
        "title": title,
        "year": _clean(str(row.get("date", ""))[:4]),
        "authors": _clean(row.get("inventors")),
        "publication_name": "patent",
        "keywords": _clean(row.get("classification_ipcr")),
        "doc_number": _clean(row.get("doc_number")),
        "country": _clean(row.get("country")),
    }
    return SourceDocument(
        source_type="patent",
        source_file=source_file,
        source_id=source_id,
        title=title,
        sections=_sections(
            _section("title", title),
            _section("classification_ipcr", row.get("classification_ipcr")),
            _section("abstract", row.get("abstract")),
            _section("description", row.get("description")),
            _section("claims", row.get("claims")),
        ),
        metadata=metadata,
    )


def iter_source_documents(data_dir: str | Path = DEFAULT_DATA_DIR) -> Iterable[SourceDocument]:
    data_path = Path(data_dir)
    articles_path = data_path / "2025_metallurgy.csv"
    patents_path = data_path / "patents_metallurgy.csv"

    if articles_path.exists():
        for row_number, row in enumerate(_read_csv(articles_path), start=1):
            yield _article_document(row, articles_path.name, row_number)

    if patents_path.exists():
        for row_number, row in enumerate(_read_csv(patents_path), start=1):
            yield _patent_document(row, patents_path.name, row_number)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> Iterable[tuple[int, int, str]]:
    normalized = text.strip()
    if not normalized:
        return

    if len(normalized) <= max_chars:
        yield 0, len(normalized), normalized
        return

    sentences = split_sentences(normalized)
    if len(sentences) <= 1:
        start = 0
        text_len = len(normalized)
        while start < text_len:
            end = min(start + max_chars, text_len)
            split_at = normalized.rfind(" ", start, end)
            if end < text_len and split_at > start + max_chars // 2:
                end = split_at
            chunk = normalized[start:end].strip()
            if chunk:
                yield start, end, chunk
            if end >= text_len:
                break
            start = max(end - overlap, start + 1)
        return

    current: list[str] = []
    current_len = 0
    cursor = 0
    chunk_start = 0
    previous_tail = ""

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > max_chars:
            chunk = " ".join(current).strip()
            chunk_end = chunk_start + len(chunk)
            yield chunk_start, chunk_end, chunk

            previous_tail = chunk[-overlap:].strip() if overlap > 0 else ""
            current = [previous_tail, sentence] if previous_tail else [sentence]
            current_len = sum(len(part) + 1 for part in current)
            found_at = normalized.find(sentence, cursor)
            chunk_start = max(found_at - len(previous_tail), 0) if found_at >= 0 else chunk_end
        else:
            if not current:
                found_at = normalized.find(sentence, cursor)
                chunk_start = found_at if found_at >= 0 else cursor
            current.append(sentence)
            current_len += sentence_len
        cursor = max(cursor, normalized.find(sentence, cursor) + len(sentence))

    if current:
        chunk = " ".join(part for part in current if part).strip()
        yield chunk_start, min(chunk_start + len(chunk), len(normalized)), chunk


def source_to_chunks(
    document: SourceDocument,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    chunks = []
    chunk_number = 0
    parent_id = f"{document.source_type}:{document.source_id}"
    for section in document.sections:
        section_text = f"{section.name}: {section.text}"
        for char_start, char_end, text in chunk_text(
            section_text,
            max_chars=max_chars,
            overlap=overlap,
        ):
            chunk_number += 1
            metadata = MetaData.from_dict({
                **document.metadata,
                "file_name": document.source_file,
                "chunk_number": chunk_number,
                "char_start": char_start,
                "char_end": char_end,
                "source_type": document.source_type,
                "source_id": document.source_id,
                "title": document.title,
                "section": section.name,
                "parent_id": parent_id,
            })
            doc_id = f"{parent_id}:{section.name}:{chunk_number}"
            chunks.append(Chunk(doc_id=doc_id, text=text, metadata=metadata))
    return chunks


def do_indexing(
    collection_name: str = DEFAULT_COLLECTION,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
    max_chars: int = 1800,
    overlap: int = 200,
    reset: bool = True,
    use_chroma: bool = True,
    limit: int | None = None,
    kg_hooks: KnowledgeGraphHooks | None = None,
    progress_callback: Callable[[dict[str, int | str]], None] | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, int | str]:
    """Индексирует CSV-корпус в локальный Tantivy index."""
    def report(stage: str, **payload: int | str) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, **payload})

    hooks = kg_hooks or KnowledgeGraphHooks()
    report("init", message="Открываю Tantivy index")
    doc_chat = TantivyDocChat(
        index_name=collection_name,
        index_dir=str(index_dir),
        reset=reset,
    )
    chroma_chat = None
    chroma_status = "disabled"
    if use_chroma:
        try:
            report("init", message="Открываю ChromaDB")
            from app.chroma_repo import ChromaDocChat

            chroma_chat = ChromaDocChat(
                collection_name=collection_name,
                chroma_persistant_dir=str(chroma_dir),
                reset=reset,
                embedding_config=embedding_config,
            )
            chroma_status = "enabled"
        except Exception as error:
            chroma_status = f"disabled: {type(error).__name__}: {error}"
            report("warning", message=f"ChromaDB отключена: {type(error).__name__}: {error}")

    documents_count = 0
    chunks_count = 0
    batch: list[Chunk] = []
    source_documents = list(iter_source_documents(data_dir))
    if limit is not None:
        source_documents = source_documents[:limit]
    total_documents = len(source_documents)
    report(
        "scan",
        message=f"Найдено документов для индексации: {total_documents}",
        total_documents=total_documents,
    )

    for source_document in source_documents:
        documents_count += 1
        chunks = source_to_chunks(
            source_document,
            max_chars=max_chars,
            overlap=overlap,
        )
        for chunk in chunks:
            entities = hooks.extract_entities(chunk)
            hooks.add_entities_to_knowledge_graph(entities, chunk)
        chunks_count += len(chunks)
        batch.extend(chunks)
        report(
            "chunk",
            message=f"Подготовлен документ: {source_document.title or source_document.source_id}",
            documents=documents_count,
            total_documents=total_documents,
            chunks=chunks_count,
            current_file=source_document.source_file,
            current_source=source_document.source_id,
            current_title=source_document.title,
        )

    if batch:
        report(
            "write_tantivy",
            message=f"Записываю {len(batch)} чанков в Tantivy",
            documents=documents_count,
            total_documents=total_documents,
            chunks=chunks_count,
        )
        doc_chat.add_records(batch)
        if chroma_chat is not None:
            report(
                "write_chroma",
                message=f"Записываю {len(batch)} embedding-векторов в ChromaDB",
                documents=documents_count,
                total_documents=total_documents,
                chunks=chunks_count,
                processed_chunks=0,
                total_chunks=len(batch),
            )
            chroma_chat.add_records(
                batch,
                progress_callback=lambda event: report(
                    "write_chroma",
                    message=(
                        "Записываю embedding-векторы в ChromaDB: "
                        f"{event.get('processed_chunks', 0)}/{event.get('total_chunks', len(batch))}"
                    ),
                    documents=documents_count,
                    total_documents=total_documents,
                    chunks=chunks_count,
                    processed_chunks=int(event.get("processed_chunks", 0) or 0),
                    total_chunks=int(event.get("total_chunks", len(batch)) or len(batch)),
                    batch_size=int(event.get("batch_size", 0) or 0),
                ),
            )

    report(
        "done",
        message="Индексация завершена",
        documents=documents_count,
        total_documents=total_documents,
        chunks=chunks_count,
    )
    return {
        "collection_name": collection_name,
        "documents": documents_count,
        "chunks": chunks_count,
        "index_dir": str(index_dir),
        "chroma_dir": str(chroma_dir),
        "chroma": chroma_status,
    }


if __name__ == "__main__":
    stats = do_indexing()
    print(stats)

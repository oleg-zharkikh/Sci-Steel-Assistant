from __future__ import annotations

from app.embeddings import EmbeddingConfig
from app.elastic_repo import TantivyDocChat
from app.indexing import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_INDEX_DIR
from app.types import Chunk


class HybridRetriever:
    """Гибридный поиск: Tantivy BM25 + Chroma vector search."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        keyword_chat: TantivyDocChat | None = None,
        vector_chat=None,
        index_dir: str = DEFAULT_INDEX_DIR,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        use_chroma: bool = True,
        embedding_config: EmbeddingConfig | None = None,
        **_: object,
    ) -> None:
        self.collection_name = collection_name
        self.keyword_chat = keyword_chat or TantivyDocChat(
            index_name=collection_name,
            index_dir=index_dir,
        )
        self.vector_chat = vector_chat
        self.chroma_error = ""

        if self.vector_chat is None and use_chroma:
            try:
                from app.chroma_repo import ChromaDocChat

                self.vector_chat = ChromaDocChat(
                    collection_name=collection_name,
                    chroma_persistant_dir=chroma_dir,
                    embedding_config=embedding_config,
                )
            except Exception as error:
                self.chroma_error = f"{type(error).__name__}: {error}"

    def retrieve_semantic(self, search_phrase: str, top_k: int = 10) -> list[Chunk]:
        """Возвращает гибридно объединенные результаты."""
        keyword_results = self.keyword_chat.search_records(search_phrase, top_k)
        vector_results = []
        if self.vector_chat is not None:
            try:
                vector_results = self.vector_chat.search_records(search_phrase, top_k)
            except Exception as error:
                self.chroma_error = f"{type(error).__name__}: {error}"
        return reciprocal_rank_fusion(
            [keyword_results, vector_results],
            top_k=top_k,
        )

    def retrieve_keywords(
        self,
        must_include: list[str],
        must_not_include: list[str] | None = None,
        top_k: int = 10,
    ) -> list[Chunk]:
        """Строгий keyword-only поиск."""
        query = build_keyword_query(must_include, must_not_include or [])
        return self.keyword_chat.search_records(query, top_k)

    def retrieve_relevant(
        self,
        query_for_semantic_search: str,
        key_words: str = "",
        top_k: int = 10,
    ) -> list[Chunk]:
        """Совместимый метод для старого кода."""
        if key_words:
            keyword_results = self.keyword_chat.search_records(key_words, top_k)
            vector_results = (
                self.vector_chat.search_records(query_for_semantic_search, top_k)
                if self.vector_chat is not None
                else []
            )
            return reciprocal_rank_fusion([keyword_results, vector_results], top_k=top_k)
        return self.retrieve_semantic(query_for_semantic_search, top_k=top_k)


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]],
    top_k: int = 10,
    k: int = 60,
) -> list[Chunk]:
    chunk_by_id: dict[str, Chunk] = {}
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_by_id[chunk.doc_id] = chunk
            scores[chunk.doc_id] = scores.get(chunk.doc_id, 0.0) + 1.0 / (k + rank)

    result = []
    for doc_id, rrf_score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        chunk = chunk_by_id[doc_id]
        chunk.score = rrf_score
        result.append(chunk)
        if len(result) >= top_k:
            break
    return result


def build_keyword_query(
    must_include: list[str],
    must_not_include: list[str] | None = None,
) -> str:
    parts = []
    for word in must_include or []:
        cleaned = str(word).strip()
        if cleaned:
            parts.append(f"+{cleaned}")
    for word in must_not_include or []:
        cleaned = str(word).strip()
        if cleaned:
            parts.append(f"-{cleaned}")
    return " ".join(parts) if parts else "*"


def format_chunks_for_llm(chunks: list[Chunk], max_chars_per_chunk: int = 900) -> str:
    """Компактно форматирует найденные чанки для LLM или UI."""
    if not chunks:
        return (
            "В локальном индексе не найдено релевантных фрагментов. "
            "Попробуйте переформулировать запрос или переиндексировать данные."
        )

    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        title = meta.title or meta.source_id or chunk.doc_id
        source = f"{meta.source_type}:{meta.source_id}".strip(":")
        text = chunk.text
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."
        blocks.append(
            "\n".join([
                f"[{idx}] {title}",
                (
                    f"Источник: {source}; файл: {meta.file_name}; "
                    f"секция: {meta.section}; чанк: {meta.chunk_number}; "
                    f"score: {chunk.score:.3f}"
                ),
                f"Год/дата: {meta.year}",
                f"Фрагмент: {text}",
            ])
        )
    return "\n\n".join(blocks)

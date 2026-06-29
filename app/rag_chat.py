from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.indexing import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_INDEX_DIR
from app.embeddings import EmbeddingConfig
from app.retrieval import HybridRetriever, format_chunks_for_llm
from app.types import Chunk

load_dotenv()


LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen/qwen3.5-9b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "lm-studio")
EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
EXTERNAL_LLM_MODEL = os.getenv("EXTERNAL_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
EXTERNAL_LLM_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1200"))
RAG_CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
RAG_CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "700"))
RAG_LLM_TIMEOUT = float(os.getenv("RAG_LLM_TIMEOUT", "80"))


@dataclass(frozen=True)
class LlmConfig:
    base_url: str = LOCAL_LLM_URL
    model: str = LOCAL_MODEL
    api_key: str = LOCAL_API_KEY
    max_tokens: int = RAG_MAX_TOKENS
    temperature: float = 0.1


def default_llm_config() -> LlmConfig:
    return LlmConfig()

SECTION_WEIGHTS = {
    "abstract": 25.0,
    "description": 18.0,
    "claims": 14.0,
    "keywords": 8.0,
    "title": 6.0,
    "classification_ipcr": 4.0,
    "publication": 0.0,
    "research_areas": 0.0,
}


@dataclass
class RagSource:
    number: int
    title: str
    source: str
    file_name: str
    section: str
    chunk_number: int
    year: str
    score: float
    text: str


@dataclass
class RagAnswer:
    answer: str
    sources: list[RagSource]
    context: str
    llm_used: bool = True
    llm_error: str = ""


def get_llm(
    timeout: float = RAG_LLM_TIMEOUT,
    llm_config: LlmConfig | None = None,
) -> ChatOpenAI:
    llm_config = llm_config or default_llm_config()
    return ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        timeout=timeout,
    )


@lru_cache(maxsize=8)
def get_cached_retriever(
    collection_name: str,
    index_dir: str,
    chroma_dir: str,
    use_chroma: bool,
    embedding_config: EmbeddingConfig | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        collection_name=collection_name,
        index_dir=index_dir,
        chroma_dir=chroma_dir,
        use_chroma=use_chroma,
        embedding_config=embedding_config,
    )


def clear_retriever_cache() -> None:
    get_cached_retriever.cache_clear()


def invoke_llm_with_timeout(
    messages: list,
    timeout: float = RAG_LLM_TIMEOUT,
    llm_config: LlmConfig | None = None,
) -> str:
    result_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            response = get_llm(timeout=timeout, llm_config=llm_config).invoke(messages)
            result_queue.put(("ok", response.content))
        except Exception as error:
            result_queue.put(("error", f"{type(error).__name__}: {error}"))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"LM Studio did not respond within {timeout:.0f} seconds"
        )

    status, payload = result_queue.get_nowait()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def chunks_to_sources(chunks: list[Chunk], max_chars: int = 1400) -> list[RagSource]:
    sources = []
    seen_chunks = set()
    for chunk in chunks:
        meta = chunk.metadata
        chunk_key = (meta.source_type, meta.source_id, meta.chunk_number)
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        text = chunk.text
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        sources.append(
            RagSource(
                number=len(sources) + 1,
                title=meta.title or meta.source_id or chunk.doc_id,
                source=f"{meta.source_type}:{meta.source_id}".strip(":"),
                file_name=meta.file_name,
                section=meta.section,
                chunk_number=meta.chunk_number,
                year=meta.year,
                score=chunk.score,
                text=text,
            )
        )
    return sources


def select_context_chunks(
    chunks: list[Chunk],
    limit: int = RAG_CONTEXT_CHUNKS,
    max_per_parent: int = 2,
) -> list[Chunk]:
    """Выбирает фрагменты для LLM, предпочитая содержательные секции."""
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            chunk.score + SECTION_WEIGHTS.get(chunk.metadata.section, 0.0)
        ),
        reverse=True,
    )
    selected = []
    parent_counts: dict[str, int] = {}
    seen = set()

    for chunk in ranked:
        parent_id = chunk.metadata.parent_id or chunk.metadata.source_id or chunk.doc_id
        key = (parent_id, chunk.metadata.section, chunk.metadata.chunk_number)
        if key in seen:
            continue
        if parent_counts.get(parent_id, 0) >= max_per_parent:
            continue
        selected.append(chunk)
        seen.add(key)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for chunk in chunks:
            parent_id = chunk.metadata.parent_id or chunk.metadata.source_id or chunk.doc_id
            key = (parent_id, chunk.metadata.section, chunk.metadata.chunk_number)
            if key in seen:
                continue
            selected.append(chunk)
            seen.add(key)
            if len(selected) >= limit:
                break

    return selected


def format_sources_for_prompt(sources: list[RagSource]) -> str:
    if not sources:
        return "Релевантные фрагменты не найдены."

    blocks = []
    for source in sources:
        blocks.append(
            "\n".join([
                f"[{source.number}] {source.title}",
                (
                    f"Источник: {source.source}; файл: {source.file_name}; "
                    f"секция: {source.section}; чанк: {source.chunk_number}; "
                    f"год/дата: {source.year}; score: {source.score:.3f}"
                ),
                f"Фрагмент: {source.text}",
            ])
        )
    return "\n\n".join(blocks)


def format_history(history: list[dict[str, str]], max_messages: int = 6) -> str:
    if not history:
        return "Истории диалога пока нет."

    lines = []
    for message in history[-max_messages:]:
        role = "Пользователь" if message.get("role") == "user" else "Ассистент"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "Истории диалога пока нет."


def build_rag_prompt(
    question: str,
    answer_mode: str,
    sources: list[RagSource],
    history: list[dict[str, str]] | None = None,
) -> list:
    system_prompt = """
/no_think
Ты — научный консультант в области металлургии.
Отвечай только по предоставленным фрагментам корпуса.
Если данных недостаточно, прямо напиши, каких данных не хватает.
Обязательно указывай ссылки на источники в формате [номер].
Не придумывай факты, режимы, свойства, составы и численные значения.
Пиши компактно: максимум 6 пунктов или 3 коротких абзаца, если пользователь не просит обзор.
"""
    user_prompt = f"""
Вопрос пользователя:
{question}

Требуемый режим ответа:
{answer_mode}

Краткая история диалога:
{format_history(history or [])}

Найденные фрагменты:
{format_sources_for_prompt(sources)}

Сформируй ответ. В конце добавь раздел "Источники" со списком использованных номеров.
"""
    return [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_prompt.strip()),
    ]


def build_extractive_fallback(sources: list[RagSource], error: Exception) -> str:
    lines = [
        "LM Studio не вернула ответ за отведенное время, поэтому показываю извлеченные фрагменты из индекса.",
        "",
        "Найдено:",
    ]
    for source in sources[:3]:
        snippet = source.text.replace("\n", " ")
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "..."
        lines.append(f"- [{source.number}] {source.title}: {snippet}")
    lines.extend([
        "",
        "Источники:",
        *[
            (
                f"[{source.number}] {source.source}; файл: {source.file_name}; "
                f"секция: {source.section}; чанк: {source.chunk_number}; "
                f"score: {source.score:.3f}"
            )
            for source in sources[:3]
        ],
        "",
        f"Техническая причина fallback: {type(error).__name__}: {error}",
    ])
    return "\n".join(lines)


def answer_question(
    question: str,
    answer_mode: str,
    collection_name: str = DEFAULT_COLLECTION,
    index_dir: str = DEFAULT_INDEX_DIR,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    use_chroma: bool = True,
    top_k: int = 8,
    llm_timeout: float = RAG_LLM_TIMEOUT,
    history: list[dict[str, str]] | None = None,
    llm_config: LlmConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> RagAnswer:
    retriever = get_cached_retriever(
        collection_name,
        index_dir,
        chroma_dir,
        use_chroma,
        embedding_config,
    )
    chunks = retriever.retrieve_semantic(question, top_k=top_k)
    context_chunks = select_context_chunks(chunks, limit=RAG_CONTEXT_CHUNKS)
    sources = chunks_to_sources(context_chunks, max_chars=RAG_CHUNK_CHARS)
    context = format_chunks_for_llm(context_chunks, max_chars_per_chunk=RAG_CHUNK_CHARS)

    if not sources:
        return RagAnswer(
            answer=(
                "В локальном индексе не найдено релевантных фрагментов. "
                "Попробуйте уточнить материал, процесс, свойство или режим."
            ),
            sources=[],
            context=context,
        )

    try:
        answer = invoke_llm_with_timeout(
            build_rag_prompt(
                question=question,
                answer_mode=answer_mode,
                sources=sources,
                history=history,
            ),
            timeout=llm_timeout,
            llm_config=llm_config,
        )
        return RagAnswer(
            answer=answer,
            sources=sources,
            context=context,
        )
    except Exception as error:
        return RagAnswer(
            answer=build_extractive_fallback(sources, error),
            sources=sources,
            context=context,
            llm_used=False,
            llm_error=f"{type(error).__name__}: {error}",
        )

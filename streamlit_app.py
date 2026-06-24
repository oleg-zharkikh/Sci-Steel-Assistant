from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.indexing import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_DIR,
    do_indexing,
)
from app.embeddings import EmbeddingConfig, default_embedding_config, get_embedding_settings
from app.rag_chat import (
    EXTERNAL_LLM_API_KEY,
    EXTERNAL_LLM_MODEL,
    EXTERNAL_LLM_URL,
    LlmConfig,
    LOCAL_LLM_URL,
    LOCAL_MODEL,
    LOCAL_API_KEY,
    RAG_MAX_TOKENS,
    RAG_LLM_TIMEOUT,
    answer_question,
    clear_retriever_cache,
)
from app.retrieval import HybridRetriever, format_chunks_for_llm


st.set_page_config(
    page_title="Sci-Steel Assistant",
    page_icon="S",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_retriever(
    collection_name: str,
    index_dir: str,
    chroma_dir: str,
    use_chroma: bool,
    embedding_config: EmbeddingConfig,
) -> HybridRetriever:
    return HybridRetriever(
        collection_name=collection_name,
        index_dir=index_dir,
        chroma_dir=chroma_dir,
        use_chroma=use_chroma,
        embedding_config=embedding_config,
    )


def show_runtime_status(
    retriever: HybridRetriever | None = None,
    llm_config: LlmConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    key_prefix: str = "runtime",
) -> None:
    llm_config = llm_config or LlmConfig()
    settings = get_embedding_settings(embedding_config)
    st.caption("Текущее состояние локального контура")
    st.metric(
        "LLM timeout",
        f"{int(RAG_LLM_TIMEOUT)} сек",
        help="Backend-дефолт hard timeout для локальной LLM. В чате его можно временно изменить слайдером.",
    )
    st.text_input(
        "LLM endpoint",
        value=llm_config.base_url,
        disabled=True,
        key=f"{key_prefix}_llm_endpoint",
        help="OpenAI-compatible адрес LLM endpoint.",
    )
    st.text_input(
        "LLM model",
        value=llm_config.model,
        disabled=True,
        key=f"{key_prefix}_llm_model",
        help="Имя модели, которое отправляется в OpenAI-compatible API.",
    )
    st.text_input(
        "Embedding backend",
        value=settings["backend"],
        disabled=True,
        key=f"{key_prefix}_embedding_backend",
        help="Источник embedding-векторов для ChromaDB. sentence-transformers использует локальную модель, hashing является быстрым fallback.",
    )
    st.text_input(
        "Embedding model",
        value=settings["model"],
        disabled=True,
        key=f"{key_prefix}_embedding_model",
        help="Путь или имя модели для sentence-transformers. После смены embedding-модели нужно переиндексировать ChromaDB.",
    )
    st.text_input(
        "Embedding endpoint",
        value=settings["base_url"],
        disabled=True,
        key=f"{key_prefix}_embedding_endpoint",
        help="Используется только для OpenAI-compatible embedding backend.",
    )
    if retriever and retriever.chroma_error:
        st.warning(f"ChromaDB отключена или вернула ошибку: {retriever.chroma_error}")


st.title("Sci-Steel Assistant")
st.caption("Локальный RAG-ассистент по статьям и патентам металлургического корпуса.")

with st.sidebar:
    st.header("Навигация")
    section = st.radio(
        "Раздел",
        ["Чат", "Поиск", "Ключевые слова", "Состояние"],
        label_visibility="collapsed",
        help="Левая вкладка переключает рабочий режим приложения без верхней панели вкладок.",
    )

    st.header("Индекс")
    collection_name = st.text_input(
        "Коллекция",
        value=DEFAULT_COLLECTION,
        help="Имя логической коллекции в Tantivy и ChromaDB.",
    )
    data_dir = st.text_input(
        "Папка данных",
        value=DEFAULT_DATA_DIR,
        help="Каталог с CSV-файлами корпуса для переиндексации.",
    )
    index_dir = st.text_input(
        "Папка индекса",
        value=DEFAULT_INDEX_DIR,
        help="Каталог полнотекстового индекса Tantivy.",
    )
    chroma_dir = st.text_input(
        "Папка ChromaDB",
        value=DEFAULT_CHROMA_DIR,
        help="Каталог persistent ChromaDB для векторного поиска.",
    )
    use_chroma = st.checkbox(
        "Гибридный поиск с ChromaDB",
        value=True,
        help="Включает объединение BM25-поиска Tantivy и векторного поиска ChromaDB.",
    )
    top_k = st.slider(
        "Количество фрагментов",
        min_value=3,
        max_value=30,
        value=8,
        help="Сколько фрагментов извлекать из индекса перед ранжированием и ответом LLM.",
    )

    st.header("Модели")
    llm_provider = st.selectbox(
        "LLM для ответа",
        ["Локальная LM Studio", "Внешняя OpenAI-compatible"],
        help="Переключает endpoint и модель, которые используются для генерации ответа.",
    )
    if llm_provider == "Локальная LM Studio":
        llm_base_url = st.text_input(
            "LLM endpoint",
            value=LOCAL_LLM_URL,
            help="OpenAI-compatible endpoint локальной LM Studio.",
        )
        llm_model = st.text_input(
            "LLM model",
            value=LOCAL_MODEL,
            help="Имя модели, видимое в /v1/models.",
        )
        llm_api_key = st.text_input(
            "LLM API key",
            value=LOCAL_API_KEY,
            type="password",
            help="Для LM Studio обычно достаточно lm-studio.",
        )
    else:
        llm_base_url = st.text_input(
            "External LLM endpoint",
            value=EXTERNAL_LLM_URL,
            help="OpenAI-compatible endpoint внешнего провайдера.",
        )
        llm_model = st.text_input(
            "External LLM model",
            value=EXTERNAL_LLM_MODEL,
            help="Имя внешней chat-модели.",
        )
        llm_api_key = st.text_input(
            "External LLM API key",
            value=EXTERNAL_LLM_API_KEY,
            type="password",
            help="API key внешнего провайдера. Не хранится в индексе.",
        )

    llm_max_tokens = st.number_input(
        "Max output tokens",
        min_value=128,
        max_value=10000,
        value=RAG_MAX_TOKENS,
        step=128,
        help="Лимит токенов ответа LLM.",
    )
    llm_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.1,
        step=0.1,
        help="Температура генерации ответа.",
    )
    llm_config = LlmConfig(
        base_url=llm_base_url,
        model=llm_model,
        api_key=llm_api_key,
        max_tokens=int(llm_max_tokens),
        temperature=float(llm_temperature),
    )

    default_embedding = default_embedding_config()
    embedding_backend_label = st.selectbox(
        "Embeddings для ChromaDB",
        [
            "Локальная bge-m3",
            "Внешняя OpenAI-compatible",
            "Hashing fallback",
        ],
        help="После смены embeddings нужно переиндексировать CSV, чтобы ChromaDB хранила векторы той же размерности.",
    )
    if embedding_backend_label == "Локальная bge-m3":
        embedding_config = EmbeddingConfig(
            backend="sentence-transformers",
            model=st.text_input(
                "Embedding model/path",
                value=default_embedding.model,
                help="Локальный путь или Hugging Face model id для sentence-transformers.",
            ),
            batch_size=int(st.number_input(
                "Embedding batch size",
                min_value=1,
                max_value=512,
                value=default_embedding.batch_size,
                step=1,
                help="Размер пачки при расчете embeddings.",
            )),
            dim=default_embedding.dim,
            base_url=default_embedding.base_url,
            api_key=default_embedding.api_key,
            timeout=default_embedding.timeout,
            device=default_embedding.device,
            local_only=default_embedding.local_only,
        )
    elif embedding_backend_label == "Внешняя OpenAI-compatible":
        embedding_config = EmbeddingConfig(
            backend="openai-compatible",
            model=st.text_input(
                "External embedding model",
                value=default_embedding.model,
                help="Имя embedding-модели внешнего OpenAI-compatible API.",
            ),
            batch_size=int(st.number_input(
                "Embedding batch size",
                min_value=1,
                max_value=512,
                value=default_embedding.batch_size,
                step=1,
                help="Размер пачки запросов к embedding API.",
            )),
            dim=default_embedding.dim,
            base_url=st.text_input(
                "Embedding endpoint",
                value=default_embedding.base_url,
                help="OpenAI-compatible endpoint с /embeddings.",
            ),
            api_key=st.text_input(
                "Embedding API key",
                value=default_embedding.api_key,
                type="password",
                help="API key для embedding endpoint.",
            ),
            timeout=default_embedding.timeout,
            device=default_embedding.device,
            local_only=default_embedding.local_only,
        )
    else:
        embedding_config = EmbeddingConfig(
            backend="hashing",
            model="hashing",
            batch_size=default_embedding.batch_size,
            dim=int(st.number_input(
                "Hashing dimension",
                min_value=64,
                max_value=4096,
                value=default_embedding.dim,
                step=64,
                help="Размерность hashing-векторов.",
            )),
            base_url=default_embedding.base_url,
            api_key=default_embedding.api_key,
            timeout=default_embedding.timeout,
            device=default_embedding.device,
            local_only=default_embedding.local_only,
        )

    llm_timeout = st.slider(
        "Hard timeout LLM, сек",
        min_value=5,
        max_value=180,
        value=min(max(int(RAG_LLM_TIMEOUT), 5), 180),
        step=5,
        help="Жесткий лимит ожидания локальной LLM. По умолчанию 80 секунд.",
    )
    chunk_chars = st.slider(
        "Размер чанка, символы",
        min_value=500,
        max_value=4000,
        value=1800,
        step=100,
        help="Максимальная длина одного индексируемого фрагмента документа.",
    )
    chunk_overlap = st.slider(
        "Overlap чанков, символы",
        min_value=0,
        max_value=800,
        value=200,
        step=50,
        help="Перекрытие соседних фрагментов, чтобы не терять смысл на границе чанков.",
    )

    if st.button(
        "Переиндексировать CSV",
        type="primary",
        help="Полностью пересобирает Tantivy и, если включено, ChromaDB из CSV-файлов.",
    ):
        progress_bar = st.progress(0, text="Готовлю индексацию...")
        status_box = st.empty()
        details_box = st.empty()

        def update_indexing_status(event: dict[str, int | str]) -> None:
            stage = str(event.get("stage", ""))
            message = str(event.get("message", "Индексирую..."))
            documents = int(event.get("documents", 0) or 0)
            total_documents = int(event.get("total_documents", 0) or 0)
            chunks = int(event.get("chunks", 0) or 0)
            processed_chunks = int(event.get("processed_chunks", 0) or 0)
            total_chunks = int(event.get("total_chunks", 0) or 0)
            current_file = str(event.get("current_file", "") or "")
            current_source = str(event.get("current_source", "") or "")
            current_title = str(event.get("current_title", "") or "")

            if total_documents > 0 and stage == "chunk":
                progress = min(documents / total_documents, 0.88)
            elif stage == "write_tantivy":
                progress = 0.92
            elif stage == "write_chroma":
                chroma_fraction = (
                    processed_chunks / total_chunks
                    if total_chunks > 0
                    else 0.0
                )
                progress = 0.94 + min(chroma_fraction, 1.0) * 0.05
            elif stage == "done":
                progress = 1.0
            else:
                progress = 0.02

            progress_bar.progress(progress, text=message)
            chroma_text = (
                f"; Chroma: {processed_chunks}/{total_chunks}"
                if stage == "write_chroma" and total_chunks
                else ""
            )
            status_box.info(
                (
                    f"Документы: {documents}/{total_documents or '?'}; "
                    f"чанки: {chunks}{chroma_text}; этап: {stage or 'init'}"
                )
            )
            if current_file or current_source or current_title:
                details_box.caption(
                    (
                        f"Файл: {current_file or '-'}; "
                        f"источник: {current_source or '-'}; "
                        f"название: {current_title or '-'}"
                    )
                )

        with st.spinner("Индексирую CSV..."):
            stats = do_indexing(
                collection_name=collection_name,
                data_dir=data_dir,
                index_dir=index_dir,
                chroma_dir=chroma_dir,
                max_chars=chunk_chars,
                overlap=chunk_overlap,
                use_chroma=use_chroma,
                reset=True,
                progress_callback=update_indexing_status,
                embedding_config=embedding_config,
            )
        get_retriever.clear()
        clear_retriever_cache()
        st.success(
            (
                f"Готово: документов {stats['documents']}, "
                f"чанков {stats['chunks']}; Chroma: {stats['chroma']}"
            )
        )

    if st.button(
        "Сбросить чат",
        help="Очищает историю сообщений и последние найденные источники.",
    ):
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("last_sources", None)
        st.session_state.pop("last_context", None)
        st.session_state.pop("last_llm_error", None)
        st.session_state.pop("last_llm_used", None)

    with st.expander("Статус", expanded=False):
        show_runtime_status(
            llm_config=llm_config,
            embedding_config=embedding_config,
            key_prefix="sidebar",
        )


if section == "Чат":
    answer_mode = st.segmented_control(
        "Режим ответа",
        [
            "краткий ответ с источниками",
            "подробный ответ с источниками",
            "обзор с пробелами в данных",
        ],
        default="подробный ответ с источниками",
        help="Задает стиль ответа LLM: короткий, подробный или обзорный с явным указанием пробелов.",
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Задайте вопрос по статьям и патентам")
    if question:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Ищу фрагменты и формирую ответ через выбранную LLM..."):
                rag_answer = answer_question(
                    question=question,
                    answer_mode=answer_mode,
                    collection_name=collection_name,
                    index_dir=index_dir,
                    chroma_dir=chroma_dir,
                    use_chroma=use_chroma,
                    top_k=top_k,
                    llm_timeout=llm_timeout,
                    history=st.session_state.chat_messages,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                )
                answer = rag_answer.answer
                st.session_state.last_sources = rag_answer.sources
                st.session_state.last_context = rag_answer.context
                st.session_state.last_llm_error = rag_answer.llm_error
                st.session_state.last_llm_used = rag_answer.llm_used
            if not rag_answer.llm_used:
                st.warning("LLM не успела сформировать ответ, показан fallback по найденным фрагментам.")
            st.write(answer)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
        })

    sources = st.session_state.get("last_sources") or []
    llm_error = st.session_state.get("last_llm_error") or ""
    if llm_error:
        st.caption(f"Последняя ошибка LLM: {llm_error}")
    if sources:
        with st.expander("Найденные источники"):
            for source in sources:
                st.markdown(f"**[{source.number}] {source.title}**")
                st.caption(
                    f"{source.source}; файл: {source.file_name}; "
                    f"секция: {source.section}; чанк: {source.chunk_number}; "
                    f"год/дата: {source.year}; score: {source.score:.3f}"
                )
                st.write(source.text)
                st.divider()

elif section == "Поиск":
    query = st.text_area(
        "Вопрос или поисковая фраза",
        placeholder="Например: corrosion resistance M2052 alloy selective laser melting",
        height=110,
        help="Свободный запрос для гибридного поиска: Tantivy BM25 плюс ChromaDB, если она включена.",
    )
    if st.button(
        "Найти",
        disabled=not query.strip(),
        help="Запускает поиск по локальному индексу без обращения к LLM.",
    ):
        retriever = get_retriever(
            collection_name,
            index_dir,
            chroma_dir,
            use_chroma,
            embedding_config,
        )
        chunks = retriever.retrieve_semantic(query, top_k=top_k)
        if retriever.chroma_error:
            st.warning(f"ChromaDB отключена или вернула ошибку: {retriever.chroma_error}")
        st.subheader("Найденные фрагменты")
        st.text(format_chunks_for_llm(chunks, max_chars_per_chunk=1200))

        for chunk in chunks:
            meta = chunk.metadata
            with st.expander(f"{meta.title or chunk.doc_id} | {meta.source_type}:{meta.source_id}"):
                st.write(chunk.text)
                st.caption(
                    f"Файл: {meta.file_name}; чанк: {meta.chunk_number}; "
                    f"секция: {meta.section}; год/дата: {meta.year}; "
                    f"score: {chunk.score:.3f}"
                )

elif section == "Ключевые слова":
    include_text = st.text_input(
        "Обязательные слова",
        placeholder="M2052 corrosion",
        help="Слова, которые должны присутствовать в полнотекстовом запросе Tantivy.",
    )
    exclude_text = st.text_input(
        "Исключить слова",
        placeholder="review",
        help="Слова, которые нужно исключить из keyword-only поиска.",
    )
    if st.button(
        "Искать по ключевым словам",
        disabled=not include_text.strip(),
        help="Запускает строгий полнотекстовый поиск без векторного ранжирования и без LLM.",
    ):
        retriever = get_retriever(
            collection_name,
            index_dir,
            chroma_dir,
            use_chroma,
            embedding_config,
        )
        chunks = retriever.retrieve_keywords(
            include_text.split(),
            exclude_text.split(),
            top_k=top_k,
        )
        st.subheader("Найденные фрагменты")
        st.text(format_chunks_for_llm(chunks, max_chars_per_chunk=1200))

else:
    retriever = get_retriever(
        collection_name,
        index_dir,
        chroma_dir,
        use_chroma,
        embedding_config,
    )
    show_runtime_status(
        retriever,
        llm_config=llm_config,
        embedding_config=embedding_config,
        key_prefix="main",
    )
    st.subheader("Оценка эффективности")
    st.markdown(
        "\n".join([
            "- LLM timeout теперь согласован с требованием: 80 секунд по умолчанию.",
            "- Поисковые клиенты кэшируются между запросами, поэтому чат и поиск не пересоздают Tantivy/Chroma на каждый вызов.",
            "- Если `models/bge-m3` доступна, embedding backend по умолчанию использует `sentence-transformers`; иначе включается hashing fallback.",
            "- После перехода с hashing на `bge-m3` нужно переиндексировать CSV, иначе старая ChromaDB может иметь несовместимый размер векторов.",
            "- При ошибке ChromaDB приложение продолжает отвечать через Tantivy BM25 и показывает предупреждение.",
        ])
    )

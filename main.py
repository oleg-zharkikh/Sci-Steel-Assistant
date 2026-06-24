from pydantic import BaseModel, Field
from langchain.tools import tool
import json
import re
import os
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
)
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
import time

from dotenv import load_dotenv
from app.indexing import DEFAULT_COLLECTION, DEFAULT_INDEX_DIR
from app.retrieval import HybridRetriever, format_chunks_for_llm

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────

# ── Конфигурация ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
"""local = LM Studio, external = OpenAI-compatible API, yandex = Yandex AI Studio."""

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen/qwen3.5-9b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "lm-studio")

EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
EXTERNAL_LLM_MODEL = os.getenv("EXTERNAL_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
EXTERNAL_LLM_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER", "")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "")
YANDEX_CLOUD_MODEL = os.getenv("YANDEX_CLOUD_MODEL", "qwen3.6-35b-a3b/latest")
YANDEX_LLM_URL = os.getenv("YANDEX_LLM_URL", "https://ai.api.cloud.yandex.net/v1")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "10000"))

# Общие параметры
MAX_SEARCH_ITER = int(os.getenv("MAX_SEARCH_ITER", "5"))
"""Лимит итераций поиска для защиты от бесконечного зацикливания"""

COLLECTION_NAME = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
INDEX_DIR = os.getenv("TANTIVY_INDEX_DIR", DEFAULT_INDEX_DIR)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))

# ── Инициализация LLM ────────────────────────────────────────────────────────
if LLM_PROVIDER == "local":
    print(f"[INFO] Используем локальную LLM: {LOCAL_MODEL}")
    llm = ChatOpenAI(
        base_url=LOCAL_LLM_URL,
        api_key=LOCAL_API_KEY,
        model=LOCAL_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
elif LLM_PROVIDER == "external":
    print(f"[INFO] Используем внешнюю OpenAI-compatible LLM: {EXTERNAL_LLM_MODEL}")
    llm = ChatOpenAI(
        base_url=EXTERNAL_LLM_URL,
        api_key=EXTERNAL_LLM_API_KEY,
        model=EXTERNAL_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
elif LLM_PROVIDER == "yandex":
    print(f"[INFO] Используем Yandex AI Studio: {YANDEX_CLOUD_MODEL}")
    llm = ChatOpenAI(
        base_url=YANDEX_LLM_URL,
        api_key=YANDEX_CLOUD_API_KEY,
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort=None,
    )
else:
    raise ValueError(
        "LLM_PROVIDER must be one of: local, external, yandex"
    )


# ── Состояние агента ─────────────────────────────────────────────────────────
class AgentState(MessagesState):
    """Расширенное состояние с памятью и счетчиком поисков по БД."""
    context: Dict[str, Any] = Field(default_factory=dict)
    search_count: int = 0


# ── Схемы аргументов для инструментов ────────────────────────────────────────
class AppendArgs(BaseModel):
    filepath: str = Field(
        ..., description='Имя текстового файла в ./agent_data')
    content: str = Field(
        ..., description='Текст для записи в файл')


# ── Схема для структурированного ответа эксперта-оценщика ────────────────────
class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(
        description=('Истинно, если информации достаточно для '
                     'исчерпывающего ответа.'))
    advice: str = Field(
        description=('Если информации недостаточно, конкретный совет, '
                     'что нужно найти. Иначе пустая строка.'))


# ── Инструменты ──────────────────────────────────────────────────────────────
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Ленивая инициализация локального поисковика."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            collection_name=COLLECTION_NAME,
            index_dir=INDEX_DIR,
        )
    return _retriever


def _as_list(value: List[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _retrieval_error_message(error: Exception) -> str:
    return (
        "Не удалось выполнить поиск в локальном индексе. "
        f"Коллекция: {COLLECTION_NAME}, индекс: {INDEX_DIR}. "
        "Если индекс еще не создан, запустите: "
        "venv/bin/python -m app.indexing. "
        f"Техническая ошибка: {error}"
    )

@tool(description='Поиск в локальной базе данных по семантической близости')
def retrieve_by_semantic_similarity(search_query: str) -> str:
    """Получает семантически-релевантную информацию из локальной базы данных."""
    try:
        chunks = get_retriever().retrieve_semantic(
            search_query,
            top_k=RETRIEVAL_TOP_K,
        )
        return format_chunks_for_llm(chunks)
    except Exception as error:
        return _retrieval_error_message(error)

@tool(description='Поиск в локальной базе данных по ключевым словам')
def retrieve_by_keywords(must_include: List[str], must_not_include: List[str]) -> str:
    """Получает информацию из локальной базы данных по ключевым словам."""
    try:
        chunks = get_retriever().retrieve_keywords(
            _as_list(must_include),
            _as_list(must_not_include),
            top_k=RETRIEVAL_TOP_K,
        )
        return format_chunks_for_llm(chunks)
    except Exception as error:
        return _retrieval_error_message(error)


ALL_TOOLS = [retrieve_by_semantic_similarity, retrieve_by_keywords]


# ── Системный промпт с TAO-циклом ────────────────────────────────────────────
REACT_SYSTEM_PROMPT = """
Ты — интеллектуальный агент - научный консультант в области металлургии.

Твой рабочий цикл (TAO):
1. THOUGHT: Проанализируй запрос пользователя, определи необходимые данные
2. ACTION: Вызови ОДИН подходящий инструмент для получения информации
3. OBSERVATION: Проанализируй результат, реши: нужен ли следующий шаг или можно ответить

Доступные инструменты:
- retrieve_by_semantic_similarity(search_query: str) → семантический поиск
- retrieve_by_keywords(must_include: List[str], must_not_include: List[str]) → поиск по ключевым словам

ВАЖНО: Ты ДОЛЖЕН вызывать инструменты для получения информации из базы данных. Не отвечай на вопросы напрямую — используй только данные из инструментов.

Пример вызова инструмента:
- retrieve_by_semantic_similarity("сплав INCONEL")
- retrieve_by_keywords(["INCONEL", "alloy"], [])

Важные правила:
1. Вызывай только ОДИН инструмент за шаг
2. Не придумывай данные — используй только ответы от инструментов
3. Если информации не хватает, продолжай поиск
4. В финальном ответе указывай, на какие найденные источники и чанки ты опираешься
"""

# ── Узлы графа ───────────────────────────────────────────────────────────────

def agent_node(state: AgentState):
    """Основной узел агента: генерация ответа или вызов инструмента."""
    messages = state['messages']
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + messages

    llm_with_tools = llm.bind_tools(
        ALL_TOOLS,
        parallel_tool_calls=False
    )
    response = llm_with_tools.invoke(messages)

    # Заставляем модель проговаривать мысль перед вызовом инструмента
    if response.tool_calls and not response.content.strip():
        tool_names = ', '.join(tc['name'] for tc in response.tool_calls)
        thought_prompt = (f'Ты выбрал инструмент: {tool_names}. '
                          'Кратко (1 предложение) объясни этот шаг. '
                          'Отвечай ТОЛЬКО рассуждением.')
        thought_response = llm.invoke(
            messages + [HumanMessage(content=thought_prompt)])
        response.content = f'{thought_response.content.strip()}'

    return {'messages': [response]}


def tools_node(state: AgentState):
    """Узел выполнения инструментов с обновлением счетчика поисков."""
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return state

    results = []
    search_increment = 0
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        print('call: ', tool_name)
        print('args: ', tool_args)
        selected_tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)

        if not selected_tool:
            results.append(ToolMessage(
                content=f"Ошибка: инструмент '{tool_name}' не найден",
                tool_call_id=tool_call['id']))
            continue

        try:
            print(f'Пробую вызвать tool {tool_name}')
            print(f'Параметры: {tool_args}')
            raw_result = selected_tool.invoke(tool_args)
            print(f'Результаты: {raw_result[0:100]}')
            results.append(ToolMessage(content=raw_result, tool_call_id=tool_call['id']))
            # Считаем только реальные поиски по БД
            if tool_name in [
                'retrieve_by_semantic_similarity', 'retrieve_by_keywords'
            ]:
                search_increment += 1
        except Exception as e:
            results.append(ToolMessage(
                content=f'Ошибка выполнения {tool_name}: {str(e)}',
                tool_call_id=tool_call['id']))

    return {
        'messages': results,
        'search_count': state.get('search_count', 0) + search_increment
    }


def parse_evaluation_response(text: str) -> Optional[dict]:
    """Парсинг JSON из текста на случай сбоев structured output."""
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def evaluate_node(state: AgentState):
    """Узел оценки достаточности информации.

    Для локальных моделей MVP использует детерминированную проверку вместо
    отдельного LLM-оценщика: Qwen в LM Studio нестабильно поддерживает
    structured output и может зависать на fallback-вызове.
    """
    context = state.get('context', {})

    if context.get('force_end'):
        return {'messages': [], 'context': {**context, 'sufficient': True}}

    search_count = state.get('search_count', 0)

    # Защита от зацикливания
    if search_count >= MAX_SEARCH_ITER:
        print((f'\n[Оценка]: Достигнут лимит поисков ({MAX_SEARCH_ITER}). '
               'Принудительное формирование ответа.'))
        limit_msg = HumanMessage(
            content="=== ЛИМИТ ИТЕРАЦИЙ === "
                    "Достигнут максимальный лимит поисков по базе данных. "
                    "Немедленно сформируй окончательный ответ на основе той информации, что уже есть. "
                    "Если данных не хватает, честно сообщи пользователю, что не удалось найти полную информацию."
        )
        return {
            'messages': [limit_msg],
            'context': {**context, 'force_end': True, 'sufficient': False}
        }

    messages = state['messages']
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
        if tool_messages:
            return {'messages': [], 'context': {**context, 'sufficient': True}}
        advice_msg = HumanMessage(
            content=(
                "=== ЭКСПЕРТНАЯ ОЦЕНКА ===\n"
                "Перед ответом нужно выполнить поиск в локальной базе данных."
            )
        )
        return {'messages': [advice_msg], 'context': {**context, 'sufficient': False}}

    if tool_messages:
        return {'messages': [], 'context': {**context, 'sufficient': True}}

    # Формируем контекст для оценки
    context_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"[Запрос пользователя]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                context_parts.append(f"[Размышления агента]: {msg.content}")
            if getattr(msg, 'tool_calls', None):
                for tc in msg.tool_calls:
                    context_parts.append(f"[Агент вызвал]: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            content_preview = (msg.content[:1500] + "..."
                               if len(msg.content) > 1500
                               else msg.content)
            context_parts.append(f"[Результат из БД ({msg.name})]: {content_preview}")

    context_text = "\n".join(context_parts)

#     eval_system_prompt = """Ты — критически мыслящий эксперт в области металлургии. 
# Оцени, достаточно ли собранной информации в контексте для исчерпывающего и научно обоснованного ответа на запрос пользователя.
# Если информации недостаточно, укажи, каких именно данных не хватает и что нужно поискать дополнительно."""
    eval_system_prompt = """Ты — критически мыслящий эксперт в области металлургии. 
Оцени, достаточно ли собранной информации в контексте для обоснованного ответа на запрос пользователя. Учитывай пожелания пользователя относительно полноты ответа. Если не полнота не задана - достатоно любой найденной информации.
Если информации недостаточно, укажи, каких именно данных не хватает и что нужно поискать дополнительно."""

    eval_user_prompt = f"""КОНТЕКСТ ДИАЛОГА:
{context_text}

Оцени достаточность информации."""

    is_sufficient = False
    advice = ""

    try:
        structured_llm = llm.with_structured_output(EvaluationResult)
        result = structured_llm.invoke(
            [
                SystemMessage(content=eval_system_prompt),
                HumanMessage(content=eval_user_prompt)
            ]
        )
        is_sufficient = result.is_sufficient
        advice = result.advice
    except Exception as e:
        print((f'[Оценка]: Structured output failed ({e}), '
               'using fallback text parsing...'))
        fallback_prompt = """Оцени достаточность информации и ответь СТРОГО в формате JSON без лишнего текста:
{
  "is_sufficient": true/false,
  "advice": "совет или пусто"
}"""
        response = llm.invoke(
            [
                SystemMessage(content=eval_system_prompt),
                HumanMessage(content=eval_user_prompt + "\n\n" + fallback_prompt)
            ]
        )
        parsed = parse_evaluation_response(response.content)

        if parsed and 'is_sufficient' in parsed:
            val = parsed['is_sufficient']
            if isinstance(val, bool):
                is_sufficient = val
            elif isinstance(val, str):
                is_sufficient = val.lower() in ('true', '1', 'yes', 'да', 'истина')
            else:
                is_sufficient = bool(val)
            advice = str(parsed.get('advice', ''))
        else:
            print('[Оценка]: Failed to parse. Defaulting to sufficient.')
            is_sufficient = True

    if is_sufficient:
        print('\n[Оценка эксперта]: Информации достаточно для ответа.')
        return {'messages': [], 'context': {**context, 'sufficient': True}}
    else:
        print(f'\n[Оценка эксперта]: Информации недостаточно. Совет: {advice}')
        advice_msg = HumanMessage(
            content=f"=== ЭКСПЕРТНАЯ ОЦЕНКА ===\nСобранной информации НЕДОСТАТОЧНО.\n"
                    f"Совет по дальнейшему поиску: {advice}\nПожалуйста, вызови инструменты для поиска недостающих данных."
        )
        return {'messages': [advice_msg], 'context': {**context, 'sufficient': False}}


def should_continue(state: AgentState):
    """Определяет следующий шаг после узла agent."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    # Если агент сгенерировал текст, отправляем на оценку эксперту
    return 'evaluate'


def route_after_evaluation(state: AgentState):
    """Определяет следующий шаг после оценки эксперта."""
    context = state.get('context', {})
    if context.get('sufficient') is True:
        return 'end'
    # Если не хватает или сработал лимит (но агент еще не сформировал ответ)
    return 'agent'


# ── Сборка графа ─────────────────────────────────────────────────────────────

def create_react_agent(tools_list=None, system_prompt=None):
    if tools_list is None:
        tools_list = ALL_TOOLS
    if system_prompt is None:
        system_prompt = REACT_SYSTEM_PROMPT

    workflow = StateGraph(AgentState)

    workflow.add_node('agent', agent_node)
    workflow.add_node('tools', tools_node)
    workflow.add_node('evaluate', evaluate_node)

    workflow.add_edge(START, 'agent')

    workflow.add_conditional_edges(
        'agent', should_continue,
        {'tools': 'tools', 'evaluate': 'evaluate'}
    )

    workflow.add_edge('tools', 'agent')

    workflow.add_conditional_edges(
        'evaluate', route_after_evaluation,
        {'agent': 'agent', 'end': END}
    )

    return workflow.compile()


# ── Запуск с трассировкой ────────────────────────────────────────────────────

def run_and_trace(
    agent: CompiledStateGraph,
    query: str,
    history: list[BaseMessage] | None = None,
):
    print(f'Пользователь: {query}')
    print('═' * 70)

    start_time = time.time()
    initial_state = {
        'messages': (history or []) + [HumanMessage(content=query)],
        'context': {},
        'search_count': 0
    }
    result = agent.invoke(initial_state)
    elapsed = time.time() - start_time

    tao_steps = 0
    tool_calls_count = 0

    for i, msg in enumerate(result['messages']):
        msg_type = type(msg).__name__

        if msg_type == 'AIMessage' and getattr(msg, 'tool_calls', None):
            tao_steps += 1
            for tc in msg.tool_calls:
                tool_calls_count += 1
                print(f'\n  Шаг TAO #{tao_steps}')
                if msg.content and msg.content.strip():
                    zip_content = "..."if len(msg.content) > 200 else ""
                    print(f'THOUGHT: {msg.content[:200]}{zip_content}')
                args_dumps = json.dumps(tc["args"], ensure_ascii=False)
                print(f'ACTION:  {tc["name"]}({args_dumps})')

        elif msg_type == 'ToolMessage':
            zip_content = "..." if len(msg.content) > 180 else msg.content
            preview = msg.content[:180] + zip_content
            print(f'OBSERVE: {preview}')

        elif (msg_type == 'HumanMessage'
              and ('ЭКСПЕРТНАЯ ОЦЕНКА' in msg.content
                   or 'ЛИМИТ ИТЕРАЦИЙ' in msg.content)):
            print(f'\n--- {msg.content} ---')

        elif msg_type == 'AIMessage' and not getattr(msg, 'tool_calls', None):
            if msg.content and i > 0:
                zip_content = "..." if len(msg.content) > 400 else ""
                print(f'\nАгент: {msg.content[:400]}{zip_content}')

    print(f'\n{"═" * 70}')
    print(f'Статистика: шагов TAO={tao_steps}, вызовов инструментов={tool_calls_count}, время={elapsed:.2f}с')

    final_answer = next(
        (msg.content for msg in reversed(result['messages']) if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None)),
        'Нет ответа'
    )
    return final_answer, result


def enrich_query_interactively(query: str) -> str:
    """Уточняет желаемую полноту ответа в CLI."""
    print('\nРежим ответа:')
    print('1 - кратко')
    print('2 - подробно с источниками (по умолчанию)')
    print('3 - обзор с пробелами в данных')
    answer_mode = input('Выберите режим или нажмите Enter: ').strip()
    modes = {
        '1': 'краткий ответ, только главное',
        '2': 'подробный ответ с указанием источников и чанков',
        '3': 'обзор: найденные факты, сравнение источников и пробелы в данных',
        '': 'подробный ответ с указанием источников и чанков',
    }
    mode_text = modes.get(answer_mode, answer_mode)
    return f'{query}\n\nТребуемая полнота ответа: {mode_text}'


def trim_history(messages: list[BaseMessage], max_messages: int = 16) -> list[BaseMessage]:
    """Ограничивает историю, чтобы не раздувать контекст."""
    return messages[-max_messages:]


# ── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Инициализация агента...')
    react_agent = create_react_agent()
    print('Агент готов к работе!\n')
    session_history: list[BaseMessage] = []

    while True:
        user_msg = input('> ')
        if user_msg.strip().lower() in ('exit', 'выход', 'ничего'):
            break
        query = enrich_query_interactively(user_msg)
        answer, state = run_and_trace(react_agent, query, session_history)
        session_history = trim_history(state['messages'])
        print(f'\nФинальный ответ: {answer}')

        follow_up = input(
            '\nПродолжить поиск в текущем контексте? '
            'Напишите направление или нажмите Enter для нового вопроса: '
        ).strip()
        while follow_up:
            follow_up_query = (
                'Продолжи анализ предыдущего запроса. '
                f'Новое направление поиска: {follow_up}'
            )
            answer, state = run_and_trace(
                react_agent,
                enrich_query_interactively(follow_up_query),
                session_history,
            )
            session_history = trim_history(state['messages'])
            print(f'\nФинальный ответ: {answer}')
            follow_up = input(
                '\nПродолжить еще? Напишите направление или нажмите Enter: '
            ).strip()


"""Пример запуска агента на данных заглушки:


Статистика: шагов TAO=1, вызовов инструментов=1, время=40.27с

Финальный ответ: На основе данных из базы данных, вот пример получения тонкой пленки BiFe(1-x)PdxO3 (BFPxO):

**Метод синтеза:** Золь-гель метод

В работе описывается получение тонких пленок BiFe(1-x)PdxO3 с последующим формированием гетероструктуры BFPxO/NiO методом золь-гель. Этот метод позволяет синтезировать сегнетоэлектрический феррит висмута с частичным замещением ионов палладия в позиции Fe, что приводит к изменению оптоэлектронных характеристик материала.

Легирование палладием в данной системе позволяет минимизировать постоянную времени отклика до значений ниже 10 мс, а также выявить скрытые фотоактивные свойства ферроидных перовскитных оксидов, обусловленные синергетической кинетикой поляризации/деполяризации.
"""

from pydantic import BaseModel, Field
from langchain.tools import tool
import json
import secrets
import string
import requests
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
)
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
import time


# ─────────────────────────────────────────────────────────────────────────────

# ── Конфигурация ─────────────────────────────────────────────────────────────
MODEL = 'qwen/qwen3.5-9b'
LLM_URL = 'http://127.0.0.1:1234/v1'
API_BASE_URL = 'http://127.0.0.1:8000'

# Лимит итераций поиска для защиты от бесконечного зацикливания
MAX_SEARCH_ITER = 5


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


# ── Вспомогательные функции для API ──────────────────────────────────────────
def _make_api_request(
    method: str, endpoint: str, token: Optional[str] = None,
    json_data: Optional[Dict] = None, params: Optional[Dict] = None
) -> Tuple[bool, Any]:
    """Выполняет API-запрос."""
    url = f'{API_BASE_URL}{endpoint}'
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            params=params,
            timeout=30
        )
        if response.status_code in [200, 201, 204]:
            return (True, response.json()
                    if response.content
                    else {'status': 'success'})
        else:
            error_msg = (response.json().get('message', response.text)
                         if response.content
                         else f'HTTP {response.status_code}')
            return (False,
                    {'error': error_msg, 'status_code': response.status_code})
    except requests.exceptions.RequestException as e:
        return False, {'error': f'Network error: {str(e)}'}


# ── Инструменты ──────────────────────────────────────────────────────────────
stub_data = """Мы успешно получили тонкие пленки BiFe(1-x)PdxO3 (BFPxO) (т.е. сегнетоэлектрический феррит висмута с частичным замещением ионов палладия в позиции Fe), а также гетероструктуру BFPxO/NiO методом золь-гель. В данной работе мы пытаемся разобраться в двух аспектах проблемы. С одной стороны, мы выясняем важную роль легирования палладием в оптоэлектронных характеристиках BFPxO. Мы устанавливаем взаимосвязи между валентностью палладия, искажением решетки и оптоэлектронными характеристиками BFPxO. Мы подтверждаем, что легирование палладием может минимизировать постоянную времени отклика BiFe(1-x)PdxO3 ниже 10 мс; при этом обнаружительная способность гетероструктуры BFPO/NiO может достигать примерно 109 Джонс. С другой стороны, для получения представления о различных стадиях затухания, существующих в BFPxO, используется метод измерения переходных процессов спада напряжения холостого хода (OCVD). Аномальное переходное явление с «чрезмерно длительным» временем релаксации, превышающим 10 с, вероятно, обусловлено процессом деполяризации BFPxO. Следовательно, мы выявляем скрытые фотоактивные свойства ферроидных перовскитных оксидов, которые могут быть обусловлены синергетической кинетикой поляризации/деполяризации, связанной с ферроэлектрическими доменами, и, в частности, могут быть вызваны легированием благородными металлами."""

# вопрос:
# Приведи хотя бы один пример получения тонкой пленки BiFe(1-x)PdxO3 (BFPxO). Достаточно общего ответа.

@tool(description='Поиск в локальной базе данных по семантической близости')
def retrieve_by_semantic_similarity(search_query: str) -> str:
    """Получает семантически-релевантную информацию из локальной базы данных."""
    print('Заглушка: вызов поиска по семантике.')
    return stub_data

@tool(description='Поиск в локальной базе данных по ключевым словам')
def retrieve_by_keywords(must_include: List[str], must_not_include: List[str]) -> str:
    """Получает информацию из локальной базы данных по ключевым словам."""
    print('Заглушка: вызов поиска по ключевым словам.')
    return stub_data


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
"""

# ── Инициализация LLM ────────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url=LLM_URL,
    api_key="lm-studio",
    model=MODEL,
    temperature=0.1,
    max_tokens=10000
)


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
    """Узел оценки достаточности информации."""
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

def run_and_trace(agent: CompiledStateGraph, query: str):
    print(f'Пользователь: {query}')
    print('═' * 70)

    start_time = time.time()
    initial_state = {
        'messages': [HumanMessage(content=query)],
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


# ── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Инициализация агента...')
    react_agent = create_react_agent()
    print('Агент готов к работе!\n')

    while True:
        user_msg = input('> ')
        if user_msg.strip().lower() in ('exit', 'выход', 'ничего'):
            break
        answer, state = run_and_trace(react_agent, user_msg)
        print(f'\nФинальный ответ: {answer}')


"""Пример запуска агента на данных заглушки:


Статистика: шагов TAO=1, вызовов инструментов=1, время=40.27с

Финальный ответ: На основе данных из базы данных, вот пример получения тонкой пленки BiFe(1-x)PdxO3 (BFPxO):

**Метод синтеза:** Золь-гель метод

В работе описывается получение тонких пленок BiFe(1-x)PdxO3 с последующим формированием гетероструктуры BFPxO/NiO методом золь-гель. Этот метод позволяет синтезировать сегнетоэлектрический феррит висмута с частичным замещением ионов палладия в позиции Fe, что приводит к изменению оптоэлектронных характеристик материала.

Легирование палладием в данной системе позволяет минимизировать постоянную времени отклика до значений ниже 10 мс, а также выявить скрытые фотоактивные свойства ферроидных перовскитных оксидов, обусловленные синергетической кинетикой поляризации/деполяризации.
"""

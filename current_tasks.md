# Текущие задачи для команды:


## 1. Парсер файлов

```
def parse(file_path: str) -> list[Chunk]:
    """Читает файл по заданному пути file_path и возвращает список чанков.

    Args:
        file_path (str): имя входного файла, допустимы txt, docx, pdf.

    Returns:
        list[Chunk]: Список чанков с метаданными.
    """
    pass
```

## 2. Парсер сущностей

```
def parse_entities(text: str) -> list[dict]:
    """Парсит текстовый фрагмент text и возвращает список сущностей и связей.

    Args:
        text (str): Текст (документа или чанка).

    Returns:
        list[dict]: Список сущностей, полученных от LLM и валидированных.
    """
    pass
```

## 3. Работа с графом -> поиск узла

```
def search_node(name: str) -> int | None:
    """Поиск узла в графе с учетов вариаций написания.

    Args:
        name (str): имя искомой сущности.

    Returns:
        int | None: ID сущности если найдена или None.
    """
    pass
```

## 4. Работа с графом -> добавление узла

```
def add_node(name: str):
    """Добавление узла в граф."""
```

## 5. Работа с графом -> удаление узла

```
def del_node(name: str):
    """Удаление узла."""
```

## 6. Работа с графом -> поиск соседей

```
def get_neighbors(node):
    """Получение соседей вершины. Возвращает строку с описанием фактов

    (сущности, связи)."""
```

## 7. Работа с графом -> формирование контекста для LLM.

```
def format_context(nodes: list[Node], ...) -> str:
    """Формирует контекст из найденных фактов для передачи в LLM."""

```

## 8. Интеграция ИИ-агента-исследователя из main.py в streamlit-приложение.


## Типы данных для чанкинга

```
@dataclass
class MetaData:
    """Метаданные чанка."""

    file_name: str
    chunk_number: int
    char_start: int
    char_end: int


class Chunk:
    """Чанк - текстовый фрагмент документа."""

    def __init__(self, doc_id: str, text: str, metadata: MetaData):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata

```



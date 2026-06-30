# Научный клубок
Задача: создать knowledge graph или поисково-аналитическую систему, которая связывает статьи, эксперименты, материалы, свойства, режимы, установки, исследовательские команды и выводы. Вам не нужно вписываться в жесткие рамки — вы сами решаете, какую архитектуру и интерфейс предложить: графовую БД, семантический поиск, гибридное решение. Главное — чтобы система отвечала на вопросы вида: «что уже делали по сплавам X при режиме Y и какой был эффект на свойство Z», показывала связанные сущности, историю решений и пробелы в данных. Для работы вы получите доступ к корпусу внутренних документов, каталогу экспериментов, справочникам материалов и оборудования, перечню сотрудников/лабораторий и тегам тематик.


## Онтология предметной области
Исходя из постановки задачи определены следующие ключевые `сущности` и связи.

### Узлы (Nodes)
* `Material` (Материал/Сплав) - Любой конструкционный или функциональный материал. Сплав, керамика, композит, покрытие. Атрибуты: `name`, `composition` (формула), `class` (сталь, титан, алюминий…), `phase`, `producer`, `form`
* `Element` (Элемент) - Химический элемент. Атрибуты: `name`
* `Property` (Свойство) - Физико-механическая характеристика. Напр.: Предел прочности, Коррозионная стойкость. Атрибуты: `name` (предел прочности, твёрдость, электропроводность…), `unit`, `measurement_method`
* `Process` (Процесс/Режим) - Напр.: Закалка 900°C + Старение 500°C 4ч. Атрибуты: `parameter` (температура, время, скорость охлаждения, давление), `value`, `unit`
* `Equipment` (Установка) — Напр.: Печь СНЗ, Разрывная машина Instron. Атрибуты: `model`, `manufacturer`, `specifications`
* `Team` (Команда/Лаборатория) — Напр.: Лаб. металловедения, Иванов И.И. Атрибуты: `name`, `type` (лаборатория/кафедра/завод), `parent`
* `Experiment` (Эксперимент) — Единичный эксперимент или серия опытов с конкретными параметрами. Атрибуты: `date`, `duration`, `protocol_id`, `status`, `objective`
* `Conclusion` (Вывод/Эффект) — Напр.: "Рост прочности на 15%", "Появление трещин". Атрибуты: `text`
* `Document` (Документ/Статья) — Источник данных. Атрибуты: `title`, `doi`, `journal`, `year`, `abstract`, `keywords`
* `Person` (Автор/Исследователь) - Персоналии, ассоциированные с сущность. Атрибуты: `full_name`, `orcid`, `affiliation`
* `Tag` (Тег) - тематический тег. Атрибуты: `name`, `category`


### Связи (Edges)
|Имя связи|От - к|Описание|
| ---- | ---- | ---- |
|`HAS_PROPERTY`|Material → Property|Свойство, измеренное для данного материала|
|`MEASURED_IN`|Property → Experiment|Свойство получено в конкретном эксперименте|
|`USED_MATERIAL`|Experiment → Material|	В эксперименте использовался данный материал|
|`APPLIED_CONDITION`|Experiment → ProcessCondition|	В эксперименте применялся данный режим (температура, скорость и т.д.)|
|`USED_EQUIPMENT`|Experiment → Equipment|Эксперимент выполнен на данной установке|
|`PERFORMED_BY`|Experiment → Person|Эксперимент выполнен исследователем (или группой)|
|`SUPERVISED_BY`|Experiment → Person|Научный руководитель эксперимента|
|`DESCRIBES`|Document → Material / Experiment / Property|Документ описывает материал, эксперимент или свойство|
|`AUTHORED_BY`|Document → Person|Автор документа|
|`AFFILIATED_TO`|Person → Organization|Сотрудник лаборатории/института|
|`OWNS_EQUIPMENT`|Organization → Equipment|Лаборатория владеет установкой|
|`TAGGED_AS`|(любая сущность) → Tag|Привязка тематического тега|
|`PART_OF_PROJECT`|Experiment / Document → Project|Эксперимент или статья принадлежит проекту|
|`DERIVED_FROM`|Material → Material|Новый сплав получен на основе базового (модификация)|
|`REFERENCES`|Document → Document|Цитирование или ссылка на предшествующую работу|


Пример запроса «что уже делали по сплаву X при режиме Y»

```
MATCH (m:Material {name: 'Inconel 718'})
MATCH (e:Experiment)-[:USED_MATERIAL]->(m)
MATCH (e)-[:APPLIED_CONDITION]->(c:ProcessCondition)
WHERE c.parameter = 'temperature' AND c.value >= 900 AND c.value <= 1000
MATCH (e)-[:HAS_PROPERTY]->(p:Property)
RETURN e, c, p.value, p.unit
```

### Формализованное описание базовых сущностей и связей в формате JSON‑словаря
Применяется для:
* Проектирования схемы графовой БД.
* Инструктирования LLM при извлечении структурированных данных из текстов.

#### Описание вершин

```
{
  "entityTypes": [
    {
      "name": "Material",
      "description": "Исследуемый материал (сплав, керамика, композит, покрытие).",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true, "description": "Технический идентификатор, генерируется системой" },
        { "name": "name", "type": "string", "required": true, "description": "Каноническое наименование (например, 'Inconel 718')" },
        { "name": "aliases", "type": "array", "items": "string", "required": false, "description": "Синонимы, марки по ГОСТ, торговые названия" },
        { "name": "composition", "type": "object", "required": false, "description": "Химический состав в формате {'element': 'wt%'}", "example": {"Ni": 52, "Cr": 19, "Fe": 18} },
        { "name": "class", "type": "string", "required": false, "description": "Класс: сталь, титан, алюминий, никелевый сплав и т.п." },
        { "name": "producer", "type": "string", "required": false, "description": "Производитель или поставщик" },
        { "name": "form", "type": "string", "required": false, "description": "Форма поставки: лист, пруток, порошок, проволока" },
        { "name": "normalized_name", "type": "string", "required": false, "description": "Нормализованное имя (строчные буквы, удалены пробелы/дефисы) для дедупликации" }
      ],
      "uniqueConstraints": ["name"],
      "indexes": ["normalized_name"]
    },
    {
      "name": "Experiment",
      "description": "Единичный эксперимент или серия опытов с фиксированными параметрами.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "protocol_id", "type": "string", "required": false, "description": "Внутренний номер протокола или шифр" },
        { "name": "date", "type": "datetime", "required": false, "description": "Дата проведения" },
        { "name": "duration", "type": "string", "required": false, "description": "Длительность в удобном формате (например, '2 ч')" },
        { "name": "objective", "type": "string", "required": false, "description": "Краткая цель эксперимента" },
        { "name": "status", "type": "string", "required": false, "enum": ["planned", "in_progress", "completed", "failed"], "description": "Статус выполнения" }
      ],
      "indexes": ["date"]
    },
    {
      "name": "Document",
      "description": "Научная статья, отчёт, патент, диссертация, техническое задание.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "title", "type": "string", "required": true, "description": "Заголовок документа" },
        { "name": "doi", "type": "string", "required": false, "description": "Digital Object Identifier" },
        { "name": "journal", "type": "string", "required": false, "description": "Название журнала или сборника" },
        { "name": "year", "type": "integer", "required": false },
        { "name": "abstract", "type": "string", "required": false },
        { "name": "keywords", "type": "array", "items": "string", "required": false },
        { "name": "type", "type": "string", "required": false, "enum": ["article", "patent", "report", "thesis", "specification"], "description": "Тип документа" }
      ],
      "uniqueConstraints": ["doi"],
      "indexes": ["title", "year"]
    },
    {
      "name": "Person",
      "description": "Исследователь, инженер, руководитель.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "full_name", "type": "string", "required": true },
        { "name": "orcid", "type": "string", "required": false, "description": "ORCID идентификатор" },
        { "name": "affiliation", "type": "string", "required": false, "description": "Текущее место работы (текст, для быстрого поиска)" },
        { "name": "email", "type": "string", "required": false }
      ],
      "uniqueConstraints": ["orcid"],
      "indexes": ["full_name"]
    },
    {
      "name": "Organization",
      "description": "Лаборатория, институт, завод, университет.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "name", "type": "string", "required": true },
        { "name": "type", "type": "string", "required": false, "enum": ["lab", "institute", "plant", "university"], "description": "Тип организации" },
        { "name": "parent", "type": "string", "required": false, "description": "Вышестоящая организация (текст)" }
      ],
      "uniqueConstraints": ["name"]
    },
    {
      "name": "Equipment",
      "description": "Установка, стенд, печь, пресс, микроскоп, дилатометр и т.д.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "model", "type": "string", "required": true, "description": "Модель оборудования" },
        { "name": "manufacturer", "type": "string", "required": false },
        { "name": "serial_number", "type": "string", "required": false, "description": "Заводской/инвентарный номер" },
        { "name": "specifications", "type": "object", "required": false, "description": "Технические характеристики (диапазон температур, давление и т.п.)" }
      ],
      "uniqueConstraints": [["model", "serial_number"]]
    },
    {
      "name": "Property",
      "description": "Измеряемое или расчётное свойство материала.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "name", "type": "string", "required": true, "description": "Название свойства (например, 'предел прочности')" },
        { "name": "unit", "type": "string", "required": false, "description": "Единица измерения (МПа, ГПа, % и т.д.)" },
        { "name": "measurement_method", "type": "string", "required": false, "description": "Метод измерения (ASTM, ГОСТ, etc.)" }
      ],
      "indexes": ["name"]
    },
    {
      "name": "ProcessCondition",
      "description": "Режим обработки (температура, время, давление, скорость, среда).",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "parameter", "type": "string", "required": true, "enum": ["temperature", "time", "pressure", "strain_rate", "cooling_rate", "atmosphere"], "description": "Тип параметра" },
        { "name": "value", "type": "number", "required": true },
        { "name": "unit", "type": "string", "required": false, "description": "Единица измерения (если не очевидна)" },
        { "name": "description", "type": "string", "required": false, "description": "Дополнительное описание, например 'нагрев в вакууме'" }
      ],
      "indexes": ["parameter", "value"]
    },
    {
      "name": "Project",
      "description": "Группировка экспериментов и документов по гранту или тематике.",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "title", "type": "string", "required": true },
        { "name": "funding_source", "type": "string", "required": false },
        { "name": "period_start", "type": "date", "required": false },
        { "name": "period_end", "type": "date", "required": false }
      ],
      "uniqueConstraints": ["title"]
    },
    {
      "name": "Tag",
      "description": "Тематический тег (например, 'жаропрочность', 'сварка', 'аддитивные технологии').",
      "attributes": [
        { "name": "id", "type": "UUID", "required": true },
        { "name": "name", "type": "string", "required": true },
        { "name": "category", "type": "string", "required": false, "enum": ["process", "property", "application"], "description": "Категория тега" }
      ],
      "uniqueConstraints": ["name"]
    }
  ]
}

```

#### Описание отношений

```
{
  "relationshipTypes": [
    {
      "type": "HAS_PROPERTY",
      "from": "Material",
      "to": "Property",
      "description": "Свойство, зафиксированное для материала (в контексте эксперимента).",
      "attributes": [
        { "name": "value", "type": "number", "required": true, "description": "Численное значение свойства" },
        { "name": "std", "type": "number", "required": false, "description": "Стандартное отклонение" },
        { "name": "condition_ref", "type": "UUID", "required": false, "description": "Ссылка на условие (режим), при котором измерено" },
        { "name": "conclusion", "type": "string", "required": false, "description": "Краткий вывод/интерпретация" }
      ]
    },
    {
      "type": "MEASURED_IN",
      "from": "Property",
      "to": "Experiment",
      "description": "Свойство получено в конкретном эксперименте.",
      "attributes": []
    },
    {
      "type": "USED_MATERIAL",
      "from": "Experiment",
      "to": "Material",
      "description": "В эксперименте использовался данный материал.",
      "attributes": [
        { "name": "quantity", "type": "string", "required": false, "description": "Количество материала" }
      ]
    },
    {
      "type": "APPLIED_CONDITION",
      "from": "Experiment",
      "to": "ProcessCondition",
      "description": "К эксперименту применён данный режим (температура, время и т.п.).",
      "attributes": []
    },
    {
      "type": "USED_EQUIPMENT",
      "from": "Experiment",
      "to": "Equipment",
      "description": "Эксперимент выполнен на данной установке.",
      "attributes": []
    },
    {
      "type": "PERFORMED_BY",
      "from": "Experiment",
      "to": "Person",
      "description": "Эксперимент выполнен исследователем (или группой).",
      "attributes": [
        { "name": "role", "type": "string", "required": false, "enum": ["executor", "supervisor"], "description": "Роль в эксперименте" }
      ]
    },
    {
      "type": "SUPERVISED_BY",
      "from": "Experiment",
      "to": "Person",
      "description": "Научный руководитель эксперимента (может быть несколько).",
      "attributes": []
    },
    {
      "type": "DESCRIBES",
      "from": "Document",
      "to": "Material",
      "description": "Документ описывает материал (может быть также на Experiment или Property).",
      "attributes": [
        { "name": "relevance", "type": "string", "required": false, "description": "Степень детализации (primary/supporting)" }
      ]
    },
    {
      "type": "DESCRIBES_EXP",
      "from": "Document",
      "to": "Experiment",
      "description": "Документ описывает эксперимент."
    },
    {
      "type": "DESCRIBES_PROP",
      "from": "Document",
      "to": "Property",
      "description": "Документ описывает свойство."
    },
    {
      "type": "AUTHORED_BY",
      "from": "Document",
      "to": "Person",
      "description": "Автор документа."
    },
    {
      "type": "AFFILIATED_TO",
      "from": "Person",
      "to": "Organization",
      "description": "Человек работает в организации."
    },
    {
      "type": "OWNS_EQUIPMENT",
      "from": "Organization",
      "to": "Equipment",
      "description": "Организация владеет оборудованием."
    },
    {
      "type": "TAGGED_AS",
      "from": "ANY",
      "to": "Tag",
      "description": "Любая сущность может быть помечена тегом.",
      "attributes": []
    },
    {
      "type": "PART_OF_PROJECT",
      "from": "Experiment",
      "to": "Project",
      "description": "Эксперимент относится к проекту."
    },
    {
      "type": "PART_OF_PROJECT_DOC",
      "from": "Document",
      "to": "Project",
      "description": "Документ относится к проекту."
    },
    {
      "type": "DERIVED_FROM",
      "from": "Material",
      "to": "Material",
      "description": "Новый материал создан на основе базового (модификация)."
    },
    {
      "type": "REFERENCES",
      "from": "Document",
      "to": "Document",
      "description": "Цитирование одной работы другой."
    }
  ]
}
```


#### Извлечение сущностей через LLM
Используя few-shot-промптинг предоставить LLM пример извлечения данных:

```
{
  "entities": [
    {
      "type": "Material",
      "id": "mat-001",
      "name": "Inconel 718",
      "composition": {"Ni": 52, "Cr": 19, "Fe": 18},
      "class": "nickel alloy"
    },
    {
      "type": "Experiment",
      "id": "exp-001",
      "date": "2024-03-15",
      "status": "completed"
    }
  ],
  "relationships": [
    {
      "from_id": "exp-001",
      "to_id": "mat-001",
      "type": "USED_MATERIAL",
      "properties": {}
    },
    {
      "from_id": "mat-001",
      "to_id": "prop-001",
      "type": "HAS_PROPERTY",
      "properties": {
        "value": 1240,
        "unit": "MPa",
        "conclusion": "Прочность выше базовой на 15%"
      }
    }
  ]
}
```

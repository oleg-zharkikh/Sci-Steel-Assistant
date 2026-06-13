# pip install pandas
# pip install kuzu
import kuzu

# 1. Создаем базу в памяти (или указываем путь к папке для сохранения на диск)
db = kuzu.Database(":memory:")
conn = kuzu.Connection(db)

# 2. Создаем схему (как в Neo4j). Types: INT64, STRING, ...
# ноды (вершины графа):
#   Lang - языки программирования
#   UseCase: варианты типового применения
#   Person - кто разработал
# ребра графа: TypicalUse - типовое применение
conn.execute('CREATE NODE TABLE Lang(name STRING, PRIMARY KEY (name))')
conn.execute('CREATE NODE TABLE UseCase(name STRING, PRIMARY KEY (name))')
conn.execute('CREATE NODE TABLE Person(name STRING, PRIMARY KEY (name))')

conn.execute('CREATE REL TABLE TypicalUse(FROM Lang TO UseCase)')
conn.execute('CREATE REL TABLE InventedBy(FROM Lang TO Person)')


# Добавляем данные
conn.execute("CREATE (u:Lang {name: 'Python'})")
conn.execute("CREATE (u:Lang {name: 'Java'})")
conn.execute("CREATE (u:Lang {name: 'Prolog'})")

conn.execute("CREATE (u:UseCase {name: 'Системы искусственного интеллекта'})")
conn.execute("CREATE (u:UseCase {name: 'Системы масштаба предприятий'})")
conn.execute("CREATE (u:UseCase {name: 'Прототипирование'})")

conn.execute("CREATE (p:Person {name: 'Гвидо ван Россум'})")

conn.execute("MATCH (a:Lang), (b:UseCase) WHERE a.name = 'Prolog' AND b.name = 'Системы искусственного интеллекта' CREATE (a)-[:TypicalUse]->(b)")
conn.execute("MATCH (a:Lang), (b:UseCase) WHERE a.name = 'Python' AND b.name = 'Системы искусственного интеллекта' CREATE (a)-[:TypicalUse]->(b)")
conn.execute("MATCH (a:Lang), (b:UseCase) WHERE a.name = 'Java' AND b.name = 'Системы масштаба предприятий' CREATE (a)-[:TypicalUse]->(b)")
conn.execute("MATCH (a:Lang), (b:UseCase) WHERE a.name = 'Python' AND b.name = 'Прототипирование' CREATE (a)-[:TypicalUse]->(b)")
conn.execute("MATCH (l:Lang), (p:Person) WHERE LOWER(l.name) = 'python' AND p.name = 'Гвидо ван Россум' CREATE (l)-[:InventedBy]->(p)")


# Делаем запрос на Cypher, найдем все языки для ИИ:
print("--- Языки для ИИ (через WHERE) ---")
query = """
MATCH (a:Lang)-[:TypicalUse]->(b:UseCase)
WHERE b.name = 'Системы искусственного интеллекта'
RETURN a.name AS Язык, b.name AS Применение
"""
result = conn.execute(query)
for i in result:
    print(type(i))
    print(i)
print("--- Языки для ИИ (через WHERE) - DataFrame ---")
print(result.get_as_df())  # можно вернуть в виде таблицы Pandas.DataFrame


# Нечеткий поиск
print("\n--- Языки, содержащие слово 'искусствен' ---")
query = """
MATCH (a:Lang)-[:TypicalUse]->(b:UseCase)
WHERE b.name CONTAINS 'искусствен'
RETURN a.name, b.name
"""
result = conn.execute(query)
print(result.get_as_df())



# Пример 2: найдем вершину Python и все связанные сущности:
# в Neo4j можно было бы так:
query_explore = """
MATCH (a:Lang)-[r]-(b)
WHERE LOWER(a.name) = 'python'
RETURN
    a.name AS Язык, 
    type(r) AS Тип_связи, 
    labels(b)[0] AS Тип_целевого_узла,
    b.name AS Связанная_сущность
"""
# но в kuzu пока нет type() и других операторов, поэтому придется отдельно искать все сущности для всех возможных связей и объединить результат
import pandas as pd

# 1. Получаем данные по первому типу связи
df1 = conn.execute("""
    MATCH (a:Lang)-[:TypicalUse]->(b:UseCase)
    WHERE LOWER(a.name) = 'python'
    RETURN a.name AS Язык, 'TypicalUse' AS Тип_связи, b.name AS Сущность
""").get_as_df()

# 2. Получаем данные по второму типу связи
df2 = conn.execute("""
    MATCH (a:Lang)-[:InventedBy]->(b:Person)
    WHERE LOWER(a.name) = 'python'
    RETURN a.name AS Язык, 'InventedBy' AS Тип_связи, b.name AS Сущность
""").get_as_df()

# 3. Объединяем их в одну таблицу
final_df = pd.concat([df1, df2], ignore_index=True)
print("--- Все связи для Python (через Pandas concat) ---")
print(final_df)



# Output:
# --- Языки для ИИ (через WHERE) ---
# <class 'list'>
# ['Python', 'Системы искусственного интеллекта']
# <class 'list'>
# ['Prolog', 'Системы искусственного интеллекта']
# --- Языки для ИИ (через WHERE) - DataFrame ---
#      Язык                         Применение
# 0  Python  Системы искусственного интеллекта
# 1  Prolog  Системы искусственного интеллекта

# --- Языки, содержащие слово 'искусствен' ---
#    a.name                             b.name
# 0  Python  Системы искусственного интеллекта
# 1  Prolog  Системы искусственного интеллекта
# --- Все связи для Python (через Pandas concat) ---
#      Язык   Тип_связи                           Сущность
# 0  Python  TypicalUse  Системы искусственного интеллекта
# 1  Python  TypicalUse                   Прототипирование
# 2  Python  InventedBy                   Гвидо ван Россум
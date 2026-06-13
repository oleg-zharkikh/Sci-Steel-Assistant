# pip install tantivy
import tantivy
from tantivy import Query, Occur

# 1. Создание схемы (аналог mapping в ES)
schema_builder = tantivy.SchemaBuilder()

# Добавление полей
schema_builder.add_text_field('title', stored=True)
schema_builder.add_text_field('content', stored=True)
schema_builder.add_text_field('category', stored=True)

schema = schema_builder.build()

# 2. Создание индекса в памяти (или на диске)
index = tantivy.Index(schema, path=None)
# index = tantivy.Index(schema, path='./index')

# 3. Добавление документов через контекстный менеджер
with index.writer() as writer:
    doc1 = tantivy.Document()
    doc1.add_text('title', 'Введение в Python')
    doc1.add_text('content', 'Python отличный язык для RAG и гибридного поиска')
    doc1.add_text('category', 'article')
    writer.add_document(doc1)

    doc2 = tantivy.Document()
    doc2.add_text('title', 'Java для начинающих')
    doc2.add_text('content', 'Java используется в enterprise-системах')
    doc2.add_text('category', 'books')
    writer.add_document(doc2)

    doc3 = tantivy.Document()
    doc3.add_text('title', 'Программирование')
    doc3.add_text('content', 'Учим Java и Python одновременно')
    doc3.add_text('category', 'books')
    writer.add_document(doc3)

# 4. Поиск по индексу
searcher = index.searcher()

print(f'Отладка. Всего документов в индексе: {searcher.num_docs}')

# СПОСОБ 1: Программная булева логика
# В parse_query 1 аргумент - поисковое слово, второй - поля в которых ищем.
query_python = index.parse_query('python', ['content'])
query_java = index.parse_query('java', ['content'])

boolean_query = Query.boolean_query([
    (Occur.Must, query_python),      # AND: должно присутствовать
    (Occur.MustNot, query_java)      # NOT: должно отсутствовать
])

top_docs = searcher.search(boolean_query, 10)
print(f'Найдено документов: {top_docs.count}')

for score, address in top_docs.hits:
    doc = searcher.doc(address)
    print(f"Score: {score:.4f}, Title: {doc['title'][0]}, Category: {doc['category'][0]}")

print("-" * 40)

# СПОСОБ 2: Синтаксис как в Elasticsearch
# '+' означает MUST (AND), '-' означает MUST_NOT (NOT)
es_style_query = index.parse_query('+python -java', ['content', 'title'])

top_docs_es = searcher.search(es_style_query, 10)
print(f'Найдено документов (ES style): {top_docs_es.count}')

for score, address in top_docs_es.hits:
    doc = searcher.doc(address)
    print(f"Score: {score:.4f}, Title: {doc['title'][0]}")


# Вывод:

# Отладка. Всего документов в индексе: 3
# Найдено документов: 1
# Score: 0.4136, Title: Введение в Python, Category: article
# ----------------------------------------
# Найдено документов (ES style): 1
# Score: 1.2918, Title: Введение в Python

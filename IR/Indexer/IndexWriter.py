from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

data = [
    {
        "title": "Introduction to Elasticsearch",
        "category": "Technology"
    },
    {
        "title": "Data Science with Python",
        "category": "Science"
    }
]


for i, doc in enumerate(data):
    result = es.index(index="my_index", doc_type="_doc", id=i+1, body=doc)
    print(f"Indexed document with ID: {result['_id']}")

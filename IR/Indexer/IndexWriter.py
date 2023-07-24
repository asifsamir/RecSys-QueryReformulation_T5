from elasticsearch import Elasticsearch

class ElasticsearchIndexer:
    def __init__(self, host='localhost', port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = "my_index"
        self.doc_type = "_doc"

    def index_data(self, data):
        for i, doc in enumerate(data):
            result = self.es.index(index=self.index_name, doc_type=self.doc_type, id=i+1, body=doc)
            print(f"Indexed document with ID: {result['_id']}")


if __name__ == "__main__":
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

    indexer = ElasticsearchIndexer()
    indexer.index_data(data)

from elasticsearch import Elasticsearch

class ElasticsearchSearcher:
    def __init__(self, host='localhost', port=9200, index_name="my_index"):
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = index_name

    def search(self, search_query):
        search_results = self.es.search(index=self.index_name, body=search_query)
        return search_results["hits"]["hits"]

    def process_search_results(self, search_results):
        for hit in search_results:
            print(f"Document ID: {hit['_id']}, Score: {hit['_score']}")
            print(hit["_source"])  # This will print the document's content


if __name__ == "__main__":
    search_query = {
        "query": {
            "multi_match": {
                "query": "Introduction Data",
                "fields": ["title"]
            }
        }
    }

    searcher = ElasticsearchSearcher()
    results = searcher.search(search_query)
    searcher.process_search_results(results)

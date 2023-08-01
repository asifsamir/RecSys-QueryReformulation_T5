from elasticsearch import Elasticsearch
import ast

class ElasticsearchSearcher:
    def __init__(self, host='localhost', port=9200, index_name="query_reform"):
        self.es = Elasticsearch('http://' + host + ':' + str(port),
                                # http_auth=("username", "password"),
                                verify_certs=False)
        self.index_name = index_name

    def search(self, query, top_K_results=10):
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["bug_title", "bug_description"]
                }
            },
            "size": top_K_results,
            "_source": ["ground_truths"]
        }

        search_results = self.es.search(index=self.index_name, body=search_query)

        ground_truths = self.process_search_results(search_results)

        return ground_truths

    def process_search_results(self, search_results):
        for hit in search_results:
            print(f"Document ID: {hit['_id']}, Score: {hit['_score']}")
            print(hit["_source"])  # This will print the document's content

        results_ground_truths = []

        for hit in search_results["hits"]["hits"]:
            source = hit.get("_source", {})
            ground_truths = source.get("ground_truths")

            # ground_truth is a string which is list of strings
            # e.g. "[\"Introduction to Elasticsearch\", \"Data Science with Python\"]"
            # need to convert it to a list of strings
            ground_truths = ast.literal_eval(ground_truths)
            results_ground_truths.append(ground_truths)
            

        return results_ground_truths


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

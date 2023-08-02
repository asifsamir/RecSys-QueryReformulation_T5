from elasticsearch import Elasticsearch
import ast

class Index_Searcher:
    def __init__(self, host='localhost', port=9200, index_name="bug_localization"):
        self.es = Elasticsearch('http://' + host + ':' + str(port),
                                # http_auth=("username", "password"),
                                verify_certs=False)
        self.index_name = index_name

    def search(self, query, top_K_results=10):
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["source_code"]
                }
            },
            "size": top_K_results,
            "_source": ["file_url"]
        }

        search_results = self.es.search(index=self.index_name, body=search_query)

        ground_truths = self.compiled_search_results(search_results)

        return ground_truths

    def compiled_search_results(self, search_results):

        suggested_all_source_files = []

        for hit in search_results["hits"]["hits"]:
            source = hit.get("_source", {})
            suggested_source_file = source.get("file_url")

            suggested_all_source_files.append(suggested_source_file)

        return suggested_all_source_files



from elasticsearch import Elasticsearch

class ElasticsearchIndexer:
    def __init__(self, host='localhost', port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = "query_reform"

    def index_data(self, bug_id, bug_title, bug_description, repo, ground_truths):
        document = {
            "bug_id": bug_id,
            "bug_title": bug_title,
            "bug_description": bug_description,
            "repo": repo,
            "ground_truths": ground_truths
        }
        result = self.es.index(index=self.index_name, body=document, refresh=True)
        # print(f"Indexed document with ID: {result['_id']}")

        return result


# if __name__ == "__main__":
#     data = [
#         {
#             "title": "Introduction to Elasticsearch",
#             "category": "Technology"
#         },
#         {
#             "title": "Data Science with Python",
#             "category": "Science"
#         }
#     ]
#
#     indexer = ElasticsearchIndexer()
#     indexer.index_data(data)

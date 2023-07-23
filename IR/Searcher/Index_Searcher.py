from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

search_query = {
    "query": {
        "match": {
            "title": "Introduction"
        }
    }
}

search_results = es.search(index="my_index", body=search_query)

# Extract the search hits from the response
hits = search_results["hits"]["hits"]

# Process the search results
for hit in hits:
    print(f"Document ID: {hit['_id']}, Score: {hit['_score']}")
    print(hit["_source"])  # This will print the document's content

from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Define the index mapping
mapping = {
    "mappings": {
        "properties": {
            "bug_id": {
                "type": "keyword"
            },
            "bug_title": {
                "type": "text"
            },
            "bug_description": {
                "type": "text"
            },
            "repo": {
                "type": "keyword"
            },
            "ground_truths": {
                "type": "keyword"
            }
        }
    }
}

# Create the index
index_name = "query_reform"
es.indices.create(index=index_name, body=mapping)

print(f"Index '{index_name}' with mapping has been created.")

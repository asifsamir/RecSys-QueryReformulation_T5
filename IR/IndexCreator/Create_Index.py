from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Define the index mapping
mapping = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text"
            },
            "category": {
                "type": "keyword"
            }
        }
    }
}

# Create the index
index_name = "my_index"
es.indices.create(index=index_name, body=mapping)

print(f"Index '{index_name}' with mapping has been created.")

from elasticsearch import Elasticsearch

# Connect to Elasticsearch
host = 'localhost'
port = 9200
es = Elasticsearch('http://' + host + ':' + str(port),
                                  # http_auth=("username", "password"),
                                  verify_certs=False)
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

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
# check if index already exists
if es.indices.exists(index=index_name):
    # delete index if it already exists
    print(f"Index '{index_name}' already exists. Deleting it...")
    es.indices.delete(index=index_name)


# create index with mapping
print(f"Creating index '{index_name}' with mapping...")
es.indices.create(index=index_name, mappings=mapping['mappings'])

print(f"Index '{index_name}' with mapping has been created.")

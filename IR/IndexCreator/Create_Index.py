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
            "source_code": {
                "type": "text"
            },
            "file_url": {
                "type": "keyword"
            }
        }
    }
}

# Create the index
index_name = "bug_localization"
# check if index already exists
if es.indices.exists(index=index_name):
    # delete index if it already exists
    print(f"Index '{index_name}' already exists. Deleting it...")
    es.indices.delete(index=index_name)


# create index with mapping
print(f"Creating index '{index_name}' with mapping...")
es.indices.create(index=index_name, mappings=mapping['mappings'])

print(f"Index '{index_name}' with mapping has been created.")

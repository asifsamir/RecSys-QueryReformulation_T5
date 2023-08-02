from elasticsearch import Elasticsearch

class IndexWriter:
    def __init__(self, host='localhost', port=9200, index_name="bug_localization"):
        self.es = Elasticsearch('http://' + host + ':' + str(port),
                           # http_auth=("username", "password"),
                           verify_certs=False)
        self.index_name = index_name

    def index_data(self, source_code, file_url):
        document = {
            "source_code": source_code,
            "file_url": file_url
        }
        result = self.es.index(index=self.index_name, body=document, refresh=True)
        # print(f"Indexed document with ID: {result['_id']}")

        return result

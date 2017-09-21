from elasticsearch import Elasticsearch
from time import sleep

INDEX_NAME = "aquaint"
DOC_TYPE = "doc"

QUERY_FILE = "data/queries.txt"
OUTPUT_FILE = "data/bm25_tweaked.txt"  # output the ranking

FIELDS = ["title", "content"]


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as fin:
        for line in fin.readlines():
            qid, query = line.strip().split(" ", 1)
            queries[qid] = query
    return queries


def change_index_settings(es, b, k):
    sim = {
        "similarity": {
            "default": {
                "type": "BM25",
                "b": b,
                "k1": k
            }
        }
    }
    es.indices.close(index=INDEX_NAME)
    es.indices.put_settings(index=INDEX_NAME, body=sim)
    es.indices.open(index=INDEX_NAME)
    sleep(1)
    print(es.indices.get_settings(index=INDEX_NAME))


def main():
    es = Elasticsearch()
    queries = load_queries(QUERY_FILE)

    change_index_settings(es, 0.065, 1.05)  # Tweak BM25 parameters

    with open(OUTPUT_FILE, 'w') as output:
        output.write("QueryId,DocumentId\n")
        for q_id, query in queries.items():
            #  generate ranking
            print("Ranking documents for [%s] '%s'" % (q_id, query))
            res = es.search(index=INDEX_NAME, q=query, df="content", _source=False, size=100)
            #  write results to file
            for hit in res['hits']['hits']:
                output.write("%s,%s\n" % (q_id, hit["_id"]))

    change_index_settings(es, 0.75, 1.2)  # Change back to default


if __name__ == "__main__":
    main()

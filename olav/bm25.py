from time import sleep


INDEX_NAME = "aquaint"
DOC_TYPE = "doc"
FIELDS = ["title", "content"]

QUERY_FILE = "data/queries.txt"
OUTPUT_BM25 = "data/bm25_optimized.txt"


def change_model(sim, es):
    print('\tClosing index')
    es.indices.close(index=INDEX_NAME)
    print('\tChanging parameters')
    es.indices.put_settings(index=INDEX_NAME, body=sim)
    print('\tOpening index')
    es.indices.open(index=INDEX_NAME)
    sleep(1.0)


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as fin:
        for line in fin.readlines():
            qid, query = line.strip().split(" ", 1)
            queries[qid] = query
    return queries


def run_query(file, es):
    queries = load_queries(QUERY_FILE)
    with open(file, 'w+') as f:
        for qid, query in queries.items():
            res = es.search(
                index=INDEX_NAME,
                q=query,
                df="content",
                _source=False,
                size=100).get("hits", {}).get("hits", {})
            f.write("QueryId,DocumentId\n")
            for doc in res:
                doc_id, score = doc.get("_id"), doc.get("_score")
                f.write(qid + "," + doc_id + "\n")

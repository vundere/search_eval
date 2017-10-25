import urllib
import requests
import json
import numpy as np

API = "http://gustav1.ux.uis.no:5002"
QUERY_FILE = "data/queries.txt"
TITLE_OUTPUT_FILE = "data/baseline_title.txt"
CONTENT_OUTPUT_FILE = "data/baseline_content.txt"

queries = {}
with open(QUERY_FILE, "r") as fin:
    for line in fin.readlines():
        qid, query = line.strip().split(" ", 1)
        queries[qid] = query


# Kaggle's (N)DCG functions:
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


def searchtitle():
    print("Searching title field . . .")
    with open(TITLE_OUTPUT_FILE, "w") as fin:
        fin.write("QueryId,DocumentId")
        for qid, query in queries.items():
            # print("Query: {}".format(query))
            res = search("clueweb12b", query, "title", size=100)
            for r in res.get("hits", {}).get("hits", {}):
                fin.write("\n{},{}".format(qid, r["_id"]))
                # print("\t{} {}".format(r["_id"], r["_score"]))
    print(". . . done")


def searchcontent():
    print("Searching content field . . .")
    with open(CONTENT_OUTPUT_FILE, "w") as fin:
        fin.write("QueryId,DocumentId")
        for qid, query in queries.items():
            # print("Query: {}".format(query))
            res = search("clueweb12b", query, "content", size=100)
            for r in res.get("hits", {}).get("hits", {}):
                fin.write("\n{},{}".format(qid, r["_id"]))
                # print("\t{} {}".format(r["_id"], r["_score"]))
    print(". . . done")


def search(indexname, query, field, size=10):
    url = "/".join([API, indexname, "_search"]) + "?" \
          + urllib.parse.urlencode({"q": query, "df": field, "size": size})
    response = requests.get(url).text
    return json.loads(response)


def main():
    res = search("clueweb12b", "united states", "content", size=5)
    for r in res.get("hits", {}).get("hits", {}):
        print("{} {}".format(r["_id"], r["_score"]))

    searchtitle()
    searchcontent()



if __name__ == '__main__':
    main()

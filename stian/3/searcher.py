import urllib
import requests
import json
import math
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


gtruth = {}
with open('data/qrels.csv', 'r') as qr:
    for line in qr.readlines():
        if line.startswith('QueryId'):
            continue
        qid, did, rel = line.strip().split(',')
        if qid not in gtruth.keys():
            gtruth[qid] = {}
        gtruth[qid][did] = rel


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
    payload = {}
    print("Searching title field . . .")
    with open(TITLE_OUTPUT_FILE, "w") as fin:
        fin.write("QueryId,DocumentId")
        for qid, query in queries.items():
            payload[qid] = []
            # print("Query: {}".format(query))
            res = search("clueweb12b", query, "title", size=100)
            for r in res.get("hits", {}).get("hits", {}):
                fin.write("\n{},{}".format(qid, r["_id"]))
                payload[qid].append(r['_id'])
                # print("\t{} {}".format(r["_id"], r["_score"]))
    print(". . . done")
    return payload


def searchcontent():
    payload = {}
    print("Searching content field . . .")
    with open(CONTENT_OUTPUT_FILE, "w") as fin:
        fin.write("QueryId,DocumentId")
        for qid, query in queries.items():
            payload[qid] = []
            # print("Query: {}".format(query))
            res = search("clueweb12b", query, "content", size=100)
            for r in res.get("hits", {}).get("hits", {}):
                fin.write("\n{},{}".format(qid, r["_id"]))
                payload[qid].append(r['_id'])
                # print("\t{} {}".format(r["_id"], r["_score"]))
    print(". . . done")
    return payload


def search(indexname, query, field, size=10):
    url = "/".join([API, indexname, "_search"]) + "?" \
          + urllib.parse.urlencode({"q": query, "df": field, "size": size})
    response = requests.get(url).text
    return json.loads(response)


def dcg(rel, p):
    dcg = rel[0]
    for i in range(1, min(p, len(rel))):
        dcg += rel[i] / math.log(i + 1, 2)  # rank position is indexed from 1..
    return dcg


def evaluate_balog(rankings):
    for qid, ranking in sorted(rankings.items()):
        gt = gtruth[qid]
        print("Query", qid)

        gains = []  # holds corresponding relevance levels for the ranked docs
        for doc_id in ranking:
            gain = gt.get(doc_id, 0)
            gains.append(gain)
        # print("\tGains:", gains)

        # relevance levels of the idealized ranking
        gain_ideal = sorted([v for _, v in gt.items()], reverse=True)
        # print("\tIdeal gains:", gain_ideal)

        ndcg10 = dcg_at_k(gains, 10) / dcg_at_k(gain_ideal, 10)
        ndcg20 = dcg_at_k(gains, 20) / dcg_at_k(gain_ideal, 20)

        print(ndcg10, ndcg20)


def evaluate(rankings):
    for qid, docid in sorted(rankings.items()):
        gt = gtruth[qid]

        gains = []
        for doc_id in docid:
            gain = gt.get(doc_id, 0)
            gains.append(gain)

        print(ndcg_at_k(gains, 10))
        print(ndcg_at_k(gains, 20))


def main():
    res = search("clueweb12b", "united states", "content", size=5)
    for r in res.get("hits", {}).get("hits", {}):
        print("{} {}".format(r["_id"], r["_score"]))

    title_res = searchtitle()
    content_res = searchcontent()
    # print(gtruth)
    # print(title_res)
    evaluate_balog(title_res)
    evaluate_balog(content_res)


if __name__ == '__main__':
    main()

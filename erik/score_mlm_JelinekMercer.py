from elasticsearch import Elasticsearch
import math
from pprint import pprint

INDEX_NAME = "aquaint"
DOC_TYPE = "doc"

QUERY_FILE = "C:/Users/EJS/OneDrive/B-DATA/DAT630_webSearch_dataMining/GitHub/slippers1-files/assignment-1/data/queries.txt"
OUTPUT_FILE = "C:/Users/EJS/OneDrive/B-DATA/DAT630_webSearch_dataMining/GitHub/slippers1-files/assignment-1/data/mlm_tweaked_JelinekMercer.txt"  # output the ranking

FIELDS = ["title", "content"]
FIELD_WEIGHTS = [0.35, 0.65]
LAMBDA = 0.05


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as fin:
        for line in fin.readlines():
            qid, query = line.strip().split(" ", 1)
            queries[qid] = query
    return queries


def analyze_query(es, query):
    tokens = es.indices.analyze(index=INDEX_NAME, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        query_terms.append(t["token"])
    return query_terms


class CollectionLM(object):
    def __init__(self, es, qterms):
        self._es = es
        self._probs = {}
        # computing P(t|C_i) for each field and for each query term
        for field in FIELDS:
            self._probs[field] = {}
            for t in qterms:
                self._probs[field][t] = self.__get_prob(field, t)

    def __get_prob(self, field, term):
        # use a boolean query to find a document that contains the term
        hits = self._es.search(index=INDEX_NAME, body={"query": {"match": {field: term}}},
                               _source=False, size=1).get("hits", {}).get("hits", {})
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is not None:
            # ask for global term statistics when requesting the term vector of that doc (`term_statistics=True`)
            tv = self._es.termvectors(index=INDEX_NAME, doc_type=DOC_TYPE, id=doc_id, fields=field,
                                      term_statistics=True)["term_vectors"][field]
            ttf = tv["terms"].get(term, {}).get("ttf", 0)  # total term count in the collection (in that field)
            sum_ttf = tv["field_statistics"]["sum_ttf"]
            return ttf / sum_ttf

        return 0  # this only happens if none of the documents contain that term

    def prob(self, field, term):
        return self._probs.get(field, {}).get(term, 0)


def score_mlm(es, clm, qterms, doc_id):
    score = 0  # log P(q|d)

    # Getting term frequency statistics for the given document field from Elasticsearch
    # Note that global term statistics are not needed (`term_statistics=False`)
    tv = es.termvectors(index=INDEX_NAME, doc_type=DOC_TYPE, id=doc_id, fields=FIELDS, term_statistics=False).get("term_vectors", {})

    # NOTE: Keep in mind that a given document field might be empty. In that case there is no tv[field].

    # scoring the query
    for t in qterms:
        Pt_theta_d = 0  # P(t|\theta_d)
        for i, field in enumerate(FIELDS):

            # TODO compute the field language model $P(t|\theta_{d_i})$ with Jelinek-Mercer smoothing
            ####################################
            tf, tf_sum, dl_sum = 0, 0, 0
            if field in tv and t in tv[field]['terms']:
                if t in tv[field]['terms']:
                    tf = tv[field]['terms'][t]['term_freq']
                    tf_sum += tf
                dl = sum(stats['term_freq'] for term, stats in tv[field]['terms'].items())  # Document length
                dl_sum += dl
                Pt_theta_di = ((1 - LAMBDA) * (tf_sum/dl_sum)) + (LAMBDA * clm.prob(field, t))
            else:
                Pt_theta_di = (LAMBDA * clm.prob(field, t))

            ####################################

            # NOTE keep in mind that the term vector will not contain `term` as a key if the document doesn't
            # contain that term; you will still need to use the background term probabilities for that term.
            # You can get the background term probability using `clm.prob(field, t)`

            Pt_theta_d += FIELD_WEIGHTS[i] * Pt_theta_di

        # TODO uncomment this line once you computed Pt_theta_d (and it is >0)
        score += math.log(Pt_theta_d)

    return score


def main():
    es = Elasticsearch()
    queries = load_queries(QUERY_FILE)

    with open(OUTPUT_FILE, "w") as fout:
        # write header
        fout.write("QueryId,DocumentId\n")
        for qid, query in queries.items():
            # get top 200 docs using BM25
            print("Get baseline ranking for [%s] '%s'" % (qid, query))
            res = es.search(index=INDEX_NAME, q=query, df="content", _source=False, size=200).get('hits', {})

            # re-score docs using MLM
            print("Re-scoring documents using MLM and Jelinek-Mercer smoothing")
            # get analyzed query
            qterms = analyze_query(es, query)
            # get collection LM
            # (this needs to be instantiated only once per query and can be used for scoring all documents)
            clm = CollectionLM(es, qterms)
            scores = {}
            for doc in res.get("hits", {}):
                doc_id = doc.get("_id")
                scores[doc_id] = score_mlm(es, clm, qterms, doc_id)

            # write top 100 results to file
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]:
                fout.write(qid + "," + doc_id + "\n")

if __name__ == "__main__":
    main()

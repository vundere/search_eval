from elasticsearch import Elasticsearch
import math

INDEX_NAME = "aquaint"
DOC_TYPE = "doc"

QUERY_FILE = "data/queries.txt"
OUTPUT_FILE = "data/mlm_default.txt"  # output the ranking
OUTPUT_BM25 = "data/bm25_optimized.txt"

FIELDS = ["title", "content"]

FIELD_WEIGHTS = [0.2, 0.8]

LAMBDA = 0.4
MU = 0.5


session_file = "data/not.finished"


def dump_args(args):
    with open(session_file, "w") as f:
        for arg in args:
            f.write('{}\n'.format(arg))


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


def score_mlm(es, clm, qterms, doc_id, lam):
    score = 0  # log P(q|d)

    # Getting term frequency statistics for the given document field from Elasticsearch
    # Note that global term statistics are not needed (`term_statistics=False`)
    tv = es.termvectors(index=INDEX_NAME, doc_type=DOC_TYPE, id=doc_id, fields=FIELDS,
                        term_statistics=False).get("term_vectors", {})

    # NOTE: Keep in mind that a given document field might be empty. In that case there is no tv[field].

    # scoring the query
    for t in qterms:
        Pt_theta_d = 0  # P(t|\theta_d)
        tf, tf_sum, dl_sum = 0, 0, 0
        for i, field in enumerate(FIELDS):
            if field in tv and t in tv[field]['terms']:
                if t in tv[field]['terms']:
                    tf = tv[field]['terms'][t]['term_freq']
                    tf_sum += tf
                dl = sum(stats['term_freq'] for term, stats in tv[field]['terms'].items())  # Document length
                dl_sum += dl
                Pt_theta_di = ((1 - lam) * (tf_sum / dl_sum)) + (lam * clm.prob(field, t))
            else:
                Pt_theta_di = (LAMBDA * clm.prob(field, t))

            Pt_theta_d += FIELD_WEIGHTS[i] * Pt_theta_di

        score += math.log(Pt_theta_d)

    return score


def score_bm25(es, qterms, doc_id, k, b):
    bm_score = 0

    tv = es.termvectors(index=INDEX_NAME, doc_type=DOC_TYPE, id=doc_id, fields=FIELDS,
                        term_statistics=False).get("term_vectors", {})
    total_docs = 0
    for t in qterms:
        tf, tf_sum, dl_sum, term_docs = 0, 0, 0, 0,
        for i, field in enumerate(FIELDS):
            if field in tv and t in tv[field]['terms']:
                if t in tv[field]['terms']:
                    tf = tv[field]['terms'][t]['term_freq']
                    tf_sum += tf
                    term_docs += 1
                dl = sum(stats['term_freq'] for term, stats in tv[field]['terms'].items())  # Document length
                total_docs += 1
                dl_sum += dl
                avdl = dl_sum/total_docs
                bm_score += (tf * (1 + k) / (tf + (k * (1 - b + (b * (dl / avdl))))))\
                            * math.log(total_docs / term_docs, 2)
    return bm_score


def score_mlm_dirichlet(es, clm, qterms, doc_id, mu):
    dir_score = 0
    tv = es.termvectors(index=INDEX_NAME, doc_type=DOC_TYPE, id=doc_id, fields=FIELDS,
                        term_statistics=False).get("term_vectors", {})

    Pt_theta_d = 0  # P(t|\theta_d)
    for t in qterms:
        tf, tf_sum = 0, 0
        for i, field in enumerate(FIELDS):
            if field in tv and t in tv[field]['terms']:
                if t in tv[field]['terms']:
                    tf = tv[field]['terms'][t]['term_freq']
                    tf_sum += tf
                dl = sum(stats['term_freq'] for term, stats in tv[field]['terms'].items())
                Pt_theta_di = (tf + mu * clm.prob(field, t)) / (dl + mu)
            else:
                Pt_theta_di = clm.prob(field, t)

            Pt_theta_d += FIELD_WEIGHTS[i] * Pt_theta_di

        dir_score += math.log(Pt_theta_d)

    return dir_score


def run(k, b, lam, mu):
    try:
        es = Elasticsearch()

        queries = load_queries(QUERY_FILE)
        with open(OUTPUT_FILE, "w") as fout, open(OUTPUT_BM25, 'w') as gout:
            fout.write("QueryId,DocumentId\n")
            gout.write("QueryId,DocumentId\n")
            for qid, query in queries.items():
                print("Get baseline ranking for [%s] '%s'" % (qid, query))
                res = es.search(index=INDEX_NAME, q=query, df="content", _source=False, size=200).get('hits', {})
                qterms = analyze_query(es, query)
                if lam <= 1:
                    print("Re-scoring documents using MLM")
                    clm = CollectionLM(es, qterms)
                    scores = {}
                    for doc in res.get("hits", {}):
                        doc_id = doc.get("_id")
                        scores[doc_id] = score_mlm(es, clm, qterms, doc_id, lam)
                    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]:
                        fout.write(qid + "," + doc_id + "\n")

                print("Re-scoring documents using optimized BM25")
                scores_bm = {}
                for doc in res.get("hits", {}):
                    doc_id = doc.get("_id")
                    scores_bm[doc_id] = score_bm25(es, qterms, doc_id, k, b)

                for doc_id, score in sorted(scores_bm.items(), key=lambda x: x[1], reverse=True)[:100]:
                    gout.write(qid + "," + doc_id + "\n")
    except Exception as e:
        dump_args([LAMBDA, b])
        print('There was an error \n {}'.format(e))


if __name__ == '__main__':
    run(1.2, 0.75, 0.1, 0.7)

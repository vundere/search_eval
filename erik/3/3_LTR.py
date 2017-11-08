import urllib
import requests
import json
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

API = "http://gustav1.ux.uis.no:5002"

BASIC_INDEX_NAME = "clueweb12b"
ANCHORS_INDEX_NAME = "clueweb12b_anchors"

NUM_DOCS = 200
NUM_FEAT = 3

QUERY_FILE = "data/queries.txt"
QRELS_FILE = "data/qrels.csv"
FEATURES_FILE = "data/features.txt"

QUERY2_FILE = "data/queries2.txt"
FEATURES2_FILE = "data/features2.txt"
OUTPUT_FILE = "data/ltr2.txt"
TOP_DOCS = 20  # this many top docs to write to output file

FIELDS = ["title", "content", "anchors"]

LAMBDA = 0.1


class CollectionLM(object):
    def __init__(self, qterms):
        self._probs = {}
        # computing P(t|C_i) for each field and for each query term
        for field in FIELDS:
            self._probs[field] = {}
            for t in qterms:
                self._probs[field][t] = self.__get_prob(field, t)

    def __get_prob(self, field, term):
        # use a boolean query to find a document that contains the term
        index_name = get_index_name(field)
        hits = search(index_name, term, field, size=1).get("hits", {}).get("hits", {})
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is not None:
            # ask for global term statistics when requesting the term vector of that doc (`term_statistics=True`)
            if term_vectors("clueweb12b", doc_id, term_statistics=True)["found"] == True:
                index_name = get_index_name(field)
                tv = term_vectors(index_name, doc_id, term_statistics=True)
                ttf = tv["term_vectors"][field]["terms"].get(term, {}).get("ttf", 0)  # total term count in the collection (in that field)
                sum_ttf = tv["term_vectors"][field]["field_statistics"]["sum_ttf"]
                return ttf / sum_ttf

        return 0  # this only happens if none of the documents contain that term

    def prob(self, field, term):
        return self._probs.get(field, {}).get(term, 0)


class PointWiseLTRModel(object):
    def __init__(self, regressor):
        """
        :param classifier: an instance of scikit-learn regressor
        """
        self.regressor = regressor

    def _train(self, X, y):
        """
        Trains and LTR model.
        :param X: features of training instances
        :param y: relevance assessments of training instances
        :return:
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(self, ft, doc_ids):
        """
        Predicts relevance labels and rank documents for a given query
        :param ft: a list of features for query-doc pairs
        :param ft: a list of document ids
        :return:
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as fin:
        for line in fin.readlines():
            qid, query = line.strip().split(" ", 1)
            queries[qid] = query
    return queries


def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, "r") as fin:
        i = 0
        for line in fin.readlines():
            i += 1
            if i == 1:  # skip header line
                continue
            qid, doc_id, rel = line.strip().split(",", 2)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel
    return qrels


def load_features(features_file):
    X, y, qids, doc_ids = [], [], [], []
    with open(features_file, "r") as f:
        i, s_qid = 0, None
        for line in f:
            items = line.strip().split()
            label = int(items[0])
            qid = items[1]
            doc_id = items[2]
            features = np.array([float(i.split(":")[1]) for i in items[3:]])
            X.append(features)
            y.append(label)
            qids.append(qid)
            doc_ids.append(doc_id)

    return X, y, qids, doc_ids


def get_index_name(field):
    return ANCHORS_INDEX_NAME if field == "anchors" else BASIC_INDEX_NAME

def search(indexname, query, field, size=10):
    url = "/".join([API, indexname, "_search"]) + "?" \
          + urllib.parse.urlencode({"q": query, "df": field, "size": size})
    response = requests.get(url).text
    return json.loads(response)


def term_vectors(indexname, doc_id, term_statistics=False):
    """
    param term_statistics: Boolean; True iff term_statistics are required.
    """
    url = "/".join([API, indexname, doc_id, "_termvectors"]) + "?" \
          + urllib.parse.urlencode({"term_statistics": str(term_statistics).lower()})
    response = requests.get(url).text

    return json.loads(response)


def analyze_query(indexname, query):
    url = "/".join([API, indexname, "_analyze"]) + "?" \
          + urllib.parse.urlencode({"text": query})
    response = requests.get(url).text
    r = json.loads(response)
    return [t["token"] for t in r["tokens"]]


def exists(indexname, docid):
    url = "/".join([API, indexname, docid, "_exists"])
    response = requests.get(url).text
    return json.loads(response)


def get_training_data(queries, qrels, output_file):
    with open(output_file, "w") as fout:
        for qid, query in sorted(queries.items()):
            # get feature vectors
            feats = get_features(qid, query)
            # assign target labels and write to file
            for doc_id, feat in feats.items():
                if doc_id in qrels[qid]: # we only consider docs where we have the target label
                    rel = qrels[qid][doc_id]
                    # we use -1 as value for the missing features
                    for fid in range(1, NUM_FEAT + 1):
                        if fid not in feat:
                            feat[fid] = -1
                    # write to file
                    feat_str = ['{}:{}'.format(k,v) for k,v in sorted(feat.items())]
                    fout.write(" ".join([str(rel), qid, doc_id] + feat_str) + "\n")


def lm(clm, qterms, doc_id, field):
    score = 0  # log P(q|d)

    # Getting term frequency statistics for the given document field from Elasticsearch
    # Note that global term statistics are not needed
    index_name = get_index_name(field)
    tv = term_vectors(index_name, doc_id)["term_vectors"]

    # compute field length $|d|$
    len_d = 0  # document field length initialization
    if field in tv:  # that document field may be NOT empty
        len_d = sum([s["term_freq"] for t, s in tv[field]["terms"].items()])

    # scoring the query
    for t in qterms:
        Pt_theta_d = 0  # P(t|\theta_d)
        if field in tv:
            Pt_d = tv[field]["terms"].get(t, {}).get("term_freq", 0) / len_d  # $P(t|d)$
        else:  # that document field is empty
            Pt_d = 0
        Pt_C = clm.prob(field, t)  # $P(t|C)$
        Pt_theta_d = (1 - LAMBDA) * Pt_d + LAMBDA * Pt_C  # $P(t|\theta_{d})$ with J-M smoothing
        # Pt_theta_d is 0 if t doesn't occur in any doc for that field, even with smoothing:
        score += math.log(Pt_theta_d) if Pt_theta_d > 0 else 0

    return score


def get_features(qid, query):
    feats = {}
    print("Getting features for query #{} '{}'".format(qid, query))

    # Analyze query (will be needed for some features)
    qterms = analyze_query("clueweb12b", query)

    # Feature 1: BM25 content score
    res1 = search("clueweb12b", query, "content", size=NUM_DOCS)
    # Initializing feature vector with values for Feature 1
    print("\tElasticsearch content field ...")
    for doc in res1.get('hits', {}).get("hits", {}):
        doc_id = doc.get("_id")
        feats[doc_id] = {1: doc.get("_score")}

    # Feature 2: BM25 title score
    print("\tElasticsearch title field ...")
    res2 = search("clueweb12b", query, "title", size=NUM_DOCS)
    for doc in res2.get('hits', {}).get("hits", {}):
        doc_id = doc.get("_id")
        if doc_id in feats:
            feats[doc_id][2] = doc.get("_score")

    # Feature 3: BM25 anchors score
    # NOTE: we retrieve more candidate documents here
    print("\tElasticsearch anchors field ...")
    res3 = search("clueweb12b_anchors", query, "anchors", size=NUM_DOCS*10)
    for doc in res3.get('hits', {}).get("hits", {}):
        doc_id = doc.get("_id")
        # check if doc is present in regular index
        # if exists("clueweb12b", doc_id)["exists"] == True:
        if doc_id in feats:
            feats[doc_id][3] = doc.get("_score")

    # TODO: computation of additional features comes here
    # Feature 4: LM content score
    print("\tLM content field ...")
    res4 = search("clueweb12b", query, "content", size=NUM_DOCS*10)
    clm = CollectionLM(qterms)
    scores = {}
    for doc in res4.get("hits", {}).get("hits", {}):
        doc_id = doc.get("_id")
        if doc_id in feats:
            scores[doc_id] = lm(clm, qterms, doc_id, "content")
    # send top 20 to feats
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:NUM_DOCS]:
        feats[doc_id][4] = score

    # Feature 5: LM title score
    print("\tLM title field ...")
    res5 = search("clueweb12b", query, "title", size=NUM_DOCS*10)
    clm = CollectionLM(qterms)
    scores = {}
    for doc in res5.get("hits", {}).get("hits", {}):
        doc_id = doc.get("_id")
        if doc_id in feats:
            scores[doc_id] = lm(clm, qterms, doc_id, "title")
    # send top 20 to feats
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:NUM_DOCS]:
        feats[doc_id][5] = score

    # Feature 6: LM anchors score
    print("\tLM anchors field ...")
    res6 = search("clueweb12b_anchors", query, "anchors", size=NUM_DOCS*10)
    clm = CollectionLM(qterms)
    scores = {}
    for doc in res6.get("hits", {}).get("hits", {}):
        doc_id = doc.get("_id")
        # check if doc is present in regular index
        # if exists("clueweb12b", doc_id)["exists"] == True:
        if doc_id in feats:
            scores[doc_id] = lm(clm, qterms, doc_id, "anchors")
    # send top 20 to feats
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:NUM_DOCS]:
        feats[doc_id][6] = score

    # TODO: we can apply feature normalization here

    return feats


def main():
    queries = load_queries(QUERY_FILE)
    qrels = load_qrels(QRELS_FILE)
    get_training_data(queries, qrels, FEATURES_FILE)

    train_X, train_y, qids, doc_ids = load_features(FEATURES_FILE)

    clf = RandomForestRegressor(max_depth=2, random_state=0)
    ltr = PointWiseLTRModel(clf)
    ltr._train(train_X, train_y)

    queries2 = load_queries(QUERY2_FILE)

    output_format = "trec"

    with open(OUTPUT_FILE, "w") as fout:
        for qid, query in sorted(queries2.items()):
            # Get feature vectors
            feats = get_features(qid, query)
            # Convert into the format required by the `PointWiseLTRModel` class
            # and deal with missing feature values
            doc_fts = []
            doc_ids = []
            for doc_id, feat in feats.items():
                for fid in range(1, NUM_FEAT + 1):
                    if fid not in feat:
                        feat[fid] = -1
                doc_fts.append(np.array([float(val) for fid, val in sorted(feat.items())]))
                doc_ids.append(doc_id)
            # Get ranking
            r = ltr.rank(doc_fts, doc_ids)
            # Write the results to file
            rank = 1
            for doc_id, score in r:
                if rank <= TOP_DOCS:
                    if output_format == "trec":
                        fout.write(("\t".join(["{}"] * 6) + "\n").format(qid, "Q0", doc_id, str(rank),
                                                                     str(score), "A3_3_Baseline"))
                    else:
                        fout.write(qid + "," + doc_id + "\n")
                rank += 1

if __name__ == "__main__":
    main()

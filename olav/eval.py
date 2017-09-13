RANKING_FILE = "data/baseline.txt"  # file with the document rankings
MLM_FILE = "data/mlm_default.txt"  # file with the MLM rankings
BM_FILE = "data/bm25_optimized.txt"  # file with tweaked BM25 rankings
QRELS_FILE = "data/qrels2.csv"  # file with the relevance judgments (ground truth)


def eval_query(ranking, gt):
    p10, ap, rr, num_rel = 0, 0, 0, 0

    # TODO
    for i, doc_id in enumerate(ranking):
        if doc_id in gt:  # doc is relevant
            num_rel += 1
            pi = num_rel / (i + 1)  # P@i
            ap += pi  # AP

            if i < 10:  # P@10
                p10 += 1
            if rr == 0:  # Reciprocal rank
                rr = 1 / (i + 1)

    p10 /= 10
    ap /= len(gt)  # divide by the number of relevant documents

    return {"P10": p10, "AP": ap, "RR": rr}


def full_eval(gt_file, output_file):
    # load data from ground truth file
    gt = {}  # holds a list of relevant documents for each queryID
    with open(gt_file, "r") as fin:
        header = fin.readline().strip()
        if header != "queryID,docIDs":
            raise Exception("Incorrect file format!")
        for line in fin.readlines():
            qid, docids = line.strip().split(",")
            gt[qid] = docids.split()

    # load data from output file
    output = {}
    with open(output_file, "r") as fin:
        header = fin.readline().strip()
        if header != "QueryId,DocumentId":
            raise Exception("Incorrect file format!")
        for line in fin.readlines():
            qid, docid = line.strip().split(",")
            if qid not in output:
                output[qid] = []
            output[qid].append(docid)

    # evaluate each query that is in the ground truth
    print("  QID  P@10   (M)AP  (M)RR")
    # TODO compute averages over the entire query set
    sum_p10, sum_ap, sum_rr, len_rankings = 0, 0, 0, 0
    for qid in sorted(gt.keys()):
        res = eval_query(output.get(qid, []), gt.get(qid, []))
        sum_p10 += res["P10"]
        sum_ap += res["AP"]
        sum_rr += res["RR"]
        len_rankings += 1
        # print("%5s %6.3f %6.3f %6.3f" % (qid, res["P10"], res["AP"], res["RR"]))

    # print averages
    avg_p10 = sum_p10 / len_rankings
    avg_ap = sum_ap / len_rankings
    avg_rr = sum_rr / len_rankings
    print("%5s %6.3f %6.3f %6.3f" % ("ALL", avg_p10, avg_ap, avg_rr))
    result = {'p10': avg_p10, 'ap': avg_ap, 'rr': avg_rr}
    return result


if __name__ == '__main__':
    # full_eval(QRELS_FILE, RANKING_FILE)
    full_eval(QRELS_FILE, 'data/bmo/results_b_0.80_k_1.50_.txt')

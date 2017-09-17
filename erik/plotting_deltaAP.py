import matplotlib.pyplot as plt

A = "data/baseline.txt"  # file with the document rankings
B = "data/mlm_tweaked_Dirichlet.txt"  # file with the document rankings
QRELS_FILE = "data/qrels2.csv"  # file with the relevance judgments (ground truth)


def eval_query(ranking, gt):
    ap, num_rel = 0, 0

    for i, doc_id in enumerate(ranking):
        if doc_id in gt:  # doc is relevant
            num_rel += 1
            pi = num_rel / (i + 1)  # P@i
            ap += pi  # AP

    ap /= len(gt)  # divide by the number of relevant documents

    return {"AP": ap}


def eval(gt_file, output_file):
    # load data from ground truth file
    gt = {}  # holds a list of relevant documents for each queryID
    ap = []
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
    sum_ap, len_rankings = 0, 0
    for qid in sorted(gt.keys()):
        res = eval_query(output.get(qid, []), gt.get(qid, []))
        sum_ap += res["AP"]
        len_rankings += 1
        ap.append(res["AP"])

    return ap


def main():
    delta_ap = []
    minus_1topoint5 = 0
    minus_point5topoint25 = 0
    minus_point25topoint05 = 0
    minus_point05topoint05 = 0
    point05topoint25 = 0
    point25topoint5 = 0
    point5to1 = 0
    intervals = ["[-1,-0.5)", "[-0.5, -0.25)", "[-0.25, -0.05)", "[-0.05, 0.05]", "(0.05, 0.25]", "(0.25, 0.5]", "(0.5, 1]"]

    # Evaluate AP for the two query sets
    a_ap = eval(QRELS_FILE, A)  # Baseline AP
    b_ap = eval(QRELS_FILE, B)  # Improved AP
    bins = []

    # find delta AP
    for i, v in enumerate(a_ap):
        delta_ap.append(b_ap[i] - v)

    # Distribute queries according to delta AP
    for _, v in enumerate(delta_ap):
        if -1.0 <= v < -0.5:
            minus_1topoint5 += 1
        elif -0.5 <= v < -0.25:
            minus_point5topoint25 += 1
        elif -0.25 <= v < -0.05:
            minus_point25topoint05 += 1
        elif -0.05 <= v <= 0.05:
            minus_point05topoint05 += 1
        elif 0.05 < v <= 0.25:
            point05topoint25 += 1
        elif 0.25 < v <= 0.5:
            point25topoint5 += 1
        elif 0.5 < v <= 1.0:
            point5to1 += 1

    # Append number of queries in each interval to 'bins'
    bins.append(minus_1topoint5)
    bins.append(minus_point5topoint25)
    bins.append(minus_point25topoint05)
    bins.append(minus_point05topoint05)
    bins.append(point05topoint25)
    bins.append(point25topoint5)
    bins.append(point5to1)

    # Plot 1: delta AP for each query
    plt.bar(range(len(delta_ap)), sorted(delta_ap, reverse=True), color='green')
    plt.title("Delta AP for each query")
    plt.xlabel('Query #')
    plt.ylabel('$\Delta$ AP')
    plt.show()

    # Plot 2: distribution of queries according to delta AP
    plt.bar(range(len(bins)), bins, color='green', align='center')
    plt.xticks(range(len(bins)), intervals)  # Change x-axis labels to the intervals matching the bin values
    plt.title("Distribution of queries according to delta AP")
    plt.xlabel('$\Delta$ AP')
    plt.ylabel('Number of queries')
    plt.show()
    print(bins)

    # Plot 2: distribution of queries according to delta AP (histogram)
    # plt.hist(delta_ap, bins=[-1.00, -0.50, -0.25, -0.05, 0.05, 0.25, 0.50, 1.00])
    # plt.title("Distribution of queries according to delta AP")
    # plt.xlabel('$\Delta$ AP')
    # plt.ylabel('Number of queries')
    # plt.show()


if __name__ == "__main__":
    main()


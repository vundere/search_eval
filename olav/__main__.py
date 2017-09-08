import eval
from mlm import run
QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE

if __name__ == '__main__':
    run()
    eval.full_eval(QRELS_FILE, RANKING_FILE)
    eval.full_eval(QRELS_FILE, MLM_FILE)
    eval.full_eval(QRELS_FILE, BM_FILE)

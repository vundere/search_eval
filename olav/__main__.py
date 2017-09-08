import eval
from mlm import run
QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE

LAMBDA = 0.1
k = 1
b = 0

if __name__ == '__main__':
    while 0.0 <= LAMBDA <= 1.0 and 0.0 <= b <= 1.0 <= k <= 2.0:
        run(k=k, b=b, lam=LAMBDA)
        baseline = eval.full_eval(QRELS_FILE, RANKING_FILE)
        mlm = eval.full_eval(QRELS_FILE, MLM_FILE)
        bm = eval.full_eval(QRELS_FILE, BM_FILE)
        with open('data/results.txt', 'a') as rfile:
            rfile.write('  SCR  P@10   (M)AP  (M)RR')
            rfile.write('BLN  {0:06.3f}  {1:06.3f}  {2:06.3f}'.format(baseline['p10'], baseline['ap'], baseline['rr']))
            rfile.write('MLM  {0:06.3f}  {1:06.3f}  {2:06.3f}'.format(mlm['p10'], mlm['ap'], mlm['rr']))
            rfile.write('BMO  {0:06.3f}  {1:06.3f}  {2:06.3f}'.format(bm['p10'], bm['ap'], bm['rr']))
        LAMBDA += 0.01
        k += 0.01
        b += 0.01

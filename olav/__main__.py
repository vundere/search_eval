import eval
from mlm import run
from timer import Timer

QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE

LAMBDA = 0.1
b = 0

if __name__ == '__main__':
    timer = Timer('main')
    timer.start()
    for j in range(0, 101):
        k = 1
        for i in range(0, 101):
            run(k=k, b=b, lam=LAMBDA)
            baseline = eval.full_eval(QRELS_FILE, RANKING_FILE)
            mlm = eval.full_eval(QRELS_FILE, MLM_FILE)
            bm = eval.full_eval(QRELS_FILE, BM_FILE)
            with open('data/results.txt', 'a') as rfile:
                rfile.write('  SCR  P@10  (M)AP  (M)RR using parameters '
                            'lambda={0}, k={1}, b={2}\n'.format(LAMBDA, k, b))
                rfile.write('BLN  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(baseline['p10'], baseline['ap'],
                                                                            baseline['rr']))
                rfile.write('MLM  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(mlm['p10'], mlm['ap'], mlm['rr']))
                rfile.write('BMO  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(bm['p10'], bm['ap'], bm['rr']))
            if LAMBDA <= 1:
                LAMBDA += 0.01
            k += 0.01
        b += 0.01
    timer.stop()
    print(timer.total_running_time_long)

import eval
import os
import ast
import multiprocessing as mp
from mlm import run
from timer import Timer

QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE


LAMBDA = 0.1
b = 0
MU = 0

session_file = "data/not.finished"
outqueue = mp.Queue()


def dump_args(args):
    with open(session_file, "w") as f:
        for arg in args:
            f.write('{}\n'.format(arg))


def check_continue():
    if os.path.isfile(session_file):
        with open(session_file, "r") as f:
            arguments = ast.literal_eval(f.readline())
            print(arguments)
        return arguments
    else:
        return None


def singleprocess_run():
    run(k=k, b=b, lam=LAMBDA, mu=MU)
    baseline = eval.full_eval(QRELS_FILE, RANKING_FILE)
    mlm = eval.full_eval(QRELS_FILE, MLM_FILE)
    bm = eval.full_eval(QRELS_FILE, BM_FILE)
    return [baseline, mlm, bm]


if __name__ == '__main__':
    timer = Timer('main')
    timer.start()

    # cont = check_continue()
    # if cont:
    #     LAMBDA, b = cont
    try:
        for j in range(0, 101):
            k = 1
            for i in range(0, 101):

                with open('data/results.txt', 'a') as rfile:
                    rfile.write('SCR   P@10  (M)AP  (M)RR using parameters '
                                'lambda={0:05.3f}, k={1:05.3f}, b={2:05.3f}\n'.format(LAMBDA, k, b))
                    rfile.write('BLN  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(baseline['p10'], baseline['ap'],
                                                                                baseline['rr']))
                    rfile.write('MLM  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(mlm['p10'], mlm['ap'], mlm['rr']))
                    rfile.write('BMO  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(bm['p10'], bm['ap'], bm['rr']))
                if LAMBDA <= 1:
                    LAMBDA += 0.01
                k += 0.05
            b += 0.05
    except Exception as e:
        # dump_args([LAMBDA, b])
        print('There was an error \n {0}\n{1} {2}'.format(e, type(e).__name__, e.args))
    timer.stop()
    print(timer.total_running_time_long)

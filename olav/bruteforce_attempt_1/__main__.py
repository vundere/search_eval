import eval
import os
import ast
from elasticsearch import Elasticsearch
from mlm import run
from timer import Timer

QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE


LAMBDA = 1
b = 0.750
MU = 0
WEIGHTS = [0.1, 0.1]
INCREMENT = 0.1  # WARNING: SETTING THIS TO 0.01 WILL GIVE YOU A RUNTIME OF SEVERAL DAYS

session_file = "data/not.finished"

es = Elasticsearch()


SIM = {
    "similarity": {
        "default": {
            "type": "BM25",
            "b": 0.75,
            "k1": 1.2
        }
    }
}


def change_model(sim):
    from mlm import INDEX_NAME
    es.indices.close(index=INDEX_NAME)
    es.indices.put_settings(index=INDEX_NAME, body=sim)
    es.indices.open(index=INDEX_NAME)


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


def run_std():
    global LAMBDA, b, MU, WEIGHTS, INCREMENT
    timer = Timer('main')
    timer.start()
    cont = check_continue()
    if cont:
        LAMBDA, b = cont
    try:
        for j in range(0, int(1 / INCREMENT)):
            k = 1
            for i in range(0, int(1 / INCREMENT)):
                run(k=k, b=b, lam=LAMBDA, mu=MU, weights=WEIGHTS)
                baseline = eval.full_eval(QRELS_FILE, RANKING_FILE)
                mlm = eval.full_eval(QRELS_FILE, MLM_FILE)
                bm = eval.full_eval(QRELS_FILE, BM_FILE)
                with open('data/results.txt', 'a') as rfile:
                    rfile.write('SCR   P@10  (M)AP  (M)RR using parameters '
                                'lambda={0:05.3f}, k={1:05.3f}, b={2:05.3f}\n'.format(LAMBDA, k, b))
                    rfile.write('BLN  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(baseline['p10'], baseline['ap'],
                                                                                baseline['rr']))
                    rfile.write('MLM  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(mlm['p10'], mlm['ap'], mlm['rr']))
                    rfile.write('BMO  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(bm['p10'], bm['ap'], bm['rr']))
                if LAMBDA <= 1:
                    LAMBDA += INCREMENT
                k += INCREMENT
            b += INCREMENT
    except Exception as e:
        dump_args([LAMBDA, b])
        print('There was an error \n {}'.format(e))
    timer.stop()
    print(timer.total_running_time_long)


def run_mlm():
    global LAMBDA, b, MU, WEIGHTS, INCREMENT
    timer = Timer('main')
    timer.start()
    try:
        for j in range(0, int(1 / INCREMENT)):
            k = 1
            for i in range(0, int(1 / INCREMENT)):
                run(k=k, b=b, lam=LAMBDA, mu=MU, weights=WEIGHTS)
                baseline = eval.full_eval(QRELS_FILE, RANKING_FILE)
                mlm = eval.full_eval(QRELS_FILE, MLM_FILE)
                with open('data/results.txt', 'a') as rfile:
                    rfile.write('SCR   P@10  (M)AP  (M)RR using parameters '
                                'lambda={0:05.3f}, weights={1:05.3f}\n'.format(LAMBDA, WEIGHTS))
                    rfile.write('BLN  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(baseline['p10'], baseline['ap'],
                                                                                baseline['rr']))
                    rfile.write('MLM  {0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(mlm['p10'], mlm['ap'], mlm['rr']))
                LAMBDA += INCREMENT
                for w in WEIGHTS:
                    w += INCREMENT
    except Exception as e:
        dump_args([LAMBDA, b])
        print('There was an error \n {}'.format(e))
    timer.stop()
    print(timer.total_running_time_long)


# TODO implement iterating over MLM field weights
if __name__ == '__main__':
    run_mlm()

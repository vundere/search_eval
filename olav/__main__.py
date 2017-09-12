import eval
import os
import logging
import pprint as pp

from elasticsearch import Elasticsearch
from bm25 import run_query, change_model
from timer import Timer
from pathlib import Path

QRELS_FILE = eval.QRELS_FILE
MLM_FILE = eval.MLM_FILE
RANKING_FILE = eval.RANKING_FILE
BM_FILE = eval.BM_FILE


DEFAULT_SIM = {
    "similarity": {
        "default": {
            "type": "BM25",
            "b": 0.75,
            "k1": 1.2
        }
    }
}


SIM = {
    "similarity": {
        "default": {
            "type": "BM25",
            "b": 0.75,
            "k1": 1.45
        }
    }
}


def get_vars(sim):
    k = SIM["similarity"]["default"]["k1"]
    b = SIM["similarity"]["default"]["b"]
    return [k, b]


def exists(fname):
    checkfile = Path(fname)
    if checkfile.is_file():
        return True
    else:
        return False


# Logging functions nabbed from my discord bot
def setup_logging():
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename='bm25.log', encoding='utf-8', mode='w')
    fmt = logging.Formatter('[%(asctime)s] :%(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(fmt)
    log.addHandler(handler)
    return log


def end_logging(log):
    handlers = log.handlers[:]
    for hdlr in handlers:
        hdlr.close()
        log.removeHandler(hdlr)

if __name__ == '__main__':
    runtimer = Timer()
    runtimer.start()
    es = Elasticsearch()
    change_model(DEFAULT_SIM, es)  # Resets the search model
    loop_no = 1
    log = setup_logging()
    for i in range(11):
        SIM["similarity"]["default"]["k1"] = 1.45
        for j in range(11):
            loop_vars = get_vars(SIM)
            print('Running loop no. {}...'.format(loop_no))
            log.debug('Querying with vars k={0}, b={1}.\nElasticsearch settings:\n{2}'.format(loop_vars[0],  # line
                                                                                              loop_vars[1],  # length
                                                                                              pp.pformat(    # lol
                                                                                                 es.indices.get_settings
                                                                                                 (index='aquaint')
                                                                                             )))
            file = 'data/bmo/results_b_{0:04.2f}_k_{1:04.2f}_.txt'.format(loop_vars[1], loop_vars[0])
            if exists(file):
                print('Results for b={0:04.2f}, k={1:04.2f} already exist'.format(loop_vars[1], loop_vars[0]))
            else:
                run_query(file, es)
            SIM["similarity"]["default"]["k1"] += 0.01
            change_model(SIM, es)
            loop_no += 1
        SIM["similarity"]["default"]["b"] += 0.01
        change_model(SIM, es)
    runtimer.lap()
    print(runtimer.total_running_time_long)
    for i in os.listdir('data/BMO/'):
        with open('data/bmo/evaluations.txt', 'a') as f:
            cur_file = 'data/bmo/{}'.format(i)
            print('Evaluating {}...'.format(cur_file))
            if cur_file.startswith('data/bmo/results'):
                feval = eval.full_eval(QRELS_FILE, cur_file)
                fvars = i.split('_')
                f.write('\tP@10  (M)AP  (M)RR using parameters b={0:04.2f}, k={1:04.2f}\n'.format(float(fvars[2]),
                                                                                                  float(fvars[4])))
                f.write('\t{0:05.3f}  {1:05.3f}  {2:05.3f}\n'.format(feval['p10'], feval['ap'], feval['rr']))
            else:
                print('Invalid file, skipping...')
    change_model(DEFAULT_SIM, es)  # Resets the search model

    runtimer.lap()
    print(runtimer.total_running_time_long)


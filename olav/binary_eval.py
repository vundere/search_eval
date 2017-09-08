rankings = {
    "q1": [1, 2, 4, 5, 3, 6, 9, 8, 10, 7],
    "q2": [1, 2, 4, 5, 3, 9, 8, 6, 10, 7],
    "q3": [1, 7, 4, 5, 3, 6, 9, 8, 10, 2]
}

gtruth = {
    "q1": [1, 3],
    "q2": [2, 4, 5, 6],
    "q3": [7]
}

q1_res = {'p_five': None, 'p_ten': None, 'MRR': None, 'avg_pres': None}
q2_res = {'p_five': None, 'p_ten': None, 'MRR': None, 'avg_pres': None}
q3_res = {'p_five': None, 'p_ten': None, 'MRR': None, 'avg_pres': None}

for qid, ranking in sorted(rankings.items()):
    gt = gtruth[qid]
    print("Evaluating", qid, "ranking:", ranking, "ground truth:", gt)

    # TODO
    # - Average precision
    num_rel = 0
    sum_pres = 0
    for i in range(len(ranking)):
        for t in gt:
            if ranking[i] == t:
                num_rel += 1
                sum_pres += num_rel / (i + 1)
    avg_pres = sum_pres / len(gt)
    if qid == 'q1':
        q1_res['avg_pres'] = avg_pres
    elif qid == 'q2':
        q2_res['avg_pres'] = avg_pres
    elif qid == 'q3':
        q3_res['avg_pres'] = avg_pres
    print('Average precision: {}'.format(avg_pres))

    # - Precision@5
    nr = 0
    for t in gt:
        for r in range(len(ranking)):
            if ranking[r] == t:
                nr += 1
            if r == 4:
                break
    p_five = nr / 5
    if qid == 'q1':
        q1_res['p_five'] = p_five
    elif qid == 'q2':
        q2_res['p_five'] = p_five
    elif qid == 'q3':
        q3_res['p_five'] = p_five
    print('Precision @ 5: {}'.format(p_five))

    # - Precision@10

    nr = 0
    for t in gt:
        for r in range(len(ranking)):
            if ranking[r] == t:
                nr += 1
            if r == 9:
                break
    p_ten = nr / 10
    if qid == 'q1':
        q1_res['p_ten'] = p_ten
    elif qid == 'q2':
        q2_res['p_ten'] = p_ten
    elif qid == 'q3':
        q3_res['p_ten'] = p_ten
    print('Precision @ 10: {}'.format(p_ten))

    # - Reciprocal rank

    rr = 0
    for q in range(len(ranking)):
        for t in gt:
            if ranking[q] == t and rr == 0:
                rr = 1 / (q + 1)

    if qid == 'q1':
        q1_res['MRR'] = rr
    elif qid == 'q2':
        q2_res['MRR'] = rr
    elif qid == 'q3':
        q3_res['MRR'] = rr
    print('RR: {} \n'.format(rr))

print('\n\n')
print("Average")
dicts = [q1_res, q2_res, q3_res]


def get_avg(num):
    sum_avg = 0
    for res in dicts:
        sum_avg += res[num]
        return sum_avg / len(dicts)
# - Mean Average precision
print('MAP: {}'.format(get_avg('avg_pres')))

# - Precision@5
print('Average P@5: {}'.format(get_avg('p_five')))

# - Precision@10
print('Average P@10: {}'.format(get_avg('p_ten')))

# - Mean Reciprocal rank
print('MRR: {}'.format(get_avg('MRR')))

for res in dicts:
    print('\n', res)

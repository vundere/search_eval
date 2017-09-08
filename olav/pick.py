RESULTS_FILE = 'data/results.txt'
WINNER_FILE = 'data/best.txt'
BEST_RESULT = {}
BEST_MAP = 0
BEST_MRR = 0
BEST_P10 = 0


def compare_all(a, b):
    if b:
        cont = a[list(a)[0]]
        best = b[list(b)[0]]
        for new, old in zip(cont, best):
            if old > new:
                return b
        return a
    else:
        return a


def compare_one(a, b):
    if float(a) > float(b):
        return a
    else:
        return b


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_data():
    global BEST_RESULT, BEST_MRR, BEST_MAP, BEST_P10
    with open(RESULTS_FILE, 'r') as rfile:
        lines = [x.strip('\n') for x in rfile.readlines()]
        params = []
        for line in lines:
            temp_res = {}
            if line.startswith('SCR'):
                elements = line.replace(',', '=').split('=')  # splits the line so it's easy to extract the parameters
                params = [float(element) for element in elements if isfloat(element)]
            else:
                elements = line.split('  ')
                temp_res[elements[0]] = [elements[i] for i in range(1, 4)]
                temp_res['params'] = ['{:05.3f}'.format(param) for param in params]
                BEST_RESULT = compare_all(temp_res, BEST_RESULT)
                BEST_P10 = compare_one(temp_res[elements[0]][0], BEST_P10)
                BEST_MAP = compare_one(temp_res[elements[0]][1], BEST_MAP)
                BEST_MRR = compare_one(temp_res[elements[0]][2], BEST_MRR)

    return lines

if __name__ == '__main__':
    read_data()
    print('Best:\nResult: {0}\nP@10: {1}\nMAP: {2}\nMRR: {3}'.format(BEST_RESULT, BEST_P10, BEST_MAP, BEST_MRR))
    with open(WINNER_FILE, 'w') as wfile:
        print('Writing results to file.')
        wfile.write('Result: {0}\nP@10: {1}\nMAP: {2}\nMRR: {3}'.format(BEST_RESULT, BEST_P10, BEST_MAP, BEST_MRR))

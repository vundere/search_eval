RESULTS_FILE = 'data/results.txt'
BEST_RESULT = {}


def compare(a, b):
    if b:
        cont = a[list(a)[0]]
        best = b[list(b)[0]]
        for new, old in zip(cont, best):
            if old > new:
                return b
        return a
    else:
        return a


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_data():
    global BEST_RESULT
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
                BEST_RESULT = compare(temp_res, BEST_RESULT)

    return lines

if __name__ == '__main__':
    read_data()
    print(BEST_RESULT)

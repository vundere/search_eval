import matplotlib.pyplot as plt
import csv
import numpy as np
import json


ATTRS_ALL = ["LOCATION", "W", "FINAL_MARGIN", "SHOT_NUMBER", "PERIOD", "GAME_CLOCK", "SHOT_CLOCK", "DRIBBLES",
             "TOUCH_TIME", "SHOT_DIST", "PTS_TYPE", "CLOSE_DEF_DIST", "Target"]
ATTRS_CATEGORICAL = ["LOCATION", "W", "PERIOD", "PTS_TYPE", "Target"]


def load_basketball_data(filename):
    records = []
    with open(filename, 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if len(row) == len(ATTRS_ALL):  # 13 attributes, including the target
                d = {}
                for i, attr in enumerate(ATTRS_ALL):
                    d[attr] = row[i] if attr in ATTRS_CATEGORICAL else float(row[i])
                records.append(d)
    return records


basketball_data = load_basketball_data("data/basketball.train.csv")

# TODO Choose 3 numeric attributes and complete the table with the summary statistics.
selected_attrs = ["DRIBBLES", "SHOT_NUMBER", "SHOT_DIST"]


def calc_attrs(attrs, data):
    for attr in attrs:
        print('Calculating attribute', attr)
        attr_values = []
        for entry in data:
            attr_values.append(entry[attr])
        mean = np.mean(attr_values)
        median = np.median(attr_values)
        rng = np.ptp(attr_values)
        variance = np.var(attr_values)
        print('Mean: {0}\nMedian: {1}\nRange: {2}\nVariance: {3}'.format(mean, median, rng, variance))

# TODO For one of those 3 attributes, plot in a single figure 2 boxplots, one per each of the 2 classes.

# TODO Choose 2 categorical attributes (different from the target) and plot each distribution in a histogram.

# TODO Binarize all the categorical attributes (different from the target) to obtain a dataset where each record is a list of zeros and non-zero values.

# TODO Choose 2 numeric attributes and compare them in a scatter plot, with different colors per each class.


if __name__ == '__main__':
    calc_attrs(selected_attrs, basketball_data)

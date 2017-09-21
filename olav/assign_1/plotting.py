import matplotlib.pyplot as plt
import numpy as np


hist = [-0.2, -0.1, 0.2, 0.4, 0.6]


def first_plot():
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    c, s = np.cos(x), np.sin(x)
    plt.plot(x, c)
    plt.plot(x, s)
    plt.show()


def plot_hist():
    plt.hist(hist, 5)
    plt.show()


if __name__ == '__main__':
    plot_hist()

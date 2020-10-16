import numpy as np
import matplotlib.pyplot as plt


def plot_rec(X, y, dict_struct):
    if not 'Wopt' in dict_struct.keys():
        return True
    b = dict_struct['Wopt'][2]
    m0 = dict_struct['Wopt'][0]
    m1 = dict_struct['Wopt'][1]
    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = -(m0 / m1) * x_vals - b / m1
    plt.plot(x_vals, y_vals, label=dict_struct['Wopt'])
    plot_rec(X, y, dict_struct['left'])
    plot_rec(X, y, dict_struct['right'])

    # plt.fill_between(x_vals, y1, y2, where=y2 > y1, facecolor='yellow', alpha=0.5)
    return True


def plot_2D(X: np.array, W: np.array, y: np.array, tree, title: str):
    b = W[2]
    m0 = W[0]
    m1 = W[1]
    # Plot
    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = -(m0 / m1) * x_vals - b / m1

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='y=1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='y=-1')

    plt.plot(x_vals, y_vals, label=tree.tree_struct['Wopt'])

    # recursive plot
    plot_rec(X, y, tree.tree_struct['left'])
    plot_rec(X, y, tree.tree_struct['right'])

    plt.title(title)
    plt.grid(True)
    y_max = np.max(X[:, 1])
    y_min = np.min(X[:, 1])
    plt.ylim(2 * y_min, 2 * y_max)
    plt.legend()
    plt.show()

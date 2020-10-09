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
    plt.plot(x_vals, y_vals, color='blue')
    plot_rec(X, y, dict_struct['left'])
    plot_rec(X, y, dict_struct['right'])
    return True


def plot_2D(X: np.array, W: np.array, y: np.array, tree):
    b = W[2]
    m0 = W[0]
    m1 = W[1]
    # Plot
    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = -(m0 / m1) * x_vals - b / m1

    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1])
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])
    plt.plot(x_vals, y_vals, color='blue')

    # recursive plot
    plot_rec(X, y, tree.tree_struct['left'])
    plot_rec(X, y, tree.tree_struct['right'])

    plt.title('Separating line with custom cost')
    plt.grid(True)
    plt.show()

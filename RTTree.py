import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp
from scipy.optimize import minimize
from scipy.stats import entropy
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def entropy1(labels):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts)


def func(w: np.array, x: np.array, y: np.array):
    return np.mean(1 / (1 + np.exp(2.5 * y * np.dot(x, w))))


def calculate_entropy(y: np.array, y_0: np.array, y_1: np.array):
    ent_left = entropy1(y_0)
    ent_right = entropy1(y_1)
    new_ent = (len(y_0) * ent_left + len(y_1) * ent_right) / (len(y_1) + len(y_0))
    init_ent = entropy1(y)
    return init_ent, ent_left, ent_right, new_ent


def split(x: np.array, y: np.array, W: np.array):  # y must be +/-1
    result = fmin_slsqp(func, W, bounds=[(-1000, 1000)] * len(W), args=(x, y), disp=False, full_output=True)
    Wopt, fW, its, imode, smode = result
    return Wopt


class RTTree():
    def __init__(self):
        # Structure of JSON:
        # {'init_ent': float, 'new_ent': float, 'Wopt': [...], 'left': {...}, 'right': {...}}
        self.tree_struct = dict()

    def fit_node(self, X: np.array, y: np.array, node_struct: dict):
        # optimizator
        # W_init = np.array([-0.1, -0.2, -0.1])  # think of optimal initialization
        W_init = np.random.rand(X.shape[1])
        Wopt = split(X, y, W_init)
        Wopt = np.array(Wopt)

        # predict
        y_pred = np.dot(X, Wopt) > 0

        # Split the data according to the predictions
        x_left = X[np.invert(y_pred)]  # np.dot(X, Wopt) <= 0
        y_left = y[np.invert(y_pred)]
        x_right = X[y_pred]  # np.dot(X, Wopt) > 0
        y_right = y[y_pred]
        # b = Wopt[2]
        # m0 = Wopt[0]
        # m1 = Wopt[1]

        # entropy
        if len(x_left) == 0 or len(x_right) == 0:
            node_struct['terminal'] = True
            if sum(y == -1) > sum(y == 1):  # просто кого больше, того и predict
                node_struct['predict'] = -1
            else:
                node_struct['predict'] = 1
            return False
        init_ent, ent_left, ent_right, new_ent = calculate_entropy(y, y_left, y_right)

        if new_ent < init_ent:
            # save to json
            node_struct['Wopt'] = Wopt
            node_struct['new_ent'] = new_ent
            node_struct['left'] = dict()
            node_struct['left']['init_ent'] = ent_left
            node_struct['left']['terminal'] = False
            node_struct['right'] = dict()
            node_struct['right']['init_ent'] = ent_right
            node_struct['right']['terminal'] = False

            # recursion left
            res_left = self.fit_node(x_left, y_left, node_struct['left'])

            # recursion right
            res_right = self.fit_node(x_right, y_right, node_struct['right'])
        else:
            return False
        return True

    def fit(self, X_train: np.array, y_train: np.array):
        # prepare data
        X_train = np.c_[X_train, np.ones(X_train.shape[0])]  # add column of ones for bias
        y_train[y_train == 0] = -1
        assert (np.unique(y_train) == np.array([-1, 1])).all()

        self.tree_struct['init_ent'] = entropy1(y_train)
        self.tree_struct['terminal'] = False
        self.fit_node(X_train, y_train, self.tree_struct)

    def predict_node(self, x, node_struct):
        if node_struct['terminal']:
            return node_struct['predict']
        if np.dot(x, node_struct['Wopt']) > 0:
            res = self.predict_node(x, node_struct['right'])
        else:
            res = self.predict_node(x, node_struct['left'])
        return res

    def predict(self, X_test: np.array):
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]
        pred = [self.predict_node(x, self.tree_struct) for x in X_test]
        return np.array(pred)


from sklearn.datasets import make_circles, make_moons, make_classification

np.random.seed(7)

X, y = make_circles(n_samples=200, noise=0.3, factor=0.2, )
transformation = [[0.6, -0.6], [-0.1, 0.6]]
X = np.dot(X, transformation)
y[y == 0] = -1

temp = RTTree()
temp.fit(X, y)
pred = temp.predict(np.array([[1, 2], [0, 0], [0.5, -0.2], [1, -1.5]]))

print(json.dumps(temp.tree_struct, sort_keys=False, indent=4, cls=NumpyEncoder))
# print(temp.tree_struct)
print(pred)

X_new = np.random.randn(100, 4)
y_new = np.random.choice([True, False], size=100).astype(int)
temp = RTTree()
temp.fit(X_new, y_new)
temp.predict(X_new)

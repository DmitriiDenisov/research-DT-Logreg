import json

from sklearn.datasets import make_circles, make_moons, make_classification
import numpy as np

from RTTree import RTTree, NumpyEncoder
from utils import plot_2D

"""
Example 0
np.random.seed(7)
X, y = make_circles(n_samples=200, noise=0.3, factor=0.2, )
transformation = [[0.6, -0.6], [-0.1, 0.6]]
X = np.dot(X, transformation)
y[y == 0] = -1
"""

"""
# Example 1a
np.random.seed(17)
n1 = 50
n2 = 50
center_x1 = np.array([-3, 0])
center_x2 = np.array([3, 0])

X1 = center_x1 + np.random.randn(n1, 2)
X2 = center_x2 + np.random.randn(n2, 2)
"""

"""
# Example 1b
np.random.seed(17)
n1 = 50
n2 = 50
center_x1 = np.array([-2, 0])
center_x2 = np.array([2, 0])

X1 = center_x1 + np.random.randn(n1, 2)
X2 = center_x2 + np.random.randn(n2, 2)
"""

"""
# Example 1c
np.random.seed(17)
n1 = 50
n2 = 50
center_x1 = np.array([-1.5, 0])
center_x2 = np.array([1.5, 0])

X1 = center_x1 + np.random.randn(n1, 2)
X2 = center_x2 + np.random.randn(n2, 2)
"""
# Example 1d
np.random.seed(17)
n1 = 50
n2 = 50
center_x1 = np.array([-1.3, 0])
center_x2 = np.array([1.3, 0])

X1 = center_x1 + np.random.randn(n1, 2)
X2 = center_x2 + np.random.randn(n2, 2)

X = np.concatenate((X1, X2), axis=0)
y = np.array([1] * n1 + [-1] * n2)

temp = RTTree()
temp.fit(X, y)
pred = temp.predict(np.array([[1, 2], [0, 0], [-1, -2], [-1, 2]]))

print(json.dumps(temp.tree_struct, sort_keys=False, indent=4, cls=NumpyEncoder))
# print(temp.tree_struct)
print(pred)

# X: np.array (n, 2), y: np.array (n,) (y consists from -1 and 1)
plot_2D(X, temp.tree_struct['Wopt'], y, temp, title='Example 1d')

# Example with (N, M)
"""
X_new = np.random.randn(100, 4)
y_new = np.random.choice([True, False], size=100).astype(int)
temp = RTTree()
temp.fit(X_new, y_new)
temp.predict(X_new)
"""

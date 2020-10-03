import json

from sklearn.datasets import make_circles, make_moons, make_classification
import numpy as np

from RTTree import RTTree, NumpyEncoder

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

import numpy as np
from regression import simple_linear_regression


def test_linear_regression():

    # Observations
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    b = simple_linear_regression.linear_regression(x, y)
    assert [b[0], b[1]] == [1.2363636363636363, 1.1696969696969697]




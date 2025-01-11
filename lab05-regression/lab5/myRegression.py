import numpy as np
from math import exp
from math import log2
from numpy.linalg import inv


class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0

    def fit(self, x, y):
        sx = sum(x)
        sy = sum(y)
        sx2 = sum(i * i for i in x)
        sxy = sum(i * j for (i, j) in zip(x, y))
        w1 = (len(x) * sxy - sx * sy) / (len(x) * sx2 - sx * sx)
        w0 = (sy - w1 * sx) / len(x)
        self.intercept_, self.coef_ = w0, w1

    def predict(self, x):
        if isinstance(x[0], list):
            return [self.intercept_ + self.coef_ * val[0] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]


class MyLinearBivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = [0.0, 0.0]

    def fit(self, x1, x2, y):
        sx1 = sum(x1)
        sx2 = sum(x2)
        sy = sum(y)
        sx1x1 = sum(i * i for i in x1)
        sx2x2 = sum(i * i for i in x2)
        sx1x2 = sum(i * j for (i, j) in zip(x1, x2))
        sx1y = sum(i * j for (i, j) in zip(x1, y))
        sx2y = sum(i * j for (i, j) in zip(x2, y))

        n = len(x1)

        determinant = n * sx1x1 * sx2x2 + 2 * sx1 * sx1x2 * sx2 - n * sx2 * sx2 * sx1x1 - sx1 * sx1 * sx2x2 - n * sx1x2 * sx1x2

        if determinant == 0:
            raise ValueError(
                "The determinant of the coefficient matrix is zero. The system is singular and cannot be solved.")

        w1 = (sy * sx1x1 * sx2x2 + sx1 * sx1x2 * sx2y + sx2 * sx1y * sx1x2 - sx2 * sx1x1 * sx2y - sx1 * sx1 * sx2y - sy * sx1x2 * sx1x2) / determinant
        w2 = (n * sx2 * sx1y * sx1x1 + sx1 * sx2x2 * sx2y + sx2 * sx1x2 * sx1y - sx2 * sx1 * sx1y - n * sx1x2 * sx2y - sx1 * sx2 * sx2x2 * sx1y) / determinant
        w0 = (sy - w1 * sx1 - w2 * sx2) / n

        self.intercept_ = w0
        self.coef_ = [w1, w2]

    def predict(self, x1, x2):
        if isinstance(x1[0], list):
            return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x1]
        else:
            return [self.intercept_ + self.coef_[0] * val1 + self.coef_[1] * val2 for val1, val2 in zip(x1, x2)]

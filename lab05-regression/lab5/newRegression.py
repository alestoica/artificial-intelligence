class MyLinearBivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = [0.0, 0.0]

    def fit(self, X, y):
        X = [[1] + point for point in X]
        y = [[val] for val in y]

        X_transpose = self.transpose(X)

        Xt_X = self.matrix_multiply(X_transpose, X)
        Xt_y = self.matrix_multiply(X_transpose, y)

        Xt_X_inv = self.matrix_inverse(Xt_X)
        coefficients = self.matrix_multiply(Xt_X_inv, Xt_y)

        self.intercept_ = coefficients[0][0]
        self.coef_ = [coefficients[i][0] for i in range(len(coefficients))]

    def predict(self, X):
        if isinstance(X[0], (int, float)):
            return self.intercept_ + sum(X[i] * self.coef_[i] for i in range(len(X)))
        else:
            return [self.intercept_ + sum(X[i] * self.coef_[i] for i in range(len(X))) for X in X]

    def transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def matrix_multiply(self, A, B):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in self.transpose(B)] for A_row in A]

    def matrix_inverse(self, matrix):
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if determinant == 0:
            raise ValueError("The matrix is singular, it cannot be inverted")
        return [[matrix[1][1] / determinant, -matrix[0][1] / determinant],
                [-matrix[1][0] / determinant, matrix[0][0] / determinant]]

    def initMatrix(self, rows, cols):
        matrix = []
        for _ in range(rows):
            matrix += [[0 for _ in range(cols)]]
        return matrix

    def multiplyMatrices(self, matrix1, matrix2):
        res = self.initMatrix(len(matrix1), len(matrix2[0]))
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    res[i][j] += matrix1[i][k] * matrix2[k][j]
        return res

    def getIdentityMatrix(self, rows, cols):
        identity = self.initMatrix(rows, cols)
        for i in range(rows):
            identity[i][i] = 1
        return identity

    def copyMatrix(self, matrix):
        mc = self.initMatrix(len(matrix), len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                mc[i][j] = matrix[i][j]
        return mc

    def invertMatrix(self, matrix):
        mc = self.copyMatrix(matrix)
        im = self.getIdentityMatrix(len(matrix), len(matrix[0]))
        imc = self.copyMatrix(im)
        indices = list(range(len(matrix)))
        for fd in range(len(matrix)):
            fdScaler = 1 / mc[fd][fd]
            for j in range(len(matrix)):
                mc[fd][j] *= fdScaler
                imc[fd][j] *= fdScaler
            for i in indices[0:fd] + indices[fd + 1:]:
                crScaler = mc[i][fd]
                for j in range(len(matrix)):
                    mc[i][j] = mc[i][j] - crScaler * mc[fd][j]
                    imc[i][j] = imc[i][j] - crScaler * imc[fd][j]
        return imc

    def my_regression(self, gdp, freedom, happiness):
        x = [[1, el1, el2] for el1, el2 in zip(gdp, freedom)]
        x_transpose = list(map(list, zip(*x)))
        first_mp = self.multiplyMatrices(x_transpose, x)
        inv_mx = self.invertMatrix(first_mp)
        second_mp = self.multiplyMatrices(inv_mx, x_transpose)
        happiness_matrix = [[el] for el in happiness]
        res = self.multiplyMatrices(second_mp, happiness_matrix)
        w0, w1, w2 = res[0][0], res[1][0], res[2][0]
        modelstr = 'f(x1, x2) = ' + str(w0) + ' + ' + str(w1) + ' * x1' + ' + ' + str(w2) + ' * x2'
        return modelstr

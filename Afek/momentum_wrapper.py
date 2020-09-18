import numpy as np


class MomentumWrapper:
    def __init__(self, data, beta, n):
        self.data = data
        self.beta = beta
        self.flag = {i: 0 for i in range(n)}

    def get(self, index, gradient) -> np.array:
        pass


class MomentumWrapper1D(MomentumWrapper):
    def __init__(self, n, beta):
        super().__init__(np.zeros(n), beta, n)

    def get(self, gradient, index) -> np.array:
        flag = self.flag.get(index)
        if flag:
            gradient = self.beta * self.data[index] + (1 - self.beta) * gradient
        self.data[index] = gradient
        return gradient


class MomentumWrapper2D(MomentumWrapper):
    def __init__(self, n, m, beta):
        super().__init__(np.zeros((n, m)), beta, n)

    def get(self, gradient, index) -> np.array:
        flag = self.flag.get(index)
        if flag:
            gradient = self.beta * self.data[index, :] + (1 - self.beta) * gradient
        self.data[index] = gradient
        return gradient

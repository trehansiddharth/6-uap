import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import classifier

class Solver:
    def __init__(self, probabilityClassifier, constraints):
        self.probabilityClassifier = probabilityClassifier
        self.constraints = constraints

    def solve(x, y0):
        f = lambda x: -np.sum(np.log(self.probabilityClassifier.classify(x)))
        J = lambda x: -np.sum((1 / sparse.diags(self.probabilityClassifier.classify(x))) * self.probabilityClassifier.derivative(x))
        result = opt.minimize(f, x0, jac=J, constraints=self.constraints)
        return result.x

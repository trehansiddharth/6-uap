import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import classifier

class Solver:
    def __init__(self, transform, constraints, maxiter=5):
        self.transform = transform
        self.constraints = constraints
        self.options = { "maxiter" : maxiter }

    def solve(self, constants, arguments, objective, goal='maximize'):
        argumentNames = arguments.keys()
        if goal is 'maximize':
            sgn = -1
        elif goal is 'minimize':
            sgn = 1
        else:
            raise ValueError('goal must be either \'maximize\' or \'minimize\'')
        f = lambda x: sgn * np.sum(self.transform.classify({ **constants, **self.transform.asDictionary(x, argumentNames) })[objective])
        J = lambda x: sgn * np.sum(self.transform.dictionaryAsMatrix(self.transform.derivative({ **constants, **self.transform.asDictionary(x, argumentNames) }), [objective], argumentNames), axis=1)
        x0 = self.transform.asVector(arguments, argumentNames)
        constraints = [(lambda x: self.constraints[k]({ **constants, **asDictionary(x, argumentNames) }, i)) for k in range(len(self.constraints)) for i in range(n)]
        result = opt.minimize(f, x0, jac=J, constraints=self.constraints, options=self.options)
        return self.transform.asDictionary(result.x, argumentNames)

class Merge(Solver):
    def __init__(self, solvers):
        self.solvers = solvers

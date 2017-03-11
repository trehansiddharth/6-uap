import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import gnn.classifier as classifier
import gnn.util as util

class Solver:
    def __init__(self, transform, maxiter=10, disp=True):
        self.transform = transform
        #self.constraints = constraints
        self.options = { "maxiter" : maxiter, "disp" : disp }

    def solve(self, constants, arguments, objective, goal='maximize'):
        argumentNames = list(arguments.keys())
        if goal is 'maximize':
            sgn = -1
        elif goal is 'minimize':
            sgn = 1
        else:
            raise ValueError('goal must be either \'maximize\' or \'minimize\'')
        def f(x):
            return sgn * np.sum(self.transform.classify({ **constants, **self.transform.asDictionary(x, argumentNames) })[objective])
        def J(x):
            return sgn * util.flatten(np.array(np.sum(self.transform.dictionaryAsMatrix(self.transform.derivative({ **constants, **self.transform.asDictionary(x, argumentNames) }, verbose=self.options["disp"]), [objective], argumentNames), axis=0)))
        x0 = self.transform.asVector(arguments, argumentNames)
        #constraints = [{ "type" : t, "fun" : lambda x: f(np.unravel_index(i, self.transform.shape), { **constants, **self.transform.asDictionary(x, argumentNames) }) } for (t, f) in self.constraints for i in range(self.transform.size)]
        result = opt.minimize(f, x0, jac=J, options=self.options, method='CG')
        return self.transform.asDictionary(result.x.reshape(self.transform.shape), argumentNames)

class Merge(Solver):
    def __init__(self, solvers):
        self.solvers = solvers

    def solve(self, constants, arguments, objective, goal='maximize'):
        pass

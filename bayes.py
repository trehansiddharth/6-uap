import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import classifier

def maximizeProbabilities(x0, p, constraints):
    f = lambda x: -np.sum(np.log(p.classify(x)))
    J = lambda x: -np.sum((1 / sparse.diags(p.classify(x))) * p.derivative(x))
    result = opt.minimize(f, x0, jac=J, constraints=constraints)
    return result.x

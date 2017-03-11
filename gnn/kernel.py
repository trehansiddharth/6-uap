import numpy as np
from functools import reduce
from math import *
import gnn.util as util
import gnn.classifier as classifier

def make(f, windowSize, center):
    kernel = np.zeros(windowSize)
    for ix in np.ndindex(windowSize):
        kernel[ix] = f(*[i - ci for (i, ci) in zip(ix, center)])
    return kernel

def extend(k, n, i):
    extendedKernel = np.zeros(k.shape + (n,))
    extendedKernel[...,i] = k
    return extendedKernel

def indices(kshape, ishape):
    ixs = list(np.ndindex(kshape))
    return list(np.ravel_multi_index(tuple(zip(*ixs)), ishape))

def gaussian(K, windowSize, center):
    A = np.linalg.inv(K)
    f = lambda *ix: exp(-0.5 * np.matrix([ix]) * A * np.matrix([ix]).T)
    kernel = make(f, windowSize, center)
    return kernel / np.sum(kernel)

def delta(windowSize, center):
    f = lambda *ix: 1.0 if np.linalg.norm(ix) == 0 else 0.0
    return make(f, windowSize, center)

def box(windowSize):
    n = util.product(windowSize)
    return 1.0 / n + np.zeros(windowSize)

def sobel(K, windowSize, pixelDistance, center):
    ci, cj = center
    dvs = [-float(pixelDistance) / 2, float(pixelDistance) / 2]
    gaussians = [gaussian(K, windowSize, (floor(ci + di), floor(cj + dj))) for di in dvs for dj in dvs]
    return gaussians[0] - gaussians[3], gaussians[2] - gaussians[1]

def toClassifier(k, offset):
    return classifier.Linear(util.flatten(k), offset)

import numpy as np
from functools import reduce
from math import *

def flatten(img):
    return np.reshape(img, img.size, order='C')

def inflate(x, shape):
    return np.reshape(x, shape)

def kernelIndices(kshape, ishape):
    ixs = list(np.ndindex(kshape))
    return list(np.ravel_multi_index(tuple(zip(*ixs)), ishape))

def makeKernel(f, windowSize, center):
    kernel = np.zeros(windowSize)
    for ix in np.ndindex(windowSize):
        kernel[ix] = f(*[i - ci for (i, ci) in zip(ix, center)])
    return kernel

def extendKernel(kernel, k, i):
    extendedKernel = np.zeros(kernel.shape + (k,))
    extendedKernel[...,i] = kernel
    return extendedKernel

def gaussianKernel(K, windowSize, center):
    A = np.linalg.inv(K)
    f = lambda *ix: exp(-0.5 * np.matrix([ix]) * A * np.matrix([ix]).T)
    kernel = makeKernel(f, windowSize, center)
    return kernel / np.sum(kernel)

def deltaKernel(windowSize, center):
    f = lambda *ix: 1.0 if np.linalg.norm(ix) == 0 else 0.0
    return makeKernel(f, windowSize, center)

def rotate(xs, k):
    return xs[-k:] + xs[:-k]

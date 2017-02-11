import numpy as np
import scipy.sparse as sparse
import util
from math import *

class Classifier:
    def __init(self):
        pass

    def classify(self, x):
        pass

    def derivative(self):
        pass

class Linear(Classifier):
    def __init__(self, w, z):
        self.w = np.matrix(w)
        self.z = np.matrix(z).T
        self.outputs = self.z.size
        self.inputs = int(self.w.size / self.z.size)

    def classify(self, x):
        return util.flatten(np.array(np.dot(self.w, x).T + self.z))

    def derivative(self, x):
        return self.w

class Parabolic(Classifier):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.inputs = len(b)
        self.outputs = 1

    def classify(self, x):
        return np.array([self.a * (np.linalg.norm(x) ** 2) + np.dot(self.b, x) + self.c])

class MergeHorizontal(Classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.inputs = self.classifiers[0].inputs
        self.outputs = sum([classifier.outputs for classifier in self.classifiers])

    def classify(self, x):
        return np.concatenate([classifier.classify(x) for classifier in self.classifiers])

    def derivative(self, x):
        return np.vstack([classifier.derivative(x) for classifier in self.classifiers])

class MergeVertical(Classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.inputs = classifiers[-1].inputs
        self.outputs = classifiers[0].outputs

    def classify(self, x):
        y = x
        for classifier in reversed(self.classifiers):
            y = classifier.classify(y)
        return y

    def derivative(self, x):
        return np.product([classifier.derivative(x) for classifier in self.classifiers])

class Softmax(Classifier):
    def __init__(self, n):
        self.inputs = n
        self.outputs = n

    def classify(self, x):
        exponents = np.exp(x)
        return exponents / sum(exponents)

    def derivative(self, x):
        n = self.outputs
        y = self.classify(x)
        A = np.matrix(np.zeros((n, n)))
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i, j] = y[i] * (1.0 - y[j])
                else:
                    A[i, j] = -y[i] * y[j]
        return A

class LogSoftmax(Classifier):
    def __init__(self, n):
        self.inputs = n
        self.outputs = n

    def classify(self, x):
        normalizer = sum(np.exp(x))
        return x - log(normalizer)

    def derivative(self, x):
        n = self.outputs
        y = np.exp(self.classify(x))
        A = np.matrix(np.zeros((n, n)))
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i, j] = 1.0 - y[j]
                else:
                    A[i, j] = -y[j]
        return A

class Concatenate(Classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.allInputs = [classifier.inputs for classifier in self.classifiers]
        self.inputs = sum(self.allInputs)
        self.allOutputs = [classifier.outputs for classifier in self.classifiers]
        self.outputs = sum(self.allOutputs)

    def classify(self, x):
        jxs = list(np.cumsum(self.allInputs))
        ixs = [0,] + jxs[:-1]
        return np.concatenate([classifier.classify(x[i:j]) for classifier, i, j in zip(self.classifiers, ixs, jxs)])

    def derivative(self, x):
        J = sparse.csr_matrix((self.outputs, self.inputs))
        jxs = list(np.cumsum(self.allInputs))
        ixs = [0,] + jxs[:-1]
        lxs = list(np.cumsum(self.allOutputs))
        kxs = [0,] + lxs[:-1]
        for classifier, i, j, k, l in zip(self.classifiers, ixs, jxs, kxs, lxs):
            J[j:k, ls] = sparse.csr_matrix(classifier.derivative(x[ls]))
        return J

class Convolutional(Classifier):
    def __init__(self, classifier, indices, skip, inputs):
        self.classifier = classifier
        self.indices = np.array(indices)
        self.skip = skip
        self.inputs = inputs

        self.numCells = int(self.inputs / skip)

        self.outputs = self.numCells * self.classifier.outputs

    def classify(self, x):
        y = np.zeros(self.outputs)
        n = self.inputs
        for i in range(self.numCells):
            j = i * self.classifier.outputs
            k = (i + 1) * self.classifier.outputs
            ls = list((i * self.skip + self.indices) % n)
            y[j:k] = self.classifier.classify(x[ls])
        return y

    def derivative(self, x):
        J = sparse.csr_matrix((self.outputs, self.inputs))
        n = self.inputs
        for i in range(self.numCells):
            j = i * self.classifier.outputs
            k = (i + 1) * self.classifier.outputs
            ls = list((i * self.skip + self.indices) % n)
            J[j:k, ls] = sparse.csr_matrix(self.classifier.derivative(x[ls]))
        return J

class Custom(Classifier):
    def __init__(self, inputs, outputs, fclassify, fderivative):
        self.inputs = inputs
        self.outputs = outputs
        self.classify = fclassify
        self.derivative = fderivative

def convolvedKernel(kernel, img, skip):
    classifier = Linear(util.flatten(kernel), 0)
    ixs = util.kernelIndices(kernel.shape, img.shape)
    return Convolutional(classifier, ixs, skip, img.size)

def constant(k, inputs):
    return Linear(np.zeros(inputs), k)

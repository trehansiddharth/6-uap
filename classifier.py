import numpy as np
import scipy.sparse as sparse

class Classifier:
    def __init(self):
        pass

    def classify(self, x):
        pass

    def derivative(self):
        pass

class Linear(Classifier):
    def __init__(self, w, z):
        self.w = w
        self.z = z
        self.inputs = len(w)
        self.outputs = 1

    def classify(self, x):
        return np.array([np.dot(self.w, x) + self.z])

    def derivative(self, x):
        return np.matrix([self.w])

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
    def __init__(self, classifier):
        self.classifier = classifier
        self.inputs = self.classifier.inputs
        self.outputs = self.classifier.outputs

    def classify(self, x):
        y = self.classifier.classify(x)
        exponents = np.exp(y)
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
        return A * self.classifier.derivative(x)

class Convolutional(Classifier):
    def __init__(self, classifier, indices, skip, inputs):
        self.classifier = classifier
        self.indices = indices
        self.skip = skip
        self.inputs = inputs

        self.numCells = int(self.inputs / skip)

        self.outputs = self.numCells * classifier.outputs

    def classify(self, x):
        y = np.zeros(self.outputs)
        n = self.inputs
        for i in range(0, self.numCells):
            j = i * self.classifier.outputs
            k = (i + 1) * self.classifier.outputs
            ls = list((i * self.skip + self.indices) % n)
            y[j:k] = self.classifier.classify(x[ls])
        return y

    def derivative(self, x):
        J = sparse.csr_matrix((self.outputs, self.inputs))
        n = self.inputs
        for i in range(0, self.numCells):
            j = i * self.classifier.outputs
            k = (i + 1) * self.classifier.outputs
            ls = list((i * self.skip + self.indices) % n)
            J[j:k, ls] = sparse.csr_matrix(self.classifier.derivative(x[ls]))
        return J

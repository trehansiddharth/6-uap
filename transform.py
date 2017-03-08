import classifier
import numpy as np
import util
import scipy.sparse as sparse

class Transform:
    def __init__(self, classifier, inputVariables, outputVariables, shape):
        self.classifier = classifier
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
        self.shape = shape
        self.size = util.product(shape)

    def classify(self, inputs, inputFormat='dictionary', outputFormat='dictionary'):
        if inputFormat is 'dictionary':
            x = self.asVector(inputs, self.inputVariables)
        elif inputFormat is 'vector':
            x = inputs
        else:
            raise ValueError('inputFormat must be either \'dictionary\' or \'vector\'')
        y = self.classifier.classify(x)
        if outputFormat is 'dictionary':
            return self.asDictionary(y, self.outputVariables)
        elif outputFormat is 'vector':
            return y
        else:
            return ValueError('outputFormat must be either \'dictionary\' or \'vector\'')

    def derivative(self, inputs, inputFormat='dictionary', outputFormat='dictionary'):
        if inputFormat is 'dictionary':
            x = self.asVector(inputs, self.inputVariables)
        elif inputFormat is 'vector':
            x = inputs
        else:
            raise ValueError('inputFormat must be either \'dictionary\' or \'vector\'')
        Dy = self.classifier.derivative(x)
        if outputFormat is 'dictionary':
            return self.matrixAsDictionary(Dy, self.outputVariables, self.inputVariables)
        elif outputFormat is 'matrix':
            return Dy
        else:
            return ValueError('outputFormat must be either \'dictionary\' or \'matrix\'')

    def asVector(self, variables, variableNames):
        return np.array([variables[variableName].reshape(self.size) for variableName in variableNames]).ravel(-1)

    def asDictionary(self, x, variableNames, inputFormat='vector'):
        n = len(variableNames)
        return {variableNames[i] : x[i::n].reshape(self.shape) for i in range(n)}

    def dictionaryAsMatrix(self, variables, rowVariableNames, columnVariableNames, outputSparse=True):
        m = len(rowVariableNames)
        n = len(columnVariableNames)
        if outputSparse:
            A = sparse.csr_matrix((self.size * m, self.size * n))
        else:
            A = np.zeros(self.size * m, self.size * n)
        for i, rowVariable in enumerate(rowVariableNames):
            for j, columnVariable in enumerate(columnVariableNames):
                A[i::m, j::n] = variables[(rowVariable, columnVariable)]
        return A

    def matrixAsDictionary(self, A, rowVariableNames, columnVariableNames):
        m = len(rowVariableNames)
        n = len(columnVariableNames)
        return {(rowVariableNames[i], columnVariableNames[j]) : A[i::m, j::n] for i in range(m) for j in range(n)}

class Merge(Transform):
    def __init__(self, transforms, inputVariables, outputVariables, shape):
        self.transforms = []
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
        self.shape = shape
        self.size = util.product(shape)

        allInputVariables = set(util.concat([transform.inputVariables for transform in transforms]))

        transformGraph = { tuple(inputState) : set() for inputState in util.powerset(allInputVariables, []) }
        for transformIndex, transform in enumerate(transforms):
            for inputState in util.powerset(allInputVariables, transform.inputVariables):
                outputState = tuple(sorted(list(set(inputState + transform.outputVariables))))
                transformGraph[tuple(inputState)].add((transformIndex, outputState))

        transformOrder = next(util.bfs(transformGraph, tuple(sorted(inputVariables)), lambda state: all([outputVariable in state for outputVariable in outputVariables])))
        self.transforms = [transforms[i] for i in transformOrder]

    def classify(self, inputs, inputFormat='dictionary', outputFormat='dictionary'):
        values = inputs.copy()
        for transform in self.transforms:
            values.update(transform.classify(values))
        return {variableName : values[variableName] for variableName in self.outputVariables}

    def derivative(self, inputs, inputFormat='dictionary', outputFormat='dictionary'):
        values = inputs.copy()
        derivatives = { }
        for transform in self.transforms:
            for (outputVariable, inputVariable), Dy in transform.derivative(values).items():
                if inputVariable in self.inputVariables:
                    derivatives[(outputVariable, inputVariable)] = Dy
                else:
                    for baseVariable in self.inputVariables:
                        if (inputVariable, baseVariable) in derivatives:
                            if (outputVariable, baseVariable) in derivatives:
                                derivatives[(outputVariable, baseVariable)] += Dy.multiply(derivatives[(inputVariable, baseVariable)])
                            else:
                                derivatives[(outputVariable, baseVariable)] = Dy.multiply(derivatives[(inputVariable, baseVariable)])
            print("Took derivative of {0} to {1}".format(transform.inputVariables, transform.outputVariables))
            values.update(transform.classify(values))
        return { (outputVariable, inputVariable) : derivatives[(outputVariable, inputVariable)] for inputVariable in self.inputVariables for outputVariable in self.outputVariables }

import numpy as np
import gnn.util as util
import gnn.classifier as classifier

def totalProbability():
    # Input is [P(X|A), P(X|NOT A), P(A)]
    # Result is P(X|A) * P(A) + P(X|A) * (1-P(A))
    A = np.matrix([[0, 0, 1], [0, 0, -1], [0, 0, 0]])
    b = np.array([0, 1, 0])
    return classifier.Parabolic(A, b, 0.0)

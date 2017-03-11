import numpy as np
import cv2
import gnn.classifier
import gnn.classifier.basic
import gnn.classifier.bayesian
import gnn.transform
import gnn.solver
import gnn.util
import sys

# To run: python examples.py {name}
# Where "name" is one of: "edgeDetection"

def edgeDetection(inputImagePath="data/sface.png", channels=[0], outputImagePath = "data/edgemap.png"):
    print("Running edge detection GNN...")
    # Load the image
    img = cv2.imread(inputImagePath)
    shape = img.shape[:-1]
    numChannels = len(channels)
    imgVariables = ["img" + str(i) for i in channels]
    imgDict = { "img" + str(i) : img[:,:,i] for i in channels }

    # Set up the difference of gaussians (dog) classifier
    dogUnitClassifier = gnn.classifier.basic.dog(1.0, 2.0, (5, 5, numChannels), 0.06, 0.0, logarithmic=False)
    dogClassifier = gnn.classifier.basic.convolved(dogUnitClassifier, (5, 5, numChannels), shape + (numChannels,))
    dogTransform = gnn.transform.Transform(dogClassifier, imgVariables, ["XgivenEdge"], shape + (numChannels,))

    # Set up null hypothesis: P(X | NOT edge) = 0.5
    nullUnitClassifier = gnn.classifier.Linear(np.array([0] * numChannels), 0.5)
    nullClassifier = gnn.classifier.basic.convolved(nullUnitClassifier, (1, 1, numChannels), shape + (numChannels,))
    nullTransform = gnn.transform.Transform(nullClassifier, imgVariables, ["XgivenNotEdge"], shape + (numChannels,))

    # Set up the bayesian edge probability classifier
    edgeProbUnitClassifier = gnn.classifier.bayesian.totalProbability()
    edgeProbClassifier = gnn.classifier.basic.convolved(edgeProbUnitClassifier, (1, 1, 3), shape + (3,))
    edgeProbTransform = gnn.transform.Transform(edgeProbClassifier, ["XgivenEdge", "XgivenNotEdge", "edgeProbMap"], ["likelihood"], shape)

    # Set up the sigmoid activation layer
    sigmoidUnitClassifier = gnn.classifier.Sigmoid(1)
    sigmoidClassifier = gnn.classifier.basic.convolved(sigmoidUnitClassifier, (1, 1, 1), shape + (1,))
    sigmoidTransform = gnn.transform.Transform(sigmoidClassifier, ["edgeMap"], ["edgeProbMap"], shape)

    # Set up the transform
    edgeTransform = gnn.transform.Merge([dogTransform, nullTransform, edgeProbTransform, sigmoidTransform], imgVariables + ["edgeMap"], ["likelihood"], shape)

    # Set up the solver
    edgeSolver = gnn.solver.Solver(edgeTransform) # [("ineq", lambda i, x: x["edgeMap"][i] >= 0), ("ineq", lambda i, x: x["edgeMap"][i] <= 1)]

    # Solve the maximum likelihood problem
    edgeMap = edgeSolver.solve(imgDict, { "edgeMap" : np.zeros(shape) }, "likelihood")["edgeMap"]

    # Save the resulting edge map
    edgeProbMap = sigmoidTransform.classify({ "edgeMap" : edgeMap })
    cv2.imwrite(outputImagePath, np.floor(255 * edgeProbMap["edgeProbMap"]))

if __name__ == "__main__":
    if sys.argv[1] == "edgeDetection":
        edgeDetection()
    else:
        raise ValueError("The method name provided is not one of the examples")

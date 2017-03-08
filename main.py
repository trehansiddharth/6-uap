import numpy as np
import cv2
import classifier
import classifier.basic
import classifier.bayesian
import transform
import solver
import util
import sys

# Options
inputImagePath = "data/sface.png"
channels = [0]
outputImagePath = "data/edgemap.png"

def main(argv):
    # Load the image
    img = cv2.imread(inputImagePath)
    shape = img.shape[:-1]
    numChannels = len(channels)
    imgVariables = ["img" + str(i) for i in channels]
    imgDict = { "img" + str(i) : img[:,:,i] for i in channels }

    # Set up the difference of gaussians (dog) classifier
    dogUnitClassifier = classifier.basic.dog(1.0, 2.0, (5, 5, numChannels), 0.06, 1.0, logarithmic=False)
    dogClassifier = classifier.basic.convolved(dogUnitClassifier, (5, 5, numChannels), shape + (numChannels,))
    dogTransform = transform.Transform(dogClassifier, imgVariables, ["XgivenEdge"], shape + (numChannels,))

    # Set up null hypothesis: P(X | NOT edge) = 0.5
    nullUnitClassifier = classifier.Linear(np.array([0] * numChannels), 0.5)
    nullClassifier = classifier.basic.convolved(nullUnitClassifier, (1, 1, numChannels), shape + (numChannels,))
    nullTransform = transform.Transform(nullClassifier, imgVariables, ["XgivenNotEdge"], shape + (numChannels,))

    # Set up the bayesian edge probability classifier
    edgeProbUnitClassifier = classifier.bayesian.totalProbability()
    edgeProbClassifier = classifier.basic.convolved(edgeProbUnitClassifier, (1, 1, 3), shape + (3,))
    edgeProbTransform = transform.Transform(edgeProbClassifier, ["XgivenEdge", "XgivenNotEdge", "edgeProbMap"], ["likelihood"], shape)

    # Set up the sigmoid activation layer
    sigmoidUnitClassifier = classifier.Sigmoid(1)
    sigmoidClassifier = classifier.basic.convolved(sigmoidUnitClassifier, (1, 1, 1), shape + (1,))
    sigmoidTransform = transform.Transform(sigmoidClassifier, ["edgeMap"], ["edgeProbMap"], shape)

    # Set up the transform
    edgeTransform = transform.Merge([dogTransform, nullTransform, edgeProbTransform, sigmoidTransform], imgVariables + ["edgeMap"], ["likelihood"], shape)

    # Set up the solver
    edgeSolver = solver.Solver(edgeTransform) # [("ineq", lambda i, x: x["edgeMap"][i] >= 0), ("ineq", lambda i, x: x["edgeMap"][i] <= 1)]

    # Solve the maximum likelihood problem
    edgeMap = edgeSolver.solve(imgDict, { "edgeMap" : np.zeros(shape) }, "likelihood")["edgeMap"]

    # Save the resulting edge map
    edgeProbMap = sigmoidTransform.classify({ "edgeMap" : edgeMap })
    cv2.imwrite(outputImagePath, np.floor(255 * edgeProbMap["edgeProbMap"]))

if __name__ == "__main__":
    main(sys.argv)

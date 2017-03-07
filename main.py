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
inputImagePath = "data/face.png"
outputImagePath = "data/edgemap.png"

def main(argv):
    # Load the image
    img = cv2.imread(inputImagePath)
    shape = img.shape[:-1]
    channels = img.shape[2]
    imgVariables = ["img" + str(i) for i in range(channels)]
    imgDict = { imgVariables[i] : img[:,:,i] for i in range(channels) }

    # Set up the difference of gaussians (dog) classifier
    dogUnitClassifier = classifier.basic.dog(1.0, 2.0, (5, 5, channels), 0.06, 1.0, logarithmic=False)
    dogClassifier = classifier.basic.convolved(dogUnitClassifier, (5, 5, channels), img.shape)
    dogTransform = transform.Transform(dogClassifier, imgVariables, ["XgivenEdge"], shape)

    # Set up null hypothesis: P(X | NOT edge) = 0.5
    nullUnitClassifier = classifier.Linear(np.array([0] * channels), 0.5)
    nullClassifier = classifier.basic.convolved(nullUnitClassifier, (1, 1, channels), img.shape)
    nullTransform = transform.Transform(nullClassifier, imgVariables, ["XgivenNotEdge"], shape)

    # Set up the bayesian edge probability classifier
    edgeProbUnitClassifier = classifier.bayesian.totalProbability()
    edgeProbClassifier = classifier.basic.convolved(edgeProbUnitClassifier, (1, 1, 3), shape + (3,))
    edgeProbTransform = transform.Transform(edgeProbClassifier, ["XgivenEdge", "XgivenNotEdge", "edgeMap"], ["likelihood"], shape)

    # Set up the transform
    edgeTransform = transform.Merge([dogTransform, nullTransform, edgeProbTransform], imgVariables + ["edgeMap"], ["likelihood"], shape)

    # Set up the solver
    edgeSolver = solver.Solver(edgeTransform, [lambda i, x: x["edgeMap"][i] >= 0, lambda i, x: x["edgeMap"][i] <= 1])

    # Solve the maximum likelihood problem
    edgeMap = edgeSolver.solve(imgDict, { "edgeMap" : np.ones(shape) / 2 }, "likelihood")["edgeMap"]

    # Save the resulting edge map
    cv2.imwrite(outputImagePath, np.floor(255 * edgeMap))

if __name__ == "__main__":
    main(sys.argv)

import numpy as np
from math import *
import util
import classifier
import kernel

def blur(sigma, windowSize, ishape):
    # Construct a classifier that blurs the signal in the window to filter out single-pixel noise
    Kblur = np.diag([sigma, sigma, sigma / 10.0])
    blurs = [util.flatten(kernel.gaussian(Kblur, windowSize, center)) for center in centers]
    return classifier.Linear(np.vstack(blurs), np.zeros(3))

def dog(sigma1, sigma2, windowSize, alpha, threshold, logarithmic=False):
    # Find the center of the window
    center = tuple([floor(i / 2) for i in windowSize[:2]])
    n = windowSize[2]

    # Construct the difference of Gaussians (dog) kernel
    gaussian1 = kernel.gaussian(np.diag([sigma1 ** 2, sigma1 ** 2]), windowSize[:2], center)
    gaussian2 = kernel.gaussian(np.diag([sigma2 ** 2, sigma2 ** 2]), windowSize[:2], center)
    dogKernel = gaussian1 - gaussian2
    dogs = [util.flatten(kernel.extend(gaussian1 - gaussian2, n, i)) for i in range(n)]
    dog = classifier.Linear(np.vstack(dogs), np.zeros(n))

    # Add up the components
    total = classifier.Linear(alpha * np.ones(n), -threshold)

    # Sigmoid function to get P(observed | edge)
    if logarithmic:
        sigmoid = classifier.LogSigmoid(1)
    else:
        sigmoid = classifier.Sigmoid(1)

    # Construct the classifier for a single window
    return classifier.MergeVertical([sigmoid, total, dog])

def sobel(sigma, windowSize, pixelDistance, alpha, threshold, logarithmic=False):
    # Find the centers of the four gaussians and construct them
    center = tuple(float(i) / 2 for i in windowSize[:2])
    n = windowSize[2]
    sobel1, sobel2 = kernel.sobel(np.diag([sigma ** 2, sigma ** 2]), windowSize[:2], pixelDistance, center)

    # Contruct the two sobel kernels
    sobel1s = [util.flatten(kernel.extend(sobel1, n, i)) for i in range(n)]
    sobel2s = [util.flatten(kernel.extend(sobel2, n, i)) for i in range(n)]
    sobel = classifier.Linear(np.vstack(sobel1s + sobel2s), np.zeros(n * 2))

    # Square it to get gradient magnitude and negative gradient magnitude
    squared = classifier.Parabolic(alpha, np.zeros(n * 2), -threshold)

    # Sigmoid function to get P(observed | edge)
    if logarithmic:
        sigmoid = classifier.LogSigmoid(1)
    else:
        sigmoid = classifier.Sigmoid(1)

    # Construct the classifier for a single window
    return classifier.MergeVertical([sigmoid, squared, sobel])

def variance(windowSize, alpha, threshold, logarithmic=False):
    # Number of channels
    n = windowSize[2]
    k = util.product(windowSize)

    # sumFirstClassifier: E[X] ** 2
    box2D = kernel.box(windowSize[:2])
    boxes = [util.flatten(kernel.extend(box2D, n, i)) for i in range(n)]
    mean = classifier.Linear(np.vstack(boxes), np.zeros(n))

    squared = classifier.Parabolic(1, np.zeros(n), 0)

    sumFirstClassifier = classifier.MergeVertical([squared, mean])

    # squareFirstClassifier: E[X ** 2]
    squareFirstClassifier = classifier.Parabolic(1 / k, np.zeros(k), 0)

    # subtract: E[X ** 2] - E[X] ** 2
    subtract = classifier.Linear(np.array([alpha, -alpha]), -threshold)

    # Sigmoid function to get P(observed | NOT single segment)
    if logarithmic:
        sigmoid = classifier.LogSigmoid(1)
    else:
        sigmoid = classifier.Sigmoid(1)

    return classifier.MergeVertical([sigmoid, subtract, classifier.MergeHorizontal([squareFirstClassifier, sumFirstClassifier])])

def convolved(unitClassifier, windowSize, ishape):
    ixs = kernel.indices(windowSize, ishape)
    return classifier.Convolutional(unitClassifier, ixs, windowSize[2])

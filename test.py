import numpy as np
import cv2
from math import *
import util
import classifier
import bayes

def threshold(img):
    img[np.where(img < 128)] = 0
    img[np.where(img >= 128)] = 255
    return img

def generateGradient(img, zero, gradient):
    gimg = np.zeros(img.shape)
    (zi, zj) = zero
    for ((i, j, k), x) in np.ndenumerate(img):
        beta = np.dot(np.array([i - zi, j - zj]), gradient)
        gimg[i, j, k] = max(min(floor(beta * x), 255), 0)
    return gimg

def generateSpotlight(img, center, amplitude, size):
    gimg = np.zeros(img.shape)
    (ci, cj) = center
    for ((i, j, k), x) in np.ndenumerate(img):
        norm2 = np.linalg.norm([i - ci, j - cj]) ** 2
        beta = amplitude * exp(-0.5 * norm2 / float(size ** 2))
        gimg[i, j, k] = max(min(floor(beta * x), 255), 0)
    return gimg

def blurClassifier(sigma, img):
    # Construct a classifier that blurs the signal in the window to filter out single-pixel noise
    Kblur = np.diag([sigma, sigma, sigma / 10.0])
    blurs = [util.flatten(util.gaussianKernel(Kblur, windowSize, center)) for center in centers]
    blurClassifier = classifier.Linear(np.vstack(blurs), np.zeros(3))

    # Return this classifier convolved to the image shape
    ixs = util.kernelIndices(windowSize, img.shape)
    return classifier.Convolutional(blurClassifier, ixs, 3, img.size)

def kernelClassifier(kernel2D, img, threshold, combination):
    # Construct the kernel classifier
    kernels = [util.flatten(util.extendKernel(kernel, 3, i)) for i in range(3)]
    kernelClassifier = classifier.Linear(np.vstack(kernels), np.zeros(3))

    # Construct a y=x^2 cost function
    squared = classifier.Parabolic(1.0, np.zeros(3), 0.0)

    # Softmax by comparing with the threshold value and choose the probability it is in a particular segment
    softmax = classifier.Softmax(2)
    compare = classifier.constant(threshold, 3)
    filterSoftmax = classifier.Linear(np.array(combination), 0)

    # Construct the classifier for a single window
    unitClassifier = classifier.MergeVertical([filterSoftmax, softmax, classifier.MergeHorizontal([squared, compare]), kernelClassifier])

    # Convolve that single-window classifier
    ixs = util.kernelIndices(windowSize, img.shape)
    return classifier.Convolutional(unitClassifier, ixs, 3, img.size)

def dogClassifier(sigma1, sigma2, windowSize, img, threshold):
    # Find the center of the window
    center = tuple([floor(i / 2) for i in windowSize[:2]])

    # Construct the difference of Gaussians (dog) kernel
    gaussian1 = util.gaussianKernel(npdiag([sigma1, sigma1]), windowSize, center)
    gaussian2 = util.gaussianKernel(npdiag([sigma2, sigma2]), windowSize, center)
    dog = gaussian1 - gaussian2

    # Construct a classifier from the dog kernel
    return kernelClassifier(smoothness, img, threshold, [0, 1])

def sobelClassifier(sigma, windowSize, img, threshold):
    # Find the center of the window
    center = tuple([floor(i / 2) for i in windowSize[:2]])

def constructGradientClassifier(img, sigmaBeta, sigmaZ, windowSize):
    dogClassifierBeta = smoothnessClassifier(sigmaBeta, img, windowSize)
    dogClassifierZ = smoothnessClassifier(sigmaZ, img, windowSize)
    x0 = None
    p = None
    constraints = None
    xOpt = bayes.maximizeProbabilities(x0, p, constraints)
    betaOpt = xOpt[::2]
    zOpt = xOpt[1::2]
    return (util.inflate(betaOpt, img.shape), util.inflate(zOpt, img.shape))

def smoothnessMap(sigmaDiff, sigmaBlur, img, windowSize, threshold):
    classifier = smoothnessClassifier(sigmaDiff, sigmaBlur, img, windowSize, threshold)
    x = util.flatten(img)
    y = classifier.classify(x)
    return np.floor(255 * np.nan_to_num(util.inflate(y, img.shape[:-1])))

import numpy as np
import cv2
from math import *
import util
import classifier
import bayes
import kernel

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

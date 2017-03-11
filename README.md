# generative-neural-networks

At the core of my senior thesis is the idea of "generative neural networks"-- neural networks that use maximum likelihood estimation in their classification process.

This library is for working with generative neural networks (specifically for image vision) and has the following modules:

* `gnn.classifier`: `Classifier` base class and several derived classes for building different types of classifiers and combining them into layered classifiers. Has the following submodules:
    * `gnn.classifier.basic`: Classifiers for common image processing kernels
    * `gnn.classifier.bayesian`: Classifiers for arrays of probabilities
* `gnn.kernel`: Functions that generate common image processing kernels with user-definable parameters
* `gnn.transform`: `Transform` class that simplifies input and output for convolutional classifiers
* `gnn.solver`: `Solver` class that runs optimization algorithms on transforms
* `examples`: Examples of how to use this library

## Concept

Neural networks typically have a straightforward classification algorithm: pass the input vector layer to layer and the output of the last layer is the result. It is therefore a simple map from the classification input vector `x` to the output vector `y`. Generative neural networks (GNNs) have an internal multi-layer classifier (the neural network) that takes as its inputs the classification input vector `x` and the output vector `y` and outputs the likelihood of `x` mapping to `y`-- it then iterates to find the maximum likelihood estimate of `y`, which it returns as its result. Every conventional neural network can also be represented as a generative neural network, but in some important cases generative neural networks are more powerful and cannot be represented in conventional form.

There are three core reasons why generative neural networks can be better suited than conventional neural networks to some problems:

* Because it is a probabilistic model, the neural network designer can incorporate domain-specific knowledge and priors
* Iteration allows for integration of top-down information into the classification process and gradual updating of belief variables
* Posing classification as an optimization problem allows for better composition of the components in an image processing pipeline

A simple example of this in action is the `edgeDetection` method in `examples.py`, which you can run by executing `python examples.py edgeDetection`. It should take several minutes to complete (runtime is unfortunately one of the biggest issues with this library at the moment). This GNN has the following structure:

![](data/edgediagram.png?raw=true)

The edge detection logic of this GNN is in the difference of Gaussians (DOG) kernel, the output of which is then passed into a sigmoid layer to convert into a probability. This probability (`XgivenEdge`) is interpreted as the probability of observed a particular pattern in the image given that it is an edge. For the null hypothesis, a simple classifier that always outputs 0.5 is used-- this probability (`XgivenNotEdge`) is the probability of observation of a particular pattern given that it is not an edge. This effectively sets the threshold value on the DOG kernel. Finally, the `sigmoidTransform` module converts the edge map of unbounded numbers into probabilities between 0 and 1, which indicates the probability that there is an edge at a particular pixel location.

The total probability classifier (in `edgeProbTransform`) determines the likelihood of observation of a given pattern in the image given the guess of the probabilities of edges at each pixel in the image (the `edgeProbMap`). This produces the `likelihood` variable, and the GNN solver will attempt to maximize this variable by finding the `edgeMap` array that maximizes the sum of the likelihoods in the likelihood array. Example input and output for this very simple edge detection GNN is shown below:

![](data/sface.png?raw=true)

![](data/edgemap.png?raw=true)

Clearly this is not much different from a traditional edge detection pipeline of convolving with a DOG kernel and thresholding. However, it demonstrates that a GNN can implement conventional CNN functions.

## API

### `gnn.classifier`

Definition of `Classifier` and some common base classes.

* `class Classifier`: Base class for all classifiers-- should not be used directly.
    * `def classify(x)`: Classifies a one-dimensional `np.array` object `x`, returning a one-dimensional `np.array` object.
    * `def derivative(x)`: Returns the derivative of the classifier around input `x`, returning an `np.matrix` object representing the Jacobian-- if the input is size `m` and the classification output is size `n`, the Jacobian is size `n × m`.

* `class Linear(Classifier)`: Linear classifier of the form `y = W x + z`.
    * `def __init__(self, w, z)`: `w` can be an array for singleton output or a matrix for multi-element output. `z` can be a number for singleton output or an array or matrix for multi-element output.

* `class Parabolic(Classifier)`: Second-order classifier of the form `y = x^T A^T A x + b^T x + c`.
    * `def __init__(self, a, b, c)`: `a` can be a number for singleton output or a matrix for multi-element output. `b` is a one-dimensional `np.array` object. `c` is a number for singleton output or a one-dimensional `np.array` for multi-element output.

* `class Sigmoid(Classifier)`: Sigmoid activation layer that converts an `n`-element input vector into an `n`-element output vector where each element is the sigmoid function evaluated on the corresponding element of the input array.
    * `def __init__(self, n)`: `n` is the number of elements desired for input and output arrays.

* `class LogSigmoid(Classifier)`: Same as `Sigmoid` except a natural log is taken of the output. Better for arrays of large positive or negative numbers.
    * `def __init__(self, n)`: `n` is the number of elements desired for input and output arrays.

* `class Softmax(Classifier)`: Runs the softmax function on an `n`-element input array and returns a single-valued `np.array`.
    * `def __init__(self, n)`: `n` is the number of elements desired for input and output arrays.

* `class LogSoftmax(Classifier)`: Same as `Softmax` except a natural log is taken of the output. Better for arrays of large positive or negative numbers.
    * `def __init__(self, n)`: `n` is the number of elements desired for input and output arrays.

* `class MergeHorizontal(Classifier)`: Merges a list of classifiers horizontally so that the same input vector gets passed to each classifier, and their output vectors are concatenated together.
    * `def __init__(self, classifiers)`: `classifiers` is the list of classifiers to merge horizontally.

* `class MergeVertical(Classifier)`: Merges a list of classifiers vertically so that the output of one classifier is the input of another.
    * `def __init__(self, classifiers)`: `classifiers` is the list of classifiers to merge vertically, from top to bottom.

* `class Concatenate(Classifier)`: Merges a list of classifiers horizontally like `MergeHorizontal` but so that the elements of the input vector are divided up between the classifiers and the outputs are concatenated together.
    * `def __init__(self, classifiers)`: `classifiers` is the list of classifiers to merge horizontally.

* `class Convolutional(Classifier)`: Turns a single classifier over a window into a convolutional classifier over an entire one-dimensional array. Written so that it works for flattened representations of multi-dimensional arrays as well.
    * `def __init__(self, classifier, indices, skip)`: `classifier` is the classifier over a particular window to convolve. `indices` is a list or `np.array` of index offsets (with respect to the index that is being operated on) that `classifier` should take as an input. `skip` is the number of indices to move `classifier` forward by in the input array to produce the next window to classify. Additional helpful functions for constructing convolutional classifiers are `gnn.kernel.indices` and `gnn.classifier.basic.convolved`.

* `class Custom(Classifier)`: Custom classifier if the classifier needed is not in the library.
    * `def __init__(self, inputs, outputs, fclassify, fderivative)`: `inputs` is the size of the input `np.array`, and `outputs` is the size of the output `np.array`. `fclassify` and `fderivative` are the functions that should be called when the `classify` and `derivative` funtions are called, respectively.

#### `gnn.classifier.basic`

* `def constant(k, inputs)`: Returns a classifier that takes in `inputs` inputs and always outputs an `np.array` of a singleton value, `k`.

The remainder of these methods construct classifiers for use over images (3-dimensional `np.array` objects)

* `def convolved(unitClassifier, windowSize, ishape)`: A useful method for constructing a convolutional classifier from a single classifier over a window in a 3-dimensional image. `unitClassifier` is the classifier to convolve, `windowSize` is a 2-dimensional tuple for the shape of the window, and `ishape` is a 3-dimensional tuple for the shape of the input image.

* `def blur(sigma, windowSize, ishape)`: Returns a convolutional classifier that takes in windows of shape `windowSize` (a 2-dimensional tuple) from an image of shape `ishape` (a 3-dimensional tuple) and blurs them using a Gaussian blur of radius `sigma`.

* `def dog(sigma1, sigma2, windowSize, alpha, threshold, logarithmic=False)`: Returns a convolutional classifier that takes in windows of shape `windowSize` (a 2-dimensional tuple) from an image of shape `ishape` (a 3-dimensional tuple), runs a difference of Gaussians (DOG) kernel over them of radii `sigma1` and `sigma2` (where `sigma1` > `sigma2`), and converts it to a probability of observation with a sigmoid function of scale `1/alpha` and a value of 0.5 corresponding to a DOG kernel output of `threshold / alpha`. If `logarithmic=True`, the natural log of this probability is returned instead of the actual probability.

* `def sobel(windowSize, alpha, threshold, logarithmic=False)`: Returns a convolutional classifier that takes in windows of shape `windowSize` (a 2-dimensional tuple) from an image of shape `ishape` (a 3-dimensional tuple), runs a Sobel operator over it, and converts it to a probability of observation witha  sigmoid function of scale `1/alpha` and a value of 0.5 corresponding to a Sobel operator output of `threshold / alpha`. If `logarithmic=True`, the natural log of this probability is returned instead of the actual probability.

* `def variance(windowSize, alpha, threshold, logarithmic=False)`: Returns a convolutional classifier that takes in windows of shape `windowSize` (a 2-dimensional tuple) from an image of shape `ishape` (a 3-dimensional tuple), takes the variance, and converts it to a probability of observation with a sigmoid function of scale `1/alpha` and a value of 0.5 corresponding to a variance of `threshold / alpha`. If `logarithmic=True`, the natural log of this probability is returned instead of the actual probability.

#### `gnn.classifier.bayesian`

* `def totalProbability()`: Returns a classifier that takes in an `np.array` of three elements (`P(X|A)`, `P(X|not A)`, and `P(A)`, respectively) and outputs an `np.array` of one element, `P(X) = P(X|A) × P(A) + P(X|not A) × (1 - P(A))`. Useful for calculating the likelihood from an array of probabilistic kernel outputs (`P(X|A)`), a null hypothesis (`P(X|not A)`), and a guess that we want to optimize (`P(A)`). The output of this is usually the objective of `gnn.solver.Solve.optimize`.

### `gnn.kernel`

* `def make(f, windowSize, center)`: Constructs an `n`-dimensional `np.array` of a kernel given the function that maps position to kernel weight (`f`), an `n`-dimensional tuple of the size of the array (`windowSize`), and an `n`-dimensional tuple for the position in the output array to take as zero for the input of `f` (`center`).

* `def extend(k, n, i)`: Constructs an `m + 1`-dimensional kernel from an `m`-dimensional kernel `k` by having `k` be in the `i`th position out of `n` total positions of the last dimension of the `m + 1`-dimensional kernel. For example, if you have a kernel `k` for 2-dimensional images and you want to extend it to 3-dimensional RGB images, you could make a kernel that runs on the R channel (the zeroeth channel out of three) by doing `gnn.kernel.extend(k, 3, 0)`. Note that `n = 3` refers to the number of channels (size of the last dimension), not the number of dimensions.

* `def indices(kshape, ishape)`: Returns a list of indices that a kernel of shape `kshape` would span in an image of shape `ishape` if both the kernel and image were flattened and if the kernel were placed so that the `(0, 0)` position of both the image and the kernel were aligned. Useful as an input to the `gnn.classifier.Convolutional` constructor when working with multi-dimensional arrays.

* `def gaussian(K, windowSize, center)`: Returns an `np.array` of shape `windowSize` and center `center` with the weights of a Gaussian kernel with covariance `K`, normalized by the sum of kernel values.

* `def delta(windowSize, center)`: Returns an `np.array` of shape `windowSize` and center `center` with the weights of a Delta function, normalized by the sum of kernel values.

* `def box(windowSize)`: Returns an `np.array` of shape `windowSize` and of constant weight at every position in the kernel, normalized by the sum of kernel values.

* `def sobel(K, windowSize, pixelDistance, center)`: Returns an `np.array` of shape `windowSize` and center `center` with the weights of two Gaussian kernels of opposite sign and of covariance matrix `K`, placed `pixelDistance` apart in the top-left and bottom-right of the window, normalized by the sum of kernel values.

* `def toClassifier(k, offset)`: Returns a `Classifier` that runs the `m × n` kernel `k` (an `np.array`) over a flattened window of size `m * n` and returns the kernel `k` run over this window, plus the number `offset`.

### `gnn.transform`

* `class Transform`: Represents transformations mapping two-dimensional array inputs to two-dimensional array outputs. Internally, it passes arrays through a network of `Classifier` objects, flattening, interspersing, inflating, and separating them to properly interface with the `Classifier`s. For convenience, inputs and outputs are dictionaries of variable names (as strings) and array values. `Transform` is the base class for different derived classes, but the `Transform` class by itself is a also useful wrapper for `Classifier`s. It supports the following functions:
    * `def __init__(self, classifier, inputVariables, outputVariables, shape)`: Creates a wrapper around the classifier `classifier`. Every `Transform` expects the input array to the `classifier` to be arranged in a specific way so that it can intersperse different input arrays and flatten arrays properly. If the input arrays to the transform are expected to be `arr0`, `arr1`, ..., `arrk`, then the input array to the classifier will be of the form `[arr0[0, 0], arr1[0, 0], ..., arrk[0, 0], arr0[0, 1], arr1[0, 1], ..., arrk[0, 1], ...]`, which is equivalent to `np.ravel(np.array([arr0, arr1, ..., arrk]))`. `inputVariables` (a list of strings) is the name of the input variables to `classifier`, in the order that they should appear in the input array. `outputVariables` (also a list of strings) is the name of the output variables from `classifier`, in the order that they appear in the output array. `shape` is a two-dimensional array representing the shape of each input array (every input and output array must be the same size, given by `shape`).
    * `def classify(inputs, inputFormat='dictionary', outputFormat='dictionary')`: Runs the inputs given by `inputs` through the transform and returns the outputs. If `inputFormat` is `"dictionary"`, then the input should be supplied as a dictionary, where the keys are variable names (one of the variable names in `inputVariables` from the constructor); if `inputFormat` is `"vector"`, `input` should be a flattened one-dimensional `np.array` that can be directly supplied to the `classifier`. If `outputFormat` is `"dictionary"`, the output is a dictionary where the keys are variable names; if `outputFormat` is `"vector"`, the output is a flattened one-dimensional `np.array` that is the direct output of the `classifier`.
    * `def derivative(self, inputs, inputFormat='dictionary', outputFormat='dictionary')`: Takes the derivative of the transform as a dictionary of Jacobian matrices. If the input variables to the transform are `["input0", "input1", ..., "inputm"]` and the output variables to the transform are `["output0", "output1", ..., "outputn"]`, then for every `0 ≤ i < m` and `0 ≤ j < n`, there is an entry in the dictionary that is the output of `derivative`, `("outputj", "inputi")` that maps to the Jacobian of the flattened `outputj` array with respect to the flattened `inputi` array. If `inputFormat` is `"dictionary"`, then the input should be supplied as a dictionary, where the keys are variable names (one of the variable names in `inputVariables` from the constructor); if `inputFormat` is `"vector"`, `input` should be a flattened one-dimensional `np.array` that can be directly supplied to the `classifier`'s `derivative` function. If `outputFormat` is `"dictionary"`, the output is a dictionary where the keys are as tuples as described above; if `outputFormat` is `"matrix"`, output is one large matrix with all Jacobians interspersed.

* `class Merge(Transform)`: Merge several transforms into one. Based on the input variable names and output variable names of each transform to merge, it computes the shortest path to get from the supplied input variables to the target output variables. Supports the following functions:
    * `def __init__(self, transforms, inputVariables, outputVariables, shape)`: `transforms` is the list of `Transform` objects to merge. `inputVariables` and `outputVariables` are the names of the input variables which should be supplied to the transform and the names of the output variables that the transform should solve for, respectively. `shape`, as with the constructor of the base class, is the two-dimensional tuple representing the shape of all the `np.array` variables. The constructor automatically computes the optimal path to map the input variables to the output variables.

### `gnn.solver`

* `class Solver`: Transforms map inputs of two-dimensional arrays to outputs of two-dimensional arrays-- `Solver` finds the optimimum values of the elements of one or more input arrays that maximizes or minimizes the sum of the elements of one of the output arrays. This is the class to use to compute the maximum likelihood estimate as part of GNN input vector classification. Supports the following functions:
    * `def __init__(self, transform, maxiter=10, disp=True)`: Constructs a `Solver` object that optimizes over `transform`. `maxiter` is the maximum number of iterations that the solver should take, and `disp` should be set to `True` if messages should be printed to `stdout` while the optimization algorithm is running.
    * `def solve(self, constants, arguments, objective, goal='maximize')`: Runs the optimization algorithm over `transform` treating the variable with the name of `objective` (a string) as the objective array (the sum of the elements of which is to be maximized or minimized), the variables with the names given by the dictionary of arrays `arguments` as the arguments to be optimized (supplied with their initial values), and the variables with the names given by the dictionary of arrays `constants` as the variables to be fed as inputs to `transform` as constants. For example, an edge-finding algorithm, the objective might be the likelihood array named `"likelihood"`, one of the constants might be the image array `"img"`, and the argument to be optimized might be probability map of edges `"edgeMap"`. `goal` determines whether the algorithm should try to find a maximum (`goal='maximum'`) or minimum (`goal='minimum'`). For maximum likelihood estimation, `goal` should be set to `'maximum'`.

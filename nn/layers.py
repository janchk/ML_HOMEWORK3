from .modules import Module
import numpy as np


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        # n_in += 1  # for bias

        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        # self.WnB = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)
        # self.b = None

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = np.add(input @ self.W.T, self.b)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput @ self.W
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradb = np.sum(gradOutput, axis=0)
        assert self.gradW.shape == (gradOutput.T @ self.gradInput).shape
        self.gradW = gradOutput.T @ input
        self.gradb = np.sum(gradOutput, axis=0)
        pass

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = "[Linear %d -> %d]" % (s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        exp = np.exp(np.subtract(input, input.max(axis=1, keepdims=True)))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = softmax
        return self.output

    def updateGradInput(self, input, gradOutput):
        exp = np.exp(np.subtract(input, input.max(axis=1, keepdims=True)))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        self.gradInput = gradOutput * softmax * (1. - softmax)
        return self.gradInput

    def __repr__(self):
        return "[SoftMax]"


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here. ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        return self.gradInput

    def __repr__(self):
        return "[LogSoftMax]"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # use self.EPS please
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        return self.gradInput

    def __repr__(self):
        return "[BatchNormalization]"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if 0 < self.p < 1:
            self.mask = np.divide(np.random.binomial(1, 1-self.p, input.shape), self.p)

        elif self.p == 1:
            self.mask = np.zeros(input.shape)

        elif self.p == 0:
            self.mask = np.ones(input.shape)

        self.output = input * self.mask
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.mask
        return self.gradInput

    def __repr__(self):
        return "[Dropout]"


class GaussianDropout(Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()

        self.p = p
        self.noise = None

    def updateOutput(self, input):
        if 0 < self.p < 1:
            stddev = np.sqrt(self.p / (1.0 - self.p))
            self.noise = np.random.normal(1.0, stddev, input.shape)

        elif self.p == 1:
            self.noise = np.zeros(input.shape)

        elif self.p == 0:
            self.noise = np.ones(input.shape)

        self.output = input * self.noise
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.noise
        return self.gradInput

    def __repr__(self):
        return "[Gaussian Dropout]"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "[ReLU]"


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def updateOutput(self, input):
        sigmoid = 1 / (1 + np.exp(-input))
        self.output = sigmoid
        return self.output

    def updateGradInput(self, input, gradOutput):
        sigmoid = 1 / (1 + np.exp(-input))
        self.gradInput = gradOutput * sigmoid * (1 - sigmoid)
        return self.gradInput

    def __repr__(self):
        return "[Sigmoid]"
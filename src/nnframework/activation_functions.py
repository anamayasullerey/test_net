"""
Activation functions and their derivatives
"""
import numpy as np

def sigmoidp(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidn(x):
    return np.exp(x) / (1.0 + np.exp(x))

def sigmoid(x):
    xp = np.multiply(x, (x>0))
    xn = np.multiply(x, (x<0))
    return sigmoidp(xp) + sigmoidn(xn) - 0.5

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    return np.exp(x-np.max(x, axis=0)) / np.sum(np.exp(x-np.max(x, axis=0)), axis=0)

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - tanh(x) * tanh(x)

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return (x > 0).astype(float)

"""
softmax output layer
"""

import numpy as np
from . import activation_functions as af
from . import layer

class SoftmaxLayer(layer.Layer):

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.act_func = getattr(af, "softmax")
        self.act_func_prime = getattr(af, "softmax_prime")

    def backward_grad(self):
        #calculate dactivation for previous layer
        self.prev_layer.dactivations = np.dot(self.act_func_prime(self.prev_layer.activations), self.dactivations)

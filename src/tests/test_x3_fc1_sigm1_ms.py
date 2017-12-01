"""
This test has a single fc neuron with weight initialized to 10
"""

import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_types as lt
import nnframework.loss_layer as ll
import nnframework.weight_update_params as wup

net = nn.NeuralNetwork("test_net", 1)

layer = lt.ldict["fc"](1)
net.add_layer(layer)
layer = lt.ldict["sigmoid"](1)
net.add_layer(layer)

layer = ll.LossLayer("sigmoid_cross_entropy_loss")
net.add_layer(layer)

np.random.seed(1)

params = wup.GradientDescentParams(0)
net.set_weight_update_function(params)
net.initialize_parameters()
net.layers[1].weights[0][0] = 1

x = np.array([[0.5]])
y = np.array([[0.5]])
net.check_gradient(x, y)

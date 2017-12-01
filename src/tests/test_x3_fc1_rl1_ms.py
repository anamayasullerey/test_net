"""
This test has a single fc neuron with weight initialized to 10
"""

import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_types as lt
import nnframework.loss_layer as ll
import nnframework.weight_update_params as wup

net = nn.NeuralNetwork("test_net", 3)

layer = lt.ldict["fc"](1)
net.add_layer(layer)
layer = lt.ldict["relu"](1)
net.add_layer(layer)

layer = ll.LossLayer("linear_mean_squared_loss")
net.add_layer(layer)

np.random.seed(1)

params = wup.GradientDescentParams(0)
net.set_weight_update_function(params)
net.initialize_parameters()

x = np.array([[2], [3], [4]])
y = np.array([[10]])
net.check_gradient(x, y)

"""
This test has a single fc neuron with weight initialized to 10
"""

import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup
import os

net = nn.NeuralNetwork("test_net", 1)

layer = ld.ldict["fc"](1)
net.add_layer(layer)
layer = ld.ldict["sigmoid"](1)
net.add_layer(layer)

layer = ld.ldict["loss"]("sigmoid_cross_entropy_loss")
net.add_layer(layer)

np.random.seed(1)

params = wup.GradientDescentParams(0)
net.set_weight_update_function(params)
net.initialize_parameters()
net.layers[1].weights[0][0] = 1

x = np.array([[0.5]])
y = np.array([[0.5]])
if (net.check_gradient(x, y)):
    print ("Test {0} passed".format(os.path.basename(__file__)))
else:
    print ("Test {0} failed".format(os.path.basename(__file__)))        

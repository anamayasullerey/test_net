"""
This test has a single fc neuron with weight initialized to 10
"""

import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup
import os

net = nn.NeuralNetwork("test_net", 4)

layer = ld.hdict["fc"](4)
net.add_layer(layer)
layer = ld.odict["softmax"](4)
net.add_layer(layer)

np.random.seed(1)

params = wup.GradientDescentParams(0)
net.set_weight_update_function(params)
net.initialize_parameters()

x = np.array([[1], [2], [3], [4]])
y = np.array([[1], [0], [0], [0]])
if (net.check_gradient(x, y)):
    print ("Test {0} passed".format(os.path.basename(__file__)))
else:
    print ("Test {0} failed".format(os.path.basename(__file__)))        

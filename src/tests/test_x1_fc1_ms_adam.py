import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup
import os

net = nn.NeuralNetwork("test_net", 1)

layer = ld.hdict["fc"](1)
net.add_layer(layer)

layer = ld.odict["loss"]("linear_mean_squared_loss")
net.add_layer(layer)

np.random.seed(1)

params = wup.AdamParams()
net.set_weight_update_function(params)
net.initialize_parameters()
net.layers[1].weights[0,0] = 10

x = np.array([[2]])
y = np.array([[10]])
net.train(x,y)

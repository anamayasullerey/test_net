import numpy as np
import neural_network.neural_network as nn
import neural_network.weight_update_params as wup

net = nn.NeuralNetwork("test_net", 3)
net.add_layer(2, "sigmoid")
net.add_layer(1, "sigmoid")

np.random.seed(1)
params = wup.GradientDescentParams()
net.set_weight_update_function(params)
net.initialize_parameters()
x = np.array([[1.], [2.], [3.]])
print(x)
y = net.forward_prop(x)
print(y)
print("--------")
print(net.layers[0].activations)
print(net.layers[0].z)
print("--------")
print(net.layers[1].activations)
print(net.layers[1].z)

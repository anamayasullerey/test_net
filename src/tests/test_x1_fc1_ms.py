import numpy as np
import neural_network.neural_network as nn
import neural_network.layer_types as lt
import neural_network.loss_layer as ll
import neural_network.weight_update_params as wup

net = nn.NeuralNetwork("test_net", 1)

layer = lt.ldict["fc"](1)
net.add_layer(layer)

layer = ll.LossLayer("linear_mean_squared_loss")
net.add_layer(layer)

np.random.seed(1)

params = wup.GradientDescentParams(0)
net.set_weight_update_function(params)
net.initialize_parameters()
net.layers[1].weights[0,0] = 10

x = np.array([[2]])
y = np.array([[10]])
net.check_gradient(x, y)

if ((net.layers[1].dweights[0][0]) != 20):
  print("Error: Expected gradient = 20.0")
  print("     : Backprop gradient = " + str(net.layers[1].dweights[0][0]))
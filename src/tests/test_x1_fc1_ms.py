import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_types as lt
import nnframework.loss_layer as ll
import nnframework.weight_update_params as wup
import os

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
if (net.check_gradient(x, y) and ((net.layers[1].dweights[0][0]) == 20)):
    print ("Test {0} passed".format(os.path.basename(__file__)))
else:
    print ("Test {0} failed".format(os.path.basename(__file__)))        

if ((net.layers[1].dweights[0][0]) != 20):
    print("Error: Expected gradient = 20.0")
    print("     : Backprop gradient = " + str(net.layers[1].dweights[0][0]))

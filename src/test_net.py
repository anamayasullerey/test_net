import numpy as np
import neural_network.neural_network as nn
import neural_network.weight_update_params as wup
import utils.load_mnist as load_mnist

training_data_zip, validation_data, test_data = load_mnist.load_mnist()
training_data = list(training_data_zip)
net = nn.NeuralNetwork("test_net", 784)
net.add_layer(40, "relu")
net.add_layer(10, "softmax")
net.set_loss_function("softmax_cross_entropy_loss", 0)

np.random.seed(1)
params = wup.GradientDescentParams()
net.set_weight_update_function(params)
net.initialize_parameters()

#mini_batch_size = 16
#random.shuffle(training_data)
#mini_batch = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
#print(training_data[1])
#for x, y in mini_batch:
x, y = training_data[1]
net.print_state()
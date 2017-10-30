import numpy as np
import neural_network.neural_network as nn
import neural_network.weight_update_params as wup
import utils.load_mnist as load_mnist
import random

training_data_zip, validation_data_zip, test_data_zip = load_mnist.load_mnist()
training_data = list(training_data_zip)
validation_data = list(validation_data_zip)
test_data = list(test_data_zip)
net = nn.NeuralNetwork("test_net", 784)
net.add_layer(40, "relu")
net.add_layer(10, "sigmoid")
net.set_loss_function("sigmoid_cross_entropy_loss", 0)

np.random.seed(1)
params = wup.GradientDescentParams()
net.set_weight_update_function(params)
net.initialize_parameters()

mini_batch_size = 2
# random.shuffle(training_data)
for epoch in range(10):
  random.shuffle(training_data)
  mini_batch = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
  print(mini_batch)
  for x, y in mini_batch:
    net.train(x, y)            
  
  accuracy = validate(validation_data, net) / 100.0
  print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))

def predict(net, x):
  net.forward_prop(x)  
  return net.argmax(net.layer[-1].activations)

def validate(self, validation_data, net):
  validation_results = [(predict(net, x) == y) for x, y in validation_data]
  return sum(result for result in validation_results)

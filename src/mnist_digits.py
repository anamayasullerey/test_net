import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup
import utils.load_mnist as load_mnist
import random

# Function for checking accuracy of a test/validation set
def calc_accuracy(data, net):
    test_results = [(net.predict_classify(x) == y) for x, y in data]
    total_correct = sum(correct for correct in test_results)
    accuracy = total_correct * 100 / len(test_results)
    return accuracy

# Function for packing data from a list to numpy array
def pack_np_array(data):
    x = np.array(data[0][0])
    y = np.array(data[0][1])
    for i in range(1, len(data)):
        x = np.append(x, data[i][0], 1)
        y = np.append(y, data[i][1], 1)
    return x, y

# Load the training, validation and test data
# Training data list has y as vectors
# Validation and test data have y as final classified values    
training_data_zip, validation_data_zip, test_data_zip = load_mnist.load_mnist()
training_data = list(training_data_zip)
validation_data = list(validation_data_zip)
test_data = list(test_data_zip)

# Creating ther Neural Network
# Step 1: Create Network - specify input layer neurons (28x28=784)
net = nn.NeuralNetwork("test_net", 784)

# Step 2: Add hidden layers in sequence

# Fully connected layer
layer = ld.ldict["fc"](800)
net.add_layer(layer)

# Relu activation layer
layer = ld.ldict["relu"](800)
net.add_layer(layer)

layer = ld.ldict["fc"](80)
net.add_layer(layer)

layer = ld.ldict["relu"](80)
net.add_layer(layer)

layer = ld.ldict["fc"](10)
net.add_layer(layer)

layer = ld.ldict["sigmoid"](10)
net.add_layer(layer)

# Add loss layer
layer = ld.ldict["loss"]("sigmoid_cross_entropy_loss")
net.add_layer(layer)

# Specify l2 loss
net.set_l2_loss_coeff(.001)

#  Neural Network definition done

# Define weight update method
params = wup.GradientDescentParams(.3)
# params = wup.MomentumParams(.3)
# params = wup.AdamParams()
net.set_weight_update_function(params)

# For repeatability during testing
np.random.seed(1)

# Initialize the network
net.initialize_parameters()

# Set training related parameters
mini_batch_size = 32
epochs = 10

# Train the network
for epoch in range(1, epochs+1):
    print("Epoch " + str(epoch))
    random.shuffle(training_data)
    mini_batches = [training_data[k:k + mini_batch_size] for k in
                    range(0, len(training_data), mini_batch_size)]

  
    for count, mini_batch in enumerate(mini_batches, start=1):
        x, y = pack_np_array(mini_batch)
        net.train(x, y)

    accuracy = calc_accuracy(validation_data, net)
    print("Epoch {0} validation data accuracy = {1} %.".format(epoch, accuracy))
    print()

accuracy = calc_accuracy(test_data, net)
print("Test data accuracy = {0} %.".format(accuracy))
print()

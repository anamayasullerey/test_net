import numpy as np
import neural_network.neural_network as nn
import neural_network.layer_types as lt
import neural_network.loss_layer as ll
import neural_network.weight_update_params as wup
import utils.load_mnist as load_mnist
import random

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def validate(validation_data, net):
  validation_results = [(net.predict_classify(x) == y) for x, y in validation_data]
  return sum(result for result in validation_results)
    
training_data_zip, validation_data_zip, test_data_zip = load_mnist.load_mnist()
#print(type(training_data_zip))
training_data = list(training_data_zip)
validation_data = list(validation_data_zip)
test_data = list(test_data_zip)
net = nn.NeuralNetwork("test_net", 784)
layer = lt.ldict["fc"](800)
net.add_layer(layer)
layer = lt.ldict["relu"](800)
net.add_layer(layer)
layer = lt.ldict["fc"](80)
net.add_layer(layer)
layer = lt.ldict["relu"](80)
net.add_layer(layer)
layer = lt.ldict["fc"](10)
net.add_layer(layer)
layer = lt.ldict["sigmoid"](10)
net.add_layer(layer)
layer = ll.LossLayer("sigmoid_cross_entropy_loss")
net.add_layer(layer)
net.set_l2_loss_coeff(.00)

np.random.seed(1)
params = wup.GradientDescentParams(.3)
#params = wup.AdamParams()
#params = wup.MomentumParams()
net.set_weight_update_function(params)
net.initialize_parameters()

# random.shuffle(training_data)
mini_batch_size = 32
epochs = 1
for epoch in range(1, epochs+1):
  print(" Epoch " + str(epoch))
  #random.shuffle(training_data)
  mini_batches = [training_data[k:k + mini_batch_size] for k in
                  range(0, len(training_data), mini_batch_size)]

  for count, mini_batch in enumerate(mini_batches, start=1):
    x = np.array(mini_batch[0][0])
    y = np.array(mini_batch[0][1])
    for i in range(1, len(mini_batch)):
      x = np.append(x, mini_batch[i][0], 1)
      y = np.append(y, mini_batch[i][1], 1)

    net.train(x, y)
  
    if (count%10 == 0):
      correct = validate(validation_data, net)
      accuracy = correct / 100.0
      print("----------------------")
      print()
      print("Epoch {0}, minibatch {1}, accuracy {2} %.".format(epoch, count, accuracy))
  
  #show(x.reshape((28,28)))           
  #if (count%100 == 0) and (epoch != 0):
  #  params.learning_rate = params.learning_rate/2
  #  print("Changing learning rate to " + str(params.learning_rate))
  #  net.set_weight_update_function(params)

#
print("Epoch {0}, accuracy {1} %.".format(epoch, accuracy))
for x, y in validation_data:
  if (net.predict_classify(x) > 9):
    show(x.reshape((28,28)))           
     

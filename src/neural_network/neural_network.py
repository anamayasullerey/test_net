from . import input_layer
from . import layer
from . import loss_functions as lf
from . import weight_update_functions as wuf
import numpy as np

class NeuralNetwork(object):
      
  def __init__(self, name, num_inputs):
    self.name = name
    self.input_layer = input_layer.InputLayer(num_inputs) 
    self.layers = []
    
  def set_loss_function(self, loss_func_name):
    self.loss_func = lf.LossFunctions.get_function(loss_func_name)
    self.loss_func_prime = lf.LossFunctions.get_function(loss_func_name + "_prime")

  def set_weight_update_function(self, params):
    wu_func = wuf.WeightUpdateFunctions.get_function(params.weight_update_func_name)
    self.weight_update_func= lambda layer: wu_func(layer, params)
    self.weight_update_init_func = wuf.WeightUpdateFunctions.get_function(params.weight_update_func_name + "_init")

  def add_layer(self, num_neurons, activation_func_name):
    layer_num = len(self.layers) 
    if (layer_num == 0):
      new_layer = layer.Layer(layer_num, num_neurons, activation_func_name, self.input_layer)
    else:
      new_layer = layer.Layer(layer_num,num_neurons, activation_func_name, self.layers[-1])
    self.layers.append(new_layer)

  def initialize_parameters(self):
    for i in range(len(self.layers)):
      self.layers[i].initialize_parameters()
      self.weight_update_init_func(self.layers[i])
      
  def forward_prop(self, x):
    self.input_layer.forward_prop(x)
    
  def backward_prop(self, da, weight_update_function):
    self.layers[-1].back_prop(self.weight_update_function)  
    
  def train(self, x, y):
      self.forward_prop(x)
      self.dactivation = self.loss_function_prime(y, self.layers[-1].activations)
      self.backward_prop(self.dactivation, self.backprop_func)

  def predict(self, x):
    self.forward_prop(x)
    return self.layers[-1].activations

  def predict_classify(self, x):
    return np.argmax(self.predict(x))

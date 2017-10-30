"""
layer (Hidden/Output) of a neural network
"""

import numpy as np
from . import activation_functions as af

class Layer(object):
  
  def __init__(self, layer_num, num_neurons, activation_func_name, prev_layer):
    
    self.layer_num = layer_num
    self.num_neurons = num_neurons

    # Set the activation function
    self.activation_func = af.ActivationFunctions.get_function(activation_func_name)

    # Set the activation function derivative
    self.activation_func_prime = af.ActivationFunctions.get_function(activation_func_name + "_prime")
      
    # initialize the weight and the bias matrix to 0s
    # this step is not must have
    self.weights = np.zeros((num_neurons, prev_layer.num_neurons))
    self.bias = np.zeros((num_neurons, 1))

    # stitch the layers in the backwards direction
    self.prev_layer = prev_layer

    # stitch the layers in the forward direction
    self.next_layer = None
    prev_layer.next_layer = self 
  
  # forward propagation function
  def forward_prop(self, input_activations):
    self.z = np.dot(self.weights, input_activations) + self.bias
    self.activations = self.activation_func(self.z)
    
    if self.next_layer is not None:
      self.next_layer.forward_prop(self.activations)

  # backwards propagation (backprop) function
  def backward_prop(self, dactivations, l2_regu_coeff):
    batch_size = self.activations.shape[1]
    self.dz = dactivations * self.activation_func_prime(self.z)
    self.dweights = np.dot(self.dz, np.transpose(self.prev_layer.activations))/batch_size
    self.dbias = np.sum(self.dz, axis=1, keepdims=True)/batch_size
    
    # update the derivatives for regularization
    if (l2_regu_coeff != 0):
      self.dweights = self.dweights + ((l2_regu_coeff * self.weights)/batch_size)
          
    if (self.layer_num > 0):
      self.dactivations_prev = np.dot(np.transpose(self.weights), self.dz)
      self.prev_layer.backward_prop(self.dactivations_prev, l2_regu_coeff)

  # intialization of weight and bias matrix
  def initialize_parameters(self):
    self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])  * 0.01
    self.bias = np.zeros(self.bias.shape)

  def print_forward(self):
    print("Layer number: " + str(self.layer_num))  
    print("weights:")
    print(self.weights)
    print("bias:")
    print(self.bias)
    print("z:")
    print(self.z)
    print("activations:")
    print(self.activations)
  
  def print_backward(self):
    print("Layer number: " + str(self.layer_num)) 
    print("dz:")
    print(self.dz)
    print("dweights:")
    print(self.dweights)
    if (hasattr(self, "dweights_numerical")):
     print("dweights_numerical:")
     print(self.dweights_numerical)
    print("dbias:")
    print(self.dbias)
    if (hasattr(self, "dbias_numerical")):
     print("dbias_numerical:")
     print(self.dbias_numerical)
    if (self.layer_num > 0):
      print("dactivations_prev:")
      print(self.dactivations_prev)
    print("weights:")
    print(self.weights)
    print("bias:")
    print(self.bias)
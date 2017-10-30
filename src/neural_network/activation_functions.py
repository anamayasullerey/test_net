"""
Activation functions and their derivatives
"""
import numpy as np

class ActivationFunctions(object):

  @staticmethod
  def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
  
  @staticmethod
  def sigmoid_prime(z):
    return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))
  
  @staticmethod
  def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
  
  @staticmethod
  def softmax_prime(z):
    return ActivationFunctions.softmax(z) * (1 - ActivationFunctions.softmax(z))
  
  @staticmethod
  def tanh():
    return np.tanh(z)
  
  @staticmethod
  def tanh_prime(z):
    return 1 - ActivationFunctions.tanh(z) * ActivationFunctions.tanh(z)
  
  @staticmethod
  def relu(z):
    return np.maximum(z, 0)
  
  @staticmethod
  def relu_prime(z):
    return (z > 0).astype(float)

  @staticmethod
  def get_function(func_name):
    try: 
      func = getattr(ActivationFunctions, func_name)
      return func
    except :
      raise ValueError('Activation function "' + func_name + '" not defined')


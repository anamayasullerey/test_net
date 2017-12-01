"""
Loss functions and their derivatives

- Loss functions calculate overall loss
- Loss derivatives calculate loss per output neuron 
  as required by the backpropagation

  # y is the observed output for the training example
  # a is the predicted output
  
"""
import numpy as np

class LossFunctions(object):

  @staticmethod
  def sigmoid_cross_entropy_loss(y, a):
    return  - np.sum((y*np.log(a) + (1-y)*np.log(1-a)))/y.shape[1]
  
  @staticmethod
  def sigmoid_cross_entropy_loss_prime(y, a):
    return - (np.divide(y, a) - np.divide(1 - y, 1 - a))

  @staticmethod
  def softmax_cross_entropy_loss(y, a):
    return - np.sum(y*np.log(a))/y.shape[1]
  
  @staticmethod
  def softmax_cross_entropy_loss_prime(y, a):
    return - np.divide(y, a)

  @staticmethod
  def linear_mean_squared_loss(y, a):
    return np.sum(np.power((a-y), 2))/(2*y.shape[1])
  
  @staticmethod
  def linear_mean_squared_loss_prime(y, a):
    return (a-y)

  @staticmethod
  def get_function(func_name):
    try: 
      return getattr(LossFunctions, func_name)
    except :
      raise ValueError('Loss function "' + func_name + '" not defined')

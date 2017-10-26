"""
Weight update  functions  
"""
import numpy as np

class WeightUpdateFunctions(object):

  @staticmethod
  def gradient_descent(layer, params):
    layer.weights = layer.weights - params.learning_rate * layer.dweights
    layer.bias = layer.bias - params.learning_rate * layer.dbias

  @staticmethod
  def gradient_descent_init(layer):
    pass
      
  @staticmethod
  def momentum(layer, params):
    layer.weight_velocity = params.beta*layer.weight_velocity + (1-params.beta)*layer.dweights
    layer.bias_velocity = params.beta*layer.bias_velocity + (1-params.beta)*layer.dbias
    layer.weights = layer.weights - params.learning_rate * layer.weight_belocity
    layer.bias = layer.bias - params.learning_rate * layer.bias_velocity

  @staticmethod
  def momentum_init(layer):
    layer.weight_velocity = np.zeros(layer.weights.shape)
    layer.bias_velocity = np.zeros(layer.bias.shape)
    
  @staticmethod
  def adam(layer, params):
    # calculate velocity
    layer.weight_velocity = params.beta1*layer.weight_velocity + (1-params.beta1)*layer.dweights
    layer.bias_velocity = params.beta1*layer.bias_velocity + (1-params.beta1)*layer.dbias
    
    # calculate square of gradients
    layer.sdw = params.beta2*layer.sdw + (1 - params.beta2)*np.power(layer.dweights, 2)
    layer.sdb = params.beta2*layer.sdb + (1 - params.beta2)*np.power(layer.dbias, 2)
    
    # adjustment
    adj = 1/(1 - np.power(params.beta1, layer.t))
    weight_velocity_adj = layer.weight_velocity * adj
    bias_velocity_adj = layer.bias_velocity * adj
    sdw_adj = layer.sdw * adj
    sdb_adj = layer.sdb * adj
    
    layer.weight = layer.weight - params.learning_rate * weight_velocity_adj/np.sqrt(sdw_adj + params.epsilon)
    layer.bias = layer.bias - params.learning_rate * bias_velocity_adj/np.sqrt(sdb_adj + params.epsilon)
  
  @staticmethod
  def adam_init(layer):
    layer.weight_velocity = np.zeros(layer.weights.shape)
    layer.bias_velocity = np.zeros(layer.bias.shape)
    layer.sdw = np.zeros(layer.weights.shape)
    layer.sdb = np.zeros(layer.bias.shape)
   
  @staticmethod
  def get_function(func_name):
    try: 
      return getattr(WeightUpdateFunctions, func_name)
    except :
      raise ValueError('Weight update function "' + func_name + '" not defined')

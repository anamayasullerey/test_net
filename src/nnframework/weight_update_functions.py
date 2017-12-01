"""
Weight update  functions  
"""
import numpy as np

def gradient_descent(layer, wu_params):
  for param in layer.parameters:
    layer.__dict__[param] -= wu_params.learning_rate * layer.__dict__["d" + param]

def gradient_descent_init(layer):
  pass
    
def momentum(layer, wu_params):
  for param in layer.parameters:
    layer.velocity[param] = wu_params.beta*layer.velocity[param] + (1-wu_params.beta)*layer.__dict__["d" + param]
    layer.__dict__[param] -= wu_params.learning_rate * layer.velocity[param]

def momentum_init(layer):
  layer.velocity = {}
  layer.sq_grad = {}
  for param in layer.parameters:
    layer.velocity[param] = np.zeros(layer.__dict__[param].shape)
    layer.sq_grad[param] = np.zeros(layer.__dict__[param].shape)
  
def adam(layer, wu_params):
  adj = 1/(1 - np.power(wu_params.beta1, wu_params.t))
  wu_params.t += 1
  for param in layer.parameters:
    # calculate velocity
    layer.velocity[param] = wu_params.beta1*layer.velocity[param] + (1-wu_params.beta1)*layer.__dict__["d" + param]
    
    # calculate square of gradients
    layer.sq_grad[param] = wu_params.beta2*layer.sq_derivative[param] + (1 - wu_params.beta2)*np.power(layer.__dict_["d" + param], 2)
    
    # adjustment
    weight_velocity_adj = layer.velocity[param] * adj
    sq_grad_adj = layer.sq_grad[param] * adj
    
    layer.__dict__[param] -= wu_params.learning_rate * weight_velocity_adj/np.sqrt(sq_grad_adj + wu_params.epsilon)

def adam_init(layer):
  layer.velocity = {}
  layer.sq_grad = {}
  for param in layer.parameters:
    layer.velocity[param] =  np.zeros(layer.__dict__[param].shape)
    layer.sq_grad[param] =  np.zeros(layer.__dict__[param].shape)

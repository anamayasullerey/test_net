"""
Loss functions and their derivatives

- Loss functions calculate overall loss
- Loss derivatives calculate loss per output neuron
  as required by the backpropagation

  # y is the observed output for the training example
  # a is the predicted output

"""
import numpy as np

def sigmoid_cross_entropy_loss(y, a):
    return  - np.sum((y*np.log(a) + (1-y)*np.log(1-a)))/y.shape[1]

def sigmoid_cross_entropy_loss_prime(y, a):
    with np.errstate(divide='ignore', invalid='ignore'):
       res = -(np.true_divide(y, a) - np.true_divide(1 - y, 1 - a))
       res[~np.isfinite(res)] = 0  # -inf inf NaN
    return res   
    #return - (np.true_divide(y, a) - np.true_divide(1 - y, 1 - a))

def softmax_cross_entropy_loss(y, a):
    return - np.sum(y*np.log(a))/y.shape[1]

def softmax_cross_entropy_loss_prime(y, a):
    with np.errstate(divide='ignore', invalid='ignore'):
       res = -np.true_divide(y, a)
       res[~np.isfinite(res)] = 0  # -inf inf NaN
    return res   

def linear_mean_squared_loss(y, a):
    return np.sum(np.power((a-y), 2))/(2*y.shape[1])

def linear_mean_squared_loss_prime(y, a):
    return (a-y)

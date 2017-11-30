"""
fully connected layer (Hidden) of a neural network
"""

from . import layer
from . import loss_functions as lf

class LossLayer(layer.Layer):
  
  def __init__(self, loss_function_name):
    super().__init__()
    self.loss_func = lf.LossFunctions.get_function(loss_function_name)
    self.loss_func_prime = lf.LossFunctions.get_function(loss_function_name+ "_prime")

  def forward_calc(self, x):
    self.activations = x
    
  # backprop calculations
  def backward_deactivations(self):
    # Set the dactivation based on loss function
    self.prev_layer.dactivations = self.loss_func_prime(self.y, self.activations)

  def backward_prop(self, y):
    self.y = y
    super().backward_prop()

  # backprop calculations
  def loss(self, y):
    loss_value = self.loss_func(y, self.activations)
    return loss_value
 
  def print_backward(self):
    pass
"""
Input layer of a neural network
"""

class InputLayer(object):
  
  def __init__(self, num_inputs):
    self.num_neurons = num_inputs
  
  def forward_prop(self, input_activations):
    self.activations = input_activations
    self.next_layer.forward_prop(input_activations)

  def print_forward(self):
    print("Input layer")  
    print("activations:")
    print(self.activations)

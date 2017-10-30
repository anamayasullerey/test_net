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
   
  def set_loss_function(self, loss_func_name, l2_regu_coeff):
    self.loss_func = lf.LossFunctions.get_function(loss_func_name)
    self.loss_func_prime = lf.LossFunctions.get_function(loss_func_name + "_prime")
    self.l2_regu_coeff = l2_regu_coeff

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
    
  def backward_prop(self, y):
    self.dactivation = self.loss_func_prime(y, self.layers[-1].activations)
    self.layers[-1].backward_prop(self.dactivation, self.l2_regu_coeff)  
 
  def weight_update(self):
    for l in self.layers:
      self.weight_update_func(l)
      
  def train(self, x, y):
    self.forward_prop(x)
    self.backward_prop(y)
    self.weight_update()

  def loss(self, y):
    loss_value = self.loss_func(y, self.layers[-1].activations)
    if (self.l2_regu_coeff):
      l2_regu_loss = 0 
      for l in self.layers:
        l2_regu_loss += l2_regu_loss + (self.l2_regu_coeff*np.sum(np.square(l.weights)))
      loss_value += l2_regu_loss
    return loss_value  
      
  def predict(self, x):
    self.forward_prop(x)
    return self.layers[-1].activations

  def predict_classify(self, x):
    return np.argmax(self.predict(x))

  def print_state(self):
    print("Printing state")
    self.input_layer.print_forward()
    for i in range(len(self.layers)):
      self.layers[i].print_forward()
      self.layers[i].print_backward()  
    print(self.dactivation)
 
  def check_gradient(self, x, y, store=0):
    # weights are not updated during the check
    # first calculate the model gradients by running forward and backward pass
    self.forward_prop(x)
    self.backward_prop(y)

    # numerically calculate gradients for each parameter
    for l in self.layers:
      if store:
         l.dweights_numerical = np.zeros(l.weights.shape)
         l.dbias_numerical = np.zeros(l.bias.shape)

      for i in range(l.weights.shape[0]):
        for j in range(l.weights.shape[1]):
          indices = (i,j)
          numerical_grad = self.calculate_numerical_gradient(x, y, l.weights, indices)
          if (store):
            l.dweights_numerical[indices] = numerical_grad
          if not NeuralNetwork.error_chk(l.dweights[indices], numerical_grad):
            print ("Error: Gradient check failed for weight matrix for layer " + str(l.layer_num))
            print ("       index = " + str(indices))
            print ("       gradient = " + str(l.dweights[indices]))
            print ("       numerical gradient = " + str(numerical_grad))
      for i in range(l.bias.shape[0]):
          indices = (i, 0)
          numerical_grad = self.calculate_numerical_gradient(x, y, l.bias, indices)
          if (store):
            l.dbias_numerical[indices] = numerical_grad
          if not NeuralNetwork.error_chk(l.dbias[indices], numerical_grad):
            print ("Error: Gradient check failed for bias matrix for layer " + str(l.layer_num))
            print ("       index = " + str(indices))
            print ("       gradient = " + str(l.bias[indices]))
            print ("       numerical gradient = " + str(numerical_grad))
         
  def calculate_numerical_gradient(self, x, y, array, indices, epsilon=1e-7):
    orig_value = array[indices]
    array[indices] += epsilon
    self.forward_prop(x)
    cost_plus = self.loss(y)
    array[indices] -= 2*epsilon
    self.forward_prop(x)
    cost_minus = self.loss(y)
    array[indices] = orig_value
    return (cost_plus-cost_minus)/(2*epsilon)
          
  @staticmethod  
  def error_chk(value0, value1, margin_fraction=1e-4):
    numerator = np.linalg.norm(value0 - value1)                                     # Step 1'
    denominator = np.linalg.norm(value0) + np.linalg.norm(value1)                   # Step 2'
    return (numerator <= margin_fraction * denominator)

 
  
          
         
         
       
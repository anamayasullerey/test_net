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
    
  def backward_prop(self, da, weight_update_function):
    self.layers[-1].backward_prop(da, self.weight_update_func, self.l2_regu_coeff)  
    
  def train(self, x, y):
    self.forward_prop(x)
    print("Printing training step")
    self.input_layer.print_forward()
    for i in range(len(self.layers)):
      self.layers[i].print_forward()
    self.dactivation = self.loss_func_prime(y, self.layers[-1].activations)
    print("output dactivation")
    print(self.dactivation)
    self.backward_prop(self.dactivation, self.weight_update_func)
    for i in range(len(self.layers)-1, -1, -1):
     self.layers[i].print_backward()  

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
    print("Printing training step")
    self.input_layer.print_forward()
    for i in range(len(self.layers)):
      self.layers[i].print_forward()
    print("output dactivation")
    print(self.dactivation)
    for i in range(len(self.layers)-1, -1, -1):
     self.layers[i].print_backward()  

  def get_gradients(self, x, y, epsilon=1e-7):
    print("Numerically calculating gradients.")
    for l in range(len(self.layers)):
      print("layer: " +str(l))
      self.layers[l].dweights_numerical = np.zeros(self.layers[l].weights.shape)
      self.layers[l].dbias_numerical = np.zeros((self.layers[l].num_neurons, 1))
      for i in range(self.layers[l].weights.shape[0]):
        for j in range(self.layers[l].weights.shape[1]):
          print(i)
          print(j)
          orig_value = self.layers[l].weights[i][j]  
          self.layers[l].weights[i][j] += epsilon
          self.forward_prop(x)
          cost_plus = self.loss(y)
          self.layers[l].weights[i][j] -= 2*epsilon
          self.forward_prop(x)
          cost_minus = self.loss(y)
          self.layers[l].dweights_numerical[i][j] = (cost_plus-cost_minus)/(2*epsilon)
          self.layers[l].weights[i][j] = orig_value
        for i in range(self.layers[l].num_neurons):
          orig_value = self.layers[l].bias[i][0]
          self.layers[l].bias[i][0] += epsilon
          self.forward_prop(x)
          cost_plus = self.loss(y)
          self.layers[l].bias[i][0] -= 2*epsilon
          self.forward_prop(x)
          cost_minus = self.loss(y)
          self.layers[l].dbias_numerical[i][0] = (cost_plus-cost_minus)/(2*epsilon)
          self.layers[l].bias[i][0] = orig_value
      print("dweights_numerical")
      print(self.layers[l].dweights_numerical)  
      print("dbias_numerical")
      print(self.layers[l].dbias_numerical)
 
  def check_gradient(self, x, y):     
   self.get_gradients(x, y)
   self.train(x,y)
   for l in self.layers:
     w_diff = np.absolute(l.dweights_numerical - l.dweights)   
     w_aver = np.absolute(l.dweights_numerical + l.dweights)/2   
     if (np.any(w_diff > .001*w_aver)):
       print("Gradient check failed for weights matrix of layer " + str(l.layer_num))
     b_diff = np.absolute(l.dbias_numerical - l.dbias)   
     b_aver = np.absolute(l.dbias_numerical + l.dbias)/2   
     if (np.any(b_diff > .001*b_aver)):
       print("Gradient check failed for bias matrix of layer " + str(l.layer_num))
      
     
          
         
         
       
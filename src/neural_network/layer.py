"""
layer 
"""
class Layer(object):
  
  def __init__(self):
    self.layer_num = -1
    self.l2_loss_coeff = 0
    self.parameters = []

  def forward_calc(self, x):
    self.activations = self.act_func(x)

  def forward_prop(self, x):
    self.forward_calc(x)
    if self.next_layer is not None: 
      self.next_layer.forward_prop(self.activations)
      
  def backward_calc(self):
    pass  

  def backward_deactivations(self):
    self.prev_layer.dactivations = self.dactivations * self.act_func_prime(self.prev_layer.activations)

  def backward_prop(self):
    self.backward_calc()
    if (self.layer_num > 1):
      self.backward_deactivations () 
      self.prev_layer.backward_prop()

  def initialize_parameters(self):
    pass      

  def print_forward(self):
    print("Layer number: " + str(self.layer_num))  
    print("activations:")
    print(self.activations)
  
  def print_backward(self):
    print("Layer number: " + str(self.layer_num)) 
    if (self.layer_num > 0):
      print("dactivations:")
      print(self.dactivations)
    
  def set_l2_loss_coeff(self, l2_loss_coeff):
    self.l2_loss_coeff = l2_loss_coeff

  def get_l2_loss(self):
    return 0

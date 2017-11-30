"""
Weight update parameters  
"""

class GradientDescentParams(object):
  weight_update_func_name = "gradient_descent"  
  def __init__(self, learning_rate=0.05):
    self.learning_rate = learning_rate  

class MomentumParams(object):
  weight_update_func_name = "momentum"  
  def __init__(self, learning_rate=0.02, beta = 0.9):
    self.beta = beta  
    self.learning_rate = learning_rate
    
class AdamParams(object):
  weight_update_func_name = "adam"  
  def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.learning_rate = learning_rate
    self.beta1 = beta1 
    self.beta2 = beta2 
    self.epsilon = epsilon
    self.t = 0
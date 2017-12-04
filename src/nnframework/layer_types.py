"""
Activation functions and their derivatives
"""
from . import activation_functions as af
from . import fc_layer
from . import layer

stateless_propagation_names = ["sigmoid", "softmax", "tanh", "relu"]
ldict = {}

class stateless_layers(layer.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

# Create all stateless propagations
for prop in stateless_propagation_names:
    ldict[prop] = type(prop,
                       (stateless_layers,),
                       {"act_func":staticmethod(getattr(af, prop)),
                        "act_func_prime":staticmethod(getattr(af, prop + "_prime"))})

ldict["fc"] = fc_layer.FcLayer

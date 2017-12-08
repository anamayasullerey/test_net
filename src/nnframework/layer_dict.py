"""
layer dictionries
"""
from . import layer
from . import fc_layer_h
from . import relu_layer_h
from . import sigmoid_layer_h
from . import tanh_layer_h

from . import loss_layer_o
#from . import sigmoid_layer_o
#from . import softmax_layer_o

hdict = {}

hdict["fc"] = fc_layer_h.FcLayerH
hdict["relu"] = relu_layer_h.ReluLayerH
hdict["sigmoid"] = sigmoid_layer_h.SigmoidLayerH
hdict["tanh"] = tanh_layer_h.TanhLayerH

odict = {}

odict["loss"] = loss_layer_o.LossLayerO
#odict["sigmoid"] = sigmoid_layer_o.SigmoidLayerO
#odict["softmax"] = softmax_layer_o.SoftmaxLayerO


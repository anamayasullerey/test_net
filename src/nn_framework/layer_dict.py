"""
layer dictionries
"""
from . import layer
from . import fc_layer_h
from . import relu_layer_h
from . import sigmoid_layer_h
from . import tanh_layer_h

from . import loss_layer_o
from . import sigmoid_layer_o
from . import softmax_layer_o

hdict = {}

hdict["fc"] = fc_layer_h.FcLayer
hdict["relu"] = relu_layer_h.ReluLayer
hdict["sigmoid"] = sigmoid_layer_h.SigmoidLayer
hdict["tanh"] = tanh_layer_h.TanhLayer

odict = {}

odict["loss"] = loss_layer_o.LossLayer
odict["sigmoid"] = sigmoid_layer_o.SigmoidLayer
odict["softmax"] = softmax_layer_o.SoftmaxLayer

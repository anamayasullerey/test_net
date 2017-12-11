import unittest
import nn_grad_test as nt
import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup

class TestTrainLin1(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 1)

        layer = ld.hdict["fc"](1)
        net.add_layer(layer)

        layer = ld.odict["loss"]("linear_mean_squared_loss")
        net.add_layer(layer)

        np.random.seed(1)

        params = wup.GradientDescentParams(.01)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        net.set_l2_loss_coeff(.001)        

        a = 4.5;

        for i in range(100):
            x = (np.random.rand(1,32) - 0.5) * 10
            y = a * x
            net.train(x,y)
        
        x = 10
        self.assertTrue((np.absolute(net.predict(x) - a*x)/a*x) < 0.1)


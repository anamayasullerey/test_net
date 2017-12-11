import unittest
import nn_grad_test as nt
import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup

class TestTrainLin2(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 4)

        layer = ld.hdict["fc"](10)
        net.add_layer(layer)

        layer = ld.hdict["fc"](40)
        net.add_layer(layer)

        layer = ld.hdict["fc"](4)
        net.add_layer(layer)

        layer = ld.odict["loss"]("linear_mean_squared_loss")
        net.add_layer(layer)

        net.set_l2_loss_coeff(.001)        

        np.random.seed(1)

        params = wup.GradientDescentParams(.01)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        a = np.array([[1], [2], [3], [4]])

        for i in range(1000):
            x = (np.random.rand(4,32) - 0.5) * 10
            y = a * x
            net.train(x,y)
        
        x = np.array([[10], [10], [10], [10]])
        self.assertTrue(((np.absolute(net.predict(x) - a*x)/a*x) < 0.1).all())



import nn_test as nt
import numpy as np
import nnframework.neural_network as nn
import nnframework.layer_dict as ld
import nnframework.weight_update_params as wup

class test_x3_fc1_sigmo1(nt.NnTest):
        
    def define_nn(self):
        self.net = nn.NeuralNetwork("test_net", 1)

        self.layer = ld.hdict["fc"](1)
        self.net.add_layer(self.layer)

        self.layer = ld.odict["sigmoid"](1)
        self.net.add_layer(self.layer)

        np.random.seed(1)

        self.params = wup.GradientDescentParams(0)
        self.net.set_weight_update_function(self.params)
        self.net.initialize_parameters()
        self.net.layers[1].weights[0,0] = 10
    
    def set_training_example(self):
        self.x = np.array([[0.5]])
        self.y = np.array([[0.5]])

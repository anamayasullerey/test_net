## neural_network
This is the class that captures overall architecture of the neural network. neural_network module is imported by the following statement.
```
import nn_framework.neural_network as nn

```
A neural network object is created by the following statement.
```
net = nn.NeuralNetwork("name", number_of_inputs)
```
Once created, layers are sequentially added to the network from input to output. The final layer is the output layer.
```
<create a layer>
net.add_layer(layer)
```
A L2 loss coefficient can be specified in the following manner.
```
net.set_l2_loss_coeff(l2_loss_coefficient)
```
A weight update method is added to the net.
```
net.set_weight_update_function(weight_update_parameters)
```
At this stage the network is defined. It can be initialized (random initialization) with the following code
```
net.initialize_parameters()
```
The following code trains the net.
```
# x : input 2D numpy array of size (number of inputs * batch size)
# y : output 2D numpy array of size (number of outputs * batch size)
net.train(x, y) 
```
The following code is used to predict the outputs.
```
y = net.predict(x) 
```
Other useful functions are,
```
loss = net.loss(y) # Returns loss. Called after net.forward_prop(x) or net.predict(x)
y = net.predict_classify(x) # Returns the index for classification based outputs
status = net.check_gradient(x, y) # Returns boolean. Numerically checks the gradient calculations.
net.print_state()
```
## layers
Layers in nn_framework is a entity that specifies forward propagation and backward propagation methods. Every layer stores activations and corresponding input derivatives (dactivations). The layers are stored in two layer dictionaries, one for hidden layers (hdict) and one for output layers (ldict). Layer dictionaries are imported by the following statement.
```
    import nn_framework.layer_dict as ld
```

### input layer
Inputl layer is a module of nn_framwork but it is automatically generated when a network instance is created. Users do not have to worry about this layer.

### hidden layers
Listed below are the input layer types and the code to generate them.
* fully connected (y = wx + b)
```
layer = ld.hdict["fc"](number_of_neurons)
```
* relu
```
layer = ld.hdict["relu"](number_of_neurons)
```
* sigmoid
```
layer = ld.hdict["sigmoid"](number_of_neurons)
```
* tanh
```
layer = ld.hdict["tanh"](number_of_neurons)
```

**_Note that "layer" frequently represents a fully connected function followed by an activation function. In nn_framework these are separate layers._**

### output layers
Listed below are the output layer types and the code to generate them.
* loss: This is the generic output layer that has an identity activation function (y=x). A loss function is specified the when this output layer is created. Following loss functions are supported for loss output layer.
o sigmoid_cross_entropy_loss
o linear_mean_squared_loss
```
layer = ld.odict["loss"]("linear_mean_squared_loss")
```
* sigmoid: This layer has a sigmoid activation function as well a sigmoid cross entropy loss.
```
layer = ld.odict["sigmoid"](number_of_neurons)
```
* softmax: This layer has a softmax activation function with a softmax cross entropy loss. When using this layer the outputs need to be logits (one hot binary set).
```
layer = ld.odict["softmax"](number_of_neurons)
```


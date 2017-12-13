# Introduction
nn_framework is a basic framework for beginner level AI learners to experiment with neural network concepts. It is inspired by and covers some of the concepts covered in first three courses of the deep learning specialization by [Prof Andrew Ng](http://www.andrewng.org/ )  ([deeplearning.ai](https://www.deeplearning.ai/)). This framework tries to make the components plug and play.

# Modules
## layers
Layers in nn_framework is a entity that specifies forward propagation and backward propagation methods. Every layer stores activations and corresponding input derivatives (dactivations). The layers are stored in two layer dictionaries, one for hidden layers (hdict) and one for output layers (ldict). Layer dictionaries are  imported by the following statement.
```
    import nn_framework.layer_dict as ld
```

### input layer
Inputl layer is a module of nn_framwork but it is automatically generated when a network instance is created. Users do not have to worry about this layer.

### hidden layers
Listed below are the input layer types and the code to generate them.
* fully connected (y = wx + b)
```
layer = ld.hdict["fc"](num_neurons) |
```
* relu
```
layer = ld.hdict["relu"](num_neurons)
```
* sigmoid
```
layer = ld.hdict["sigmoid"](num_neurons) |
```
* tanh
```
layer = ld.hdict["tanh"](num_neurons) |
```

**_Note that "layer" frequently represents a fully connected function followed by an activation function. In nn_framework these are separate layers. _**

### hidden layers
Listed below are the output layer types and the code to generate them.
* loss: This is the generic output layer that has an identity activation function (y=x). A loss function is specified the when this output layer is created. Following loss functions are supported for loss output layer.
o sigmoid_cross_entropy_loss
o linear_mean_squared_loss
```
self.layer = ld.odict["loss"]("linear_mean_squared_loss")
```
* sigmoid: This layer has a sigmoid activation function as well a sigmoid cross entropy loss.
* softmax: This layer has a softmax activation function with a softmax cross entropy loss. When using this layer the outputs need to be logits (one hot bit vectors)

## neural_network
This is the class that captures overall architecture of the neural network. neural_network module can 

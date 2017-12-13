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
Listed below are the input layer types and the code to generate them.  many frameworks represent a layer as a compo
* fully connected : Wx + b 
```
layer = ld.hdict\["fc"\](num_neurons) |
```
* relu :
```
layer = ld.hdict\["relu"\](num_neurons)
```
* sigmoid : 
```
layer = ld.hdict\["sigmoid"\](num_neurons) |
```
* tanh : tanh(x) | layer = ld.hdict\["tanh"\](num_neurons) |
```
layer = ld.hdict\["tanh"\](num_neurons) |
```

**_Note that "layer" frequently represents a fully connected function followed by an activation function. In nn_framework these are separate layers. _**

The following table lists out the output layers.

|  layer | activation | loss function |  code to create the layer |
|---------|--------------|-------------------------|----------------|
| sigmoid | sigmoid(x) | sigmoid cross entropy loss | ld.odict\["sigmoid"\](num_neurons)
| softmax | softmax(x) | softmax cross entropy loss | ld.odict\["softmax"\](num_neurons)
| loss | None | selectable | ld.odict\["loss"\](num_neurons)



## neural_network
This is the class that captures overall architecture of the neural network. neural_network module can 

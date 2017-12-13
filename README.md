# Introduction
nn_framework is a basic framework for beginner level AI learners to experiment with neural network concepts. It is inspired by and covers some of the concepts covered in first three courses of the deep learning specialization by [Prof Andrew Ng](http://www.andrewng.org/ )  ([deeplearning.ai](https://www.deeplearning.ai/)). This framework tries to make the components plug and play.

# Components
## layers
Layers in nn_framework is a entity that specifies forward propagation and backward propagation methods. Every layer stores activations and corresponding input derivatives (dactivations). The layers are stored in two layer dictionaries, one for hidden layers (hdict) and one for output layers (ldict). Layer dictionaries can be imported by the following statement.

    import nn_framework.layer_dict as ld

The following table lists out the hidden layers.

|  layer | activation | code to create the layer |
|---------|--------------|-------------------------|
| fully connected | Wx + b | layer = ld.hdict\["fc"\](num_neurons) |
| relu | relu(x) | layer = ld.hdict\["relu"\](num_neurons) |
| sigmoid | sigmoid(x) | layer = ld.hdict\["sigmoid"\](num_neurons) |
| tanh | tanh(x) | layer = ld.hdict\["tanh"\](num_neurons) |


## neural_network
This is the class that captures overall architecture of the neural network

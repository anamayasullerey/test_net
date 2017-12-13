# Introduction
nn_framework is designed for beginner level AI learners to experiment with neural network concepts. It is inspired by and covers the concepts covered in first three courses of the deep learning specialization by [Prof Andrew Ng](http://www.andrewng.org/ )  ([deeplearning.ai](https://www.deeplearning.ai/)).

# How to start
After downloading the git repository **"nnf/src"** directory needs to be added to **$PYTHONPATH**.

## 1. MNIST digit classifier example 
MNIST digit classfier example using nn_framework gives a feel of the architecture of nn_framework. It is recommended to follow the documentation and create a classifier in Jupyter notebook. The example, **mnist_net.py** is also checked in as python module and can be run in a shell.

## 2. [nn_framework documentation](nn_framework_manual.md)
The documentation on nn_framework explains the featrues of the nn_framework. It also covers the important functions of the API.

# Modifying nn_framework
## Architecture of nn_framework
nn_framework builds a network as an array of layers. There layers have handles to their previous and next layers. Forward propagation and backward propagation is done as a recursion across the layers. Weight update functionality is separated from the layers. The state for weight update is maintained within the layers.

## Unit testing
nn_framework comes with unit tests. These tests are done on very small scale nets which can be debugged manually. It is **highly recommended** to add unit tests for any new features. The unit tests that come with the framework are divided into two categories.

1. **Gradient Tests** : These tests check the gradients of various combinations of layers and cost functs
2. **Training Tests** : These tests train the network and check the preditions of the network. Training data for these tests is generated by a model function.

The tests can be run with the following command in the tests directory
```
python -m unittest discover -v
```

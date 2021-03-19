# TinyNeuralNet
An implementation of a Neural Network in C++


## Requirements
* C++17
* Eigen
* MNIST Handwritten Digits data file

## Design

This is an implementation of a neural network as designed based off 3blue1brown videos. It uses `Eigen` to do the linear algebra, and identifies 28x28 pixel handwritten digits.
It currently supports three different activation functions: sigmoid, tanh and relu. The network can be made up of arbitrarily many dense layers.

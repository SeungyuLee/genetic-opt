# Machine Learning Optimization using DEAP library

## Network

This network is trained to recognize MNIST data. It has one input, one output layer and 3 hidden layers.
Optimization is for the number of neurons for each hidden layer, momentum value, and regularization(lambda) value.

## Example

python brute.py # brute-force algorithm to find optimal hyper-parameters
python genetic.py # genetic algorithm to find optimal hyper-parameters

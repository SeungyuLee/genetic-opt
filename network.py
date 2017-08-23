"""
stochastic gradient descent, backpropagation
"""

import random, sys, json
import numpy as np

class CrossEntropyCost(object):
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)

class Network(object):

	def __init__(self, sizes, cost=CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.cost=cost
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
		self.weights = [np.random.randn(y, x)/np.sqrt(x)
						for x, y in zip(sizes[:-1], sizes[1:])]
		self.vb = [np.zeros(b.shape) for b in self.biases]
		self.vw = [np.zeros(w.shape) for w in self.weights]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, 
			lmbda=0.0, 
			momentum=0.0,
			evaluation_data=None,
			monitor_evaluation_accuracy=False):

		if evaluation_data: n_data = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size]
							for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(
						mini_batch, eta, lmbda, momentum, len(training_data))
			"""
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data: {} / {}".format(
						self.accuracy(evaluation_data), n_data)
			"""
		return self.accuracy(evaluation_data) / float(n_data)

#		result = self.evaluate(evaluation_data) / float(n_test)
#		return result

	def update_mini_batch(self, mini_batch, eta, lmbda, momentum, n):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		
		self.vw = [momentum*vw-(eta/len(mini_batch))*nw
					for vw, nw in zip(self.vw, nabla_w)]
		self.weights = [(1-eta*(lmbda/n))*w+vw
						for w, vw in zip(self.weights, self.vw)]
		self.vb = [momentum*vb-(eta/len(mini_batch))*nb
					for vb, nb in zip(self.vb, nabla_b)]
		self.biases = [b+vb for b, vb in zip(self.biases, self.vb)]
		
		"""
		self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
						for b, nb in zip(self.biases, nabla_b)]
		"""

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = (self.cost).delta(zs[-1], activations[-1], y)
#		delta = self.cost_derivative(activations[-1], y) * \
#			sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def accuracy(self, data, convert=False):
		if convert:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
		return sum(int(x==y) for (x, y) in results)

	def total_cost(self, data, lmbda, convert=False):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a, y)/len(data)
		cost += 0.5*(lmbda/len(data))*sum(
			np.linalg.norm(w)**2 for w in self.weights)
		return cost

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
	return np.tanh(z)

def tanh_prime(z):
	return 1-(np.tanh(z)**2)

def relu(z):
	return np.maximum(x, 0.0, x)

def relu_prime(z):
	return np.where(x > 0, 1.0, 0.0)


import mnist_loader

def execute(x, y, z, a, b, file_name):
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	net = Network([784, x, y, z, 10])
	accuracy = net.SGD(training_data, 10, 10, 0.2, lmbda=a, momentum=b, evaluation_data=test_data, monitor_evaluation_accuracy=False)
	data = {"layer 1": x, "layer 2": y, "layer 3": z, "lmbda": a, "momentum": b, "accuracy": accuracy}
	f = open(file_name, "a")
	json.dump(data, f)
	f.write("\n")
	f.close()
	return accuracy
